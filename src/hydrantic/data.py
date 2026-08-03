from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import lightning
import torch
from pydantic import field_validator, model_validator
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, random_split

from hydrantic.hparams import Hparams, HparamsModule


class DataHparams(Hparams):
    """Base hyperparameters for `DataModule`."""

    split: tuple[float, float] | tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42

    loader: dict[str, Any] = {"batch_size": 64}
    train_loader: dict[str, Any] = {}
    val_loader: dict[str, Any] = {}
    test_loader: dict[str, Any] = {}

    @field_validator("split")
    @classmethod
    def _split_sums_to_one(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if abs(sum(value) - 1.0) > 1e-9:
            raise ValueError(f"split must sum to 1, got {sum(value)}")
        if any(fraction < 0 for fraction in value):
            raise ValueError(f"split fractions must be non-negative, got {value}")
        return value

    @field_validator("loader")
    @classmethod
    def _no_shuffle_in_shared_loader(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "shuffle" in value:
            raise ValueError(
                "`shuffle` is stage specific: set it in `train_loader`, `val_loader` or "
                "`test_loader`, not in the shared `loader` block"
            )
        return value

    @model_validator(mode="after")
    def _persistent_workers_need_workers(self) -> "DataHparams":
        for name in ("loader", "train_loader", "val_loader", "test_loader"):
            kwargs = getattr(self, name)
            if not kwargs.get("persistent_workers"):
                continue
            workers = kwargs.get("num_workers", self.loader.get("num_workers", 0))
            if not workers:
                raise ValueError(f"{name}.persistent_workers=True requires num_workers > 0")
        return self


D = TypeVar("D", bound=DataHparams)


class DataModule(HparamsModule[D], lightning.LightningDataModule):
    """A `LightningDataModule` configured by a validated `DataHparams` schema.

    Implement `get_dataset` to have one dataset split by `thparams.split`, or override
    `get_split_datasets` when the splits already exist separately. Datasets are built in
    `setup`, which Lightning calls on every rank once the process group exists; put downloads
    and other rank-zero work in `prepare_data`."""

    # The model wins the logger; these hyperparameters still go into the checkpoint, and the
    # hydrantic CLI logs the whole resolved configuration anyway.
    log_hparams_to_logger = False

    def __init__(self, hparams: D | dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Stores the hyperparameters without touching the data.

        :param hparams: A `DataHparams` instance, or a mapping of field values.
        :param kwargs: Field values, as an alternative to passing a mapping."""

        super().__init__(hparams, **kwargs)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def get_dataset(self) -> Dataset:
        """Returns the full dataset, to be split according to `thparams.split`.

        :return: The full dataset."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement `get_dataset`, or override "
            f"`get_split_datasets` if the splits already exist separately."
        )

    def get_split_datasets(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Returns the train, validation and test datasets.

        By default this splits `get_dataset` by `thparams.split`. Override it when the splits
        already exist separately, in which case `get_dataset` need not be implemented.

        :return: The train, validation and optional test datasets."""

        subsets = split_dataset(self.get_dataset(), self.thparams.split, self.thparams.seed)
        return subsets[0], subsets[1], subsets[2] if len(subsets) > 2 else None

    def setup(self, stage: str) -> None:
        """Builds the datasets. Called by Lightning on every rank.

        :param stage: The stage Lightning is preparing for."""

        if self.train_dataset is None:
            self.train_dataset, self.val_dataset, self.test_dataset = self.get_split_datasets()

    def train_dataloader(self) -> DataLoader | None:
        """Returns the training dataloader.

        :return: The dataloader, or None if there is no training split."""

        return self._build_dataloader("train")

    def val_dataloader(self) -> DataLoader | None:
        """Returns the validation dataloader.

        :return: The dataloader, or None if there is no validation split."""

        return self._build_dataloader("val")

    def test_dataloader(self) -> DataLoader | None:
        """Returns the test dataloader.

        :return: The dataloader, or None if there is no test split."""

        return self._build_dataloader("test")

    def _build_dataloader(self, stage: Literal["train", "val", "test"]) -> DataLoader | None:
        """Builds one stage's dataloader from the shared and per-stage loader settings.

        :param stage: The stage to build the dataloader for.
        :return: The dataloader, or None if the stage has no dataset."""

        if (dataset := getattr(self, f"{stage}_dataset")) is None:
            return None

        kwargs: dict[str, Any] = {"shuffle": stage == "train"}
        kwargs |= self.thparams.loader
        kwargs |= getattr(self.thparams, f"{stage}_loader")

        if isinstance(dataset, IterableDataset):
            kwargs.pop("shuffle", None)

        return DataLoader(dataset, **kwargs)


def split_dataset(
    dataset: Dataset,
    split: Sequence[float],
    seed: int,
) -> list[Subset]:
    """Splits a dataset into fractions, reproducibly and independently of the global RNG.

    A local generator is used rather than seeding the global RNG, so the split is identical on
    every rank no matter how many random numbers were drawn beforehand.

    :param dataset: The dataset to split.
    :param split: The split fractions, which must sum to 1.
    :param seed: The seed used for the split.
    :return: The subsets, in the order of `split`."""

    if abs(sum(split) - 1.0) > 1e-9:
        raise ValueError(f"split must sum to 1, got {sum(split)}")
    if isinstance(dataset, IterableDataset):
        raise TypeError(
            "Cannot fraction-split an IterableDataset; override `get_split_datasets` instead."
        )

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, list(split), generator=generator)
