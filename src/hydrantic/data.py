from typing import Any, TypeVar

import lightning
from pydantic import field_validator
from torch.utils.data import Dataset, Subset

from hydrantic.hparams import Hparams, HparamsModule
from hydrantic.utils import split_dataset


class SplitHparams(Hparams):
    """Opt-in hyperparameters for splitting one dataset into train/validation/test.

    Inherit this instead of `Hparams` when your datamodule splits a single dataset. Datamodules
    that load their splits from separate files should not: they would carry two fields they
    never read, and both would land in the checkpoint and the logged configuration."""

    split: tuple[float, float] | tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42

    @field_validator("split")
    @classmethod
    def _split_sums_to_one(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if abs(sum(value) - 1.0) > 1e-9:
            raise ValueError(f"split must sum to 1, got {sum(value)}")
        if any(fraction < 0 for fraction in value):
            raise ValueError(f"split fractions must be non-negative, got {value}")
        return value

    def split_dataset(self, dataset: Dataset) -> list[Subset]:
        """Splits a dataset using these fractions and seed.

        :param dataset: The dataset to split.
        :return: One subset per entry in `split`."""

        return split_dataset(dataset, self.split, self.seed)


class LoaderHparams(Hparams):
    """Opt-in hyperparameters for a PyTorch `DataLoader`.

    Inherit this to get the settings almost every datamodule needs as typed, validated fields,
    then hand them to a `DataLoader` with `loader_kwargs`. Anything not covered here goes in
    your own schema and straight into your `DataLoader` call."""

    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False

    def loader_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Returns these settings as `DataLoader` keyword arguments.

        `shuffle` is deliberately not a field: it differs per stage, and a baked-in default is
        how a training loader silently starts or stops shuffling. Pass it explicitly::

            DataLoader(self.train, **self.thparams.loader_kwargs(shuffle=True))

        :param overrides: Extra or replacement keyword arguments.
        :return: Keyword arguments for `DataLoader`."""

        # Field list comes from this class, not the instance, so a subclass's own
        # hyperparameters are not smuggled into the DataLoader call.
        settings = {name: getattr(self, name) for name in LoaderHparams.model_fields}
        return settings | overrides


H = TypeVar("H", bound=Hparams)


class DataModule(HparamsModule[H], lightning.LightningDataModule):
    """A `LightningDataModule` with a validated `Hparams` schema on `thparams`.

    That is all it adds. Implement `prepare_data`, `setup` and the `*_dataloader` hooks exactly
    as you would in plain Lightning. Compose `SplitHparams` and `LoaderHparams` into your schema
    for the common cases; `hydrantic.split_dataset` does the split.

    Build datasets in `setup`, not `__init__`: Lightning calls it on every rank once the process
    group exists, and once per stage, so guard the work if it is expensive. Downloads belong in
    `prepare_data`, which runs on one process per node::

        class MyHparams(SplitHparams, LoaderHparams):
            path: str

        class MyData(DataModule[MyHparams]):
            def setup(self, stage):
                if getattr(self, "train", None) is not None:
                    return
                dataset = torch.load(self.thparams.path)
                self.train, self.val, self.test = self.thparams.split_dataset(dataset)

            def train_dataloader(self):
                return DataLoader(self.train, **self.thparams.loader_kwargs(shuffle=True))

            def val_dataloader(self):
                return DataLoader(self.val, **self.thparams.loader_kwargs())
    """

    # Lightning merges a module's and a datamodule's `hparams_initial` and raises if a shared
    # key holds different values; both are keyed `hparams`, so the model wins the logger. These
    # hyperparameters still reach the checkpoint, and the CLI logs the whole resolved config.
    log_hparams_to_logger = False
