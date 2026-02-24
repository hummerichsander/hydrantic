from typing import TypeVar

from abc import abstractmethod, ABC

import torch
from torch.utils.data import Subset, Dataset, ConcatDataset, DataLoader, random_split

from .hparams import DataHparams
from ..hparams.module import HparamsModule


_T = TypeVar("_T")
H = TypeVar("H", bound=DataHparams)


class PyTorchData(HparamsModule[H], ABC):
    """This is the base class for all torch datasets. Internally it handles the splitting of the
    dataset into train, validation (and test) sets and makes the datasets and corresponding loaders
    available as properties.
    """

    def __init__(self, hparams: DataHparams | dict):
        super().__init__(hparams)

        self.pre_init()
        self.split = self._configure_split()

        self.train_loader = self._configure_dataloader(self.train)
        self.val_loader = self._configure_dataloader(self.val)
        self.test_loader = self._configure_dataloader(self.test)

    @abstractmethod
    def get_dataset(self) -> Dataset[object]:
        """Configures the dataset. Must be implemented in subclasses of Data and should return an
        instance of Dataset.

        :return: The configured dataset."""
        raise NotImplementedError("configure_dataset must be implemented in subclasses of Data")

    def _configure_split(self) -> list[Subset[object]]:
        """Configures the dataset from the module name and kwargs specified in the hparams.
        Furthermore splits the dataset into the specified fractions with the specified random seed.

        :return: The split data subsets"""

        dataset = self.get_dataset()
        return self.split_dataset(dataset, self.thparams.split, self.thparams.seed)

    def _configure_dataloader(self, dataset: Dataset[object] | None) -> DataLoader[object] | None:
        """Configures the dataloader from the module name and kwargs specified in the hparams.


        :dataset: The dataset to use for the dataloader.
        :return: The configured dataloader."""

        if dataset is None:
            return None

        if self.thparams.loader is None:
            return None

        return DataLoader(
            dataset,
            batch_size=self.thparams.loader.batch_size,
            shuffle=self.thparams.loader.shuffle,
            pin_memory=self.thparams.loader.pin_memory,
            num_workers=self.thparams.loader.num_workers,
            persistent_workers=self.thparams.loader.persistent_workers,
            drop_last=self.thparams.loader.drop_last,
        )

    @property
    def full(self) -> Dataset[object]:
        """Returns the full dataset.

        :return: The full dataset."""

        return ConcatDataset(self.split)

    @property
    def train(self) -> Dataset[object]:
        """Returns the train dataset.

        :return: The train dataset."""

        return self.split[0]

    @property
    def val(self) -> Dataset[object]:
        """Returns the validation dataset.

        :return: The validation dataset."""

        return self.split[1]

    @property
    def test(self) -> Dataset[object] | None:
        """Returns the test dataset. If no test dataset is specified, returns None.

        :return: The test dataset."""

        if len(self.split) < 3:
            return None
        return self.split[2]

    @staticmethod
    def split_dataset(
        dataset: Dataset[object],
        dataset_split: tuple[float, float] | tuple[float, float, float],
        dataset_split_seed: int,
    ) -> list[Subset[object]]:
        """Split the dataset into train, validation (and test) sets.

        :param dataset: The dataset to split.
        :param dataset_split: The split fractions.
        :param dataset_split_seed: The random seed to use for splitting.

        :return: The train, validation and test sets."""

        if sum(dataset_split) != 1:
            raise ValueError("dataset_split must sum to 1")

        with torch.random.fork_rng():
            torch.manual_seed(dataset_split_seed)
            split = random_split(dataset, dataset_split)

        return split

    def pre_init(self) -> None:
        """Pre-initializes the dataset. This method is called before the dataset is initialized and
        can be used to perform any necessary setup.

        :return: None"""
        pass
