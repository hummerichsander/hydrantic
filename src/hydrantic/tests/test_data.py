import pytest
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from ..data.data import PyTorchData
from ..data.hparams import DataHparams, DataLoaderHparams


class TestPyTorchDataInitialization:
    """Test PyTorchData initialization and basic functionality."""

    def test_basic_initialization(self, mock_pytorch_data, sample_tensor_dataset, data_hparams):
        """Test basic PyTorchData initialization.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param data_hparams: DataHparams fixture."""
        data = mock_pytorch_data(data_hparams, sample_tensor_dataset)

        assert data.thparams == data_hparams
        assert len(data.split) == 3
        assert data.train_loader is not None
        assert data.val_loader is not None
        assert data.test_loader is not None

    def test_initialization_without_test_split(
        self, mock_pytorch_data, sample_tensor_dataset, simple_data_hparams
    ):
        """Test initialization with two-way split.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param simple_data_hparams: Simple DataHparams fixture."""
        data = mock_pytorch_data(simple_data_hparams, sample_tensor_dataset)

        assert len(data.split) == 2
        assert data.train_loader is not None
        assert data.val_loader is not None
        assert data.test is None
        assert data.test_loader is None

    def test_initialization_without_loader(self, mock_pytorch_data, sample_tensor_dataset):
        """Test initialization without dataloader configuration.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture."""
        hparams = DataHparams(loader=None, seed=42, split=(0.8, 0.2))
        data = mock_pytorch_data(hparams, sample_tensor_dataset)

        assert data.train_loader is None
        assert data.val_loader is None
        assert data.test_loader is None


class TestPyTorchDataSplitting:
    """Test dataset splitting functionality."""

    def test_split_dataset_three_way(self, sample_tensor_dataset):
        """Test three-way dataset split.

        :param sample_tensor_dataset: Sample dataset fixture."""
        splits = PyTorchData.split_dataset(
            dataset=sample_tensor_dataset, dataset_split=(0.7, 0.2, 0.1), dataset_split_seed=42
        )

        assert len(splits) == 3
        assert len(splits[0]) == 70
        assert len(splits[1]) == 20
        assert len(splits[2]) == 10

    def test_split_dataset_two_way(self, sample_tensor_dataset):
        """Test two-way dataset split.

        :param sample_tensor_dataset: Sample dataset fixture."""
        splits = PyTorchData.split_dataset(
            dataset=sample_tensor_dataset, dataset_split=(0.8, 0.2), dataset_split_seed=123
        )

        assert len(splits) == 2
        assert len(splits[0]) == 80
        assert len(splits[1]) == 20

    def test_split_reproducibility(self, sample_tensor_dataset):
        """Test that splits are reproducible with same seed.

        :param sample_tensor_dataset: Sample dataset fixture."""
        splits1 = PyTorchData.split_dataset(
            dataset=sample_tensor_dataset, dataset_split=(0.7, 0.3), dataset_split_seed=42
        )
        splits2 = PyTorchData.split_dataset(
            dataset=sample_tensor_dataset, dataset_split=(0.7, 0.3), dataset_split_seed=42
        )

        # Check that indices are the same
        assert splits1[0].indices == splits2[0].indices
        assert splits1[1].indices == splits2[1].indices

    def test_split_randomness(self, sample_tensor_dataset):
        """Test that different seeds produce different splits.

        :param sample_tensor_dataset: Sample dataset fixture."""
        splits1 = PyTorchData.split_dataset(
            dataset=sample_tensor_dataset, dataset_split=(0.7, 0.3), dataset_split_seed=42
        )
        splits2 = PyTorchData.split_dataset(
            dataset=sample_tensor_dataset, dataset_split=(0.7, 0.3), dataset_split_seed=999
        )

        assert splits1[0].indices != splits2[0].indices

    def test_split_invalid_fractions(self, sample_tensor_dataset):
        """Test that invalid split fractions raise error.

        :param sample_tensor_dataset: Sample dataset fixture."""
        with pytest.raises(ValueError, match="must sum to 1"):
            PyTorchData.split_dataset(
                dataset=sample_tensor_dataset, dataset_split=(0.5, 0.3), dataset_split_seed=42
            )


class TestPyTorchDataProperties:
    """Test PyTorchData properties and accessors."""

    def test_train_property(self, mock_pytorch_data, sample_tensor_dataset, data_hparams):
        """Test train dataset property.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param data_hparams: DataHparams fixture."""
        data = mock_pytorch_data(data_hparams, sample_tensor_dataset)

        assert len(data.train) == 70
        assert isinstance(data.train, Dataset)

    def test_val_property(self, mock_pytorch_data, sample_tensor_dataset, data_hparams):
        """Test validation dataset property.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param data_hparams: DataHparams fixture."""
        data = mock_pytorch_data(data_hparams, sample_tensor_dataset)

        assert len(data.val) == 20
        assert isinstance(data.val, Dataset)

    def test_test_property(self, mock_pytorch_data, sample_tensor_dataset, data_hparams):
        """Test test dataset property.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param data_hparams: DataHparams fixture."""
        data = mock_pytorch_data(data_hparams, sample_tensor_dataset)

        assert len(data.test) == 10
        assert isinstance(data.test, Dataset)

    def test_test_property_none(
        self, mock_pytorch_data, sample_tensor_dataset, simple_data_hparams
    ):
        """Test test property returns None for two-way split.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param simple_data_hparams: Simple DataHparams fixture."""
        data = mock_pytorch_data(simple_data_hparams, sample_tensor_dataset)

        assert data.test is None

    def test_full_property(self, mock_pytorch_data, sample_tensor_dataset, data_hparams):
        """Test full dataset property.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param data_hparams: DataHparams fixture."""
        data = mock_pytorch_data(data_hparams, sample_tensor_dataset)

        assert len(data.full) == 100
        assert isinstance(data.full, Dataset)


class TestPyTorchDataLoaders:
    """Test DataLoader configuration and properties."""

    def test_train_loader_configuration(
        self, mock_pytorch_data, sample_tensor_dataset, data_hparams
    ):
        """Test train loader configuration.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture.
        :param data_hparams: DataHparams fixture."""
        data = mock_pytorch_data(data_hparams, sample_tensor_dataset)

        assert isinstance(data.train_loader, DataLoader)
        assert data.train_loader.batch_size == 32
        assert data.train_loader.dataset == data.train

    def test_loader_batch_iteration(self, mock_pytorch_data, sample_tensor_dataset):
        """Test iterating through loader batches.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture."""
        hparams = DataHparams(
            loader=DataLoaderHparams(batch_size=10, shuffle=False), seed=42, split=(0.8, 0.2)
        )
        data = mock_pytorch_data(hparams, sample_tensor_dataset)

        batches = list(data.train_loader)
        assert len(batches) == 8  # 80 samples / 10 batch_size

        x_batch, y_batch = batches[0]
        assert x_batch.shape == (10, 10)
        assert y_batch.shape == (10,)

    def test_loader_drop_last(self, mock_pytorch_data, sample_tensor_dataset):
        """Test drop_last parameter.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture."""
        hparams = DataHparams(
            loader=DataLoaderHparams(batch_size=15, drop_last=True), seed=42, split=(0.8, 0.2)
        )
        data = mock_pytorch_data(hparams, sample_tensor_dataset)

        batches = list(data.train_loader)
        assert len(batches) == 5  # 80 // 15, dropping remainder

    def test_loader_shuffle(self, mock_pytorch_data, sample_tensor_dataset):
        """Test shuffle parameter.

        :param mock_pytorch_data: Mock PyTorchData class fixture.
        :param sample_tensor_dataset: Sample dataset fixture."""
        hparams = DataHparams(
            loader=DataLoaderHparams(batch_size=10, shuffle=True), seed=42, split=(0.8, 0.2)
        )
        data = mock_pytorch_data(hparams, sample_tensor_dataset)

        # Shuffle affects iteration order but not dataset size
        batches = list(data.train_loader)
        assert len(batches) == 8


class TestPyTorchDataCustomization:
    """Test PyTorchData customization hooks."""

    def test_pre_init_hook(self, sample_tensor_dataset, data_hparams):
        """Test pre_init hook is called.

        :param sample_tensor_dataset: Sample dataset fixture.
        :param data_hparams: DataHparams fixture."""

        class CustomPyTorchData(PyTorchData):
            hparams_schema = DataHparams

            def __init__(self, thparams: DataHparams, dataset: Dataset):
                self._dataset = dataset
                self.pre_init_called = False
                super().__init__(thparams)

            def get_dataset(self) -> Dataset:
                return self._dataset

            def pre_init(self) -> None:
                self.pre_init_called = True

        data = CustomPyTorchData(data_hparams, sample_tensor_dataset)
        assert data.pre_init_called is True

    def test_get_dataset_must_be_implemented(self, data_hparams):
        """Test that get_dataset raises NotImplementedError.

        :param data_hparams: DataHparams fixture."""

        class IncompleteData(PyTorchData):
            hparams_schema = DataHparams

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteData(data_hparams)  # type: ignore
