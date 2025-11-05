import pytest
import torch
from pathlib import Path
from torch.utils.data import Dataset, TensorDataset

from ..hparams import Hparams
from ..data.hparams import DataHparams, DataLoaderHparams
from ..data.data import PyTorchData
from ..model.hparams import (
    ModelHparams,
    OptimizerHparams,
    SchedulerHparams,
    CheckpointHparams,
    EarlyStoppingHparams,
)
from ..model.model import Model


@pytest.fixture
def device() -> torch.device:
    """Provide a torch device for testing.

    :return: CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_config_path(tmp_path: Path) -> Path:
    """Provide a temporary directory for config files.

    :param tmp_path: Pytest's temporary directory fixture.
    :return: Path to temporary config directory."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    return config_dir


@pytest.fixture
def tmp_download_path(tmp_path: Path) -> Path:
    """Provide a temporary directory for download tests.

    :param tmp_path: Pytest's temporary directory fixture.
    :return: Path to temporary download directory."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(exist_ok=True)
    return download_dir


@pytest.fixture
def sample_hparams():
    """Provide a sample Hparams dataclass for testing.

    :return: Sample Hparams instance."""

    class SampleHparams(Hparams):
        learning_rate: float = 1e-3
        batch_size: int = 32
        hidden_dim: int = 128
        dropout: float = 0.1

    return SampleHparams()


@pytest.fixture
def nested_hparams():
    """Provide nested Hparams for testing hierarchical structures.

    :return: Nested Hparams instance."""

    class OptimizerHparams(Hparams):
        lr: float = 1e-3
        weight_decay: float = 1e-4

    class ModelHparams(Hparams):
        hidden_dim: int = 256
        num_layers: int = 3

    class NestedHparams(Hparams):
        optimizer: OptimizerHparams
        model: ModelHparams
        batch_size: int = 64

    return NestedHparams(optimizer=OptimizerHparams(), model=ModelHparams())


@pytest.fixture
def sample_tensor_dataset() -> Dataset:
    """Provide a simple tensor dataset for testing.

    :return: TensorDataset with 100 samples."""
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return TensorDataset(x, y)


@pytest.fixture
def data_hparams() -> DataHparams:
    """Provide basic DataHparams for testing.

    :return: DataHparams instance."""
    return DataHparams(
        module_name="test_module",
        loader=DataLoaderHparams(batch_size=32, shuffle=True),
        seed=42,
        split=(0.7, 0.2, 0.1),
    )


@pytest.fixture
def simple_data_hparams() -> DataHparams:
    """Provide simple DataHparams without test split.

    :return: DataHparams instance."""
    return DataHparams(
        loader=DataLoaderHparams(batch_size=16),
        seed=123,
        split=(0.8, 0.2),
    )


@pytest.fixture
def mock_pytorch_data(sample_tensor_dataset: Dataset) -> type[PyTorchData]:
    """Provide a mock PyTorchData subclass for testing.

    :param sample_tensor_dataset: Sample dataset fixture.
    :return: MockPyTorchData class."""

    class MockPyTorchData(PyTorchData):
        hparams_schema = DataHparams

        def __init__(self, thparams: DataHparams, dataset: Dataset):
            self._dataset = dataset
            super().__init__(thparams)

        def get_dataset(self) -> Dataset:
            return self._dataset

    return MockPyTorchData


@pytest.fixture
def optimizer_hparams() -> OptimizerHparams:
    """Provide basic OptimizerHparams for testing.

    :return: OptimizerHparams instance."""
    return OptimizerHparams(module_name="torch.optim.SGD", kwargs={"lr": 0.01, "momentum": 0.9})


@pytest.fixture
def scheduler_hparams() -> SchedulerHparams:
    """Provide basic SchedulerHparams for testing.

    :return: SchedulerHparams instance."""
    return SchedulerHparams(
        module_name="torch.optim.lr_scheduler.StepLR", kwargs={"step_size": 10, "gamma": 0.1}
    )


@pytest.fixture
def checkpoint_hparams(tmp_path: Path) -> CheckpointHparams:
    """Provide basic CheckpointHparams for testing.

    :param tmp_path: Pytest temporary path fixture.
    :return: CheckpointHparams instance."""
    return CheckpointHparams(
        dirpath=str(tmp_path / "checkpoints"),
        monitor="validation/loss",
        mode="min",
        save_top_k=2,
    )


@pytest.fixture
def early_stopping_hparams() -> EarlyStoppingHparams:
    """Provide basic EarlyStoppingHparams for testing.

    :return: EarlyStoppingHparams instance."""
    return EarlyStoppingHparams(monitor="validation/loss", patience=5, mode="min")


@pytest.fixture
def model_hparams(
    optimizer_hparams: OptimizerHparams, scheduler_hparams: SchedulerHparams
) -> ModelHparams:
    """Provide basic ModelHparams for testing.

    :param optimizer_hparams: OptimizerHparams fixture.
    :param scheduler_hparams: SchedulerHparams fixture.
    :return: ModelHparams instance."""
    return ModelHparams(
        module_name="test_module",
        optimizer=optimizer_hparams,
        scheduler=scheduler_hparams,
        matmul_precision="high",
    )


@pytest.fixture
def simple_model_hparams() -> ModelHparams:
    """Provide simple ModelHparams without scheduler.

    :return: ModelHparams instance."""
    return ModelHparams(
        optimizer=OptimizerHparams(module_name="torch.optim.Adam", kwargs={"lr": 1e-3})
    )


@pytest.fixture
def mock_model() -> type[Model]:
    """Provide a mock Model subclass for testing.

    :return: MockModel class."""

    class MockModel(Model):
        hparams_schema = ModelHparams

        def __init__(self, thparams: ModelHparams):
            super().__init__(thparams)
            self.layer = torch.nn.Linear(10, 2)

        def compute_metrics(self, batch, batch_idx):
            x, y = batch
            y_pred = self.layer(x)
            loss = torch.nn.functional.cross_entropy(y_pred, y)
            acc = (y_pred.argmax(dim=1) == y).float().mean()
            return {"loss": loss, "accuracy": acc}

    return MockModel


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Provide a sample batch for testing.

    :return: Tuple of input and target tensors."""
    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    return x, y
