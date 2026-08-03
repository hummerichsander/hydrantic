from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset

from hydrantic import DataHparams, DataModule, Hparams, Model, ModelHparams


@pytest.fixture
def device() -> torch.device:
    """Provides a torch device for testing.

    :return: A CUDA device if available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_config_path(tmp_path: Path) -> Path:
    """Provides a temporary directory for config files.

    :param tmp_path: Pytest's temporary directory fixture.
    :return: Path to the temporary config directory."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    return config_dir


@pytest.fixture
def tmp_download_path(tmp_path: Path) -> Path:
    """Provides a temporary directory for download tests.

    :param tmp_path: Pytest's temporary directory fixture.
    :return: Path to the temporary download directory."""

    download_dir = tmp_path / "downloads"
    download_dir.mkdir(exist_ok=True)
    return download_dir


class SampleHparams(Hparams):
    """A flat schema used across the hparams tests."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    hidden_dim: int = 128
    dropout: float = 0.1


class InnerHparams(Hparams):
    """The nested half of `NestedHparams`."""

    lr: float = 1e-3
    weight_decay: float = 1e-4


class NestedHparams(Hparams):
    """A schema with a nested schema field."""

    inner: InnerHparams = InnerHparams()
    batch_size: int = 64


@pytest.fixture
def sample_hparams() -> SampleHparams:
    """Provides a flat schema instance.

    :return: A `SampleHparams` instance with default values."""

    return SampleHparams()


@pytest.fixture
def nested_hparams() -> NestedHparams:
    """Provides a nested schema instance.

    :return: A `NestedHparams` instance with default values."""

    return NestedHparams()


@pytest.fixture
def sample_tensor_dataset() -> Dataset:
    """Provides a small tensor dataset.

    :return: A `TensorDataset` with 100 samples."""

    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return TensorDataset(x, y)


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Provides a sample batch.

    :return: A tuple of inputs and targets."""

    return torch.randn(8, 10), torch.randint(0, 2, (8,))


class TinyHparams(ModelHparams):
    """Hyperparameters of `TinyModel`."""

    input_dim: int = 10
    output_dim: int = 2


class TinyModel(Model[TinyHparams]):
    """A one-layer model implementing the standard Lightning hooks."""

    def __init__(self, hparams=None, **kwargs):
        super().__init__(hparams, **kwargs)
        self.layer = nn.Linear(self.thparams.input_dim, self.thparams.output_dim)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log_metrics({"loss": loss}, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log_metrics(
            {"loss": F.cross_entropy(logits, y), "accuracy": (logits.argmax(-1) == y).float().mean()},
            "validation",
        )


class RenamedArgModel(TinyModel):
    """A model whose `__init__` renames the hyperparameter argument.

    Lightning restores a checkpoint by filtering the stored hyperparameters against `__init__`'s
    signature, so the stored key has to follow this rename."""

    def __init__(self, settings=None, **kwargs):
        super().__init__(settings, **kwargs)


@pytest.fixture
def tiny_model_cls() -> type[TinyModel]:
    """Provides the tiny model class.

    :return: The `TinyModel` class."""

    return TinyModel


class TinyDataHparams(DataHparams):
    """Hyperparameters of `TinyData`."""

    num_samples: int = 100
    input_dim: int = 10
    num_classes: int = 2


class TinyData(DataModule[TinyDataHparams]):
    """A datamodule over a generated tensor dataset."""

    #: Incremented on every `get_dataset` call, so tests can assert `setup` caches.
    build_count = 0

    def get_dataset(self) -> Dataset:
        type(self).build_count += 1
        generator = torch.Generator().manual_seed(self.thparams.seed)
        x = torch.randn(self.thparams.num_samples, self.thparams.input_dim, generator=generator)
        y = torch.randint(
            self.thparams.num_classes, (self.thparams.num_samples,), generator=generator
        )
        return TensorDataset(x, y)


@pytest.fixture
def tiny_data_cls() -> type[TinyData]:
    """Provides the tiny datamodule class with a reset build counter.

    :return: The `TinyData` class."""

    TinyData.build_count = 0
    return TinyData


@pytest.fixture
def model_config() -> DictConfig:
    """Provides a `_target_` config node for `TinyModel`.

    :return: The config node."""

    return OmegaConf.create(
        {
            "_target_": "conftest.TinyModel",
            "input_dim": 10,
            "output_dim": 2,
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-3},
        }
    )


@pytest.fixture
def data_config() -> DictConfig:
    """Provides a `_target_` config node for `TinyData`.

    :return: The config node."""

    return OmegaConf.create(
        {
            "_target_": "conftest.TinyData",
            "num_samples": 100,
            "loader": {"batch_size": 8},
        }
    )


@pytest.fixture
def run_config(model_config: DictConfig, data_config: DictConfig) -> DictConfig:
    """Provides a minimal but complete run config.

    :param model_config: The model node fixture.
    :param data_config: The data node fixture.
    :return: The composed config."""

    return OmegaConf.create(
        {
            "stage": "fit",
            "seed": 42,
            "model": model_config,
            "data": data_config,
            "trainer": {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "logger": False,
                "enable_checkpointing": False,
            },
        }
    )
