"""Typed, Hydra-configurable PyTorch Lightning training."""

from importlib.metadata import PackageNotFoundError, version

from hydrantic.data import DataHparams, DataModule, split_dataset
from hydrantic.hparams import Hparam, Hparams, HparamsModule
from hydrantic.model import Model, ModelHparams
from hydrantic.targets import ConfigError, ConfigProblem, Target, check_config

try:
    __version__ = version("hydrantic")
except PackageNotFoundError:  # source checkout without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "ConfigError",
    "ConfigProblem",
    "DataHparams",
    "DataModule",
    "Hparam",
    "Hparams",
    "HparamsModule",
    "Model",
    "ModelHparams",
    "Target",
    "__version__",
    "check_config",
    "split_dataset",
]
