"""Typed, Hydra-configurable PyTorch Lightning training."""

from importlib.metadata import PackageNotFoundError, version

from hydrantic.data import DataModule, LoaderHparams, SplitHparams
from hydrantic.hparams import Hparam, Hparams, HparamsModule
from hydrantic.model import Model, ModelHparams
from hydrantic.targets import ConfigError, ConfigProblem, Target, check_config
from hydrantic.utils import split_dataset

try:
    __version__ = version("hydrantic")
except PackageNotFoundError:  # source checkout without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "ConfigError",
    "ConfigProblem",
    "DataModule",
    "Hparam",
    "Hparams",
    "HparamsModule",
    "LoaderHparams",
    "Model",
    "ModelHparams",
    "SplitHparams",
    "Target",
    "__version__",
    "check_config",
    "split_dataset",
]
