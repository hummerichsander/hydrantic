import logging
import sys
from typing import Any, Literal, get_args

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from hydrantic.targets import ConfigError, check_config, instantiate_module

log = logging.getLogger("hydrantic")

Stage = Literal["fit", "validate", "test", "predict"]
STAGES: tuple[str, ...] = get_args(Stage)

# Lightning resolves these as sentinels rather than filesystem paths.
_CKPT_SENTINELS = frozenset({"best", "last", "hpc", "registry"})


def _resolve_ckpt_path(ckpt_path: str | None) -> str | None:
    """Makes a relative checkpoint path absolute, leaving Lightning's sentinels alone.

    Hydra does not change the working directory by default, but the path is still resolved
    explicitly so it survives DDP relaunches and multirun subdirectories.

    :param ckpt_path: The raw `ckpt_path` config value.
    :return: An absolute path, a sentinel, a URL, or None."""

    if ckpt_path is None:
        return None
    if ckpt_path in _CKPT_SENTINELS or "://" in ckpt_path:
        return ckpt_path
    return hydra.utils.to_absolute_path(ckpt_path)


def _add_cwd_to_path() -> None:
    """Puts the invocation directory on `sys.path`.

    A console script puts the venv's `bin` directory on `sys.path[0]`, never the directory the
    command was run from, so `_target_: my_project.models.Foo` would not import for a project
    that has not been installed."""

    cwd = HydraConfig.get().runtime.cwd
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def run_stage(cfg: DictConfig) -> Any:
    """Validates the configuration, builds the objects it names, and runs one Trainer stage.

    :param cfg: The composed configuration.
    :return: The value of `cfg.optimized_metric` if set, so Hydra sweepers can optimize it."""

    _add_cwd_to_path()

    stage = cfg.get("stage") or "fit"
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}, got {stage!r}")

    if problems := check_config(cfg):
        raise ConfigError(problems)

    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    if (seed := cfg.get("seed")) is not None:
        seed_everything(seed, workers=True)
    if (precision := cfg.get("matmul_precision")) is not None:
        torch.set_float32_matmul_precision(precision)

    model: LightningModule = instantiate_module(cfg.model)
    datamodule: LightningDataModule = instantiate_module(cfg.data)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    if not isinstance(datamodule, LightningDataModule):
        raise TypeError(
            f"cfg.data must instantiate a LightningDataModule, got {type(datamodule).__name__}."
        )

    # Lightning logs the model's own hyperparameters; this adds the trainer, logger and
    # callback half of the configuration, which would otherwise not be recorded anywhere.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    for logger in trainer.loggers:
        logger.log_hyperparams({"hydrantic": resolved})

    getattr(trainer, stage)(
        model, datamodule=datamodule, ckpt_path=_resolve_ckpt_path(cfg.get("ckpt_path"))
    )

    if (metric := cfg.get("optimized_metric")) is None:
        return None
    if (value := trainer.callback_metrics.get(metric)) is None:
        log.warning("optimized_metric %r was not found in the logged metrics", metric)
        return None
    return value.item()


def check_only(cfg: DictConfig) -> int:
    """Validates the configuration and reports the problems, without building anything.

    :param cfg: The composed configuration.
    :return: A process exit code."""

    _add_cwd_to_path()

    if problems := check_config(cfg):
        print(ConfigError(problems), file=sys.stderr)
        return 1

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("Configuration is valid.")
    return 0


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig) -> Any:
    """Hydra entry point shared by every hydrantic subcommand.

    :param cfg: The composed configuration.
    :return: The optimized metric for a stage run, or None."""

    if cfg.get("stage") == "check":
        sys.exit(check_only(cfg))
    return run_stage(cfg)
