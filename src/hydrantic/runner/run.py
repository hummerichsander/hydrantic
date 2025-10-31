import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

import logging
from omegaconf import OmegaConf

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger

from .hparams import RunHparams
from ..utils.utils import import_from_string


console_logger = logging.getLogger("hydrantic.runner.run")


class ModelConfig(OmegaConf):
    module_name: str


class DataConfig(OmegaConf):
    module_name: str


class RunConfig(OmegaConf):
    model_config: ModelConfig
    data_config: DataConfig
    run_config: OmegaConf


@hydra.main(version_base=None, config_path=(Path(os.environ["HYDRA_CONFIG_PATH"]).as_posix()))
def main(cfg: RunConfig) -> None:
    """Launch a model from a config file."""
    base_dir = Path(hydra.utils.get_original_cwd())
    sys.path.append(str(base_dir))

    model_class = import_from_string(cfg.model_config.module_name)
    model_hparams = model_class.hparams_schema.from_config(cfg.model_config)
    run_hparams = RunHparams.from_config(cfg.run_config)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_hparams.logger.save_dir = str(output_dir)
    model_hparams.checkpoint.dirpath = str(output_dir / "checkpoints")
    run_hparams.logger.name = output_dir.parts[-2] + "/" + output_dir.parts[-1]

    model = model_class(model_hparams)

    data_class = import_from_string(cfg.data_config.module_name)
    data_hparams = data_class.hparams_schema.from_config(cfg.data_config)
    data = data_class(data_hparams)
    console_logger.info(
        f"Data split: "
        f"train={len(data.train_loader)}, val={len(data.val_loader)}, test={len(data.test_loader)}"
    )
    console_logger.info(f"Split seed: {data_hparams.seed}")

    logger = WandbLogger(**run_hparams.logger)
    trainer = Trainer(logger=logger, **run_hparams.trainer)

    # Get resume_from_checkpoint from run_hparams (handled via Hydra config)
    resume_from_checkpoint = run_hparams.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        console_logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.fit(model, data.train_loader, data.val_loader, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(model, data.train_loader, data.val_loader)


if __name__ == "__main__":
    main()
