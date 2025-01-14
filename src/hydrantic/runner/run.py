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
from ..model.hparams import ModelHparams
from ..data.hparams import DataHparams
from ..utils.utils import import_from_string


console_logger = logging.getLogger("hydrantic.runner.run")


class RunConfig(OmegaConf):
    model_hparams: ModelHparams
    run_hparams: RunHparams
    data_hparams: DataHparams


@hydra.main(version_base=None, config_path=(Path(os.environ["HYDRA_CONFIG_PATH"])).as_posix())
def main(cfg: RunConfig):
    """Launch a model from a config file."""
    base_dir = Path(hydra.utils.get_original_cwd())
    sys.path.append(str(base_dir))

    model_class = import_from_string(cfg.model_hparams.module_name)
    model_hparams = model_class.hparams_schema.from_config(cfg.model_hparams)
    run_hparams = RunHparams.from_config(cfg.run_hparams).attribute_dict

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_hparams.logger.save_dir = str(output_dir)
    model_hparams.checkpoint.dirpath = str(output_dir / "checkpoints")
    run_hparams.logger.name = output_dir.parts[-2] + "/" + output_dir.parts[-1]

    model = model_class(model_hparams)

    data_class = import_from_string(cfg.data_hparams.module_name)
    data_hparams = data_class.hparams_schema.from_config(cfg.data_hparams)
    data = data_class(data_hparams)
    console_logger.info(
        f"Data split: train={len(data.train_loader)}, val={len(data.val_loader)}, test={len(data.test_loader)}"
    )
    console_logger.info(f"Split seed: {data.hparams.seed}")

    logger = WandbLogger(**run_hparams.logger)
    trainer = Trainer(logger=logger, **run_hparams.trainer)
    trainer.fit(model, data.train_loader, data.val_loader)


if __name__ == "__main__":
    main()
