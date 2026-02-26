import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

import logging
from omegaconf import OmegaConf

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.profilers import SimpleProfiler

from .hparams import RunHparams
from ..utils.utils import import_from_string


console_logger = logging.getLogger("hydrantic.runner.profile")


class ModelConfig(OmegaConf):
    module_name: str


class DataConfig(OmegaConf):
    module_name: str


class ProfileConfig(OmegaConf):
    model_config: ModelConfig
    data_config: DataConfig
    run_config: OmegaConf
    profile_config: OmegaConf


@hydra.main(version_base=None, config_path=(Path(os.environ["HYDRA_CONFIG_PATH"]).as_posix()))
def main(cfg: ProfileConfig) -> None:
    """Profile model training performance using PyTorch Lightning profiler.

    This command runs a single training epoch to analyze performance bottlenecks.
    It generates a detailed profile report showing time spent in each component.

    Usage:
        hydrantic.cli.profile config_name=my_config
        hydrantic.cli.profile config_name=my_config model_config.batch_size=32

    Environment Variables:
        HYDRA_CONFIG_PATH: Path to directory containing Hydra config files

    Output:
        Creates profile.txt in the output directory with detailed timing information.
    """
    base_dir = Path(hydra.utils.get_original_cwd())
    sys.path.append(str(base_dir))

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    console_logger.info(f"Output directory: {output_dir}")

    # Initialize model
    console_logger.info("Initializing model...")
    model_class = import_from_string(cfg.model_config.module_name)
    model_hparams = model_class.hparams_schema.from_config(cfg.model_config)
    model = model_class(model_hparams)
    model.has_logger = False
    console_logger.info(f"Model parameters: {model.num_params:,}")

    # Initialize data
    console_logger.info("Loading data...")
    data_class = import_from_string(cfg.data_config.module_name)
    data_hparams = data_class.hparams_schema.from_config(cfg.data_config)
    data = data_class(data_hparams)
    console_logger.info(
        f"Data split: "
        f"train={len(data.train_loader)}, val={len(data.val_loader)}, "
        f"test={len(data.test_loader) if data.test_loader else 0}"
    )
    console_logger.info(f"Split seed: {data_hparams.seed}")

    # Configure profiler
    console_logger.info("\nConfiguring profiler...")
    profiler = SimpleProfiler(dirpath=output_dir.as_posix(), filename="profile")

    # Get run hparams and configure trainer
    run_hparams = RunHparams.from_config(cfg.run_config)

    # Convert trainer hparams to dict and filter out conflicting parameters
    trainer_dict = dict(run_hparams.trainer)
    trainer_kwargs = {
        k: v
        for k, v in trainer_dict.items()
        if k
        not in [
            "max_epochs",
            "profiler",
            "enable_checkpointing",
            "logger",
            "enable_progress_bar",
            "limit_train_batches",
            "limit_val_batches",
        ]
    }

    trainer = Trainer(
        logger=False,
        max_epochs=1,
        profiler=profiler,
        enable_progress_bar=True,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=1,
        **trainer_kwargs,
    )

    console_logger.info("Starting profiling...")
    trainer.fit(model, data.train_loader, data.val_loader)

    console_logger.info("\nProfiling complete!")
    console_logger.info(f"Profile summary: {output_dir / 'profile.txt'}")


if __name__ == "__main__":
    main()
