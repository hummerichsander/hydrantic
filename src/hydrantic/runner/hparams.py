from typing import Literal

from ..hparams import Hparams, Hparam


# Todo: add Hydra Hparams via pydantic settings


class TrainerHparams(Hparams):
    """This is the base class for all hyperparameters of a trainer."""

    accelerator: Literal["cpu", "gpu", "tpu", "hpu", "auto"] = Hparam(
        default_factory=lambda: "auto"
    )
    devices: int = 1
    precision: Literal[64, 32, 16, "bf16"] = 32

    accumulate_grad_batches: int = 1
    gradient_clip_val: float | None = None

    max_epochs: int = Hparam(default_factory=lambda: 10)
    max_time: str | None = None
    benchmark: bool = False
    deterministic: bool = False
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    enable_progress_bar: bool = True
    profiler: Literal["simple", "advanced"] | None = None

    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0


class LoggerHparams(Hparams):
    """This is the base class for all hyperparameters of a logger."""

    name: str | None = ""
    save_dir: str = ""
    version: str | None = None
    offline: bool = True
    anonymous: bool = False
    project: str | None = None
    log_model: bool | Literal["all"] = False


class RunHparams(Hparams):
    trainer: TrainerHparams = Hparam(default_factory=TrainerHparams)
    logger: LoggerHparams = Hparam(default_factory=LoggerHparams)
    resume_from_checkpoint: str | None = None
