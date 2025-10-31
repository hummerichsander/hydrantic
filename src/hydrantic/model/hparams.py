from typing import Literal, Any
from ..hparams import Hparams, Hparam


class OptimizerHparams(Hparams):
    """This is the base class for all hyperparameters of the optimizer."""

    module_name: str = "torch.optim.Adam"
    kwargs: dict[str, Any] = {}


class SchedulerHparams(Hparams):
    """This is the base class for all hyperparameters of the scheduler."""

    module_name: str | None = None
    kwargs: dict[str, Any] = {}


class CheckpointHparams(Hparams):
    """This is the base class for all hyperparameters of a checkpoint."""

    dirpath: str | None = None
    filename: str | None = "{epoch}-{val_loss:.2f}"
    monitor: str | None = "validation/loss"
    mode: Literal["min", "max"] = "min"
    save_last: bool = True
    save_top_k: int = 3
    every_n_epochs: int | None = 1


class EarlyStoppingHparams(Hparams):
    """This is the base class for all hyperparameters of early stopping."""

    monitor: str | None = "validation/loss"
    min_delta: float = 0.0
    patience: int = 3
    mode: Literal["min", "max"] = "min"
    stopping_threshold: float | None = None
    divergence_threshold: float | None = None
    check_finite: bool = True


class ModelHparams(Hparams):
    module_name: str | None = None

    matmul_precision: Literal["highest", "high", "medium"] = "highest"

    optimizer: OptimizerHparams = Hparam(default_factory=OptimizerHparams)
    scheduler: SchedulerHparams = Hparam(default_factory=SchedulerHparams)
    checkpoint: CheckpointHparams | None = Hparam(default_factory=CheckpointHparams)
    early_stopping: EarlyStoppingHparams | None = None
