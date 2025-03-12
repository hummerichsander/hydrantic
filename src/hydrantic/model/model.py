import stat
from typing import Type, Self, Literal, Any

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn, Tensor

import wandb

import lightning
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

from .hparams import ModelHparams, OptimizerHparams
from ..utils.utils import import_from_string


class Model(lightning.LightningModule, ABC):
    hparams_schema: Type[ModelHparams]
    has_logger: bool = True

    def __init__(self, hparams: ModelHparams | dict):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = self.hparams_schema(**hparams)
        elif not isinstance(hparams, ModelHparams):
            hparams = self.hparams_schema(**hparams.__dict__)

        if not isinstance(hparams, self.hparams_schema):
            raise ValueError("hparams must be an instance of the specified hparams_schema")

        # Hyperparameters need to be stored under the 'hparams' key in order to load them via `load_from_checkpoint`
        self.save_hyperparameters(dict(hparams=hparams.attribute_dict))
        self._set_hparams(hparams.attribute_dict)

    @abstractmethod
    def compute_metrics(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """Computes metrics on a batch of data. Must be implemented in subclasses of Model.

        :param batch: batch of data
        :param batch_idx: index of batch
        :return: dictionary of metric names and values"""

        raise NotImplementedError("compute_metrics must be implemented in subclasses of Model")

    def _evaluate_metrics(self, metrics: dict[str, torch.Tensor], log_prefix: str) -> None:
        """Evaluates metrics and logs them to the Lightning logger.

        :param metrics: dictionary of metric names and values
        :param log_prefix: prefix to use when logging metrics"""

        if metrics is None:
            raise ValueError("metrics must not be None")
        if "loss" not in metrics:
            raise ValueError(f"metrics dictionary must contain a 'loss' key, got {metrics.keys()}")

        for metric_name, metric_value in metrics.items():
            self.log(
                f"{log_prefix}/{metric_name}",
                metric_value,
                prog_bar=(metric_name == "loss"),
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Computes the metrics on a batch of training data and logs it to the Lightning logger.

        :param batch: batch of data
        :param batch_idx: index of batch
        :return: loss tensor"""

        metrics = self.compute_metrics(batch, batch_idx)
        self._evaluate_metrics(metrics, "training")
        return metrics["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Computes the metrics on a batch of validation data and logs it to the Lightning logger.

        :param batch: batch of data
        :param batch_idx: index of batch"""

        metrics = self.compute_metrics(batch, batch_idx)
        self._evaluate_metrics(metrics, "validation")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Computes the metrics on a batch of test data and logs it to the Lightning logger.

        :param batch: batch of data
        :param batch_idx: index of batch"""

        metrics = self.compute_metrics(batch, batch_idx)
        self._evaluate_metrics(metrics, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures the optimizer and (optionally) learning rate scheduler. Utilizes the optimizer and scheduler
        defined in the hparams.

        :return: dictionary of optimizer and (optionally) learning rate scheduler"""

        optimizer_module = import_from_string(self.hparams.optimizer.module_name)
        optimizer = optimizer_module(self.parameters(), **self.hparams.optimizer.kwargs)
        if self.hparams.scheduler.module_name is not None:
            scheduler_module = import_from_string(self.hparams.scheduler.module_name)
            scheduler = scheduler_module(optimizer, **self.hparams.scheduler.kwargs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

    def configure_callbacks(self) -> list[Callback]:
        """Configures the callbacks to use during training.

        :return: list of callbacks to use during training"""

        callbacks: list[Callback] = []

        if self.hparams.checkpoint is not None:
            callbacks.append(ModelCheckpoint(**self.hparams.checkpoint))
        if self.hparams.early_stopping is not None:
            callbacks.append(EarlyStopping(**self.hparams.early_stopping))

        if self.has_logger:
            callbacks.append(LearningRateMonitor())

        return callbacks

    @property
    def num_params_by_module(self) -> dict[str, int]:
        """Returns the number of parameters in each module in the model.

        :return: dictionary of module names and number of parameters"""

        num_params_by_module_: dict[str, int] = {}
        for name, module in self.named_modules():
            if name == "":
                name = "total"
            if isinstance(module, nn.Module):
                num_params = 0
                for param in module.parameters():
                    num_params += param.numel()
                num_params_by_module_[name] = num_params
        return num_params_by_module_

    @property
    def num_params(self) -> int:
        """Returns the total number of parameters in the model.

        :return: number of parameters"""

        return sum(self.num_params_by_module.values())

    def fit_fast(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        n_epochs: int = 1,
        lr: float = 1e-3,
        accelerator: Literal["cpu", "gpu", "tpu", "hpu", "auto"] = "cpu",
        precision: Literal[64, 32, 16, "bf16"] = 32,
        verbose: bool = True,
    ) -> Self:
        """Fits the model to the training data using the Adam optimizer

        :param train_loader: Training dataloader
        :param val_dataloader: Validation dataloader
        :param n_epochs: Number of epochs to train
        :param lr: Learning rate to use in the Adam optimizer
        :param accelerator: Accelerator to use to train on
        :param precision: Precision to use to train on
        :param verbose: whether to log the training progress in a progress bar
        :return: the trained model"""

        from lightning.pytorch.trainer import Trainer

        self.has_logger = False

        self.hparams.optimizer = OptimizerHparams(module_name="torch.optim.Adam", kwargs={"lr": lr})

        trainer = Trainer(
            logger=False, accelerator=accelerator, precision=precision, max_epochs=n_epochs, enable_progress_bar=verbose
        )
        trainer.fit(self, train_loader, val_loader)
        return self
    
    @classmethod
    def load_from_wandb_artifact(cls, artifact_name: str, dl_path: str | None = None) -> Self:
        """Loads the model from a Weights & Biases artifact.

        :param artifact_name: Name of the artifact to load
        :param dl_path: Path to download the artifact to
        :return: the loaded model"""

        run = wandb.init()
        artifact = run.use_artifact(artifact_name)
        artifact_dir = Path(artifact.download(dl_path))
        return cls.load_from_checkpoint(artifact_dir / "model.ckpt")
