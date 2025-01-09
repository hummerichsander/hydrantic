from typing import Type, Any

from abc import ABC, abstractmethod

import torch
from torch import nn

import lightning
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from .hparams import ModelHparams
from ..utils.utils import import_from_string


class Model(lightning.LightningModule, ABC):
    hparams_schema: Type[ModelHparams]

    def __init__(self, hparams: ModelHparams):
        super().__init__()

        if not isinstance(hparams, self.hparams_schema):
            raise ValueError("hparams must be an instance of the specified hparams_schema")

        self.save_hyperparameters(hparams.attribute_dict)

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
