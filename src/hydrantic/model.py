from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypeVar

import lightning
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset

from hydrantic.hparams import Hparams, HparamsModule
from hydrantic.targets import Target, instantiate_target


class ModelHparams(Hparams):
    """Base hyperparameters for `Model`."""

    optimizer: Target = {"_target_": "torch.optim.Adam"}
    scheduler: Target | None = None


H = TypeVar("H", bound=ModelHparams)


class Model(HparamsModule[H], lightning.LightningModule):
    """A `LightningModule` configured by a validated `ModelHparams` schema.

    Subclasses implement the usual Lightning hooks (`forward`, `training_step`,
    `validation_step`, ...) and read their configuration from the typed `thparams` attribute.
    The optimizer and scheduler are built from the `_target_` mappings in the hyperparameters."""

    def get_optimizer_params(self) -> Iterable[Parameter] | Iterable[dict[str, Any]]:
        """Returns the parameters or parameter groups handed to the optimizer.

        Override this to give different parameter groups different settings.

        :return: The parameters to optimize."""

        return self.parameters()

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Builds the optimizer and optional scheduler from the hyperparameters.

        The `scheduler` hyperparameter is either a bare `_target_` mapping or Lightning's
        `lr_scheduler` dict with a `_target_` mapping under its `scheduler` key, which is how
        `monitor`, `interval` and `frequency` are set.

        :return: The Lightning optimizer configuration."""

        optimizer: Optimizer = instantiate_target(
            self.thparams.optimizer, self.get_optimizer_params()
        )

        if (scheduler_config := self.thparams.scheduler) is None:
            return {"optimizer": optimizer}

        if "_target_" in scheduler_config:
            lr_scheduler: dict[str, Any] = {
                "scheduler": instantiate_target(scheduler_config, optimizer)
            }
        else:
            lr_scheduler = dict(scheduler_config)
            lr_scheduler["scheduler"] = instantiate_target(lr_scheduler["scheduler"], optimizer)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def log_metrics(
        self,
        metrics: Mapping[str, Tensor | float],
        stage: Literal["training", "validation", "test", "predict"],
        **kwargs: Any,
    ) -> None:
        """Logs metrics under a `stage/name` prefix.

        The prefix is the convention the shipped `ModelCheckpoint` and `EarlyStopping` configs
        monitor, e.g. `validation/loss`. A metric named `loss` is also shown in the progress bar.

        :param metrics: Metric names and values.
        :param stage: The stage the metrics were computed in.
        :param kwargs: Extra arguments forwarded to `self.log`."""

        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, prog_bar=(name == "loss"), **kwargs)

    @property
    def num_params(self) -> int:
        """Returns the total number of parameters in the model.

        :return: The number of parameters."""

        return sum(p.numel() for p in self.parameters())

    @property
    def num_params_by_module(self) -> dict[str, int]:
        """Returns the number of parameters of each submodule.

        :return: Submodule names and their parameter counts, with the total under `"total"`."""

        return {
            (name or "total"): sum(p.numel() for p in module.parameters())
            for name, module in self.named_modules()
        }

    def fit_inline(
        self,
        train_data: Dataset | DataLoader | lightning.LightningDataModule,
        val_data: Dataset | DataLoader | None = None,
        *,
        batch_size: int = 32,
        **trainer_kwargs: Any,
    ) -> lightning.Trainer:
        """Fits the model with a throwaway `Trainer`, for debugging and smoke tests.

        Uses the optimizer configured in `thparams`; nothing on the model is modified. Datasets
        are wrapped in `DataLoader`s, dataloaders are used as given, and a `LightningDataModule`
        is passed straight through. To train with a different optimizer, build the model with
        `hparams.model_copy(update={"optimizer": {...}})`.

        :param train_data: Training dataset, dataloader or datamodule.
        :param val_data: Validation dataset or dataloader, unused for a datamodule.
        :param batch_size: Batch size used only when wrapping a raw `Dataset`.
        :param trainer_kwargs: Overrides forwarded to the `Trainer`.
        :return: The `Trainer` that ran the fit."""

        defaults: dict[str, Any] = {
            "max_epochs": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_model_summary": False,
        }
        trainer = lightning.Trainer(**(defaults | trainer_kwargs))

        if isinstance(train_data, lightning.LightningDataModule):
            if val_data is not None:
                raise ValueError(
                    "val_data must be None when train_data is a LightningDataModule; "
                    "the datamodule provides its own validation dataloader."
                )
            trainer.fit(self, datamodule=train_data)
        else:
            trainer.fit(
                self,
                _as_dataloader(train_data, batch_size=batch_size, shuffle=True),
                _as_dataloader(val_data, batch_size=batch_size, shuffle=False),
            )

        return trainer


def _as_dataloader(
    data: Dataset | DataLoader | None,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader | None:
    """Wraps a `Dataset` in a `DataLoader`, passing dataloaders and None through.

    :param data: The dataset or dataloader.
    :param batch_size: Batch size used when wrapping a dataset.
    :param shuffle: Whether to shuffle, ignored for iterable datasets.
    :return: The dataloader, or None."""

    if data is None or isinstance(data, DataLoader):
        return data
    if isinstance(data, IterableDataset):
        return DataLoader(data, batch_size=batch_size)
    if isinstance(data, Dataset):
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    raise TypeError(
        f"Expected a Dataset, DataLoader or LightningDataModule, got {type(data).__name__}."
    )
