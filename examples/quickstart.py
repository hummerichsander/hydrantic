"""A complete hydrantic project in one file.

Run it from this directory:

    hydrantic check --config-dir conf --config-name train
    hydrantic fit   --config-dir conf --config-name train
    hydrantic fit   --config-dir conf --config-name train trainer.max_epochs=5
"""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hydrantic import DataModule, LoaderHparams, Model, ModelHparams, SplitHparams


class ClassifierHparams(ModelHparams):
    """Hyperparameters of the toy classifier."""

    input_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 3
    dropout: float = 0.0


class Classifier(Model[ClassifierHparams]):
    """A small MLP classifier."""

    def __init__(self, hparams: ClassifierHparams | dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(hparams, **kwargs)

        self.net = nn.Sequential(
            nn.Linear(self.thparams.input_dim, self.thparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.thparams.dropout),
            nn.Linear(self.thparams.hidden_dim, self.thparams.output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Computes class logits.

        :param x: A batch of inputs.
        :return: The logits."""

        return self.net(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Computes and logs the training loss.

        :param batch: A batch of inputs and labels.
        :param batch_idx: The index of the batch.
        :return: The loss to optimize."""

        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log_metrics({"loss": loss}, "training")
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Computes and logs the validation loss and accuracy.

        :param batch: A batch of inputs and labels.
        :param batch_idx: The index of the batch."""

        x, y = batch
        logits = self(x)
        self.log_metrics(
            {
                "loss": F.cross_entropy(logits, y),
                "accuracy": (logits.argmax(-1) == y).float().mean(),
            },
            "validation",
        )

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Computes and logs the test loss and accuracy.

        :param batch: A batch of inputs and labels.
        :param batch_idx: The index of the batch."""

        x, y = batch
        logits = self(x)
        self.log_metrics(
            {
                "loss": F.cross_entropy(logits, y),
                "accuracy": (logits.argmax(-1) == y).float().mean(),
            },
            "test",
        )


class BlobsDataHparams(SplitHparams, LoaderHparams):
    """Hyperparameters of the synthetic blob dataset.

    `split`/`seed` come from `SplitHparams`, the `DataLoader` settings from `LoaderHparams`."""

    num_samples: int = 2048
    input_dim: int = 16
    num_classes: int = 3


class BlobsData(DataModule[BlobsDataHparams]):
    """Gaussian blobs, one per class, generated on the fly.

    Plain Lightning hooks; `DataModule` only supplies the typed `thparams`."""

    def setup(self, stage: str) -> None:
        """Generates the dataset and splits it.

        :param stage: The stage Lightning is preparing for."""

        if getattr(self, "train", None) is not None:
            return  # setup runs once per stage; generate only once

        h = self.thparams
        generator = torch.Generator().manual_seed(h.seed)

        labels = torch.randint(h.num_classes, (h.num_samples,), generator=generator)
        centers = torch.randn(h.num_classes, h.input_dim, generator=generator) * 3.0
        features = centers[labels] + torch.randn(h.num_samples, h.input_dim, generator=generator)

        dataset = TensorDataset(features, labels)
        self.train, self.val, self.test = h.split_dataset(dataset)

    def train_dataloader(self) -> DataLoader:
        """:return: The training dataloader."""
        return DataLoader(self.train, **self.thparams.loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        """:return: The validation dataloader."""
        return DataLoader(self.val, **self.thparams.loader_kwargs())

    def test_dataloader(self) -> DataLoader:
        """:return: The test dataloader."""
        return DataLoader(self.test, **self.thparams.loader_kwargs())


if __name__ == "__main__":
    # The inline API: no config files, no CLI.
    model = Classifier(hidden_dim=32)
    data = BlobsData(num_samples=512)

    trainer = model.fit_inline(data, max_epochs=3, accelerator="cpu")
    print("final training loss:", trainer.callback_metrics["training/loss"].item())
