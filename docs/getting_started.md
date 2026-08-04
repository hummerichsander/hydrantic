# Getting started

A complete project is two classes and their two schemas. The runnable version of everything
below is [`examples/quickstart.py`](https://github.com/hummerichsander/hydrantic/blob/main/examples/quickstart.py).

## A model

Declare the hyperparameters as a pydantic schema, then bind it by parametrising
{class}`hydrantic.model.Model`. `self.thparams` is that schema, typed and validated.

```python
from torch import nn
from torch.nn import functional as F

from hydrantic import Model, ModelHparams


class ClassifierHparams(ModelHparams):
    input_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 3


class Classifier(Model[ClassifierHparams]):
    def __init__(self, hparams=None, **kwargs):
        super().__init__(hparams, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(self.thparams.input_dim, self.thparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.thparams.hidden_dim, self.thparams.output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log_metrics({"loss": loss}, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log_metrics({"loss": F.cross_entropy(self(x), y)}, "validation")
```

These are ordinary
[Lightning hooks](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) —
hydrantic adds nothing to them. {meth}`~hydrantic.model.Model.log_metrics` is an optional
helper that prefixes names with the stage, giving `training/loss` and `validation/loss`; those
are the names the shipped `ModelCheckpoint` config monitors.

The optimizer comes from the hyperparameters, so it is configurable without touching code.
{class}`~hydrantic.model.ModelHparams` provides `optimizer` and `scheduler`; you never write
`configure_optimizers` yourself.

## A datamodule

{class}`hydrantic.data.DataModule` is a
[LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) with a
typed `thparams` — and nothing else. You write `setup` and the dataloader hooks exactly as you
would in plain Lightning.

Two optional schemas cover the usual needs. Inherit
{class}`~hydrantic.data.SplitHparams` when you split one dataset into train/validation/test, and
{class}`~hydrantic.data.LoaderHparams` for the standard `DataLoader` settings:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from hydrantic import DataModule, LoaderHparams, SplitHparams


class BlobsDataHparams(SplitHparams, LoaderHparams):
    num_samples: int = 2048
    input_dim: int = 16
    num_classes: int = 3


class BlobsData(DataModule[BlobsDataHparams]):
    def setup(self, stage):
        if getattr(self, "train", None) is not None:
            return  # setup runs once per stage

        h = self.thparams
        generator = torch.Generator().manual_seed(h.seed)
        labels = torch.randint(h.num_classes, (h.num_samples,), generator=generator)
        centers = torch.randn(h.num_classes, h.input_dim, generator=generator) * 3.0
        features = centers[labels] + torch.randn(h.num_samples, h.input_dim, generator=generator)

        self.train, self.val, self.test = h.split_dataset(TensorDataset(features, labels))

    def train_dataloader(self):
        return DataLoader(self.train, **self.thparams.loader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val, **self.thparams.loader_kwargs())
```

Both schemas are opt-in, so a datamodule that loads its validation set from a separate file just
inherits `Hparams` and carries no `split`/`seed` it never reads.
{func}`~hydrantic.utils.split_dataset` is also available as a plain function.

Build datasets in `setup`, not `__init__`: Lightning calls it on every rank once the process
group exists, which is what makes this DDP-correct. Downloads belong in `prepare_data`.

`shuffle` is deliberately not a `LoaderHparams` field — it differs per stage, and a baked-in
default is how a training loader silently starts or stops shuffling. Pass it explicitly.

## Running it

Inline, for a notebook or a smoke test:

```python
model = Classifier(hidden_dim=32)
data = BlobsData(num_samples=512)

trainer = model.fit_inline(data, max_epochs=3, accelerator="cpu")
print(trainer.callback_metrics["training/loss"])
```

{meth}`~hydrantic.model.Model.fit_inline` accepts a `Dataset`, a `DataLoader` or a
`LightningDataModule`, forwards any other keyword argument to the `Trainer`, and returns that
`Trainer`. It changes nothing on the model.

For real runs, use the CLI:

```bash
cd examples
hydrantic check --config-dir conf --config-name train
hydrantic fit   --config-dir conf --config-name train
```

See [Configuration](configuration.md) and [CLI](cli.md).
