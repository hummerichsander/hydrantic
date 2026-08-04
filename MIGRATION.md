# Migration

- [0.1.0 → 0.2.0](#from-010-to-020) — the data layer only.
- [0.0.x → 0.1.0](#from-00x-to-010) — the large one.

---

# From 0.1.0 to 0.2.0

Only the data layer changed. Models, the CLI and the configuration scheme are untouched.

`DataModule` is now a `LightningDataModule` with typed hyperparameters and nothing else: no
`get_dataset`, no `loader` dicts, no generated dataloaders. You write `setup` and the dataloader
hooks as in plain Lightning, and two opt-in schemas cover the usual settings.

## Config

```yaml
# 0.1.0
data:
  _target_: my_project.MyData
  split: [0.8, 0.1, 0.1]
  seed: 42
  loader: {batch_size: 1024, num_workers: 0}
  train_loader: {shuffle: false}

# 0.2.0
data:
  _target_: my_project.MyData
  split: [0.8, 0.1, 0.1]
  seed: 42
  batch_size: 1024
  num_workers: 0
```

`train_loader: {shuffle: false}` disappears because shuffling is no longer something hydrantic
decides — see the note at the end.

## Code

```python
# 0.1.0
class MyDataHparams(DataHparams):
    path: str


class MyData(DataModule[MyDataHparams]):
    def get_dataset(self):
        return torch.load(self.thparams.path)


# 0.2.0
class MyDataHparams(SplitHparams, LoaderHparams):
    path: str


class MyData(DataModule[MyDataHparams]):
    def setup(self, stage):
        if getattr(self, "train", None) is not None:
            return  # setup runs once per stage; guard expensive work yourself
        dataset = torch.load(self.thparams.path)
        self.train, self.val, self.test = self.thparams.split_dataset(dataset)

    def train_dataloader(self):
        return DataLoader(self.train, **self.thparams.loader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val, **self.thparams.loader_kwargs())

    def test_dataloader(self):
        return DataLoader(self.test, **self.thparams.loader_kwargs())
```

| 0.1.0 | 0.2.0 |
| --- | --- |
| `DataHparams` | `SplitHparams` if you split a dataset, `LoaderHparams` for `DataLoader` settings, or plain `Hparams` |
| `get_dataset()` | build the dataset in `setup(stage)` |
| `get_split_datasets()` | assign `self.train` / `self.val` / `self.test` in `setup` |
| `loader: {...}` config dict | typed fields from `LoaderHparams` |
| `train_loader` / `val_loader` / `test_loader` config dicts | arguments to your own `DataLoader` calls |
| generated `*_dataloader()` methods | write them; `loader_kwargs()` supplies the settings |
| `data.train_dataset` / `.val_dataset` / `.test_dataset` | whatever you name them |
| `from hydrantic.data import split_dataset` | `from hydrantic.utils import split_dataset` (or `from hydrantic import split_dataset`, unchanged) |

Overriding `_build_dataloader` to get a different `DataLoader` class — a `torch_geometric` one,
say — is no longer needed: just construct it in `train_dataloader`.

## Two behaviour changes to check

**Shuffling.** 0.1.0 forced `shuffle=True` on the training loader, which silently flipped
behaviour for anything that ran `shuffle: False` under 0.0.x. If your model reshuffles its own
dataset in `on_train_epoch_end`, pass `shuffle=False` (or simply omit it — `loader_kwargs()` adds
nothing) rather than relying on a default.

**`setup` is no longer guarded for you.** Lightning calls `setup` once per stage, so a `fit`
followed by a `test` calls it twice. If building the dataset is expensive, guard it as shown
above.

`split_dataset` still returns `Subset`s, so code reaching through the split with
`trainer.train_dataloader.dataset.dataset` keeps working.

---

# From 0.0.x to 0.1.0

0.1.0 is a clean break with no compatibility shims. The changes fall into three groups:
configuration moves to Hydra's `_target_`, the data class becomes a real `LightningDataModule`,
and `compute_metrics` gives way to ordinary Lightning hooks.

Pin `hydrantic==0.0.2` until you are ready to migrate.

## Imports

| Before | After |
| --- | --- |
| `from hydrantic.model.model import Model` | `from hydrantic import Model` |
| `from hydrantic.model.hparams import ModelHparams` | `from hydrantic import ModelHparams` |
| `from hydrantic.data.data import PyTorchData` | `from hydrantic import DataModule` |
| `from hydrantic.data.hparams import DataHparams` | `from hydrantic import DataHparams` |
| `from hydrantic.utils.utils import download_url` | `from hydrantic.utils import download_url` |
| `from hydrantic.hparams import Hparams` | unchanged |

`DataLoaderHparams`, `OptimizerHparams`, `SchedulerHparams`, `CheckpointHparams`,
`EarlyStoppingHparams`, `TrainerHparams`, `LoggerHparams`, `RunHparams` and
`utils.import_from_string` are gone; each mirrored a signature that Hydra can now read directly.

## Training steps

`compute_metrics` is gone. Write the Lightning hooks you actually want.

```python
# before
def compute_metrics(self, batch, batch_idx):
    x, y = batch
    return {"loss": F.cross_entropy(self(x), y)}


# after
def training_step(self, batch, batch_idx):
    x, y = batch
    loss = F.cross_entropy(self(x), y)
    self.log_metrics({"loss": loss}, "training")
    return loss


def validation_step(self, batch, batch_idx):
    x, y = batch
    self.log_metrics({"loss": F.cross_entropy(self(x), y)}, "validation")
```

`log_metrics` keeps the old `training/`, `validation/` and `test/` prefixes, so existing
checkpoint monitors and logged metric names are unchanged. `Model` is no longer abstract.

## Optimizers and schedulers

`module_name` plus a `kwargs` dict becomes a `_target_` mapping.

```yaml
# before
optimizer:
  module_name: torch.optim.Adam
  kwargs:
    lr: 0.001
scheduler:
  module_name: torch.optim.lr_scheduler.ExponentialLR
  kwargs:
    gamma: 0.999

# after
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
scheduler:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.999
```

Override paths shorten accordingly: `model_config.optimizer.kwargs.lr=1e-4` becomes
`model.optimizer.lr=1e-4`.

For a scheduler needing `monitor` or `interval`, use Lightning's own `lr_scheduler` dict with a
`_target_` in the `scheduler` slot — see [docs/configuration.md](docs/configuration.md).

## Callbacks

`ModelHparams.checkpoint` and `.early_stopping` are gone; callbacks are the `Trainer`'s
business. Either select the shipped `callbacks` group or declare them yourself:

```yaml
callbacks:
  model_checkpoint:
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${hydra:runtime.output_dir}/checkpoints
    monitor: validation/loss
    save_top_k: 3
    save_last: true
  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: validation/loss
    patience: 10
```

`Model.configure_callbacks` was removed deliberately: Lightning drops any `Trainer` callback
whose type the model also returns, so keeping it would have silently deleted the
`ModelCheckpoint` you configured.

`matmul_precision` moves from `ModelHparams` to a top-level config key — it is a process-global
torch setting, not a model hyperparameter.

## Data

`PyTorchData` becomes `DataModule`, a real `LightningDataModule`.

```python
# before
class MyData(PyTorchData[MyDataHparams]):
    def pre_init(self):
        download(...)

    def get_dataset(self):
        return TensorDataset(...)


trainer.fit(model, data.train_loader, data.val_loader)


# after
class MyData(DataModule[MyDataHparams]):
    def prepare_data(self):
        download(...)

    def get_dataset(self):
        return TensorDataset(...)


trainer.fit(model, datamodule=data)
```

| Before | After |
| --- | --- |
| `data.train` / `.val` / `.test` | `data.train_dataset` / `.val_dataset` / `.test_dataset` |
| `data.train_loader` / `.val_loader` / `.test_loader` | `data.train_dataloader()` / `.val_dataloader()` / `.test_dataloader()` |
| `data.full` | removed — it concatenated the shuffled subsets, so it was never the original dataset |
| `pre_init()` | `prepare_data()` for downloads, `setup(stage)` for state |
| `PyTorchData.split_dataset(...)` | `hydrantic.split_dataset(...)` |
| `loader: {batch_size: 64, shuffle: true}` | `loader: {batch_size: 64}` plus `train_loader: {...}` per stage |

Datasets are now built in `setup`, not `__init__`, which is what makes DDP correct. Two
consequences: nothing is loaded until Lightning (or your own `setup("fit")` call) asks for it,
and a missing `get_dataset` now raises at `setup` time rather than at construction.

Three behaviour fixes worth knowing about:

- `shuffle` no longer leaks onto the validation and test loaders. It is on for train and off
  elsewhere unless a per-stage block says otherwise, and setting it in the shared `loader`
  block is now an error.
- `DataHparams.seed` defaults to `42` instead of a fresh random integer per instantiation, so
  default runs are reproducible. Set it explicitly if you relied on the randomness.
- The split uses a local `torch.Generator` instead of seeding the global RNG, so it no longer
  perturbs anything else and is identical on every rank.

## Inline fitting

`fit_fast` is now `fit_inline`, and it no longer overwrites your configured optimizer.

```python
# before
model.fit_fast(train_loader=loader, n_epochs=10, lr=1e-3, accelerator="gpu", verbose=True)

# after
model.fit_inline(loader, max_epochs=10, accelerator="gpu", enable_progress_bar=True)
```

Any `Trainer` argument passes through, `train_data` also accepts a `Dataset` or a
`LightningDataModule`, and the `Trainer` is returned so you can read
`trainer.callback_metrics`. To train with a different learning rate, change the hyperparameters
rather than the call:

```python
model = Classifier(
    hparams.model_copy(update={"optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-2}})
)
```

## Hyperparameters

`Hparams` is now frozen and rejects unknown fields.

- `hparams.field = value` raises. Use `hparams.model_copy(update={"field": value})`.
- A misspelled key raises instead of being silently dropped. This is the point — it is what
  turns a typo into an error message.
- `{**hparams}` and `dict(hparams)` no longer work (the `Mapping` mixin is gone, and it was
  already broken on pydantic ≥ 2.11). Use `hparams.model_dump()`.
- `Hparams.from_yaml(path, key=...)` is unchanged, and now resolves interpolations and raises on
  a misspelled `key`.
- `Hparams.from_dict(d)` → `MyHparams(**d)`. `Hparams.from_config(cfg)` → let the CLI or
  `hydra.utils.instantiate` build the object.

## CLI

```bash
# before
export HYDRA_CONFIG_PATH=/path/to/conf
python -m hydrantic.cli.fit --config-name train model_config.optimizer.kwargs.lr=1e-4
python -m hydrantic.cli.profile --config-name train

# after
hydrantic fit --config-dir /path/to/conf --config-name train model.optimizer.lr=1e-4
hydrantic fit --config-dir /path/to/conf --config-name train +experiment=profile
```

- One console script, `hydrantic`, with `fit`, `validate`, `test`, `predict` and `check`
  commands. The `hydrantic.cli`, `hydrantic.cli.fit` and `hydrantic.cli.profile` scripts are
  gone.
- `HYDRA_CONFIG_PATH` is replaced by Hydra's own `--config-dir`, or `HYDRANTIC_CONFIG_DIR` if
  you prefer an environment variable. The old variable was read at import time, so merely
  importing the module raised `KeyError` when it was unset.
- `hydrantic.cli.profile` is gone; profiling is configuration (`trainer.profiler=simple` or
  `+experiment=profile`).
- `python -m hydrantic.runner.run` is gone. That module had already been empty for some time,
  so any script still calling it was already broken.
- Section names: `model_config` → `model`, `data_config` → `data`, `run_config.trainer` →
  `trainer`, `run_config.logger` → `logger` (now a dict of named loggers),
  `run_config.resume_from_checkpoint` → `ckpt_path`.
- `WandbLogger` is no longer hardcoded, and `wandb` is no longer a required dependency. Install
  `hydrantic[wandb]` and select `logger=wandb`.

Start from [`examples/conf/train.yaml`](examples/conf/train.yaml) rather than porting an old
`config_blueprint.yaml` by hand, and run `hydrantic check` — it will name every remaining
mismatch at once.

## Weights & Biases artifacts

`Model.load_from_wandb_artifact` is gone. It called `wandb.init()` — creating a run just to
download a file — and hardcoded the artifact filename.

```python
from lightning.pytorch.loggers import WandbLogger

directory = WandbLogger.download_artifact("entity/project/model-abc123:v3", save_dir="artifacts")
model = Classifier.load_from_checkpoint(Path(directory) / "model.ckpt")
```

`download_artifact` is rank-zero only and returns `None` on other ranks, so call it outside a
distributed context or broadcast the result.

## Downloads

`download_url` no longer disables TLS certificate verification. Previously it passed
`ssl._create_unverified_context()` unconditionally, which silently accepted any certificate.
Self-signed hosts now fail loudly; pass an explicit `ssl_context=` if you genuinely need to
relax verification. It also downloads to a `.part` file and renames on success, so an
interrupted download can no longer leave a truncated file that later looks complete.
