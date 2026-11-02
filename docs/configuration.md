# Configuration

A run configuration is a Hydra config. Any mapping with a `_target_` key is imported and called
with the remaining keys as keyword arguments, so anything constructible is configurable —
including classes hydrantic has never heard of. See
[`examples/conf/train.yaml`](https://github.com/hummerichsander/hydrantic/blob/main/examples/conf/train.yaml)
for a complete file.

## Sections

| Key | Meaning |
| --- | --- |
| `model` | your {class}`~hydrantic.model.Model`; its remaining keys are the hyperparameter schema |
| `data` | your {class}`~hydrantic.data.DataModule`; likewise, its remaining keys are its schema |
| `trainer` | keyword arguments for [`lightning.Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html) |
| `logger` | a dict of named loggers, converted to a list for the `Trainer` |
| `callbacks` | a dict of named callbacks, likewise |
| `seed` | passed to `lightning.seed_everything(..., workers=True)`; `null` to skip |
| `ckpt_path` | a `.ckpt` path, or Lightning's `"last"` / `"best"` |
| `matmul_precision` | `highest`, `high` or `medium` |
| `optimized_metric` | a logged metric name, returned to Hydra sweepers |

Hyperparameters sit directly under `model` and `data`, so overrides are short:

```bash
hydrantic fit ... model.hidden_dim=256 model.optimizer.lr=1e-4 data.batch_size=32
```

## Shipped config groups

hydrantic ships Hydra groups for `trainer`, `logger`, `callbacks` and `experiment`. Make them
resolvable with a `hydra.searchpath` entry in your primary config, then select them in
`defaults`:

```yaml
defaults:
  - _self_
  - trainer: default      # default | debug
  - logger: csv           # csv | tensorboard | wandb | beamtime | none, or a list: [csv, wandb]
  - callbacks: default    # default | early_stopping | none

hydra:
  searchpath:
    - pkg://hydrantic.conf
```

Your own `--config-dir` takes precedence, so a group of the same name in your project wins.

Two interpolation forms are in play, and mixing them raises an error: inside the `hydra:` block
use `${hydra.job.name}`; in the rest of the config use the resolver form
`${hydra:runtime.output_dir}`. The shipped groups use the latter to put checkpoints and logs
inside the run directory, which keeps multirun and DDP correct without any Python code.

## Optimizer and scheduler

The short form is a `_target_` node:

```yaml
model:
  optimizer:
    _target_: torch.optim.AdamW
    lr: 1e-3
  scheduler:
    _target_: torch.optim.lr_scheduler.ExponentialLR
    gamma: 0.99
```

For a scheduler needing Lightning's extras, use
[Lightning's own `lr_scheduler` dict](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
with a `_target_` in the `scheduler` slot:

```yaml
model:
  scheduler:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      mode: min
    monitor: validation/loss
    interval: epoch
```

Do not add `_partial_: true` — hydrantic passes the model parameters itself. It is tolerated if
present.

## Validation

`hydrantic check` — and every run, automatically — imports each `_target_` and binds its keys
against the real signature, and validates `model` and `data` against their pydantic schemas.
All problems are reported at once, before anything is constructed:

```
Configuration invalid (3 problems):

  model.optimizer
    _target_: cannot import 'torch.optim.Adan' (did you mean 'torch.optim.Adam'?)

  model.hiden_dim
    unexpected field for ClassifierHparams (did you mean 'hidden_dim'?)

  trainer.max_epocs
    unexpected keyword for lightning.Trainer (did you mean 'max_epochs'?)
```

Because schemas are declared with `extra="forbid"`, a misspelled hyperparameter is an error
rather than a silently ignored key. Values that are only checkable at construction time — a
`ModelCheckpoint` monitoring a metric you never log, for instance — are not caught here.
