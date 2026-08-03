# Hydrantic

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Tests](https://github.com/hummerichsander/hydrantic/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/hummerichsander/hydrantic/branch/main/graph/badge.svg)](https://codecov.io/gh/hummerichsander/hydrantic)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://hummerichsander.github.io/hydrantic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Typed, Hydra-configurable [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
training — a CLI and an inline API, with [pydantic](https://docs.pydantic.dev/latest/)
validation that catches configuration errors before a run starts.

You write ordinary Lightning. hydrantic adds four things:

- **a CLI** — `hydrantic fit|validate|test|predict`, driven by
  [Hydra](https://hydra.cc/docs/intro/) configs, with multirun and sweeps for free;
- **an inline API** — `model.fit_inline(data)` for notebooks and smoke tests;
- **typed hyperparameters** — one pydantic schema per model and datamodule, so `self.thparams.`
  autocompletes;
- **configuration checking** — every `_target_` is imported and matched against the real
  signature up front, so a typo costs a second rather than an hour.

## Install

```bash
pip install git+https://github.com/hummerichsander/hydrantic.git
```

## Taste

```python
class Classifier(Model[ClassifierHparams]):
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log_metrics({"loss": loss}, "training")
        return loss


trainer = Classifier(hidden_dim=32).fit_inline(dataset, max_epochs=3)
```

```bash
hydrantic fit --config-dir conf --config-name train -m model.optimizer.lr=1e-3,1e-4
```

```
$ hydrantic check --config-dir conf --config-name train

Configuration invalid (2 problems):

  model.hiden_dim
    unexpected field for ClassifierHparams (did you mean 'hidden_dim'?)

  trainer.max_epocs
    unexpected keyword for lightning.Trainer (did you mean 'max_epochs'?)
```

A complete runnable project is in [`examples/`](examples/).

## Documentation

[hummerichsander.github.io/hydrantic](https://hummerichsander.github.io/hydrantic/) —
getting started, configuration, CLI and API reference. Upgrading from 0.0.x? See
[MIGRATION.md](MIGRATION.md).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```
