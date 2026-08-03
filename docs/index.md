# hydrantic

Typed, Hydra-configurable [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
training.

hydrantic is a thin layer over Lightning, [Hydra](https://hydra.cc/docs/intro/) and
[pydantic](https://docs.pydantic.dev/latest/). It gives you

- **a CLI** — `hydrantic fit|validate|test|predict`, driven by Hydra configs, with multirun and
  sweeps for free;
- **an inline API** — `model.fit_inline(data)` for notebooks and smoke tests;
- **typed hyperparameters** — one pydantic schema per model and datamodule, so `self.thparams.`
  autocompletes and a bad config fails immediately;
- **configuration checking** — every `_target_` in the config is imported and matched against
  the real signature *before* the run starts, so a typo costs a second, not an hour.

You write ordinary Lightning: your own `forward`, your own `training_step`. hydrantic does not
wrap or replace anything Lightning already provides.

```{toctree}
:maxdepth: 1

installation
getting_started
configuration
cli
api
```
