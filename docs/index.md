# Welcome to hydrantic's documentation!

Hydrantic is a wrapper for [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/)'s `LightningModule`, implementing some default training and validation steps. Furthermore, it sets up [weights & biases](https://docs.wandb.ai/) for model logging and checkpointing.

Using [pydantic](https://docs.pydantic.dev/latest/)'s well-engineered schema definition and validation mechanisms, hydrantic offers an easy and typed hyperparameter configuration system.

Last, hydrantic enables straight forward command line interface access to model configuration and fitting using [hydra](https://hydra.cc/docs/intro/)'s configuration parsing and overwriting logic.

### Overview

```{toctree}
:maxdepth: 1

installation
getting_started
cli
modules
```
