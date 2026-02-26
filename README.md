# Hydrantic

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Tests](https://github.com/hummerichsander/hydrantic/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/hummerichsander/hydrantic/branch/main/graph/badge.svg)](https://codecov.io/gh/hummerichsander/hydrantic)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://hummerichsander.github.io/hydrantic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## About

Hydrantic is a wrapper for [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/)'s `LightningModule`, implementing some default training and validation steps.

Using [pydantic](https://docs.pydantic.dev/latest/)'s well-engineered schema definition and validation mechanisms, hydrantic offers an easy and typed hyperparameter configuration system.

Last, hydrantic enables straight forward command line interface access to model configuration and fitting using [hydra](https://hydra.cc/docs/intro/)'s configuration parsing and overwriting logic. Furthermore, it sets up [weights & biases](https://docs.wandb.ai/) for model logging and checkpointing.

## Documentation

A short getting started guide, introduction to the CLI module, and full API documentation, are available at [https://hummerichsander.github.io/hydrantic/](https://hummerichsander.github.io/hydrantic/).
