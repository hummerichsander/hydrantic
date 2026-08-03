# Installation

```bash
pip install git+https://github.com/hummerichsander/hydrantic.git
```

Python 3.10+ is required.

## Extras

Loggers other than the CSV logger need their own package:

| Extra | Enables |
| --- | --- |
| `hydrantic[wandb]` | `logger=wandb` |
| `hydrantic[tensorboard]` | `logger=tensorboard` |
| `hydrantic[rich]` | Lightning's rich progress bar and model summary |
| `hydrantic[all]` | all of the above |

```bash
pip install "hydrantic[wandb] @ git+https://github.com/hummerichsander/hydrantic.git"
```

## Development

```bash
git clone git@github.com:hummerichsander/hydrantic.git
cd hydrantic
pip install -e ".[dev]"

pytest
ruff check . && ruff format --check .
```
