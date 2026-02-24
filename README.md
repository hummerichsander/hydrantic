# Hydrantic

## About

Hydrantic is an easy-to-use model prototyping and training interface combing pydantic's well-engineered schema
definition with the configuration capabilities offered by hydra. It furthermore uses pytorch lightning together with
weights & biases for fast model training and logging.

## Installation

## Hyperparameters

Hyperparameters in Hydrantic are managed by an extension of pydantic's BaseModel. This enables powerfull type-checking and validation of hyperparamters.

### The Hparams Class

The `Hparams` class is the base class for all hyperparameter definitions. It combines Pydantic's type validation with dictionary-like access and provides convenient methods for loading configurations from various sources.

**Key Features:**

- **Type Safety**: Leverages Pydantic's validation to ensure hyperparameters match expected types
- **Multiple Input Sources**: Load from OmegaConf/DictConfig, dictionaries, or YAML files
- **Dictionary-like Access**: Implements `Mapping` interface for flexible parameter access
- **Field Validation**: Use `Hparam` (alias for Pydantic's `Field`) to add constraints and descriptions (optional)

**Example Usage:**

```python
from hydrantic.hparams import Hparams, Hparam

class TestHparams(Hparams):
    """Hyperparameters for a neural network model."""
    learning_rate: float = Hparam(default=1e-3, gt=0.0)
    hidden_dim: int = Hparam(default=128, ge=1)
    num_layers: int = Hparam(default=3, ge=1, le=10)
    dropout: float = Hparam(default=0.1, ge=0.0, lt=1.0)

# Create as class instance
hparams = ModelHparams(learning_rate=0.001, hidden_dim=256, num_layers=4, dropout=0.2)

# Create from dictionary
dict_config = dict(learning_rate=0.001, hidden_dim=256, num_layers=4, dropout=0.2)
hparams.from_dict(dict_config)

# Create from YAML file
hparams = ModelHparams.from_yaml("config.yaml", key="model")

# Create from Hydra/OmegaConf
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")
hparams = ModelHparams.from_config(cfg.model)

# Access as class
print(hparams.hidden_dim)  # 256

# Access like a dictionary
print(hparams["learning_rate"])  # 0.001
print(hparams.keys())  # dict_keys(['learning_rate', 'hidden_dim', 'num_layers', 'dropout'])
for key, value in hparams.items():
    print(f"{key}: {value}")
print(**hparams)  # Unpack all hyperparameters as keyword arguments
```

**Validation Example:**

```python
# This will raise a ValidationError because learning_rate must be > 0
try:
    hparams = ModelHparams(learning_rate=-0.001)
except ValueError as e:
    print(f"Validation error: {e}")
```

## Models

### The Model Class

The `Model` class is the base for all Hydrantic models, extending PyTorch Lightning’s `LightningModule`. It provides shared functionality like parameter counting and training loops, so subclasses only need to define their architecture.

Each subclass must implement `compute_metrics`, returning a dictionary of metric names to tensors, with `"loss"` used for optimization.

Passing the `ModelHparams` type to the parent `Model` class constructor ensures that all hyperparameters are properly initialized, typed, and accessible within the model.

**Example Usage:**

```python
from hydrantic.model.model import Model, ModelHparams

class MyModelHparams(ModelHparams):
    """Hyperparameters specific to MyModel."""
    hidden_dim: int = 128
    num_layers: int = 3

class MyModel(Model[MyModelHparams]):  # Passing the MyModelHparams type to Model
    """A simple neural network model."""
    def __init__(self, hparams: MyModelHparams):
        # The superclass Model takes care of validating the hparams and provides a typed version
        # of the hparams via self.thparams attribute.
        super().__init__(hparams)

        # Define model layers here
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(self.thparams.hidden_dim, self.thparams.hidden_dim)
            for _ in range(self.thparams.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

    def compute_metrics(self, outputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        # Define metric computation here
        loss = torch.nn.functional.mse_loss(outputs, targets)
        return {"loss": loss}
```

### The ModelHparams Class

Besides custom model hyperparameters, the `ModelHparams` class manages all configuration aspects of model training, including optimizer settings, learning rate scheduling, checkpointing, and early stopping. It uses nested `Hparams` classes to organize related hyperparameters into logical groups.

**Structure:**

- **OptimizerHparams**: Configures the optimizer (default: Adam)
- **SchedulerHparams**: Configures the learning rate scheduler (optional)
- **CheckpointHparams**: Configures model checkpointing behavior
- **EarlyStoppingHparams**: Configures early stopping criteria (optional)

**Example Usage:**

```python
from hydrantic.model.hparams import (
    ModelHparams,
    OptimizerHparams,
    SchedulerHparams,
    CheckpointHparams,
    EarlyStoppingHparams
)

# Create with default settings
model_hparams = ModelHparams()

# Customize optimizer
model_hparams = ModelHparams(
    optimizer=OptimizerHparams(
        module_name="torch.optim.AdamW",
        kwargs={"lr": 1e-4, "weight_decay": 0.01}
    )
)

# Add learning rate scheduler
model_hparams = ModelHparams(
    optimizer=OptimizerHparams(
        module_name="torch.optim.Adam",
        kwargs={"lr": 1e-3}
    ),
    scheduler=SchedulerHparams(
        module_name="torch.optim.lr_scheduler.StepLR",
        kwargs={"step_size": 10, "gamma": 0.1}
    )
)

# Configure checkpointing and early stopping
model_hparams = ModelHparams(
    checkpoint=CheckpointHparams(
        dirpath="checkpoints/",
        monitor="validation/loss",
        mode="min",
        save_top_k=3,
        save_last=True
    ),
    early_stopping=EarlyStoppingHparams(
        monitor="validation/loss",
        patience=5,
        mode="min"
    )
)

# Set matrix multiplication precision
model_hparams = ModelHparams(
    matmul_precision="high"  # Options: "highest", "high", "medium"
)
```

**Loading from Configuration Files:**

```python
# From YAML file
model_hparams = ModelHparams.from_yaml("config.yaml", key="model_config")

# From Hydra/OmegaConf
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")
model_hparams = ModelHparams.from_config(cfg.model)
```

**Example YAML Configuration:**

```yaml
model_config:
  ...  # your custom hyperparameters should be specified here
  matmul_precision: high
  optimizer:
    module_name: torch.optim.AdamW
    kwargs:
      lr: 0.001
      weight_decay: 0.01
  scheduler:
    module_name: torch.optim.lr_scheduler.CosineAnnealingLR
    kwargs:
      T_max: 100
  checkpoint:
    dirpath: checkpoints/
    monitor: validation/loss
    mode: min
    save_top_k: 3
  early_stopping:
    monitor: validation/loss
    patience: 10
    mode: min
```

## Data

Hydrantic does provide a built-in data module

## Model Training

The `Model` class provides built-in training loops using PyTorch Lightning. You can customize training behavior through the `ModelHparams` configuration (see above).

Per default, training uses a Weights & Biases logger for experiment tracking. Make sure to have it configured in your environment, especially the API key set up, before starting training. See [Weights & Biases Documentation](https://docs.wandb.ai/) for details.

### Command Line Training

Hydrantic provides a command line interface to train models defined using the `Model` class. You can specify the configuration file and other parameters directly from the command line. For hyperparameter-parsing and overrides, Hydrantic leverages Hydra's powerful configuration management system.

Hydra requires a configuration yaml file to define the model and training parameters. You can create a configuration file starting from the `config_blueprint.yaml` provided in the repository, which includes all necessary fields and default values. Adjust the parameters as needed for your specific model and training setup. In order for hydra to locate your configuration file, make sure to set the `HYDRA_CONFIG_PATH` environment variable to the directory containing your config file before running the training script.

Training from a configuration file `config.yaml` in the configuration directory specified under `HYDRA_CONFIG_PATH` can be started using the following command:

```bash
python -m hydrantic.cli.fit --config-name your_config
```

Using hydras overwrite logic, you can also override specific hyperparameters directly from the command line. For example, to change the learning rate and batch size, you can run:

```bash
python -m hydrantic.cli.fit --config-name your_config \
optimizer.kwargs.lr=0.0005 \
data.batch_size=64
```

To add a brand-new key to the configuration, use the syntax:

```bash
python -m hydrantic.cli.fit --config-name your_config \
+new_key=new_value
```

An extensive guide on Hydra's configuration management and command line overrides can be found in the [Hydra Documentation](https://hydra.cc/docs/intro/).
