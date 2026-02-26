# Command line interface

Hydrantic provides a command line interface in the `hydrantic.cli` module. It allows to fit instances of {class}`hydrantic.model.Model` via the CLI. Hyperparameters can be managed using [Hydra](https://hydra.cc/docs/intro/)'s powerful configuration management system. Hydrantic trains the model using pytorch-lightning with [weights & biases](https://docs.wandb.ai/) logging.

## Setup

Hydra allows to parse configurations of model, training, and logging fom configuration files. To setup hydra we have to specify the path to the configuration files, e.g., `/<project-path>/config/`, by setting the environment variable:

```bash
HYDRA_CONFIG_PATH=<path-to-your-config-folder>
```

Furtermore, to connect the logger of the training process, one has to create an API key for your weight & biases profile and set the following environment variable:

```bash
WANDB_API_KEY=<your-api-key>
```

## Configuration files

Hydra requires a configuration yaml file to define the model and training parameters. You can create a configuration file starting from the `config_blueprint.yaml` provided in the repository, which includes all necessary fields and default values. Adjust the parameters as needed for your specific model and training setup. There are four main sections in the configuration file:

### model_config

Under the model configuration key, you can specify you custom model parameters, implemented in your hyperparameter class, as well as the default hyperparameters provided in the {class}`hydrantic.model.ModelHparams` class.

### run_config

The run config includes settings for the training process and the logging configuration.

### data_config

In the data config section you can specify your custom hypereparameters for your implementation of a {class}`hydrantic.data.DataHparams` class. Additionaly, the default settings include splitting hyperparameters and setup for the dataloader.

### hydra

In this section you can specify hydra-specific settings, such as the output directory for logs and checkpoints.

## Fitting your model via the CLI

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
