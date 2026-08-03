# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-03

A breaking release that drops hydrantic's re-implementations of Lightning, Hydra and pydantic
functionality. See [MIGRATION.md](MIGRATION.md) for before/after recipes.

### Added

- Configuration checking: every `_target_` in a composed config is imported and bound against
  the real signature, and `model`/`data` are validated against their pydantic schemas, before
  anything is constructed. All problems are reported at once with `did you mean` suggestions.
  Runs automatically on every run, and available as `hydrantic check`.
- `Target`, an annotated pydantic type for signature-checked `_target_` mappings.
- A single `hydrantic` console script with `fit`, `validate`, `test`, `predict` and `check`
  commands, plus `python -m hydrantic`.
- Shipped Hydra config groups for `trainer`, `logger`, `callbacks` and `experiment`, discoverable
  via `hydra.searchpath: [pkg://hydrantic.conf]`.
- `DataModule`, a real `LightningDataModule` with DDP-correct `setup`, per-stage dataloader
  settings, and support for pre-split datasets via `get_split_datasets`.
- `Model.log_metrics`, an optional helper preserving the `stage/name` logging convention.
- `Model.get_optimizer_params`, for per-group optimizer settings.
- `hydrantic.split_dataset` and `hydrantic.utils.extract_archive`.
- A runnable end-to-end example in `examples/`.

### Changed

- `Hparams` is frozen and rejects unknown fields, so a misspelled hyperparameter is an error.
- `Hparams.from_yaml` resolves interpolations, returns plain containers, and raises on a
  misspelled `key` instead of silently validating an empty dict.
- Optimizer and scheduler configuration uses Hydra `_target_` mappings instead of
  `module_name` plus a `kwargs` dict. Lightning's `lr_scheduler` dict form is accepted as-is.
- `fit_fast` is now `fit_inline`: it no longer mutates the model's hyperparameters, accepts a
  `Dataset`, `DataLoader` or `LightningDataModule`, forwards arbitrary `Trainer` arguments, and
  returns the `Trainer`.
- `wandb` is an optional extra rather than a required dependency, and the CLI no longer
  hardcodes `WandbLogger`.
- Checkpoint and log directories are wired to the Hydra run directory declaratively via
  `${hydra:runtime.output_dir}`.
- The package is flat: `hydrantic.model`, `hydrantic.data`, `hydrantic.hparams`,
  `hydrantic.utils` are modules, and the public API is importable from `hydrantic`.
- Tests moved out of the installed package into a top-level `tests/`.
- Requires Python 3.10+ and Lightning 2.5+. Dependency pins loosened.

### Removed

- `Model.compute_metrics` and the `training_step`/`validation_step`/`test_step` implementations
  that called it. Write the Lightning hooks directly; `Model` is no longer abstract.
- `Model.configure_callbacks` and `has_logger`. Lightning drops any `Trainer` callback whose
  type the model also returns, so this silently deleted configured callbacks.
- `PyTorchData`, `DataLoaderHparams`, `OptimizerHparams`, `SchedulerHparams`,
  `CheckpointHparams`, `EarlyStoppingHparams`, `TrainerHparams`, `LoggerHparams`, `RunHparams` —
  each mirrored an upstream signature by hand.
- `Model.load_from_wandb_artifact`, in favour of `WandbLogger.download_artifact`.
- `utils.import_from_string`, in favour of `hydra.utils.instantiate`.
- `hydrantic.cli.profile`; profiling is configuration (`trainer.profiler=simple`).
- The `HYDRA_CONFIG_PATH` environment variable, in favour of Hydra's `--config-dir`.
- `Hparams.from_config`, `Hparams.from_dict` and the `Mapping` mixin; use `model_dump()`.
- The empty `hydrantic.runner` package.

### Fixed

- `shuffle` no longer applied to the validation and test dataloaders.
- `DataHparams.seed` no longer defaults to a fresh random integer per instantiation, which made
  every default run irreproducible.
- Dataset splitting no longer perturbs the global RNG, and is identical across ranks.
- Hyperparameters stored in checkpoints are plain primitives, so `torch.load` with torch 2.6+'s
  default `weights_only=True` and `load_from_checkpoint` both work.
- `Hparams`' `Mapping` interface was broken on pydantic 2.11+; it is gone.
- `download_url` no longer disables TLS certificate verification, and no longer leaves a
  truncated file behind on an interrupted download.
- Importing the CLI no longer raises `KeyError` when an environment variable is unset.
- The CLI no longer crashes on a two-way dataset split, or when no checkpoint is configured.

## [0.0.2] - 2026-02-26

Initial documented release.
