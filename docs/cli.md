# Command line interface

```
hydrantic <command> [hydra flags] [key=value ...]
```

Commands are `fit`, `validate`, `test`, `predict` — the four
[Trainer stages](https://lightning.ai/docs/pytorch/stable/common/trainer.html) — plus `check`,
which validates the configuration and exits.

Point it at your config directory with Hydra's `--config-dir`, or set `HYDRANTIC_CONFIG_DIR`
once. Everything after the command belongs to Hydra, so
[its override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) and flags all work
unchanged.

```bash
hydrantic check --config-dir conf --config-name train
hydrantic fit   --config-dir conf --config-name train
hydrantic fit   --config-dir conf --config-name train trainer.max_epochs=50
hydrantic test  --config-dir conf --config-name train ckpt_path=last
hydrantic fit   --config-dir conf --config-name train -m model.optimizer.lr=1e-3,1e-4
hydrantic fit   --config-dir conf --config-name train --cfg job --resolve
hydrantic fit   --config-dir conf --config-name train --hydra-help
```

`python -m hydrantic <command>` is equivalent.

Your project directory is put on `sys.path`, so `_target_: my_project.models.Foo` imports
without installing the project. Installing it with `pip install -e .` is still the more robust
choice.

## What a run does

1. validates the whole configuration and reports every problem at once
   ([Configuration](configuration.md));
2. seeds with `lightning.seed_everything(cfg.seed, workers=True)`;
3. builds `model`, `data` and `trainer` from their `_target_` nodes;
4. logs the fully resolved configuration — Hydra's own `.hydra/config.yaml` is saved
   *unresolved*, so interpolations are opaque there;
5. runs the stage, returning `cfg.optimized_metric` so
   [Hydra sweepers](https://hydra.cc/docs/plugins/optuna_sweeper/) can optimize it.

## Profiling

There is no separate profile command; `Trainer(profiler=...)` is just configuration.

```bash
hydrantic fit --config-dir conf --config-name train +experiment=profile
hydrantic fit --config-dir conf --config-name train trainer.profiler=simple
```

## Distributed runs

DDP works as usual — Lightning re-executes the command in each worker, and the stage survives
that round trip.

```bash
hydrantic fit ... trainer.devices=2 trainer.strategy=ddp
```
