import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra._internal.config_repository import ConfigRepository
from hydra._internal.utils import create_config_search_path

from hydrantic.cli.app import STAGES, _resolve_ckpt_path
from hydrantic.cli.main import COMMANDS, normalize_args
from hydrantic.targets import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples"


class TestNormalizeArgs:
    """Tests the argv rewrite that fronts Hydra."""

    def test_command_becomes_an_override(self) -> None:
        assert normalize_args(["fit"], env={}) == ["++stage=fit"]

    @pytest.mark.parametrize("command", COMMANDS)
    def test_every_command_is_accepted(self, command: str) -> None:
        assert normalize_args([command], env={}) == [f"++stage={command}"]

    def test_hydra_arguments_are_preserved_in_order(self) -> None:
        args = ["fit", "--config-dir", "conf", "-m", "model.h=8,16"]
        assert normalize_args(args, env={}) == [
            "--config-dir",
            "conf",
            "-m",
            "model.h=8,16",
            "++stage=fit",
        ]

    def test_idempotent(self) -> None:
        # Lightning's DDP launcher re-executes sys.argv[1:] verbatim in every worker, so a
        # second pass must not strip the command again or inject a second config directory.
        once = normalize_args(["fit", "--config-dir", "conf", "-m", "x=1,2"], env={})
        assert normalize_args(once, env={}) == once

    def test_idempotent_with_ddp_suffix(self) -> None:
        once = normalize_args(["fit", "--config-dir", "conf"], env={})
        relaunched = [*once, "hydra.run.dir=/tmp/x", "hydra.job.name=train_ddp_process_1"]
        assert normalize_args(relaunched, env={}) == relaunched

    def test_stage_override_form_passes_through(self) -> None:
        assert normalize_args(["stage=test", "model.h=4"], env={}) == ["stage=test", "model.h=4"]

    def test_bare_hydra_flag_passes_through(self) -> None:
        assert normalize_args(["--config-dir", "conf"], env={}) == ["--config-dir", "conf"]

    def test_unknown_command_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown command 'fti'"):
            normalize_args(["fti"], env={})

    def test_empty_args(self) -> None:
        assert normalize_args([], env={}) == []

    def test_env_config_dir_is_injected(self) -> None:
        args = normalize_args(["fit"], env={"HYDRANTIC_CONFIG_DIR": "/etc/conf"})
        assert args == ["--config-dir", "/etc/conf", "++stage=fit"]

    @pytest.mark.parametrize("flag", ["--config-dir", "-cd", "--config-path", "-cp"])
    def test_explicit_config_dir_wins_over_env(self, flag: str) -> None:
        args = normalize_args(["fit", flag, "mine"], env={"HYDRANTIC_CONFIG_DIR": "/etc/conf"})
        assert "/etc/conf" not in args

    def test_equals_form_of_config_dir_wins_over_env(self) -> None:
        args = normalize_args(["fit", "--config-dir=mine"], env={"HYDRANTIC_CONFIG_DIR": "/etc"})
        assert "/etc" not in args


class TestResolveCkptPath:
    """Tests checkpoint path resolution."""

    def test_none(self) -> None:
        assert _resolve_ckpt_path(None) is None

    @pytest.mark.parametrize("sentinel", ["best", "last", "hpc", "registry"])
    def test_sentinels_are_untouched(self, sentinel: str) -> None:
        assert _resolve_ckpt_path(sentinel) == sentinel

    def test_urls_are_untouched(self) -> None:
        url = "s3://bucket/model.ckpt"
        assert _resolve_ckpt_path(url) == url

    def test_relative_path_becomes_absolute(self) -> None:
        resolved = _resolve_ckpt_path("checkpoints/last.ckpt")
        assert Path(resolved).is_absolute()


class TestStages:
    """Tests the advertised commands."""

    def test_stages_match_trainer_methods(self) -> None:
        import lightning

        for stage in STAGES:
            assert callable(getattr(lightning.Trainer, stage))

    def test_check_is_a_command_but_not_a_stage(self) -> None:
        assert "check" in COMMANDS
        assert "check" not in STAGES


class TestShippedConfigGroups:
    """Tests that the packaged Hydra config groups are discoverable."""

    @pytest.fixture
    def repository(self) -> ConfigRepository:
        return ConfigRepository(config_search_path=create_config_search_path("pkg://hydrantic.conf"))

    @pytest.mark.parametrize("group", ["trainer", "logger", "callbacks", "experiment"])
    def test_group_exists(self, repository: ConfigRepository, group: str) -> None:
        assert repository.group_exists(group)

    def test_trainer_options(self, repository: ConfigRepository) -> None:
        assert set(repository.get_group_options("trainer")) >= {"default", "debug"}

    def test_logger_options(self, repository: ConfigRepository) -> None:
        options = set(repository.get_group_options("logger"))
        assert options >= {"csv", "tensorboard", "wandb", "none"}

    def test_callbacks_options(self, repository: ConfigRepository) -> None:
        options = set(repository.get_group_options("callbacks"))
        assert options >= {"default", "early_stopping", "none"}

    def test_trainer_default_targets_the_trainer(self, repository: ConfigRepository) -> None:
        # Loaded directly, the file's own keys are at the top level; group packaging happens
        # during composition. Only non-interpolated keys are readable here, because the
        # `hydra:` resolver is registered by a running Hydra app.
        loaded = repository.load_config("trainer/default.yaml")
        assert loaded.config["_target_"] == "lightning.Trainer"
        assert loaded.config["max_epochs"] == 100

    def test_conf_package_has_an_init(self) -> None:
        # ImportlibResourcesConfigSource.available() literally checks for __init__.py.
        import hydrantic.conf

        assert Path(hydrantic.conf.__file__).name == "__init__.py"

    def test_every_group_file_is_packaged(self) -> None:
        # Guards the wheel: `[tool.setuptools.package-data]` must ship conf/**/*.yaml.
        from importlib.resources import files

        root = files("hydrantic.conf")
        for group in ("trainer", "logger", "callbacks", "experiment"):
            yaml_files = [f.name for f in (root / group).iterdir() if f.name.endswith(".yaml")]
            assert yaml_files, f"no YAML shipped for the {group!r} group"


class TestRunStage:
    """Tests the Hydra task function in process."""

    @pytest.fixture(autouse=True)
    def _no_sys_path_mutation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `_add_cwd_to_path` needs a running Hydra app for HydraConfig.get().
        monkeypatch.setattr("hydrantic.cli.app._add_cwd_to_path", lambda: None)

    def test_fit(self, run_config) -> None:
        from hydrantic.cli.app import run_stage

        assert run_stage(run_config) is None

    def test_invalid_config_raises_before_instantiating(self, run_config) -> None:
        from hydrantic.cli.app import run_stage

        run_config.trainer.max_epocs = 1
        with pytest.raises(ConfigError, match="trainer.max_epocs"):
            run_stage(run_config)

    def test_unknown_stage_raises(self, run_config) -> None:
        from hydrantic.cli.app import run_stage

        run_config.stage = "frobnicate"
        with pytest.raises(ValueError, match="stage must be one of"):
            run_stage(run_config)

    def test_stage_defaults_to_fit(self, run_config) -> None:
        from hydrantic.cli.app import run_stage

        del run_config.stage
        assert run_stage(run_config) is None

    def test_optimized_metric_is_returned(self, run_config) -> None:
        from hydrantic.cli.app import run_stage

        run_config.optimized_metric = "validation/loss"
        assert isinstance(run_stage(run_config), float)

    def test_unknown_optimized_metric_warns(self, run_config, caplog) -> None:
        from hydrantic.cli.app import run_stage

        run_config.optimized_metric = "validation/nope"
        with caplog.at_level("WARNING", logger="hydrantic"):
            assert run_stage(run_config) is None
        assert "was not found" in caplog.text

    def test_non_datamodule_data_raises(self, run_config) -> None:
        from hydrantic.cli.app import run_stage

        run_config.data = {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1}
        with pytest.raises(TypeError, match="must instantiate a LightningDataModule"):
            run_stage(run_config)

    def test_matmul_precision_is_applied(self, run_config) -> None:
        from hydrantic.cli.app import run_stage

        original = torch.get_float32_matmul_precision()
        try:
            run_config.matmul_precision = "high"
            run_stage(run_config)
            assert torch.get_float32_matmul_precision() == "high"
        finally:
            torch.set_float32_matmul_precision(original)

    def test_check_only_accepts_a_valid_config(self, run_config, capsys) -> None:
        from hydrantic.cli.app import check_only

        assert check_only(run_config) == 0
        assert "Configuration is valid." in capsys.readouterr().out

    def test_check_only_rejects_an_invalid_config(self, run_config, capsys) -> None:
        from hydrantic.cli.app import check_only

        run_config.model.input_dimm = 1
        assert check_only(run_config) == 1
        assert "model.input_dimm" in capsys.readouterr().err


def run_cli(*args: str, cwd: Path = EXAMPLE_DIR) -> subprocess.CompletedProcess:
    """Runs the hydrantic CLI in a subprocess.

    :param args: Arguments to pass to the CLI.
    :param cwd: Working directory to run in.
    :return: The completed process."""

    return subprocess.run(
        [sys.executable, "-m", "hydrantic", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.skipif(not EXAMPLE_DIR.exists(), reason="examples/ is not present")
class TestCliEndToEnd:
    """Tests the CLI against the shipped example project."""

    def test_help(self) -> None:
        result = run_cli("--help")
        assert result.returncode == 0
        assert "hydrantic <command>" in result.stdout

    def test_version(self) -> None:
        result = run_cli("--version")
        assert result.returncode == 0
        assert result.stdout.startswith("hydrantic ")

    def test_unknown_command(self) -> None:
        result = run_cli("frobnicate")
        assert result.returncode == 2
        assert "unknown command" in result.stderr

    def test_check_accepts_the_example(self) -> None:
        result = run_cli("check", "--config-dir", "conf", "--config-name", "train")
        assert result.returncode == 0, result.stderr
        assert "Configuration is valid." in result.stdout

    def test_check_rejects_a_model_typo(self) -> None:
        result = run_cli(
            "check", "--config-dir", "conf", "--config-name", "train", "++model.hiden_dim=1"
        )
        assert result.returncode == 1
        assert "model.hiden_dim" in result.stderr
        assert "did you mean 'hidden_dim'" in result.stderr

    def test_check_rejects_a_trainer_typo(self) -> None:
        result = run_cli(
            "check", "--config-dir", "conf", "--config-name", "train", "++trainer.max_epocs=1"
        )
        assert result.returncode == 1
        assert "did you mean 'max_epochs'" in result.stderr

    def test_check_reports_every_problem_at_once(self) -> None:
        result = run_cli(
            "check",
            "--config-dir",
            "conf",
            "--config-name",
            "train",
            "++model.input_dimm=1",
            "++trainer.max_epocs=1",
            "++data.n_samples=1",
        )
        assert result.returncode == 1
        assert "3 problems" in result.stderr

    def test_check_rejects_an_unimportable_target(self) -> None:
        result = run_cli(
            "check",
            "--config-dir",
            "conf",
            "--config-name",
            "train",
            "model.optimizer._target_=torch.optim.Adan",
        )
        assert result.returncode == 1
        assert "cannot import 'torch.optim.Adan'" in result.stderr

    @pytest.mark.slow
    def test_fit_runs(self, tmp_path: Path) -> None:
        result = run_cli(
            "fit",
            "--config-dir",
            "conf",
            "--config-name",
            "train",
            "trainer.fast_dev_run=true",
            f"hydra.run.dir={tmp_path}",
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.slow
    def test_fit_writes_checkpoints_into_the_run_dir(self, tmp_path: Path) -> None:
        result = run_cli(
            "fit",
            "--config-dir",
            "conf",
            "--config-name",
            "train",
            "trainer.max_epochs=1",
            "data.num_samples=256",
            "data.batch_size=16",
            f"hydra.run.dir={tmp_path}",
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "checkpoints" / "last.ckpt").exists()
        assert (tmp_path / "metrics.csv").exists()
