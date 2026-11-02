import functools

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from hydrantic.hparams import Hparams
from hydrantic.targets import (
    ConfigError,
    Target,
    check_config,
    check_optional_requirement,
    check_target_signature,
    instantiate_module,
    instantiate_target,
    resolve_target,
    validate_target,
)


class WithTarget(Hparams):
    """A schema with a signature-checked `_target_` field."""

    optimizer: Target = {"_target_": "torch.optim.Adam"}


def problems_at(config: DictConfig, path: str) -> list[str]:
    """Returns the messages reported for one config path.

    :param config: The config to check.
    :param path: The exact problem path to filter on.
    :return: The messages reported at that path."""

    return [problem.message for problem in check_config(config) if problem.path == path]


class TestResolveTarget:
    """Tests importing a dotted path."""

    def test_resolves_class(self) -> None:
        import torch

        assert resolve_target("torch.optim.Adam") is torch.optim.Adam

    def test_unknown_attribute_suggests(self) -> None:
        with pytest.raises(ImportError, match=r"did you mean 'torch.optim.Adam'"):
            resolve_target("torch.optim.Adan")

    def test_unknown_module_raises(self) -> None:
        with pytest.raises(ImportError, match="cannot import 'nope_xyz.Thing'"):
            resolve_target("nope_xyz.Thing")


class TestCheckTargetSignature:
    """Tests binding a node's keys against the real signature."""

    def test_valid_node(self) -> None:
        node = {"_target_": "torch.optim.Adam", "lr": 1e-3}
        assert check_target_signature(node, skip_first=True) == []

    def test_unexpected_keyword(self) -> None:
        node = {"_target_": "torch.optim.Adam", "lr_rate": 1e-3}
        problems = check_target_signature(node, skip_first=True)
        assert problems == [("lr_rate", "unexpected keyword for torch.optim.Adam")]

    def test_unexpected_keyword_suggests(self) -> None:
        node = {"_target_": "lightning.Trainer", "max_epocs": 1}
        (key, message), *rest = check_target_signature(node)
        assert (key, rest) == ("max_epocs", [])
        assert "did you mean 'max_epochs'" in message

    def test_missing_required_argument(self) -> None:
        node = {"_target_": "torch.nn.Linear"}
        problems = check_target_signature(node)
        assert ("in_features", "missing required argument for torch.nn.Linear") in problems

    def test_skip_first_ignores_leading_parameter(self) -> None:
        node = {"_target_": "torch.optim.Adam", "lr": 1e-3}
        # `params` is supplied positionally, so it must not be reported as missing.
        assert not check_target_signature(node, skip_first=True)
        assert ("params", "missing required argument for torch.optim.Adam") in (
            check_target_signature(node)
        )

    def test_underscore_keys_are_ignored(self) -> None:
        node = {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3}
        assert check_target_signature(node, skip_first=True) == []

    def test_target_accepting_kwargs_accepts_anything(self) -> None:
        # TinyModel.__init__ takes **kwargs, so nothing can be an unexpected keyword.
        node = {"_target_": "conftest.TinyModel", "anything": 1}
        assert check_target_signature(node) == []

    def test_non_string_target(self) -> None:
        problems = check_target_signature({"_target_": 42})
        assert problems == [("_target_", "must be a dotted path string, got int")]


class TestOptionalRequirements:
    """Tests the missing-optional-package check for the shipped logger groups."""

    def test_absent_package_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        message = check_optional_requirement("lightning.pytorch.loggers.WandbLogger")
        assert message is not None
        assert "pip install 'hydrantic[wandb]'" in message

    def test_present_package_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
        assert check_optional_requirement("lightning.pytorch.loggers.WandbLogger") is None

    def test_unimportable_target_gets_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # BeamtimeLogger lives inside its optional package, so the import itself fails.
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        node = {"_target_": "beamtime.lightning.BeamtimeLogger"}
        ((key, message),) = check_target_signature(node)
        assert key == "_target_"
        assert "pip install 'hydrantic[beamtime]'" in message

    def test_unknown_target_is_ignored(self) -> None:
        assert check_optional_requirement("torch.optim.Adam") is None

    def test_reported_through_check_config(
        self, run_config: DictConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        run_config.trainer.logger = {"_target_": "lightning.pytorch.loggers.WandbLogger"}

        (message,) = problems_at(run_config, "trainer.logger._target_")
        assert "hydrantic[wandb]" in message


class TestValidateTarget:
    """Tests the `Target` annotated type."""

    def test_accepts_short_form(self) -> None:
        assert WithTarget(optimizer={"_target_": "torch.optim.SGD", "lr": 0.1})

    def test_accepts_lightning_scheduler_dict_form(self) -> None:
        node = {
            "scheduler": {"_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau", "mode": "min"},
            "monitor": "validation/loss",
            "interval": "epoch",
        }
        assert validate_target(node) == node

    def test_rejects_missing_target(self) -> None:
        with pytest.raises(ValueError, match="needs a `_target_` key"):
            validate_target({"lr": 1e-3})

    def test_rejects_callable(self) -> None:
        partial = functools.partial(dict, a=1)
        with pytest.raises(ValueError, match="expected a `_target_` mapping"):
            validate_target(partial)

    def test_rejects_callable_with_recursive_hint(self) -> None:
        with pytest.raises(ValueError, match="_recursive_"):
            validate_target(functools.partial(dict, a=1))

    def test_normalises_dict_config(self) -> None:
        node = OmegaConf.create({"_target_": "torch.optim.Adam", "lr": 1e-3})
        validated = validate_target(node)
        assert type(validated) is dict

    def test_surfaces_signature_problems(self) -> None:
        with pytest.raises(ValidationError, match="unexpected keyword for torch.optim.Adam"):
            WithTarget(optimizer={"_target_": "torch.optim.Adam", "lr_rate": 1e-3})

    def test_surfaces_import_problems(self) -> None:
        with pytest.raises(ValidationError, match="cannot import 'torch.optim.Adan'"):
            WithTarget(optimizer={"_target_": "torch.optim.Adan"})


class TestCheckConfig:
    """Tests validating a whole composed configuration."""

    def test_valid_config_has_no_problems(self, run_config: DictConfig) -> None:
        assert check_config(run_config) == []

    def test_model_field_typo(self, run_config: DictConfig) -> None:
        run_config.model.hiden_dim = 256
        (message,) = problems_at(run_config, "model.hiden_dim")
        assert "did you mean 'hidden_dim'" not in message  # TinyHparams has no hidden_dim
        assert "unexpected field for TinyHparams" in message

    def test_model_field_typo_suggests(self, run_config: DictConfig) -> None:
        run_config.model.input_dimm = 4
        (message,) = problems_at(run_config, "model.input_dimm")
        assert "did you mean 'input_dim'" in message

    def test_data_field_typo(self, run_config: DictConfig) -> None:
        run_config.data.n_samples = 4
        (message,) = problems_at(run_config, "data.n_samples")
        assert "did you mean 'num_samples'" in message

    def test_trainer_keyword_typo(self, run_config: DictConfig) -> None:
        run_config.trainer.max_epocs = 5
        (message,) = problems_at(run_config, "trainer.max_epocs")
        assert "did you mean 'max_epochs'" in message

    def test_optimizer_target_typo(self, run_config: DictConfig) -> None:
        run_config.model.optimizer._target_ = "torch.optim.Adan"
        (message,) = problems_at(run_config, "model.optimizer")
        assert "cannot import 'torch.optim.Adan'" in message

    def test_all_problems_are_reported_together(self, run_config: DictConfig) -> None:
        run_config.model.input_dimm = 4
        run_config.data.n_samples = 4
        run_config.trainer.max_epocs = 5

        paths = {problem.path for problem in check_config(run_config)}
        assert paths == {"model.input_dimm", "data.n_samples", "trainer.max_epocs"}

    def test_nested_callback_target_is_checked(self, run_config: DictConfig) -> None:
        run_config.trainer.callbacks = [
            {"_target_": "lightning.pytorch.callbacks.EarlyStopping", "patients": 3}
        ]
        (message,) = problems_at(run_config, "trainer.callbacks[0].patients")
        assert "did you mean 'patience'" in message

    def test_missing_target_in_model(self, run_config: DictConfig) -> None:
        del run_config.model._target_
        assert problems_at(run_config, "model") == ["needs a `_target_` key"]

    def test_unimportable_model_target(self, run_config: DictConfig) -> None:
        run_config.model._target_ = "nope_xyz.Model"
        (message,) = problems_at(run_config, "model._target_")
        assert "cannot import" in message

    def test_target_without_schema_is_ignored(self, run_config: DictConfig) -> None:
        run_config.model = {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1}
        assert check_config(run_config) == []

    def test_absent_sections_are_skipped(self) -> None:
        assert check_config(OmegaConf.create({"seed": 1})) == []


class TestConfigError:
    """Tests the aggregated error message."""

    def test_message_lists_every_problem(self, run_config: DictConfig) -> None:
        run_config.model.input_dimm = 4
        run_config.trainer.max_epocs = 5

        message = str(ConfigError(check_config(run_config)))
        assert "Configuration invalid (2 problems)" in message
        assert "model.input_dimm" in message
        assert "trainer.max_epocs" in message

    def test_singular_wording(self, run_config: DictConfig) -> None:
        run_config.trainer.max_epocs = 5
        assert "(1 problem)" in str(ConfigError(check_config(run_config)))


class TestInstantiateTarget:
    """Tests building an object from a `_target_` mapping."""

    def test_positional_argument(self) -> None:
        import torch

        parameters = [torch.nn.Parameter(torch.zeros(1))]
        optimizer = instantiate_target({"_target_": "torch.optim.Adam", "lr": 0.5}, parameters)
        assert optimizer.param_groups[0]["lr"] == 0.5

    def test_tolerates_partial_flag(self) -> None:
        import torch

        parameters = [torch.nn.Parameter(torch.zeros(1))]
        node = {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.25}
        optimizer = instantiate_target(node, parameters)
        assert optimizer.param_groups[0]["lr"] == 0.25

    def test_does_not_mutate_input(self) -> None:
        node = {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 3}
        instantiate_target(node)
        assert node == {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 3}

    def test_unimportable_target_raises(self) -> None:
        with pytest.raises(Exception, match="nope_xyz.Thing"):
            instantiate_target({"_target_": "nope_xyz.Thing"})


class TestInstantiateModule:
    """Tests that a module's hyperparameters reach it as data."""

    def test_nested_targets_stay_data(self, model_config: DictConfig) -> None:
        model = instantiate_module(model_config)
        assert type(model.thparams.optimizer) is dict
        assert model.thparams.optimizer["_target_"] == "torch.optim.Adam"

    def test_no_omegaconf_containers_survive(self, model_config: DictConfig) -> None:
        model_config.scheduler = {
            "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1},
            "monitor": "validation/loss",
        }
        model = instantiate_module(model_config)

        assert type(model.thparams.scheduler) is dict
        assert type(model.thparams.scheduler["scheduler"]) is dict

    def test_hparams_are_json_primitive(self, model_config: DictConfig) -> None:
        model = instantiate_module(model_config)
        stored = model.hparams["hparams"]

        def assert_plain(value: object) -> None:
            assert type(value) in (dict, list, str, int, float, bool, type(None))
            if isinstance(value, dict):
                for item in value.values():
                    assert_plain(item)
            elif isinstance(value, list):
                for item in value:
                    assert_plain(item)

        assert_plain(stored)
