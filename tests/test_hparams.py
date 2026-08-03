from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from conftest import InnerHparams, NestedHparams, SampleHparams

from hydrantic.hparams import (
    Hparams,
    HparamsModule,
    _hparams_arg_name,
    _resolve_hparams_schema,
    to_plain_container,
)


class Module(HparamsModule[SampleHparams]):
    """A plain (non-Lightning) module used to test the mixin in isolation."""


class TestHparamsInitialization:
    """Tests construction and defaults of `Hparams`."""

    def test_defaults(self, sample_hparams: SampleHparams) -> None:
        assert sample_hparams.learning_rate == 1e-3
        assert sample_hparams.batch_size == 32

    def test_overrides(self) -> None:
        hparams = SampleHparams(learning_rate=0.1, batch_size=8)
        assert hparams.learning_rate == 0.1
        assert hparams.batch_size == 8

    def test_nested(self, nested_hparams: NestedHparams) -> None:
        assert isinstance(nested_hparams.inner, InnerHparams)
        assert nested_hparams.inner.lr == 1e-3

    def test_equality_and_repr(self) -> None:
        assert SampleHparams() == SampleHparams()
        assert SampleHparams(batch_size=1) != SampleHparams(batch_size=2)
        assert "batch_size" in repr(SampleHparams())


class TestHparamsValidation:
    """Tests that invalid configurations are rejected."""

    def test_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SampleHparams(batch_size="not an int")

    def test_coercible_type_is_accepted(self) -> None:
        assert SampleHparams(batch_size="8").batch_size == 8

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SampleHparams(hiden_dim=256)

    def test_frozen_rejects_assignment(self, sample_hparams: SampleHparams) -> None:
        with pytest.raises(ValidationError, match="frozen"):
            sample_hparams.batch_size = 64

    def test_model_copy_replaces_values(self, sample_hparams: SampleHparams) -> None:
        updated = sample_hparams.model_copy(update={"batch_size": 64})
        assert updated.batch_size == 64
        assert sample_hparams.batch_size == 32


class TestHparamsFromYaml:
    """Tests loading a schema straight from a YAML file."""

    def test_from_yaml(self, tmp_config_path: Path) -> None:
        path = tmp_config_path / "hparams.yaml"
        path.write_text("learning_rate: 0.5\nbatch_size: 4\n")

        hparams = SampleHparams.from_yaml(path)
        assert hparams.learning_rate == 0.5
        assert hparams.batch_size == 4
        assert hparams.hidden_dim == 128  # untouched default

    def test_from_yaml_accepts_str_path(self, tmp_config_path: Path) -> None:
        path = tmp_config_path / "hparams.yaml"
        path.write_text("batch_size: 4\n")

        assert SampleHparams.from_yaml(str(path)).batch_size == 4

    def test_from_yaml_with_key(self, tmp_config_path: Path) -> None:
        path = tmp_config_path / "run.yaml"
        path.write_text("model:\n  batch_size: 16\ndata:\n  batch_size: 99\n")

        assert SampleHparams.from_yaml(path, key="model").batch_size == 16

    def test_from_yaml_with_nested_key(self, tmp_config_path: Path) -> None:
        path = tmp_config_path / "run.yaml"
        path.write_text("outer:\n  inner:\n    batch_size: 7\n")

        assert SampleHparams.from_yaml(path, key="outer.inner").batch_size == 7

    def test_from_yaml_resolves_interpolations(self, tmp_config_path: Path) -> None:
        path = tmp_config_path / "hparams.yaml"
        path.write_text("batch_size: 8\nhidden_dim: ${batch_size}\n")

        assert SampleHparams.from_yaml(path).hidden_dim == 8

    def test_from_yaml_missing_key_raises(self, tmp_config_path: Path) -> None:
        path = tmp_config_path / "run.yaml"
        path.write_text("model:\n  batch_size: 16\n")

        with pytest.raises(KeyError, match="has no key 'modle'"):
            SampleHparams.from_yaml(path, key="modle")

    def test_from_yaml_validates(self, tmp_config_path: Path) -> None:
        path = tmp_config_path / "hparams.yaml"
        path.write_text("hiden_dim: 256\n")

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SampleHparams.from_yaml(path)

    def test_from_yaml_returns_plain_containers(self, tmp_config_path: Path) -> None:
        class WithMapping(Hparams):
            extras: dict = {}

        path = tmp_config_path / "hparams.yaml"
        path.write_text("extras:\n  nested:\n    a: 1\n")

        loaded = WithMapping.from_yaml(path)
        assert type(loaded.extras["nested"]) is dict


class TestModelDump:
    """Tests that a dumped schema is safe to put in a checkpoint."""

    def test_dump_contains_only_primitives(self, nested_hparams: NestedHparams) -> None:
        dumped = nested_hparams.model_dump()
        assert type(dumped) is dict
        assert type(dumped["inner"]) is dict
        assert dumped["inner"]["lr"] == 1e-3


class TestToPlainContainer:
    """Tests the OmegaConf normalisation helper."""

    def test_dict_config(self) -> None:
        converted = to_plain_container(OmegaConf.create({"a": {"b": [1, 2]}}))
        assert type(converted) is dict
        assert type(converted["a"]) is dict
        assert type(converted["a"]["b"]) is list

    def test_nested_dict_config_inside_plain_dict(self) -> None:
        converted = to_plain_container({"a": OmegaConf.create({"b": 1})})
        assert type(converted["a"]) is dict

    def test_passes_through_plain_values(self) -> None:
        assert to_plain_container(5) == 5
        assert to_plain_container("x") == "x"
        assert to_plain_container((1, 2)) == (1, 2)


class TestHparamsSchemaResolution:
    """Tests how `HparamsModule` binds its schema."""

    def test_direct_binding(self) -> None:
        assert Module.hparams_schema is SampleHparams
        assert Module().thparams == SampleHparams()

    def test_inherited_binding(self) -> None:
        class Sub(Module):
            pass

        assert Sub.hparams_schema is SampleHparams

    def test_explicit_class_attribute_wins(self) -> None:
        class Explicit(HparamsModule[SampleHparams]):
            hparams_schema = NestedHparams

        assert Explicit.hparams_schema is NestedHparams

    def test_unparametrised_subclass_raises(self) -> None:
        class Unparametrised(HparamsModule):
            pass

        with pytest.raises(TypeError, match="does not bind a hyperparameter schema"):
            Unparametrised()

    def test_resolve_returns_none_when_unparametrised(self) -> None:
        class Unparametrised(HparamsModule):
            pass

        assert _resolve_hparams_schema(Unparametrised) is None


class TestHparamsModuleInputs:
    """Tests the accepted forms of the `hparams` argument."""

    def test_instance(self) -> None:
        hparams = SampleHparams(batch_size=1)
        assert Module(hparams).thparams is hparams

    def test_mapping(self) -> None:
        assert Module({"batch_size": 2}).thparams.batch_size == 2

    def test_kwargs(self) -> None:
        assert Module(batch_size=3).thparams.batch_size == 3

    def test_dict_config(self) -> None:
        assert Module(OmegaConf.create({"batch_size": 4})).thparams.batch_size == 4

    def test_mapping_and_kwargs_merge(self) -> None:
        module = Module({"batch_size": 4}, hidden_dim=8)
        assert (module.thparams.batch_size, module.thparams.hidden_dim) == (4, 8)

    def test_instance_and_kwargs_conflict(self) -> None:
        with pytest.raises(TypeError, match="Pass one or the other"):
            Module(SampleHparams(), batch_size=5)

    def test_wrong_schema_instance(self) -> None:
        with pytest.raises(TypeError, match="expects SampleHparams"):
            Module(NestedHparams())

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="must be a SampleHparams or a mapping"):
            Module(42)

    def test_validation_errors_propagate(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Module(hiden_dim=1)


class TestHparamsArgName:
    """Tests the checkpoint key derivation."""

    def test_default_name(self) -> None:
        assert _hparams_arg_name(Module) == "hparams"

    def test_renamed_argument(self) -> None:
        class Renamed(HparamsModule[SampleHparams]):
            def __init__(self, settings=None, **kwargs):
                super().__init__(settings, **kwargs)

        assert _hparams_arg_name(Renamed) == "settings"

    def test_kwargs_only_falls_back(self) -> None:
        class KwargsOnly(HparamsModule[SampleHparams]):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        assert _hparams_arg_name(KwargsOnly) == "hparams"
