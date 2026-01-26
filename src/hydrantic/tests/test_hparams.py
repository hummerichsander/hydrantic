import pytest
import yaml

from ..hparams import Hparams


class TestHparamsInitialization:
    """Test Hparams initialization and basic functionality."""

    def test_simple_initialization(self, sample_hparams):
        """Test basic Hparams initialization.

        :param sample_hparams: Sample Hparams fixture."""
        assert sample_hparams.learning_rate == 1e-3
        assert sample_hparams.batch_size == 32
        assert sample_hparams.hidden_dim == 128
        assert sample_hparams.dropout == 0.1

    def test_nested_initialization(self, nested_hparams):
        """Test nested Hparams initialization.

        :param nested_hparams: Nested Hparams fixture."""
        assert nested_hparams.batch_size == 64
        assert nested_hparams.optimizer.lr == 1e-3
        assert nested_hparams.model.hidden_dim == 256
        assert nested_hparams.model.num_layers == 3

    def test_custom_values(self):
        """Test Hparams with custom initialization values."""

        class CustomHparams(Hparams):
            value: int = 10

        hparams = CustomHparams(value=42)
        assert hparams.value == 42


class TestHparamsSerialization:
    """Test Hparams serialization and deserialization."""

    def test_to_dict(self, sample_hparams):
        """Test conversion to dictionary using dict unpacking.

        :param sample_hparams: Sample Hparams fixture."""
        hparams_dict = {**sample_hparams}
        assert isinstance(hparams_dict, dict)
        assert hparams_dict["learning_rate"] == 1e-3
        assert hparams_dict["batch_size"] == 32

    def test_model_dump(self, sample_hparams):
        """Test Pydantic's model_dump method.

        :param sample_hparams: Sample Hparams fixture."""
        hparams_dict = sample_hparams.model_dump()
        assert isinstance(hparams_dict, dict)
        assert hparams_dict["learning_rate"] == 1e-3
        assert hparams_dict["batch_size"] == 32

    def test_from_dict(self, sample_hparams):
        """Test creation from dictionary.

        :param sample_hparams: Sample Hparams fixture."""
        hparams_dict = {**sample_hparams}
        reconstructed = type(sample_hparams).from_dict(hparams_dict)
        assert reconstructed.learning_rate == sample_hparams.learning_rate
        assert reconstructed.batch_size == sample_hparams.batch_size

    def test_from_yaml(self, sample_hparams, tmp_config_path):
        """Test loading from YAML file.

        :param sample_hparams: Sample Hparams fixture.
        :param tmp_config_path: Temporary config directory."""
        yaml_path = tmp_config_path / "config.yaml"

        # Write YAML file manually
        with open(yaml_path, "w") as f:
            yaml.dump(sample_hparams.model_dump(), f)

        assert yaml_path.exists()
        loaded = type(sample_hparams).from_yaml(str(yaml_path))
        assert loaded.learning_rate == sample_hparams.learning_rate
        assert loaded.batch_size == sample_hparams.batch_size

    def test_nested_serialization(self, nested_hparams, tmp_config_path):
        """Test serialization of nested Hparams.

        :param nested_hparams: Nested Hparams fixture.
        :param tmp_config_path: Temporary config directory."""
        yaml_path = tmp_config_path / "nested.yaml"

        # Write YAML file with nested structure
        nested_dict = nested_hparams.model_dump()
        with open(yaml_path, "w") as f:
            yaml.dump(nested_dict, f)

        loaded = type(nested_hparams).from_yaml(str(yaml_path))
        assert loaded.optimizer.lr == nested_hparams.optimizer.lr
        assert loaded.model.hidden_dim == nested_hparams.model.hidden_dim

    def test_from_yaml_with_key(self, sample_hparams, tmp_config_path):
        """Test loading from YAML file with specific key.

        :param sample_hparams: Sample Hparams fixture.
        :param tmp_config_path: Temporary config directory."""
        yaml_path = tmp_config_path / "config_with_key.yaml"

        # Write YAML file with nested key
        config_dict = {"model_config": sample_hparams.model_dump()}
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        loaded = type(sample_hparams).from_yaml(str(yaml_path), key="model_config")
        assert loaded.learning_rate == sample_hparams.learning_rate
        assert loaded.batch_size == sample_hparams.batch_size


class TestHparamsValidation:
    """Test Hparams validation and type checking."""

    def test_type_validation(self):
        """Test that type validation works correctly."""

        class TypedHparams(Hparams):
            count: int = 10
            ratio: float = 0.5

        hparams = TypedHparams(count=20, ratio=0.8)
        assert hparams.count == 20
        assert hparams.ratio == 0.8

    def test_pydantic_validation_error(self):
        """Test that Pydantic raises validation errors for invalid types."""
        from pydantic import ValidationError

        class StrictHparams(Hparams):
            count: int = 10

        with pytest.raises(ValidationError):
            StrictHparams(count="not_an_int")

    def test_extra_fields_ignored(self, sample_hparams):
        """Test handling of extra fields during construction.

        :param sample_hparams: Sample Hparams fixture."""
        # Pydantic by default ignores extra fields
        hparams_dict = {**sample_hparams}
        hparams_dict["invalid_field"] = 42

        reconstructed = type(sample_hparams).from_dict(hparams_dict)
        assert not hasattr(reconstructed, "invalid_field")


class TestHparamsUtilities:
    """Test Hparams utility methods."""

    def test_mapping_interface(self, sample_hparams):
        """Test that Hparams implements Mapping interface.

        :param sample_hparams: Sample Hparams fixture."""
        assert sample_hparams["learning_rate"] == 1e-3
        assert sample_hparams["batch_size"] == 32
        assert len(sample_hparams) == 4

    def test_keys_method(self, sample_hparams):
        """Test keys method.

        :param sample_hparams: Sample Hparams fixture."""
        keys = list(sample_hparams.keys())
        assert "learning_rate" in keys
        assert "batch_size" in keys
        assert "hidden_dim" in keys
        assert "dropout" in keys

    def test_iteration(self, sample_hparams):
        """Test iteration over Hparams.

        :param sample_hparams: Sample Hparams fixture."""
        items = dict(sample_hparams)
        assert items["learning_rate"] == 1e-3
        assert items["batch_size"] == 32

    def test_repr(self, sample_hparams):
        """Test string representation.

        :param sample_hparams: Sample Hparams fixture."""
        repr_str = repr(sample_hparams)
        assert "learning_rate" in repr_str
        assert "batch_size" in repr_str

    def test_equality(self, sample_hparams):
        """Test equality comparison.

        :param sample_hparams: Sample Hparams fixture."""
        other = type(sample_hparams)(learning_rate=1e-3, batch_size=32, hidden_dim=128, dropout=0.1)
        assert sample_hparams == other

        different = type(sample_hparams)(learning_rate=5e-4)
        assert sample_hparams != different

    def test_model_copy(self, sample_hparams):
        """Test Pydantic's model_copy method.

        :param sample_hparams: Sample Hparams fixture."""
        copy = sample_hparams.model_copy(update={"learning_rate": 5e-4})
        assert copy.learning_rate == 5e-4
        assert copy.batch_size == 32
        assert sample_hparams.learning_rate == 1e-3  # Original unchanged
