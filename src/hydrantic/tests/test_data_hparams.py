import pytest
import random

from ..data.hparams import DataHparams, DataLoaderHparams


class TestDataLoaderHparams:
    """Test DataLoaderHparams configuration."""

    def test_default_initialization(self):
        """Test default DataLoaderHparams values."""
        hparams = DataLoaderHparams()
        assert hparams.batch_size == 64
        assert hparams.shuffle is False
        assert hparams.pin_memory is False
        assert hparams.num_workers == 0
        assert hparams.persistent_workers is False
        assert hparams.drop_last is False

    def test_custom_initialization(self):
        """Test custom DataLoaderHparams values."""
        hparams = DataLoaderHparams(
            batch_size=32,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            drop_last=True,
        )
        assert hparams.batch_size == 32
        assert hparams.shuffle is True
        assert hparams.pin_memory is True
        assert hparams.num_workers == 4
        assert hparams.persistent_workers is True
        assert hparams.drop_last is True

    def test_serialization(self):
        """Test DataLoaderHparams serialization."""
        hparams = DataLoaderHparams(batch_size=128, shuffle=True)
        hparams_dict = hparams.model_dump()

        assert hparams_dict["batch_size"] == 128
        assert hparams_dict["shuffle"] is True

        reconstructed = DataLoaderHparams.from_dict(hparams_dict)
        assert reconstructed.batch_size == 128
        assert reconstructed.shuffle is True


class TestDataHparams:
    """Test DataHparams configuration."""

    def test_default_initialization(self):
        """Test default DataHparams values."""
        hparams = DataHparams()
        assert hparams.module_name is None
        assert hparams.loader is None
        assert isinstance(hparams.seed, int)
        assert 0 <= hparams.seed <= 10_000
        assert hparams.split == (0.8, 0.1, 0.1)

    def test_custom_initialization(self):
        """Test custom DataHparams values."""
        loader = DataLoaderHparams(batch_size=32)
        hparams = DataHparams(
            module_name="custom.module",
            loader=loader,
            seed=42,
            split=(0.7, 0.3),
        )
        assert hparams.module_name == "custom.module"
        assert hparams.loader.batch_size == 32
        assert hparams.seed == 42
        assert hparams.split == (0.7, 0.3)

    def test_nested_loader_hparams(self):
        """Test nested DataLoaderHparams initialization."""
        hparams = DataHparams(loader=DataLoaderHparams(batch_size=16, shuffle=True, num_workers=2))
        assert hparams.loader.batch_size == 16
        assert hparams.loader.shuffle is True
        assert hparams.loader.num_workers == 2

    def test_split_two_way(self):
        """Test two-way split configuration."""
        hparams = DataHparams(split=(0.8, 0.2))
        assert len(hparams.split) == 2
        assert sum(hparams.split) == 1.0

    def test_split_three_way(self):
        """Test three-way split configuration."""
        hparams = DataHparams(split=(0.7, 0.2, 0.1))
        assert len(hparams.split) == 3
        assert sum(hparams.split) == 1.0

    def test_seed_randomness(self):
        """Test that default seeds are random."""
        seeds = {DataHparams().seed for _ in range(10)}
        assert len(seeds) > 1  # Should have different seeds

    def test_serialization(self, data_hparams):
        """Test DataHparams serialization.

        :param data_hparams: DataHparams fixture."""
        hparams_dict = data_hparams.model_dump()

        assert hparams_dict["module_name"] == "test_module"
        assert hparams_dict["seed"] == 42
        assert hparams_dict["split"] == (0.7, 0.2, 0.1)

        reconstructed = DataHparams.from_dict(hparams_dict)
        assert reconstructed.module_name == data_hparams.module_name
        assert reconstructed.seed == data_hparams.seed
        assert reconstructed.split == data_hparams.split

    def test_from_yaml(self, tmp_config_path):
        """Test loading DataHparams from YAML.

        :param tmp_config_path: Temporary config directory fixture."""
        import yaml

        yaml_path = tmp_config_path / "data_config.yaml"
        config = {
            "module_name": "torch.utils.data.TensorDataset",
            "seed": 999,
            "split": [0.6, 0.3, 0.1],
            "loader": {"batch_size": 64, "shuffle": True, "num_workers": 2},
        }

        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        loaded = DataHparams.from_yaml(str(yaml_path))
        assert loaded.module_name == "torch.utils.data.TensorDataset"
        assert loaded.seed == 999
        assert loaded.split == (0.6, 0.3, 0.1)
        assert loaded.loader.batch_size == 64
        assert loaded.loader.shuffle is True
        assert loaded.loader.num_workers == 2
