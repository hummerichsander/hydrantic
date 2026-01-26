import pytest
import yaml

from ..model.hparams import (
    ModelHparams,
    OptimizerHparams,
    SchedulerHparams,
    CheckpointHparams,
    EarlyStoppingHparams,
)


class TestOptimizerHparams:
    """Test OptimizerHparams configuration."""

    def test_default_initialization(self):
        """Test default OptimizerHparams values."""
        hparams = OptimizerHparams()
        assert hparams.module_name == "torch.optim.Adam"
        assert hparams.kwargs == {}

    def test_custom_initialization(self):
        """Test custom OptimizerHparams values."""
        hparams = OptimizerHparams(
            module_name="torch.optim.SGD", kwargs={"lr": 0.01, "momentum": 0.9}
        )
        assert hparams.module_name == "torch.optim.SGD"
        assert hparams.kwargs["lr"] == 0.01
        assert hparams.kwargs["momentum"] == 0.9

    def test_serialization(self):
        """Test OptimizerHparams serialization."""
        hparams = OptimizerHparams(module_name="torch.optim.AdamW", kwargs={"lr": 1e-4})
        hparams_dict = hparams.model_dump()

        reconstructed = OptimizerHparams.from_dict(hparams_dict)
        assert reconstructed.module_name == hparams.module_name
        assert reconstructed.kwargs == hparams.kwargs


class TestSchedulerHparams:
    """Test SchedulerHparams configuration."""

    def test_default_initialization(self):
        """Test default SchedulerHparams values."""
        hparams = SchedulerHparams()
        assert hparams.module_name is None
        assert hparams.kwargs == {}

    def test_custom_initialization(self):
        """Test custom SchedulerHparams values."""
        hparams = SchedulerHparams(
            module_name="torch.optim.lr_scheduler.StepLR", kwargs={"step_size": 5, "gamma": 0.5}
        )
        assert hparams.module_name == "torch.optim.lr_scheduler.StepLR"
        assert hparams.kwargs["step_size"] == 5
        assert hparams.kwargs["gamma"] == 0.5

    def test_serialization(self):
        """Test SchedulerHparams serialization."""
        hparams = SchedulerHparams(
            module_name="torch.optim.lr_scheduler.CosineAnnealingLR", kwargs={"T_max": 100}
        )
        hparams_dict = hparams.model_dump()

        reconstructed = SchedulerHparams.from_dict(hparams_dict)
        assert reconstructed.module_name == hparams.module_name
        assert reconstructed.kwargs == hparams.kwargs


class TestCheckpointHparams:
    """Test CheckpointHparams configuration."""

    def test_default_initialization(self):
        """Test default CheckpointHparams values."""
        hparams = CheckpointHparams()
        assert hparams.dirpath is None
        assert hparams.filename == "{epoch}-{val_loss:.2f}"
        assert hparams.monitor == "validation/loss"
        assert hparams.mode == "min"
        assert hparams.save_last is True
        assert hparams.save_top_k == 3
        assert hparams.every_n_epochs == 1

    def test_custom_initialization(self):
        """Test custom CheckpointHparams values."""
        hparams = CheckpointHparams(
            dirpath="/tmp/checkpoints",
            monitor="validation/accuracy",
            mode="max",
            save_top_k=5,
        )
        assert hparams.dirpath == "/tmp/checkpoints"
        assert hparams.monitor == "validation/accuracy"
        assert hparams.mode == "max"
        assert hparams.save_top_k == 5

    def test_serialization(self, checkpoint_hparams):
        """Test CheckpointHparams serialization.

        :param checkpoint_hparams: CheckpointHparams fixture."""
        hparams_dict = checkpoint_hparams.model_dump()

        reconstructed = CheckpointHparams.from_dict(hparams_dict)
        assert reconstructed.monitor == checkpoint_hparams.monitor
        assert reconstructed.mode == checkpoint_hparams.mode


class TestEarlyStoppingHparams:
    """Test EarlyStoppingHparams configuration."""

    def test_default_initialization(self):
        """Test default EarlyStoppingHparams values."""
        hparams = EarlyStoppingHparams()
        assert hparams.monitor == "validation/loss"
        assert hparams.min_delta == 0.0
        assert hparams.patience == 3
        assert hparams.mode == "min"
        assert hparams.stopping_threshold is None
        assert hparams.divergence_threshold is None
        assert hparams.check_finite is True

    def test_custom_initialization(self):
        """Test custom EarlyStoppingHparams values."""
        hparams = EarlyStoppingHparams(
            monitor="validation/accuracy",
            patience=10,
            mode="max",
            min_delta=0.001,
        )
        assert hparams.monitor == "validation/accuracy"
        assert hparams.patience == 10
        assert hparams.mode == "max"
        assert hparams.min_delta == 0.001

    def test_serialization(self, early_stopping_hparams):
        """Test EarlyStoppingHparams serialization.

        :param early_stopping_hparams: EarlyStoppingHparams fixture."""
        hparams_dict = early_stopping_hparams.model_dump()

        reconstructed = EarlyStoppingHparams.from_dict(hparams_dict)
        assert reconstructed.monitor == early_stopping_hparams.monitor
        assert reconstructed.patience == early_stopping_hparams.patience


class TestModelHparams:
    """Test ModelHparams configuration."""

    def test_default_initialization(self):
        """Test default ModelHparams values."""
        hparams = ModelHparams()
        assert hparams.module_name is None
        assert hparams.matmul_precision == "highest"
        assert isinstance(hparams.optimizer, OptimizerHparams)
        assert isinstance(hparams.scheduler, SchedulerHparams)
        assert isinstance(hparams.checkpoint, CheckpointHparams)
        assert hparams.early_stopping is None

    def test_custom_initialization(self, optimizer_hparams, scheduler_hparams):
        """Test custom ModelHparams values.

        :param optimizer_hparams: OptimizerHparams fixture.
        :param scheduler_hparams: SchedulerHparams fixture."""
        hparams = ModelHparams(
            module_name="custom.model",
            matmul_precision="medium",
            optimizer=optimizer_hparams,
            scheduler=scheduler_hparams,
        )
        assert hparams.module_name == "custom.model"
        assert hparams.matmul_precision == "medium"
        assert hparams.optimizer.module_name == "torch.optim.SGD"
        assert hparams.scheduler.module_name == "torch.optim.lr_scheduler.StepLR"

    def test_nested_hparams(self):
        """Test nested hparams initialization."""
        hparams = ModelHparams(
            optimizer=OptimizerHparams(module_name="torch.optim.Adam", kwargs={"lr": 1e-3}),
            scheduler=SchedulerHparams(
                module_name="torch.optim.lr_scheduler.StepLR", kwargs={"step_size": 10}
            ),
            checkpoint=CheckpointHparams(save_top_k=1),
            early_stopping=EarlyStoppingHparams(patience=7),
        )
        assert hparams.optimizer.kwargs["lr"] == 1e-3
        assert hparams.scheduler.kwargs["step_size"] == 10
        assert hparams.checkpoint.save_top_k == 1
        assert hparams.early_stopping.patience == 7

    def test_serialization(self, model_hparams):
        """Test ModelHparams serialization.

        :param model_hparams: ModelHparams fixture."""
        hparams_dict = model_hparams.model_dump()

        reconstructed = ModelHparams.from_dict(hparams_dict)
        assert reconstructed.module_name == model_hparams.module_name
        assert reconstructed.matmul_precision == model_hparams.matmul_precision
        assert reconstructed.optimizer.module_name == model_hparams.optimizer.module_name

    def test_from_yaml(self, tmp_config_path):
        """Test loading ModelHparams from YAML.

        :param tmp_config_path: Temporary config directory fixture."""
        yaml_path = tmp_config_path / "model_config.yaml"
        config = {
            "module_name": "custom.model.MyModel",
            "matmul_precision": "high",
            "optimizer": {"module_name": "torch.optim.Adam", "kwargs": {"lr": 0.001}},
            "scheduler": {
                "module_name": "torch.optim.lr_scheduler.StepLR",
                "kwargs": {"step_size": 5},
            },
        }

        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        loaded = ModelHparams.from_yaml(str(yaml_path))
        assert loaded.module_name == "custom.model.MyModel"
        assert loaded.matmul_precision == "high"
        assert loaded.optimizer.kwargs["lr"] == 0.001
        assert loaded.scheduler.kwargs["step_size"] == 5
