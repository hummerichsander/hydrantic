import pytest
import torch
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from ..model.model import Model
from ..model.hparams import ModelHparams, OptimizerHparams, SchedulerHparams


class TestModelInitialization:
    """Test Model initialization and basic functionality."""

    def test_basic_initialization(self, mock_model, simple_model_hparams):
        """Test basic Model initialization.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)

        assert model.thparams == simple_model_hparams
        assert isinstance(model.layer, torch.nn.Linear)
        assert model.has_logger is True

    def test_initialization_with_dict(self, mock_model):
        """Test Model initialization with dictionary.

        :param mock_model: Mock Model class fixture."""
        hparams_dict = {
            "optimizer": {"module_name": "torch.optim.Adam", "kwargs": {"lr": 0.001}},
            "matmul_precision": "high",
        }
        model = mock_model(hparams_dict)

        assert model.thparams.optimizer.module_name == "torch.optim.Adam"
        assert model.thparams.matmul_precision == "high"

    def test_initialization_with_none_raises_error(self, mock_model):
        """Test that initialization with None raises TypeError.

        :param mock_model: Mock Model class fixture."""
        with pytest.raises(TypeError, match="thparams must not be None"):
            mock_model(None)

    def test_initialization_with_wrong_type_raises_error(self, mock_model):
        """Test that initialization with wrong type raises TypeError.

        :param mock_model: Mock Model class fixture."""
        with pytest.raises(TypeError, match="must be of type"):
            mock_model("invalid_type")

    def test_matmul_precision_is_set(self, mock_model, simple_model_hparams):
        """Test that matmul precision is set correctly.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        simple_model_hparams.matmul_precision = "medium"
        model = mock_model(simple_model_hparams)

        assert model.thparams.matmul_precision == "medium"


class TestModelMetrics:
    """Test Model compute_metrics and evaluation."""

    def test_compute_metrics_returns_dict(self, mock_model, simple_model_hparams, sample_batch):
        """Test that compute_metrics returns dictionary.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture.
        :param sample_batch: Sample batch fixture."""
        model = mock_model(simple_model_hparams)
        metrics = model.compute_metrics(sample_batch, 0)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], torch.Tensor)
        assert isinstance(metrics["accuracy"], torch.Tensor)

    def test_evaluate_metrics_requires_loss(self, mock_model, simple_model_hparams):
        """Test that _evaluate_metrics requires 'loss' key.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)

        with pytest.raises(ValueError, match="must contain a 'loss' key"):
            model._evaluate_metrics({"accuracy": torch.tensor(0.9)}, "test")

    def test_evaluate_metrics_with_none_raises_error(self, mock_model, simple_model_hparams):
        """Test that _evaluate_metrics with None raises ValueError.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)

        with pytest.raises(ValueError, match="metrics must not be None"):
            model._evaluate_metrics(None, "test")

    def test_training_step(self, mock_model, simple_model_hparams, sample_batch):
        """Test training_step method.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture.
        :param sample_batch: Sample batch fixture."""
        model = mock_model(simple_model_hparams)
        loss = model.training_step(sample_batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_validation_step(self, mock_model, simple_model_hparams, sample_batch):
        """Test validation_step method.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture.
        :param sample_batch: Sample batch fixture."""
        model = mock_model(simple_model_hparams)
        result = model.validation_step(sample_batch, 0)

        assert result is None

    def test_test_step(self, mock_model, simple_model_hparams, sample_batch):
        """Test test_step method.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture.
        :param sample_batch: Sample batch fixture."""
        model = mock_model(simple_model_hparams)
        result = model.test_step(sample_batch, 0)

        assert result is None


class TestModelOptimizer:
    """Test Model optimizer configuration."""

    def test_configure_optimizer_without_scheduler(self, mock_model, simple_model_hparams):
        """Test optimizer configuration without scheduler.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)
        optimizer_config = model.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" not in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.Adam)

    def test_configure_optimizer_with_scheduler(self, mock_model, model_hparams):
        """Test optimizer configuration with scheduler.

        :param mock_model: Mock Model class fixture.
        :param model_hparams: ModelHparams fixture."""
        model = mock_model(model_hparams)
        optimizer_config = model.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.SGD)

    def test_optimizer_parameters(self, mock_model, simple_model_hparams):
        """Test that optimizer receives model parameters.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config["optimizer"]

        param_groups = optimizer.param_groups
        assert len(param_groups) > 0


class TestModelCallbacks:
    """Test Model callback configuration."""

    def test_configure_callbacks_with_all(
        self, mock_model, model_hparams, checkpoint_hparams, early_stopping_hparams
    ):
        """Test callback configuration with all callbacks.

        :param mock_model: Mock Model class fixture.
        :param model_hparams: ModelHparams fixture.
        :param checkpoint_hparams: CheckpointHparams fixture.
        :param early_stopping_hparams: EarlyStoppingHparams fixture."""
        model_hparams.checkpoint = checkpoint_hparams
        model_hparams.early_stopping = early_stopping_hparams
        model = mock_model(model_hparams)

        callbacks = model.configure_callbacks()

        assert len(callbacks) == 3
        assert any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
        assert any(isinstance(cb, EarlyStopping) for cb in callbacks)
        assert any(isinstance(cb, LearningRateMonitor) for cb in callbacks)

    def test_configure_callbacks_without_checkpoint(self, mock_model, simple_model_hparams):
        """Test callback configuration without checkpoint.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        simple_model_hparams.checkpoint = None
        model = mock_model(simple_model_hparams)

        callbacks = model.configure_callbacks()

        assert not any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
        assert any(isinstance(cb, LearningRateMonitor) for cb in callbacks)

    def test_configure_callbacks_without_logger(self, mock_model, simple_model_hparams):
        """Test callback configuration without logger.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)
        model.has_logger = False

        callbacks = model.configure_callbacks()

        assert not any(isinstance(cb, LearningRateMonitor) for cb in callbacks)


class TestModelProperties:
    """Test Model properties and utilities."""

    def test_num_params_property(self, mock_model, simple_model_hparams):
        """Test num_params property.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)
        num_params = model.num_params

        assert isinstance(num_params, int)
        assert num_params > 0
        # Linear(10, 2) has 10*2 + 2 = 22 parameters
        expected_params = 10 * 2 + 2
        assert num_params == expected_params

    def test_num_params_by_module(self, mock_model, simple_model_hparams):
        """Test num_params_by_module property.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture."""
        model = mock_model(simple_model_hparams)
        num_params_by_module = model.num_params_by_module

        assert isinstance(num_params_by_module, dict)
        assert "total" in num_params_by_module
        assert "layer" in num_params_by_module
        assert num_params_by_module["layer"] == 22


class TestModelFitFast:
    """Test Model fit_fast convenience method."""

    def test_fit_fast_basic(self, mock_model, simple_model_hparams, sample_tensor_dataset):
        """Test fit_fast with basic configuration.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture.
        :param sample_tensor_dataset: Sample tensor dataset fixture."""
        model = mock_model(simple_model_hparams)
        train_loader = DataLoader(sample_tensor_dataset, batch_size=16)

        model.fit_fast(train_loader, n_epochs=1, lr=1e-3, verbose=False)

        assert model.has_logger is False
        assert model.thparams.optimizer.module_name == "torch.optim.Adam"
        assert model.thparams.optimizer.kwargs["lr"] == 1e-3

    def test_fit_fast_with_validation(
        self, mock_model, simple_model_hparams, sample_tensor_dataset
    ):
        """Test fit_fast with validation loader.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture.
        :param sample_tensor_dataset: Sample tensor dataset fixture."""
        model = mock_model(simple_model_hparams)
        train_subset = Subset(sample_tensor_dataset, range(80))
        val_subset = Subset(sample_tensor_dataset, range(80, 100))
        train_loader = DataLoader(train_subset, batch_size=16)
        val_loader = DataLoader(val_subset, batch_size=16)

        model.fit_fast(train_loader, val_loader, n_epochs=1, verbose=False)

        assert model.has_logger is False

    def test_fit_fast_custom_lr(self, mock_model, simple_model_hparams, sample_tensor_dataset):
        """Test fit_fast with custom learning rate.

        :param mock_model: Mock Model class fixture.
        :param simple_model_hparams: Simple ModelHparams fixture.
        :param sample_tensor_dataset: Sample tensor dataset fixture."""
        model = mock_model(simple_model_hparams)
        train_loader = DataLoader(sample_tensor_dataset, batch_size=16)

        model.fit_fast(train_loader, n_epochs=1, lr=5e-4, verbose=False)

        assert model.thparams.optimizer.kwargs["lr"] == 5e-4
