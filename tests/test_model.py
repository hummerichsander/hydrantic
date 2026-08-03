import io

import lightning
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset

from conftest import RenamedArgModel, TinyData, TinyHparams, TinyModel

from hydrantic import Model, ModelHparams
from hydrantic.model import _as_dataloader


class TestModelInitialization:
    """Tests the accepted forms of model construction."""

    def test_defaults(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls()
        assert isinstance(model.thparams, TinyHparams)
        assert model.thparams.input_dim == 10

    def test_kwargs(self, tiny_model_cls: type[TinyModel]) -> None:
        assert tiny_model_cls(input_dim=4).thparams.input_dim == 4

    def test_mapping(self, tiny_model_cls: type[TinyModel]) -> None:
        assert tiny_model_cls({"input_dim": 5}).thparams.input_dim == 5

    def test_instance(self, tiny_model_cls: type[TinyModel]) -> None:
        hparams = TinyHparams(input_dim=6)
        assert tiny_model_cls(hparams).thparams is hparams

    def test_dict_config(self, tiny_model_cls: type[TinyModel]) -> None:
        config = OmegaConf.create({"input_dim": 7})
        assert tiny_model_cls(config).thparams.input_dim == 7

    def test_is_a_lightning_module(self, tiny_model_cls: type[TinyModel]) -> None:
        assert isinstance(tiny_model_cls(), lightning.LightningModule)

    def test_typo_is_rejected(self, tiny_model_cls: type[TinyModel]) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            tiny_model_cls(input_dimm=4)

    def test_hparams_are_saved(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls(input_dim=3)
        assert model.hparams["hparams"]["input_dim"] == 3

    def test_layers_are_registered(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls(input_dim=4, output_dim=2)
        assert model.layer.weight.shape == (2, 4)


class TestModelHparamsSchema:
    """Tests `ModelHparams` itself."""

    def test_default_optimizer(self) -> None:
        assert ModelHparams().optimizer == {"_target_": "torch.optim.Adam"}

    def test_default_scheduler_is_none(self) -> None:
        assert ModelHparams().scheduler is None

    def test_optimizer_needs_a_target(self) -> None:
        with pytest.raises(ValidationError, match="needs a `_target_` key"):
            ModelHparams(optimizer={"lr": 1e-3})

    def test_optimizer_kwargs_are_signature_checked(self) -> None:
        with pytest.raises(ValidationError, match="unexpected keyword for torch.optim.Adam"):
            ModelHparams(optimizer={"_target_": "torch.optim.Adam", "lr_rate": 1e-3})

    def test_scheduler_accepts_short_form(self) -> None:
        hparams = ModelHparams(
            scheduler={"_target_": "torch.optim.lr_scheduler.ExponentialLR", "gamma": 0.9}
        )
        assert hparams.scheduler["gamma"] == 0.9

    def test_scheduler_accepts_lightning_dict_form(self) -> None:
        hparams = ModelHparams(
            scheduler={
                "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1},
                "monitor": "validation/loss",
            }
        )
        assert hparams.scheduler["monitor"] == "validation/loss"


class TestConfigureOptimizers:
    """Tests optimizer and scheduler construction."""

    def test_optimizer_only(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls(optimizer={"_target_": "torch.optim.AdamW", "lr": 0.5})
        config = model.configure_optimizers()

        assert isinstance(config["optimizer"], AdamW)
        assert config["optimizer"].param_groups[0]["lr"] == 0.5
        assert "lr_scheduler" not in config

    def test_default_optimizer_is_adam(self, tiny_model_cls: type[TinyModel]) -> None:
        assert isinstance(tiny_model_cls().configure_optimizers()["optimizer"], Adam)

    def test_scheduler_short_form(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls(
            scheduler={"_target_": "torch.optim.lr_scheduler.ExponentialLR", "gamma": 0.9}
        )
        config = model.configure_optimizers()

        assert isinstance(config["lr_scheduler"]["scheduler"], ExponentialLR)

    def test_scheduler_lightning_dict_form(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls(
            scheduler={
                "scheduler": {"_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau"},
                "monitor": "validation/loss",
                "interval": "epoch",
            }
        )
        config = model.configure_optimizers()

        assert isinstance(config["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
        assert config["lr_scheduler"]["monitor"] == "validation/loss"
        assert config["lr_scheduler"]["interval"] == "epoch"

    def test_partial_flag_is_tolerated(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls(
            optimizer={"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.25}
        )
        assert model.configure_optimizers()["optimizer"].param_groups[0]["lr"] == 0.25

    def test_get_optimizer_params_override(self) -> None:
        class GroupedModel(TinyModel):
            def get_optimizer_params(self):
                return [{"params": list(self.layer.parameters()), "lr": 0.123}]

        optimizer = GroupedModel().configure_optimizers()["optimizer"]
        assert optimizer.param_groups[0]["lr"] == 0.123

    def test_configure_optimizers_does_not_mutate_hparams(
        self, tiny_model_cls: type[TinyModel]
    ) -> None:
        model = tiny_model_cls()
        before = model.thparams.model_dump()
        model.configure_optimizers()
        assert model.thparams.model_dump() == before


class TestNoLegacySurface:
    """Tests that the removed hooks are really gone."""

    @pytest.mark.parametrize(
        "name", ["compute_metrics", "has_logger", "fit_fast", "load_from_wandb_artifact"]
    )
    def test_removed(self, name: str) -> None:
        assert not hasattr(Model, name)

    def test_configure_callbacks_is_not_overridden(self, tiny_model_cls: type[TinyModel]) -> None:
        # Lightning drops any trainer callback whose type the model also returns, so overriding
        # this hook would silently delete callbacks configured in YAML.
        assert "configure_callbacks" not in Model.__dict__
        assert tiny_model_cls().configure_callbacks() == []

    def test_model_is_not_abstract(self) -> None:
        class Bare(Model[TinyHparams]):
            pass

        assert Bare() is not None

    @pytest.mark.parametrize("name", ["module_name", "matmul_precision", "checkpoint", "early_stopping"])
    def test_removed_hparams_fields(self, name: str) -> None:
        assert name not in ModelHparams.model_fields


class TestLogMetrics:
    """Tests the metric logging helper."""

    def test_prefixes_metric_names(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls()
        logged: dict[str, object] = {}
        model.log = lambda name, value, **kwargs: logged.__setitem__(name, kwargs)

        model.log_metrics({"loss": torch.tensor(1.0), "accuracy": torch.tensor(0.5)}, "validation")

        assert set(logged) == {"validation/loss", "validation/accuracy"}

    def test_loss_goes_to_the_progress_bar(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls()
        logged: dict[str, object] = {}
        model.log = lambda name, value, **kwargs: logged.__setitem__(name, kwargs)

        model.log_metrics({"loss": torch.tensor(1.0), "accuracy": torch.tensor(0.5)}, "training")

        assert logged["training/loss"]["prog_bar"] is True
        assert logged["training/accuracy"]["prog_bar"] is False

    def test_forwards_extra_kwargs(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls()
        logged: dict[str, object] = {}
        model.log = lambda name, value, **kwargs: logged.__setitem__(name, kwargs)

        model.log_metrics({"loss": torch.tensor(1.0)}, "test", sync_dist=True)

        assert logged["test/loss"]["sync_dist"] is True


class TestModelProperties:
    """Tests the parameter counting properties."""

    def test_num_params(self, tiny_model_cls: type[TinyModel]) -> None:
        model = tiny_model_cls(input_dim=10, output_dim=2)
        assert model.num_params == 10 * 2 + 2

    def test_num_params_by_module(self, tiny_model_cls: type[TinyModel]) -> None:
        counts = tiny_model_cls(input_dim=10, output_dim=2).num_params_by_module
        assert counts["total"] == 22
        assert counts["layer"] == 22


class TestAsDataloader:
    """Tests the dataset-to-dataloader coercion."""

    def test_none_passes_through(self) -> None:
        assert _as_dataloader(None, batch_size=4, shuffle=False) is None

    def test_dataloader_passes_through(self, sample_tensor_dataset: Dataset) -> None:
        loader = DataLoader(sample_tensor_dataset, batch_size=2)
        assert _as_dataloader(loader, batch_size=4, shuffle=True) is loader

    def test_dataset_is_wrapped(self, sample_tensor_dataset: Dataset) -> None:
        loader = _as_dataloader(sample_tensor_dataset, batch_size=4, shuffle=True)
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 4

    def test_iterable_dataset_is_not_shuffled(self) -> None:
        class Stream(IterableDataset):
            def __iter__(self):
                yield torch.zeros(10), torch.tensor(0)

        loader = _as_dataloader(Stream(), batch_size=1, shuffle=True)
        assert isinstance(loader, DataLoader)

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Expected a Dataset, DataLoader or"):
            _as_dataloader(42, batch_size=1, shuffle=False)


class TestFitInline:
    """Tests the inline training API."""

    @pytest.fixture
    def loader(self) -> DataLoader:
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        return DataLoader(TensorDataset(x, y), batch_size=8)

    def test_returns_the_trainer(self, tiny_model_cls: type[TinyModel], loader: DataLoader) -> None:
        trainer = tiny_model_cls().fit_inline(loader, accelerator="cpu")

        assert isinstance(trainer, lightning.Trainer)
        assert "training/loss" in trainer.callback_metrics

    def test_accepts_a_dataset(self, tiny_model_cls: type[TinyModel]) -> None:
        dataset = TensorDataset(torch.randn(16, 10), torch.randint(0, 2, (16,)))
        trainer = tiny_model_cls().fit_inline(dataset, batch_size=4, accelerator="cpu")
        assert trainer.state.finished

    def test_accepts_a_datamodule(self, tiny_model_cls: type[TinyModel], tiny_data_cls) -> None:
        data = tiny_data_cls(num_samples=32, loader={"batch_size": 8})
        trainer = tiny_model_cls().fit_inline(data, accelerator="cpu")
        assert "validation/loss" in trainer.callback_metrics

    def test_datamodule_with_val_data_raises(
        self, tiny_model_cls: type[TinyModel], tiny_data_cls, loader: DataLoader
    ) -> None:
        data = tiny_data_cls(num_samples=32)
        with pytest.raises(ValueError, match="val_data must be None"):
            tiny_model_cls().fit_inline(data, loader)

    def test_validation_loader(self, tiny_model_cls: type[TinyModel], loader: DataLoader) -> None:
        trainer = tiny_model_cls().fit_inline(loader, loader, accelerator="cpu")
        assert "validation/loss" in trainer.callback_metrics

    def test_trainer_kwargs_are_forwarded(
        self, tiny_model_cls: type[TinyModel], loader: DataLoader
    ) -> None:
        trainer = tiny_model_cls().fit_inline(loader, accelerator="cpu", max_epochs=3)
        assert trainer.max_epochs == 3

    def test_fast_dev_run_passes_through(
        self, tiny_model_cls: type[TinyModel], loader: DataLoader
    ) -> None:
        trainer = tiny_model_cls().fit_inline(loader, accelerator="cpu", fast_dev_run=True)
        assert trainer.state.finished

    def test_does_not_mutate_hparams(
        self, tiny_model_cls: type[TinyModel], loader: DataLoader
    ) -> None:
        model = tiny_model_cls(optimizer={"_target_": "torch.optim.SGD", "lr": 0.05})
        before = model.thparams.model_dump()
        before_initial = dict(model.hparams_initial)

        model.fit_inline(loader, accelerator="cpu")

        assert model.thparams.model_dump() == before
        assert dict(model.hparams_initial) == before_initial

    def test_uses_the_configured_optimizer(
        self, tiny_model_cls: type[TinyModel], loader: DataLoader
    ) -> None:
        model = tiny_model_cls(optimizer={"_target_": "torch.optim.SGD", "lr": 0.05})
        trainer = model.fit_inline(loader, accelerator="cpu")

        assert type(trainer.optimizers[0]).__name__ == "SGD"
        assert trainer.optimizers[0].param_groups[0]["lr"] == 0.05

    def test_does_not_log_or_checkpoint_by_default(
        self, tiny_model_cls: type[TinyModel], loader: DataLoader
    ) -> None:
        trainer = tiny_model_cls().fit_inline(loader, accelerator="cpu")
        assert trainer.loggers == []
        assert trainer.checkpoint_callbacks == []


class TestCheckpointRoundtrip:
    """Tests that a model survives a save/load cycle."""

    @staticmethod
    def _save(model: Model) -> io.BytesIO:
        buffer = io.BytesIO()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams_initial),
                "pytorch-lightning_version": lightning.__version__,
            },
            buffer,
        )
        buffer.seek(0)
        return buffer

    def test_weights_only_load_succeeds(self, tiny_model_cls: type[TinyModel]) -> None:
        # torch >= 2.6 defaults `weights_only=True`, which refuses to unpickle a DictConfig,
        # a pydantic model or a functools.partial stored in the hyperparameters.
        buffer = self._save(tiny_model_cls(input_dim=4))
        assert torch.load(buffer, weights_only=True)["hyper_parameters"]["hparams"]["input_dim"] == 4

    @pytest.mark.parametrize("cls", [TinyModel, RenamedArgModel])
    def test_load_from_checkpoint(self, cls: type[TinyModel]) -> None:
        original = cls(input_dim=4, output_dim=3)
        buffer = self._save(original)

        restored = cls.load_from_checkpoint(buffer)

        assert restored.thparams == original.thparams
        assert restored.layer.weight.shape == (3, 4)

    def test_restored_model_configures_optimizers(self, tiny_model_cls: type[TinyModel]) -> None:
        original = tiny_model_cls(optimizer={"_target_": "torch.optim.AdamW", "lr": 0.5})
        restored = tiny_model_cls.load_from_checkpoint(self._save(original))

        assert isinstance(restored.configure_optimizers()["optimizer"], AdamW)

    def test_hparams_come_back_as_the_schema(self, tiny_model_cls: type[TinyModel]) -> None:
        restored = tiny_model_cls.load_from_checkpoint(self._save(tiny_model_cls()))
        assert isinstance(restored.thparams, TinyHparams)


class TestInstantiatedFromHydra:
    """Tests a model built the way the CLI builds it."""

    def test_end_to_end(self, model_config: DictConfig, tiny_data_cls: type[TinyData]) -> None:
        from hydrantic.targets import instantiate_module

        model = instantiate_module(model_config)
        data = tiny_data_cls(num_samples=32, loader={"batch_size": 8})

        trainer = model.fit_inline(data, accelerator="cpu")
        assert "training/loss" in trainer.callback_metrics
