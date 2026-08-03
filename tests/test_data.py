import lightning
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, TensorDataset

from conftest import TinyData, TinyDataHparams

from hydrantic import DataHparams, DataModule, split_dataset


class TestDataHparamsSchema:
    """Tests `DataHparams` validation."""

    def test_defaults(self) -> None:
        hparams = DataHparams()
        assert hparams.split == (0.8, 0.1, 0.1)
        assert hparams.seed == 42
        assert hparams.loader == {"batch_size": 64}

    def test_seed_default_is_deterministic(self) -> None:
        assert DataHparams().seed == DataHparams().seed == 42

    def test_two_way_split(self) -> None:
        assert DataHparams(split=(0.8, 0.2)).split == (0.8, 0.2)

    def test_split_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="split must sum to 1"):
            DataHparams(split=(0.5, 0.2))

    def test_split_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            DataHparams(split=(1.5, -0.5))

    def test_shared_loader_rejects_shuffle(self) -> None:
        with pytest.raises(ValidationError, match="`shuffle` is stage specific"):
            DataHparams(loader={"batch_size": 4, "shuffle": True})

    def test_stage_loader_accepts_shuffle(self) -> None:
        assert DataHparams(train_loader={"shuffle": False}).train_loader == {"shuffle": False}

    def test_persistent_workers_need_workers(self) -> None:
        with pytest.raises(ValidationError, match="requires num_workers > 0"):
            DataHparams(loader={"persistent_workers": True})

    def test_persistent_workers_with_workers_is_fine(self) -> None:
        assert DataHparams(loader={"num_workers": 2, "persistent_workers": True})

    def test_persistent_workers_inherits_shared_workers(self) -> None:
        assert DataHparams(
            loader={"num_workers": 2}, train_loader={"persistent_workers": True}
        )

    def test_typo_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DataHparams(spilt=(0.5, 0.5))

    def test_removed_loader_schema(self) -> None:
        import hydrantic.data as data_module

        assert not hasattr(data_module, "DataLoaderHparams")
        assert not hasattr(data_module, "PyTorchData")


class TestSplitDataset:
    """Tests the fractional split helper."""

    def test_three_way_sizes(self, sample_tensor_dataset: Dataset) -> None:
        subsets = split_dataset(sample_tensor_dataset, (0.8, 0.1, 0.1), seed=0)
        assert [len(s) for s in subsets] == [80, 10, 10]

    def test_two_way_sizes(self, sample_tensor_dataset: Dataset) -> None:
        subsets = split_dataset(sample_tensor_dataset, (0.7, 0.3), seed=0)
        assert [len(s) for s in subsets] == [70, 30]

    def test_reproducible(self, sample_tensor_dataset: Dataset) -> None:
        first = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=7)
        second = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=7)
        assert first[0].indices == second[0].indices

    def test_different_seeds_differ(self, sample_tensor_dataset: Dataset) -> None:
        first = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=1)
        second = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=2)
        assert first[0].indices != second[0].indices

    def test_independent_of_global_rng(self, sample_tensor_dataset: Dataset) -> None:
        torch.manual_seed(1234)
        first = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=7)

        torch.manual_seed(9999)
        torch.randn(100)  # advance the global RNG
        second = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=7)

        assert first[0].indices == second[0].indices

    def test_does_not_disturb_global_rng(self, sample_tensor_dataset: Dataset) -> None:
        torch.manual_seed(0)
        expected = torch.randn(3)

        torch.manual_seed(0)
        split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=7)
        assert torch.equal(torch.randn(3), expected)

    def test_bad_split_raises(self, sample_tensor_dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="split must sum to 1"):
            split_dataset(sample_tensor_dataset, (0.5, 0.2), seed=0)

    def test_iterable_dataset_raises(self) -> None:
        class Stream(IterableDataset):
            def __iter__(self):
                yield 1

        with pytest.raises(TypeError, match="Cannot fraction-split an IterableDataset"):
            split_dataset(Stream(), (0.8, 0.2), seed=0)


class TestDataModuleLifecycle:
    """Tests that data is built in `setup`, not in `__init__`."""

    def test_init_does_no_work(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()

        assert data.train_dataset is None
        assert data.val_dataset is None
        assert data.test_dataset is None
        assert tiny_data_cls.build_count == 0

    def test_is_a_lightning_datamodule(self, tiny_data_cls: type[TinyData]) -> None:
        assert isinstance(tiny_data_cls(), lightning.LightningDataModule)

    def test_setup_builds_the_datasets(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()
        data.setup("fit")

        assert isinstance(data.train_dataset, Subset)
        assert len(data.train_dataset) == 80
        assert tiny_data_cls.build_count == 1

    def test_setup_is_idempotent(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()
        data.setup("fit")
        data.setup("test")

        assert tiny_data_cls.build_count == 1

    def test_split_is_consistent_across_instances(self, tiny_data_cls: type[TinyData]) -> None:
        # The DDP invariant: every rank constructs its own instance and must agree on the split.
        first, second = tiny_data_cls(), tiny_data_cls()
        first.setup("fit")
        second.setup("fit")

        assert first.train_dataset.indices == second.train_dataset.indices

    def test_hparams_are_saved_but_not_logged(self, tiny_data_cls: type[TinyData]) -> None:
        # Lightning refuses to merge a module's and a datamodule's hparams when a shared key
        # differs, and both are keyed `hparams`, so the datamodule keeps out of the logger.
        data = tiny_data_cls(num_samples=32)

        assert data.hparams["hparams"]["num_samples"] == 32
        assert data._log_hyperparams is False


class TestDataModuleDataloaders:
    """Tests dataloader construction and per-stage settings."""

    def test_train_dataloader_shuffles(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()
        data.setup("fit")

        from torch.utils.data import RandomSampler

        assert isinstance(data.train_dataloader().sampler, RandomSampler)

    def test_val_dataloader_does_not_shuffle(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()
        data.setup("fit")

        from torch.utils.data import SequentialSampler

        assert isinstance(data.val_dataloader().sampler, SequentialSampler)

    def test_shared_loader_settings_apply(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls(loader={"batch_size": 5})
        data.setup("fit")

        assert data.train_dataloader().batch_size == 5
        assert data.val_dataloader().batch_size == 5

    def test_stage_settings_override_shared(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls(loader={"batch_size": 5}, val_loader={"batch_size": 2})
        data.setup("fit")

        assert data.train_dataloader().batch_size == 5
        assert data.val_dataloader().batch_size == 2

    def test_stage_settings_can_enable_shuffle(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls(val_loader={"shuffle": True})
        data.setup("fit")

        from torch.utils.data import RandomSampler

        assert isinstance(data.val_dataloader().sampler, RandomSampler)

    def test_two_way_split_has_no_test_dataloader(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls(split=(0.8, 0.2))
        data.setup("fit")

        assert data.test_dataset is None
        assert data.test_dataloader() is None

    def test_three_way_split_has_a_test_dataloader(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()
        data.setup("fit")

        assert isinstance(data.test_dataloader(), DataLoader)

    def test_dataloader_before_setup_is_none(self, tiny_data_cls: type[TinyData]) -> None:
        assert tiny_data_cls().train_dataloader() is None


class TestDataModuleHooks:
    """Tests the dataset-providing hooks."""

    def test_missing_get_dataset_raises_in_setup(self) -> None:
        class Incomplete(DataModule[DataHparams]):
            pass

        data = Incomplete()
        with pytest.raises(NotImplementedError, match="must implement `get_dataset`"):
            data.setup("fit")

    def test_get_split_datasets_override(self) -> None:
        class PreSplit(DataModule[DataHparams]):
            def get_split_datasets(self):
                make = lambda n: TensorDataset(torch.randn(n, 4), torch.zeros(n, dtype=torch.long))
                return make(10), make(5), None

        data = PreSplit(loader={"batch_size": 2})
        data.setup("fit")

        assert len(data.train_dataset) == 10
        assert len(data.val_dataset) == 5
        assert data.test_dataloader() is None

    def test_prepare_data_is_available(self, tiny_data_cls: type[TinyData]) -> None:
        assert hasattr(tiny_data_cls(), "prepare_data")

    def test_pre_init_is_gone(self, tiny_data_cls: type[TinyData]) -> None:
        assert not hasattr(tiny_data_cls(), "pre_init")


class TestDataModuleWithTrainer:
    """Tests the datamodule driving a real fit."""

    def test_fit_runs(self, tiny_data_cls: type[TinyData], tiny_model_cls) -> None:
        data = tiny_data_cls(num_samples=32, loader={"batch_size": 8})
        trainer = lightning.Trainer(
            max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False
        )
        trainer.fit(tiny_model_cls(), datamodule=data)

        assert trainer.state.finished
        assert "validation/loss" in trainer.callback_metrics

    def test_trainer_setup_builds_data(self, tiny_data_cls: type[TinyData], tiny_model_cls) -> None:
        data = tiny_data_cls(num_samples=32, loader={"batch_size": 8})
        assert data.train_dataset is None

        lightning.Trainer(
            fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
        ).fit(tiny_model_cls(), datamodule=data)

        assert data.train_dataset is not None
