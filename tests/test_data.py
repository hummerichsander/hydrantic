import io

import lightning
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, TensorDataset

from conftest import TinyData

from hydrantic import DataModule, Hparams, LoaderHparams, SplitHparams, split_dataset


class TestSplitHparams:
    """Tests the opt-in split schema."""

    def test_defaults(self) -> None:
        hparams = SplitHparams()
        assert hparams.split == (0.8, 0.1, 0.1)
        assert hparams.seed == 42

    def test_carries_only_split_and_seed(self) -> None:
        assert set(SplitHparams.model_fields) == {"split", "seed"}

    def test_seed_default_is_deterministic(self) -> None:
        assert SplitHparams().seed == SplitHparams().seed == 42

    def test_two_way_split(self) -> None:
        assert SplitHparams(split=(0.8, 0.2)).split == (0.8, 0.2)

    def test_split_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="split must sum to 1"):
            SplitHparams(split=(0.5, 0.2))

    def test_split_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            SplitHparams(split=(1.5, -0.5))

    def test_typo_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SplitHparams(spilt=(0.5, 0.5))

    def test_split_dataset_method(self, sample_tensor_dataset: Dataset) -> None:
        subsets = SplitHparams(split=(0.6, 0.4), seed=3).split_dataset(sample_tensor_dataset)
        assert [len(s) for s in subsets] == [60, 40]

    def test_split_dataset_method_matches_the_function(
        self, sample_tensor_dataset: Dataset
    ) -> None:
        hparams = SplitHparams(seed=11)
        from_method = hparams.split_dataset(sample_tensor_dataset)
        from_function = split_dataset(sample_tensor_dataset, hparams.split, hparams.seed)
        assert from_method[0].indices == from_function[0].indices


class TestLoaderHparams:
    """Tests the opt-in DataLoader schema."""

    def test_defaults(self) -> None:
        hparams = LoaderHparams()
        assert hparams.batch_size == 64
        assert hparams.num_workers == 0
        assert hparams.pin_memory is False

    def test_fields(self) -> None:
        assert set(LoaderHparams.model_fields) == {
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "drop_last",
        }

    def test_shuffle_is_not_a_field(self) -> None:
        # shuffle differs per stage; a baked-in default is how a train loader silently
        # starts or stops shuffling.
        assert "shuffle" not in LoaderHparams.model_fields

    def test_typed_validation(self) -> None:
        assert LoaderHparams(batch_size="32").batch_size == 32
        with pytest.raises(ValidationError):
            LoaderHparams(batch_size="lots")

    def test_typo_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            LoaderHparams(batch_sze=32)

    def test_loader_kwargs(self) -> None:
        kwargs = LoaderHparams(batch_size=8, num_workers=2).loader_kwargs()
        assert kwargs == {
            "batch_size": 8,
            "num_workers": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
        }

    def test_loader_kwargs_overrides(self) -> None:
        kwargs = LoaderHparams(batch_size=8).loader_kwargs(shuffle=True, batch_size=1)
        assert kwargs["shuffle"] is True
        assert kwargs["batch_size"] == 1

    def test_loader_kwargs_excludes_subclass_fields(self) -> None:
        # A subclass's own hyperparameters must not be smuggled into the DataLoader call.
        class MyHparams(LoaderHparams):
            path: str = "data.pt"
            num_samples: int = 10

        kwargs = MyHparams().loader_kwargs()
        assert "path" not in kwargs
        assert "num_samples" not in kwargs

    def test_loader_kwargs_reach_a_dataloader(self, sample_tensor_dataset: Dataset) -> None:
        hparams = LoaderHparams(batch_size=5, drop_last=True)
        loader = DataLoader(sample_tensor_dataset, **hparams.loader_kwargs(shuffle=True))

        assert loader.batch_size == 5
        assert loader.drop_last is True

    def test_composes_with_split_hparams(self) -> None:
        class Both(SplitHparams, LoaderHparams):
            path: str = "data.pt"

        hparams = Both(batch_size=4, split=(0.9, 0.1))
        assert hparams.batch_size == 4
        assert hparams.split == (0.9, 0.1)
        assert hparams.loader_kwargs()["batch_size"] == 4


class TestDataModuleSurface:
    """Tests that `DataModule` adds typed hyperparameters and nothing else."""

    def test_is_a_lightning_datamodule(self, tiny_data_cls: type[TinyData]) -> None:
        assert isinstance(tiny_data_cls(), lightning.LightningDataModule)

    def test_thparams_is_typed(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls(num_samples=32)
        assert data.thparams.num_samples == 32
        assert data.thparams.split == (0.8, 0.1, 0.1)

    def test_typo_is_rejected(self, tiny_data_cls: type[TinyData]) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            tiny_data_cls(num_smaples=32)

    def test_hparams_are_saved_but_not_logged(self, tiny_data_cls: type[TinyData]) -> None:
        # Lightning refuses to merge a module's and a datamodule's hparams when a shared key
        # differs, and both are keyed `hparams`, so the datamodule stays out of the logger.
        data = tiny_data_cls(num_samples=32)

        assert data.hparams["hparams"]["num_samples"] == 32
        assert data._log_hyperparams is False

    @pytest.mark.parametrize(
        "name",
        ["get_dataset", "get_split_datasets", "dataloader_kwargs", "_build_dataloader"],
    )
    def test_no_hydrantic_specific_data_hooks(self, name: str) -> None:
        assert not hasattr(DataModule, name)

    def test_lightning_hooks_are_inherited(self) -> None:
        for name in ("prepare_data", "setup", "train_dataloader", "val_dataloader"):
            assert hasattr(DataModule, name)

    def test_binds_to_plain_hparams(self) -> None:
        # No data-specific base is required; split/loader schemas are opt-in.
        class Bare(DataModule[Hparams]):
            pass

        assert Bare().thparams == Hparams()

    def test_removed_data_hparams(self) -> None:
        import hydrantic

        assert not hasattr(hydrantic, "DataHparams")


class TestSplitDataset:
    """Tests the split helper, which now lives in `hydrantic.utils`."""

    def test_importable_from_utils(self) -> None:
        from hydrantic.utils import split_dataset as from_utils

        assert from_utils is split_dataset

    def test_three_way_sizes(self, sample_tensor_dataset: Dataset) -> None:
        subsets = split_dataset(sample_tensor_dataset, (0.8, 0.1, 0.1), seed=0)
        assert [len(s) for s in subsets] == [80, 10, 10]

    def test_two_way_sizes(self, sample_tensor_dataset: Dataset) -> None:
        subsets = split_dataset(sample_tensor_dataset, (0.7, 0.3), seed=0)
        assert [len(s) for s in subsets] == [70, 30]

    def test_returns_subsets(self, sample_tensor_dataset: Dataset) -> None:
        # Consumers reach through the split with `loader.dataset.dataset`; keep it a Subset.
        subsets = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=0)
        assert all(isinstance(s, Subset) for s in subsets)
        assert subsets[0].dataset is sample_tensor_dataset

    def test_unpacks_to_the_split_length(self, sample_tensor_dataset: Dataset) -> None:
        train, val = split_dataset(sample_tensor_dataset, (0.8, 0.2), seed=0)
        assert len(train) == 80 and len(val) == 20

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


class TestUserWrittenDataModule:
    """Tests a datamodule written with plain Lightning hooks, as users now write them."""

    def test_init_does_no_work(self, tiny_data_cls: type[TinyData]) -> None:
        tiny_data_cls()
        assert tiny_data_cls.build_count == 0

    def test_setup_builds_and_splits(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()
        data.setup("fit")

        assert isinstance(data.train, Subset)
        assert (len(data.train), len(data.val), len(data.test)) == (80, 10, 10)

    def test_user_guard_makes_setup_idempotent(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls()
        data.setup("fit")
        data.setup("test")

        assert tiny_data_cls.build_count == 1

    def test_split_is_consistent_across_instances(self, tiny_data_cls: type[TinyData]) -> None:
        # The DDP invariant: every rank builds its own instance and must agree on the split.
        first, second = tiny_data_cls(), tiny_data_cls()
        first.setup("fit")
        second.setup("fit")

        assert first.train.indices == second.train.indices

    def test_dataloaders(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls(batch_size=4)
        data.setup("fit")

        assert data.train_dataloader().batch_size == 4
        assert isinstance(data.val_dataloader(), DataLoader)

    def test_shuffling_is_explicit(self, tiny_data_cls: type[TinyData]) -> None:
        from torch.utils.data import RandomSampler, SequentialSampler

        data = tiny_data_cls()
        data.setup("fit")

        assert isinstance(data.train_dataloader().sampler, RandomSampler)
        assert isinstance(data.val_dataloader().sampler, SequentialSampler)

    def test_two_way_split(self, tiny_data_cls: type[TinyData]) -> None:
        data = tiny_data_cls(split=(0.8, 0.2))
        data.setup("fit")

        assert data.test is None
        assert data.test_dataloader() is None

    def test_pre_split_data_needs_neither_mixin(self) -> None:
        # The case that motivated dropping split/seed from the base: separate files.
        class PreSplitHparams(LoaderHparams):
            n_train: int = 10
            n_val: int = 5

        class PreSplit(DataModule[PreSplitHparams]):
            def setup(self, stage):
                make = lambda n: TensorDataset(torch.randn(n, 4), torch.zeros(n, dtype=torch.long))
                self.train, self.val = make(self.thparams.n_train), make(self.thparams.n_val)

            def train_dataloader(self):
                return DataLoader(self.train, **self.thparams.loader_kwargs(shuffle=True))

        data = PreSplit(batch_size=2)
        data.setup("fit")

        assert "split" not in data.thparams.model_dump()
        assert len(data.train) == 10
        assert data.train_dataloader().batch_size == 2


class TestDataModuleWithTrainer:
    """Tests the datamodule driving a real fit."""

    def test_fit_runs(self, tiny_data_cls: type[TinyData], tiny_model_cls) -> None:
        data = tiny_data_cls(num_samples=32, batch_size=8)
        trainer = lightning.Trainer(
            max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False
        )
        trainer.fit(tiny_model_cls(), datamodule=data)

        assert trainer.state.finished
        assert "validation/loss" in trainer.callback_metrics

    def test_trainer_calls_setup(self, tiny_data_cls: type[TinyData], tiny_model_cls) -> None:
        data = tiny_data_cls(num_samples=32, batch_size=8)
        assert tiny_data_cls.build_count == 0

        lightning.Trainer(
            fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
        ).fit(tiny_model_cls(), datamodule=data)

        assert tiny_data_cls.build_count == 1

    def test_load_from_checkpoint_restores_hparams(self, tiny_data_cls: type[TinyData]) -> None:
        # Datamodules store their hyperparameters under their own checkpoint key.
        data = tiny_data_cls(num_samples=48)
        buffer = io.BytesIO()
        torch.save(
            {
                tiny_data_cls.CHECKPOINT_HYPER_PARAMS_KEY: dict(data.hparams_initial),
                "pytorch-lightning_version": lightning.__version__,
            },
            buffer,
        )
        buffer.seek(0)

        restored = tiny_data_cls.load_from_checkpoint(buffer)
        assert restored.thparams == data.thparams

    def test_hparams_land_in_a_real_checkpoint(
        self, tiny_data_cls: type[TinyData], tiny_model_cls, tmp_path
    ) -> None:
        # The end-to-end version: Trainer writes both the model's and the datamodule's
        # hyperparameters, under their separate keys, and both survive weights_only=True.
        data = tiny_data_cls(num_samples=32, batch_size=8)
        trainer = lightning.Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            default_root_dir=tmp_path,
            enable_progress_bar=False,
        )
        trainer.fit(tiny_model_cls(), datamodule=data)
        trainer.save_checkpoint(tmp_path / "out.ckpt")

        blob = torch.load(tmp_path / "out.ckpt", weights_only=True)
        assert blob["hyper_parameters"]["hparams"]["input_dim"] == 10
        assert blob["datamodule_hyper_parameters"]["hparams"]["num_samples"] == 32

        restored = tiny_data_cls.load_from_checkpoint(tmp_path / "out.ckpt")
        assert restored.thparams == data.thparams
