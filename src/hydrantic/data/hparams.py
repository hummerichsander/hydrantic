import random

from ..hparams import Hparams, Hparam


class DataLoaderHparams(Hparams):
    batch_size: int
    shuffle: bool = False
    pin_memory: bool = False
    num_workers: int = 0
    persistent_workers: bool = False
    drop_last: bool = False


class DataHparams(Hparams):
    module_name: str
    loader: DataLoaderHparams
    seed: int = Hparam(default_factory=lambda: random.randint(0, 10_000))
    split: tuple[float, float] | tuple[float, float, float]
