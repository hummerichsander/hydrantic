from omegaconf import OmegaConf

from ...hparams import Hparams


def test_hparams():
    class TestHparams(Hparams):
        a: int
        b: str

    # check if basic hparams work
    t = TestHparams(a=1, b="test")
    assert t.a == 1
    assert t.b == "test"

    # check methods
    assert TestHparams.from_dict({"a": 1, "b": "test"}) == t

    cfg = OmegaConf.create({"a": 1, "b": "test"})
    assert TestHparams.from_config(cfg) == t

    assert t.attribute_dict == {"a": 1, "b": "test"}
    assert t.attribute_dict.a == 1
    assert t.attribute_dict.b == "test"
