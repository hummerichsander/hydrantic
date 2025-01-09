from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import Field

from ..utils.attribute_dict import AttributeDict


class Hparams(BaseModel):
    """This is the base class for all hyperparameters. It uses the pydantic library to validate the hyperparameters."""

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def from_config(cls, config: OmegaConf):
        """Create an instance of the class from a config object.

        :param config: The config object.
        :return: The instance of the class."""

        dict_config: dict[str, Any] = OmegaConf.to_object(config)
        return cls(**dict_config)

    @classmethod
    def from_dict(cls, dict_config: dict[str, Any]):
        """Create an instance of the class from a dictionary.

        :param dict_config: The dictionary.
        :return: The instance of the class."""

        return cls(**dict_config)

    @property
    def attribute_dict(self):
        """Return the attributes of the class as an attributedict."""

        return AttributeDict(**self.model_dump(mode="dict"))


Hparam = Field
