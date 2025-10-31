from typing import Any
from typing_extensions import Self

from yaml import safe_load
from omegaconf import OmegaConf, DictConfig
from collections.abc import Mapping
from pydantic import BaseModel
from pydantic import Field


class Hparams(BaseModel, Mapping):
    """This is the base class for all hyperparameters. It uses the pydantic library to validate the
    hyperparameters."""

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def from_config(cls, config: OmegaConf | DictConfig) -> Self:
        """Create an instance of the class from a config object.

        :param config: The config object.
        :return: The instance of the class."""

        dict_config: dict[str, Any] = OmegaConf.to_object(config)  # type: ignore
        return cls(**dict_config)

    @classmethod
    def from_dict(cls, dict_config: dict[str, Any]) -> Self:
        """Create an instance of the class from a dictionary.

        :param dict_config: The dictionary.
        :return: The instance of the class."""

        return cls(**dict_config)

    @classmethod
    def from_yaml(cls, yaml_path: str, key: str | None = None) -> Self:
        """Create an instance of the class from a YAML file.

        :param yaml_path: The path to the YAML file.
        :return: The instance of the class."""

        config = safe_load(open(yaml_path, "r"))
        if key is not None:
            config = config.get(key, {})
        return cls.from_dict(config)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self):
        for field in self.model_fields:
            yield field, getattr(self, field)

    def keys(self):
        return self.model_fields.keys()

    def __len__(self) -> int:
        return len(self.model_fields)


Hparam = Field
