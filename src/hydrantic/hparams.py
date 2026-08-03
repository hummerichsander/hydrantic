import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast, get_args

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self


class Hparams(BaseModel):
    """Typed, validated hyperparameters for a `HparamsModule`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_yaml(cls, path: str | Path, key: str | None = None) -> Self:
        """Loads and validates hyperparameters from a YAML file.

        Interpolations such as `${oc.env:HOME}` are resolved on load.

        :param path: Path to the YAML file.
        :param key: Dotted key to read the hyperparameters from, e.g. `"model"`.
        :return: The validated hyperparameters."""

        config = OmegaConf.load(path)

        if key is not None:
            selected = OmegaConf.select(config, key, throw_on_missing=True)
            if selected is None:
                available = (
                    ", ".join(map(str, config.keys())) if isinstance(config, DictConfig) else ""
                )
                raise KeyError(f"{path} has no key {key!r}. Available top-level keys: {available}.")
            config = selected

        return cls.model_validate(to_plain_container(config))


Hparam = Field


def to_plain_container(value: Any) -> Any:
    """Recursively converts OmegaConf containers into plain Python containers.

    Hyperparameters end up in the checkpoint, and `torch.load` defaults to `weights_only=True`
    on torch >= 2.6, which refuses to unpickle a `DictConfig`. Normalising on the way in keeps
    checkpoints loadable no matter how the hyperparameters were built.

    :param value: The value to convert.
    :return: The value with every OmegaConf container replaced by a dict or list."""

    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_object(value)
    if isinstance(value, Mapping):
        return {key: to_plain_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_container(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain_container(item) for item in value)
    return value


def _resolve_hparams_schema(cls: type) -> type[Hparams] | None:
    """Finds the concrete `Hparams` subclass bound to a generic base of `cls`.

    :param cls: The class to inspect.
    :return: The bound schema, or None if the class is not parametrised."""

    for klass in cls.__mro__:
        for base in klass.__dict__.get("__orig_bases__", ()):
            for arg in get_args(base):
                if isinstance(arg, type) and issubclass(arg, Hparams):
                    return arg
    return None


def _hparams_arg_name(cls: type) -> str:
    """Returns the name of the first positional `__init__` parameter of `cls`.

    Lightning restores a checkpoint by filtering the stored hyperparameters against
    `__init__`'s signature, so the stored dict has to be keyed by that parameter's name.

    :param cls: The class whose `__init__` to inspect.
    :return: The parameter name, defaulting to `"hparams"`."""

    parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            return parameter.name
    return "hparams"


_H = TypeVar("_H", bound=Hparams)


class HparamsModule(Generic[_H]):
    """Mixin that validates an `Hparams` schema into the typed `thparams` attribute.

    Bind the schema by parametrising the generic base, `class MyModel(Model[MyHparams])`, or by
    setting the `hparams_schema` class attribute explicitly."""

    thparams: _H
    hparams_schema: ClassVar[type[Hparams]]

    # Lightning merges a module's and a datamodule's `hparams_initial` and raises if a shared
    # key holds different values. Both are keyed by their `__init__` parameter name, which is
    # `hparams` for both, so exactly one of the two may send its hyperparameters to the logger.
    log_hparams_to_logger: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Binds `hparams_schema` from the generic parameter of a base class.

        :param kwargs: Class keyword arguments forwarded to `super()`."""

        super().__init_subclass__(**kwargs)

        if "hparams_schema" in cls.__dict__:
            return
        if (schema := _resolve_hparams_schema(cls)) is not None:
            cls.hparams_schema = schema

    def __init__(self, hparams: _H | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Validates `hparams` against the bound schema and stores it on `self.thparams`.

        :param hparams: An `Hparams` instance, or a mapping of field values.
        :param kwargs: Field values, as an alternative to passing a mapping."""

        super().__init__()

        schema = getattr(type(self), "hparams_schema", None)
        if schema is None:
            raise TypeError(
                f"{type(self).__name__} does not bind a hyperparameter schema. Parametrise a "
                f"generic base, e.g. class {type(self).__name__}(Model[YourHparams])."
            )

        if hparams is None:
            self.thparams = cast(_H, schema(**to_plain_container(kwargs)))
        elif isinstance(hparams, schema):
            if kwargs:
                raise TypeError(
                    f"Got both a {schema.__name__} instance and the keyword arguments "
                    f"{sorted(kwargs)}. Pass one or the other."
                )
            self.thparams = hparams
        elif isinstance(hparams, Hparams):
            raise TypeError(
                f"{type(self).__name__} expects {schema.__name__}, got {type(hparams).__name__}."
            )
        elif isinstance(hparams, Mapping):
            merged = {**to_plain_container(hparams), **to_plain_container(kwargs)}
            self.thparams = cast(_H, schema(**merged))
        else:
            raise TypeError(
                f"hparams must be a {schema.__name__} or a mapping, got {type(hparams).__name__}."
            )

        # Only Lightning classes have this; the mixin stays usable on plain objects.
        if hasattr(self, "save_hyperparameters"):
            key = _hparams_arg_name(type(self))
            self.save_hyperparameters(
                {key: self.thparams.model_dump()}, logger=self.log_hparams_to_logger
            )
