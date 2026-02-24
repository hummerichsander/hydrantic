from typing import TypeVar, Generic, Type, cast, get_args

from .hparams import Hparams


H = TypeVar("H", bound=Hparams)


class HparamsModule(Generic[H]):
    """This is the base class for all modules that carry hparams."""

    thparams: H

    def __init__(self, hparams: Hparams | dict):
        # Validate input
        if hparams is None:
            raise TypeError("thparams must not be None")

        # Infer the hparams schema from the generic type parameter
        hparams_schema = self._get_hparams_schema()

        # If already the desired schema instance, accept it directly
        if isinstance(hparams, hparams_schema):
            hparams = cast(H, hparams)
        elif isinstance(hparams, dict):
            hparams = cast(H, hparams_schema(**hparams))
        else:
            raise TypeError(
                f"thparams must be of type {hparams_schema} or dict, got {type(hparams)}"
            )

        self.thparams = hparams

    def _get_hparams_schema(self) -> Type[Hparams]:
        """Extract the hparams schema from the generic type parameter H.

        :return: The hparams schema class."""

        for base in self.__class__.__orig_bases__:  # type: ignore
            if hasattr(base, "__origin__"):
                origin = base.__origin__
                if origin is HparamsModule or (
                    isinstance(origin, type) and issubclass(origin, HparamsModule)
                ):
                    args = get_args(base)
                    if args:
                        return args[0]
        raise TypeError(
            f"Could not infer hparams schema from generic type parameter for {self.__class__.__name__}"
        )

    @property
    def hparams_schema(self) -> Type[Hparams]:
        """Returns the `hparams_schema` attribute for legacy."""
        return self._get_hparams_schema()
