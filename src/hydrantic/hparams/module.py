from typing import TypeVar, Generic, Type, cast, get_args

from .hparams import Hparams


H = TypeVar("H", bound=Hparams)


class HparamsModule(Generic[H]):
    """This is the base class for all modules that carry hparams."""

    thparams: H
    hparams_schema: Type[Hparams]

    def __init_subclass__(cls, **kwargs):
        """Automatically set hparams_schema class attribute when subclass is created."""
        super().__init_subclass__(**kwargs)

        # Skip if already set (legacy compatibility)
        if "hparams_schema" in cls.__dict__:
            return

        # Try to infer from generic type parameter
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:  # type: ignore
                if hasattr(base, "__origin__"):
                    origin = base.__origin__
                    # Check if this base is HparamsModule or a subclass of it
                    if origin is HparamsModule or (
                        isinstance(origin, type) and issubclass(origin, HparamsModule)
                    ):
                        args = get_args(base)
                        if args:
                            schema = args[0]
                            # Check if the type parameter is still the generic TypeVar
                            if not isinstance(schema, TypeVar):
                                cls.hparams_schema = schema
                                return

    def __init__(self, hparams: Hparams | dict):
        # Validate input
        if hparams is None:
            raise TypeError("thparams must not be None")

        # Get the hparams schema (now a class attribute set by __init_subclass__)
        if not hasattr(self.__class__, "hparams_schema"):
            raise TypeError(
                f"{self.__class__.__name__} must specify a type parameter when inheriting from HparamsModule. "
                f"Example: class {self.__class__.__name__}(HparamsModule[YourHparamsClass])"
            )

        hparams_schema = self.__class__.hparams_schema

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
