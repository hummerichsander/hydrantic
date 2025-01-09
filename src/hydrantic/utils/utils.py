from typing import Any
from importlib import import_module


def import_from_string(string: str) -> Any:
    """Imports an object from a string.

    :param string: String representation of an object to import.
    :return: The imported object."""

    module_name, name = string.rsplit(".", 1)
    module = import_module(module_name)
    object_ = getattr(module, name)
    return object_
