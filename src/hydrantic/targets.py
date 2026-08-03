import difflib
import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from hydra._internal.utils import _locate
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from pydantic import AfterValidator

from hydrantic.hparams import Hparams, HparamsModule, to_plain_container

RECURSIVE_HINT = (
    "hydrantic instantiates `_target_` nodes inside the model itself, so they must reach it as "
    "data. Let the hydrantic CLI build the model, or pass `_recursive_=False` to "
    "`hydra.utils.instantiate`."
)

# Targets that import cleanly but fail at construction when an optional package is absent.
# Deliberately limited to the loggers hydrantic ships config groups for: selecting
# `logger=wandb` is an expectation hydrantic itself created, so it should say how to meet it.
# Anything else surfaces the underlying library's own message at instantiation time.
_OPTIONAL_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "lightning.pytorch.loggers.WandbLogger": ("wandb", "hydrantic[wandb]"),
    "lightning.pytorch.loggers.wandb.WandbLogger": ("wandb", "hydrantic[wandb]"),
    "lightning.pytorch.loggers.TensorBoardLogger": ("tensorboard", "hydrantic[tensorboard]"),
    "lightning.pytorch.loggers.tensorboard.TensorBoardLogger": (
        "tensorboard",
        "hydrantic[tensorboard]",
    ),
}


def check_optional_requirement(target: str) -> str | None:
    """Checks whether a known target's optional package is installed.

    :param target: The dotted path from a `_target_` key.
    :return: A problem description, or None if nothing is missing."""

    if (requirement := _OPTIONAL_REQUIREMENTS.get(target)) is None:
        return None

    package, extra = requirement
    if importlib.util.find_spec(package) is not None:
        return None

    return f"{target} needs the {package!r} package: pip install '{extra}'"


def resolve_target(target: str) -> Any:
    """Imports the object named by a `_target_` string.

    :param target: A dotted path such as `"torch.optim.Adam"`.
    :return: The imported object.
    :raises ImportError: If the path cannot be imported."""

    try:
        return _locate(target)
    except Exception as error:
        raise ImportError(f"cannot import {target!r}{_suggest_target(target)}") from error


def _suggest_target(target: str) -> str:
    """Builds a `did you mean` fragment for an unimportable dotted path.

    :param target: The dotted path that failed to import.
    :return: A parenthesised suggestion, or an empty string."""

    module_path, _, name = target.rpartition(".")
    if not module_path:
        return ""
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return ""

    candidates = [n for n in dir(module) if not n.startswith("_")]
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return f" (did you mean '{module_path}.{matches[0]}'?)" if matches else ""


def _suggest(name: str, candidates: object) -> str:
    """Builds a `did you mean` fragment for a misspelled name.

    :param name: The name that was not found.
    :param candidates: The valid names.
    :return: A parenthesised suggestion, or an empty string."""

    matches = difflib.get_close_matches(name, [str(c) for c in candidates], n=1, cutoff=0.6)
    return f" (did you mean {matches[0]!r}?)" if matches else ""


def check_target_signature(
    node: Mapping[str, Any], *, skip_first: bool = False
) -> list[tuple[str, str]]:
    """Checks a `_target_` node against the real signature of its target.

    :param node: The mapping holding `_target_` and the keyword arguments.
    :param skip_first: Whether the target's first parameter is supplied positionally later,
        as for an optimizer's `params` or a scheduler's `optimizer`.
    :return: `(key, message)` pairs, where `key` is the offending key within the node, or an
        empty string for problems about the node as a whole. Empty if the node is valid."""

    target = node["_target_"]
    if not isinstance(target, str):
        return [("_target_", f"must be a dotted path string, got {type(target).__name__}")]

    try:
        resolved = resolve_target(target)
    except ImportError as error:
        return [("_target_", str(error))]

    if (missing := check_optional_requirement(target)) is not None:
        return [("_target_", missing)]

    if not callable(resolved):
        return [("_target_", f"{target!r} is not callable")]

    try:
        signature = inspect.signature(resolved)
    except (TypeError, ValueError):
        return []  # builtins and C extensions expose no signature; nothing to check

    parameters = list(signature.parameters.values())
    if inspect.isclass(resolved) and parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]
    parameters = list(parameters[1:] if skip_first else parameters)

    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    accepted = {p.name for p in parameters if p.kind in keyword_kinds}
    supplied = {key for key in node if not key.startswith("_")}

    problems: list[tuple[str, str]] = []

    if not accepts_any:
        for key in sorted(supplied - accepted):
            problems.append(
                (key, f"unexpected keyword for {target}{_suggest(key, sorted(accepted))}")
            )

    required = {
        p.name
        for p in parameters
        if p.default is inspect.Parameter.empty and p.kind in keyword_kinds
    }
    for name in sorted(required - supplied):
        problems.append((name, f"missing required argument for {target}"))

    return problems


def validate_target(value: Any) -> dict[str, Any]:
    """Validates a `_target_` mapping, used by the `Target` annotated type.

    Accepts either a bare `_target_` node or Lightning's `lr_scheduler` dict form, which nests
    the `_target_` node under a `scheduler` key alongside `monitor`, `interval` and friends.

    :param value: The value to validate.
    :return: The value as a plain dict.
    :raises ValueError: If the node is not a valid `_target_` mapping."""

    if callable(value) and not isinstance(value, Mapping):
        raise ValueError(
            f"expected a `_target_` mapping, got {type(value).__name__}. {RECURSIVE_HINT}"
        )

    if not isinstance(value, Mapping):
        raise ValueError(f"expected a `_target_` mapping, got {type(value).__name__}")

    node = to_plain_container(value)

    inner = node if "_target_" in node else node.get("scheduler")
    if not isinstance(inner, Mapping) or "_target_" not in inner:
        raise ValueError(
            f"needs a `_target_` key, either at the top level or nested under `scheduler`; "
            f"got keys {sorted(node)}"
        )

    if problems := check_target_signature(inner, skip_first=True):
        raise ValueError(
            "; ".join(f"{key}: {message}" if key else message for key, message in problems)
        )

    return node


Target = Annotated[dict[str, Any], AfterValidator(validate_target)]
"""A `_target_` mapping that is import- and signature-checked at validation time."""


@dataclass(frozen=True)
class ConfigProblem:
    """A single problem found in a composed configuration."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}\n    {self.message}"


class ConfigError(Exception):
    """Raised when a composed configuration fails validation."""

    def __init__(self, problems: list[ConfigProblem]) -> None:
        self.problems = problems
        body = "\n\n".join(f"  {problem}" for problem in problems)
        plural = "problem" if len(problems) == 1 else "problems"
        super().__init__(f"Configuration invalid ({len(problems)} {plural}):\n\n{body}\n")


def _check_hparams_schema(node: Mapping[str, Any], path: str) -> list[ConfigProblem]:
    """Validates a `model`/`data` node against the `Hparams` schema of its target.

    :param node: The node holding `_target_` and the hyperparameter values.
    :param path: The config path of the node, used in problem messages.
    :return: The problems found."""

    try:
        resolved = resolve_target(node["_target_"])
    except ImportError as error:
        return [ConfigProblem(f"{path}._target_", str(error))]

    if not (isinstance(resolved, type) and issubclass(resolved, HparamsModule)):
        return []

    schema = getattr(resolved, "hparams_schema", None)
    if schema is None:
        return [
            ConfigProblem(
                path,
                f"{resolved.__name__} does not bind a hyperparameter schema; parametrise a "
                f"generic base, e.g. class {resolved.__name__}(Model[YourHparams])",
            )
        ]

    values = {key: item for key, item in node.items() if not key.startswith("_")}
    try:
        schema.model_validate(to_plain_container(values))
    except Exception as error:
        return _problems_from_validation_error(error, path, schema)
    return []


def _problems_from_validation_error(
    error: Exception, path: str, schema: type[Hparams]
) -> list[ConfigProblem]:
    """Converts a pydantic `ValidationError` into per-field problems.

    :param error: The exception raised by `model_validate`.
    :param path: The config path of the validated node.
    :param schema: The schema that was validated against.
    :return: The problems found."""

    errors = getattr(error, "errors", None)
    if not callable(errors):
        return [ConfigProblem(path, str(error))]

    problems = []
    for entry in errors():
        location = ".".join(str(part) for part in entry["loc"])
        full_path = f"{path}.{location}" if location else path

        if entry["type"] == "extra_forbidden":
            message = (
                f"unexpected field for {schema.__name__}"
                f"{_suggest(location, sorted(schema.model_fields))}"
            )
        else:
            # pydantic prefixes messages raised by custom validators; the path already says
            # this is a validation problem, so the prefix is noise.
            message = entry["msg"].removeprefix("Value error, ")

        problems.append(ConfigProblem(full_path, message))
    return problems


def _iter_target_nodes(value: Any, path: str) -> list[tuple[str, Mapping[str, Any]]]:
    """Collects every `_target_` node in a config subtree, depth first.

    :param value: The subtree to walk.
    :param path: The config path of the subtree.
    :return: `(path, node)` pairs for each `_target_` mapping found."""

    found: list[tuple[str, Mapping[str, Any]]] = []

    if isinstance(value, Mapping):
        if "_target_" in value:
            found.append((path, value))
        # Sorted so that a node reachable by two paths is reported at a stable one. The shipped
        # groups interpolate `logger` and `callbacks` into `trainer`, so each such node appears
        # twice; sorting means the source node wins, which is the one the user edits.
        for key in sorted(value, key=str):
            if not str(key).startswith("_"):
                child = f"{path}.{key}" if path else str(key)
                found.extend(_iter_target_nodes(value[key], child))
    elif isinstance(value, list | ListConfig):
        for index, item in enumerate(value):
            found.extend(_iter_target_nodes(item, f"{path}[{index}]"))

    return found


def check_config(config: DictConfig | Mapping[str, Any]) -> list[ConfigProblem]:
    """Validates a composed configuration without instantiating anything.

    Every `_target_` node is imported and its keyword arguments are bound against the real
    signature, and the `model` and `data` nodes are validated against their `Hparams` schemas.
    All problems are collected rather than raising on the first, and nothing is constructed, so
    this is fast enough to run before every job.

    :param config: The composed configuration.
    :return: The problems found, empty if the configuration is valid."""

    try:
        resolved = to_plain_container(config)
    except Exception as error:  # unresolvable interpolation or a MISSING value
        return [ConfigProblem("<config>", f"cannot resolve configuration: {error}")]

    problems: list[ConfigProblem] = []

    for section in ("model", "data"):
        node = resolved.get(section)
        if node is None:
            continue
        if not isinstance(node, Mapping) or "_target_" not in node:
            problems.append(ConfigProblem(section, "needs a `_target_` key"))
            continue
        problems.extend(_check_hparams_schema(node, section))

    # `model`/`data` hyperparameters are validated by their schema above; their nested
    # `_target_` nodes are checked there too, via the `Target` annotated type.
    checked_by_schema = {"model", "data"}
    seen: set[tuple[str, str]] = set()

    for path, node in _iter_target_nodes(resolved, ""):
        if path.split(".")[0].split("[")[0] in checked_by_schema:
            continue
        for key, message in check_target_signature(node):
            # One node can be reachable by two paths through interpolation; report it once.
            if (key, message) in seen:
                continue
            seen.add((key, message))
            full_path = ".".join(part for part in (path, key) if part)
            problems.append(ConfigProblem(full_path or "<root>", message))

    return problems


def instantiate_target(node: Mapping[str, Any], *args: Any) -> Any:
    """Instantiates a `_target_` mapping, tolerating a stray `_partial_: true`.

    :param node: The mapping holding `_target_` and the keyword arguments.
    :param args: Positional arguments passed to the target, such as a model's parameters.
    :return: The instantiated object."""

    config = dict(to_plain_container(node))
    partial = config.pop("_partial_", False)

    try:
        # `_partial_: true` is the idiom other Hydra templates use for optimizers. hydrantic
        # passes the positional arguments itself, so honour the flag but apply them either way.
        if partial:
            return instantiate(config, _partial_=True)(*args)
        return instantiate(config, *args)
    except InstantiationException as error:
        raise InstantiationException(
            f"could not instantiate {config.get('_target_')!r}: {error}"
        ) from error


def instantiate_module(config: DictConfig | Mapping[str, Any]) -> Any:
    """Instantiates a `model` or `data` node, keeping its hyperparameters as data.

    `_recursive_=False` stops Hydra from building the nested `_target_` nodes so they reach the
    pydantic schema as plain mappings, and `_convert_="all"` strips the OmegaConf containers
    that `torch.load(weights_only=True)` would later refuse to unpickle.

    :param config: The node to instantiate.
    :return: The instantiated object."""

    return instantiate(config, _recursive_=False, _convert_="all")
