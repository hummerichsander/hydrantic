import os
import sys
from importlib.metadata import PackageNotFoundError, version

from hydrantic.cli.app import STAGES
from hydrantic.cli.app import main as hydra_main

COMMANDS: tuple[str, ...] = (*STAGES, "check")

_CONFIG_DIR_FLAGS = ("--config-dir", "-cd", "--config-path", "-cp")

_HELP = f"""usage: hydrantic <command> [hydra flags] [key=value ...]

commands:
  {", ".join(COMMANDS)}

examples:
  hydrantic fit   --config-dir conf --config-name train
  hydrantic fit   --config-dir conf --config-name train trainer.max_epochs=10
  hydrantic check --config-dir conf --config-name train
  hydrantic test  --config-dir conf --config-name train ckpt_path=last
  hydrantic fit   --config-dir conf --config-name train -m model.optimizer.lr=1e-3,1e-4
  hydrantic fit   --config-dir conf --config-name train +experiment=profile

The command is passed to Hydra as `++stage=<command>`, so `hydrantic stage=test ...` is
equivalent. Everything else is handled by Hydra; see `hydrantic fit --hydra-help`.
The config directory can be set once via the HYDRANTIC_CONFIG_DIR environment variable.
"""


def _get_version() -> str:
    """Returns the installed hydrantic version.

    :return: The version string, or a sentinel when running from an uninstalled tree."""

    try:
        return version("hydrantic")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _has_config_dir_flag(args: list[str]) -> bool:
    """Checks whether Hydra has already been pointed at a config directory.

    :param args: The argument list without the program name.
    :return: True if any config-dir or config-path flag is present."""

    return any(
        arg == flag or arg.startswith(f"{flag}=") for arg in args for flag in _CONFIG_DIR_FLAGS
    )


def normalize_args(args: list[str], env: dict[str, str] | None = None) -> list[str]:
    """Rewrites a hydrantic argument list into a pure Hydra argument list.

    The rewrite is idempotent, which is required because Lightning's DDP launcher re-executes
    `sys.argv[1:]` verbatim in every worker: a second pass must not strip the command again or
    inject a second config directory.

    :param args: The argument list without the program name.
    :param env: Environment to read HYDRANTIC_CONFIG_DIR from, defaulting to `os.environ`.
    :return: The argument list to hand to Hydra.
    :raises ValueError: If the first token is neither a command nor a Hydra argument."""

    env = os.environ if env is None else env
    args = list(args)

    if args and args[0] in COMMANDS:
        args = [*args[1:], f"++stage={args[0]}"]
    elif args and not (args[0].startswith("-") or "=" in args[0]):
        raise ValueError(f"unknown command {args[0]!r}; expected one of {', '.join(COMMANDS)}")

    if not _has_config_dir_flag(args) and (config_dir := env.get("HYDRANTIC_CONFIG_DIR")):
        args[:0] = ["--config-dir", config_dir]

    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `hydrantic` console script.

    :param argv: Argument list to use instead of `sys.argv[1:]`.
    :return: A process exit code."""

    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(_HELP)
        return 0
    if args[0] in ("-V", "--version"):
        print(f"hydrantic {_get_version()}")
        return 0

    try:
        args = normalize_args(args)
    except ValueError as error:
        print(f"hydrantic: {error}\n\n{_HELP}", file=sys.stderr)
        return 2

    sys.argv = [sys.argv[0], *args]
    hydra_main()  # hydra.main owns sys.argv from here and exits on failure
    return 0


if __name__ == "__main__":
    sys.exit(main())
