"""Main CLI entry point for hydrantic."""

import argparse
import sys


def main():
    """Main entry point for the hydrantic CLI.

    Provides access to hydrantic commands like fit and profile.
    """
    parser = argparse.ArgumentParser(
        prog="hydrantic.cli",
        description="Hydrantic - A PyTorch Lightning wrapper with Hydra and Pydantic integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  fit       Train a model using the configuration system
  profile   Profile model training performance
  
For command-specific help, use:
  python -m hydrantic.cli.fit --help
  python -m hydrantic.cli.profile --help
  
Example usage:
  python -m hydrantic.cli.fit config_name=my_config
  python -m hydrantic.cli.profile config_name=my_config
""",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.0.2")

    args = parser.parse_args()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
