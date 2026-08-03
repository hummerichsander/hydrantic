"""Allows `python -m hydrantic <command> ...`."""

import sys

from hydrantic.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
