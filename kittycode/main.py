"""CLI entry point wrappers for KittyCode."""

from __future__ import annotations

import sys

from .cli import main as _cli_main
from .config.tui import run_config_tui


def main() -> int:
    if "--config" in sys.argv[1:]:
        run_config_tui()
        return 0
    _cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
