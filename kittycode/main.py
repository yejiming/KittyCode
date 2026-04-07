"""CLI entry point wrappers for KittyCode."""

from __future__ import annotations

from .cli import main as _cli_main


def main() -> int:
    _cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())