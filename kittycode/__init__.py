"""KittyCode public package surface."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

__version__ = "0.2.0"

from .config import Config
from .llm import LLM
from .runtime.agent import Agent
from .tools import ALL_TOOLS


def _register_legacy_module(alias: str, target: str) -> None:
    module = sys.modules.setdefault(
        f"{__name__}.{alias}",
        importlib.import_module(target, __name__),
    )
    setattr(sys.modules[__name__], alias, module)


class _MainModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package_name: str, module_path: Path):
        self.package_name = package_name
        self.module_name = f"{package_name}.__main__"
        self.module_path = module_path

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.module_name:
            return None
        return importlib.util.spec_from_file_location(fullname, self.module_path)


def _register_module_entrypoint() -> None:
    module_path = Path(__file__).with_name("main.py")
    for finder in sys.meta_path:
        if isinstance(finder, _MainModuleFinder) and finder.package_name == __name__:
            return
    sys.meta_path.insert(0, _MainModuleFinder(__name__, module_path))


for _alias, _target in (
    ("agent", ".runtime.agent"),
    ("context", ".runtime.context"),
    ("interrupts", ".runtime.interrupts"),
    ("session", ".runtime.session"),
    ("logging_utils", ".runtime.logging"),
):
    _register_legacy_module(_alias, _target)


_register_module_entrypoint()


__all__ = ["Agent", "Config", "LLM", "ALL_TOOLS", "__version__"]