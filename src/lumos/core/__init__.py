"""Core building blocks of LUMOS: parameters, ports, elements and the registry."""

import importlib
from typing import Any

#: Modules of the core, imported on first access (``lumos.core.context``...).
__all__ = [
    "action",
    "clock",
    "context",
    "driver",
    "element",
    "graph",
    "group",
    "journal",
    "param",
    "port",
    "priors",
    "registry",
    "sequence",
    "signal",
]


def __getattr__(name: str) -> Any:
    """Import the modules of the core on first access."""
    if name in __all__:
        return importlib.import_module(f"lumos.core.{name}")
    raise AttributeError(f"module 'lumos.core' has no attribute {name!r}")
