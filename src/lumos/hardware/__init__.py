"""Operation of real and emulated equipment (phase 12, ADR-0014).

- :mod:`~lumos.hardware.transport`: TCP, serial, VISA and in-process transports of ASCII lines;
- :mod:`~lumos.hardware.protocol`: the LumosLabs ASCII protocol;
- :mod:`~lumos.hardware.emulators`: emulators of the LumosLabs devices and their TCP server;
- :mod:`~lumos.hardware.drivers`: drivers of the LumosLabs devices (and plugins);
- :mod:`~lumos.hardware.discovery`: discovery of devices and association with elements.
"""

import importlib
from typing import Any

#: Modules of the operation of equipment, imported on first access.
__all__ = ["discovery", "drivers", "emulators", "protocol", "transport"]


def __getattr__(name: str) -> Any:
    """Import the modules of :mod:`lumos.hardware` on first access."""
    if name in __all__:
        return importlib.import_module(f"lumos.hardware.{name}")
    raise AttributeError(f"module 'lumos.hardware' has no attribute {name!r}")
