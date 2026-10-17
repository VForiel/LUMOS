"""Drivers of equipment: the LumosLabs drivers and the registry of drivers by model (steps P12.5, P12.9).

:func:`driver_for` returns the driver class of a device model (``"LL-LS100"``, or a vendor model). Other
packages add drivers with :func:`register_driver` or through the ``lumos.drivers`` entry points
(:func:`load_plugins`), as described in the guide "Writing a driver".
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from lumos.core.driver import Driver
from lumos.hardware.drivers import lumoslabs

__all__ = ["DRIVERS", "PLUGIN_GROUP", "driver_for", "load_plugins", "lumoslabs", "register_driver"]

logger = logging.getLogger("lumos.hardware")

#: Entry point group of driver plugins: ``model = "package.module:DriverClass"``.
PLUGIN_GROUP = "lumos.drivers"

#: Driver classes by device model.
DRIVERS: dict[str, type[Driver]] = {}

_plugins_loaded = False


def register_driver(model: str, cls: type[Driver]) -> type[Driver]:
    """Register ``cls`` as the driver of the devices of ``model``."""
    if not (isinstance(cls, type) and issubclass(cls, Driver)):
        raise TypeError(f"a driver must be a subclass of lumos.core.driver.Driver, got {cls!r}")
    existing = DRIVERS.get(model)
    if existing is not None and existing is not cls:
        raise ValueError(f"model {model!r} already has the driver {existing.__qualname__}")
    DRIVERS[model] = cls
    return cls


def load_plugins(group: str = PLUGIN_GROUP) -> list[str]:
    """Register the drivers of the entry points ``group`` of installed packages; return their models."""
    global _plugins_loaded
    loaded = []
    for entry in entry_points(group=group):
        try:
            cls = entry.load()
            register_driver(entry.name, cls)
            loaded.append(entry.name)
        except Exception:  # a broken plugin must not prevent the others from loading
            logger.exception("cannot load the driver plugin %s", entry.value)
    _plugins_loaded = True
    return loaded


def driver_for(model: str) -> type[Driver]:
    """Return the driver class of ``model`` (plugins are loaded on the first unknown model)."""
    if model not in DRIVERS and not _plugins_loaded:
        load_plugins()
    try:
        return DRIVERS[model]
    except KeyError:
        raise LookupError(f"no driver for the model {model!r}; known: {', '.join(sorted(DRIVERS))}") from None


for _cls in (
    lumoslabs.LinearStageDriver,
    lumoslabs.RotationStageDriver,
    lumoslabs.PiezoDriver,
    lumoslabs.ShutterDriver,
    lumoslabs.FilterWheelDriver,
    lumoslabs.LaserDriver,
    lumoslabs.PowerMeterDriver,
    lumoslabs.CameraDriver,
    lumoslabs.PhaseControllerDriver,
):
    register_driver(_cls.model, _cls)
