"""Discovery of equipment and association with the elements of a context (step P12.6, REQ-HW-2).

:func:`discover` opens each candidate connection (TCP addresses, serial ports, VISA resources), asks ``*IDN?``
and keeps the devices that identify themselves as ``VENDOR,MODEL,SERIAL,FIRMWARE``; connections that fail or
answer something else are skipped. :func:`associate` attaches to every element declaring a ``lumoslabs_model``
(or ``device_model``) the driver of a discovered device of that model, and connects it. When several devices
or elements share a model, ``assignments`` maps element paths to serial numbers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lumos.core.driver import DriverError
from lumos.hardware import protocol
from lumos.hardware.transport import Transport, open_transport

__all__ = ["DeviceInfo", "associate", "discover", "identify"]

logger = logging.getLogger("lumos.hardware")


@dataclass(frozen=True)
class DeviceInfo:
    """A discovered device: identity and URL of its connection."""

    vendor: str
    model: str
    serial: str
    firmware: str
    url: str


def identify(transport: Transport, *, timeout: float = 0.5) -> DeviceInfo | None:
    """Ask ``*IDN?`` on ``transport`` (opened and closed here); ``None`` if nothing identifiable answers."""
    try:
        transport.open()
        values = protocol.parse_reply(transport.query("*IDN?", timeout))
        identity = protocol.parse_identity(str(values[0]))
    except (DriverError, OSError, ValueError, IndexError) as error:
        logger.debug("no device identified at %s: %s", transport.url, error)
        return None
    finally:
        transport.close()
    return DeviceInfo(identity.vendor, identity.model, identity.serial, identity.firmware, transport.url)


def _serial_ports(serial: bool | Iterable[str]) -> list[str]:
    if serial is False:
        return []
    if serial is True:
        from serial.tools import list_ports

        return [port.device for port in list_ports.comports()]
    return list(serial)


def _visa_resources(visa: bool | Iterable[str]) -> list[str]:
    if visa is False:
        return []
    if visa is True:
        import pyvisa

        return list(pyvisa.ResourceManager().list_resources())
    return list(visa)


def discover(
    *,
    tcp: Iterable[str] = (),
    serial: bool | Iterable[str] = False,
    visa: bool | Iterable[str] = False,
    timeout: float = 0.5,
) -> list[DeviceInfo]:
    """Return the devices that answer ``*IDN?``.

    Parameters
    ----------
    tcp
        Addresses ``host:port`` or ``tcp://host:port``.
    serial
        ``True`` to try every serial port of the computer, or a list of ports (pyserial URLs).
    visa
        ``True`` to try every VISA resource, or a list of resources.
    timeout
        Time allowed to each candidate to answer, in seconds.
    """
    urls = [address if "://" in address else f"tcp://{address}" for address in tcp]
    urls += [f"serial://{port}" for port in _serial_ports(serial)]
    urls += [f"visa://{resource}" for resource in _visa_resources(visa)]
    found = []
    for url in urls:
        device = identify(open_transport(url, timeout=timeout), timeout=timeout)
        if device is not None:
            logger.info("found %s %s at %s", device.model, device.serial, url)
            found.append(device)
    return found


def _model(element: Any) -> str | None:
    return getattr(element, "device_model", None) or getattr(element, "lumoslabs_model", None)


def associate(
    ctx: Any,
    devices: Iterable[DeviceInfo],
    *,
    assignments: Mapping[str, str] | None = None,
    connect: bool = True,
) -> dict[str, DeviceInfo]:
    """Attach the drivers of ``devices`` to the elements of ``ctx`` that represent them.

    Parameters
    ----------
    ctx
        The context.
    devices
        Discovered devices (:func:`discover`).
    assignments
        Serial number of the device of each element, by element path, when a model is shared.
    connect
        Connect the drivers.

    Returns
    -------
    dict
        The device of each associated element, by path.
    """
    from lumos.hardware.drivers import driver_for

    assignments = dict(assignments or {})
    available = list(devices)
    chosen: dict[str, DeviceInfo] = {}
    pending: dict[str, list[str]] = {}
    elements = {path: element for path, element in ctx.walk() if _model(element)}
    for path, element in elements.items():
        model = str(_model(element))
        if path in assignments:
            serial = assignments[path]
            matches = [d for d in available if d.model == model and d.serial == serial]
            if not matches:
                raise DriverError(f"{path}: no {model} with the serial number {serial!r} was discovered")
            chosen[path] = matches[0]
        elif element.driver is None:
            pending.setdefault(model, []).append(path)
    used = {device.url for device in chosen.values()}
    for model, paths in pending.items():
        candidates = [d for d in available if d.model == model and d.url not in used]
        if not candidates:
            raise DriverError(f"no {model} device was discovered for {', '.join(paths)}")
        if len(paths) > 1 or len(candidates) > 1:
            serials = ", ".join(d.serial for d in candidates)
            raise DriverError(
                f"{', '.join(paths)} could use the {model} devices {serials}: "
                f"give assignments={{path: serial}}"
            )
        chosen[paths[0]] = candidates[0]
        used.add(candidates[0].url)
    for path, device in chosen.items():
        driver = driver_for(device.model)(device.url)  # type: ignore[call-arg]
        elements[path].driver = driver
        if connect:
            driver.connect()
        logger.info("%s -> %s %s at %s", path, device.model, device.serial, device.url)
    return chosen
