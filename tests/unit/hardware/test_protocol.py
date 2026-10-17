"""Tests for the LumosLabs protocol (step P12.3, REQ-HW-3, ADR-0014).

Requirement: a documented ASCII protocol (commands, replies, errors) shared by the whole LumosLabs range. The
expected strings are those of the specification in ``docs/hardware/lumoslabs_protocol.md``.
"""

import numpy as np
import pytest
from astropy import units as u

from lumos.core.driver import DriverStopped, LimitError
from lumos.hardware import protocol
from lumos.hardware.protocol import DeviceError, ProtocolError


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (12.5 * u.mm, "12.5[mm]"),
        (-3 * u.deg, "-3[deg]"),
        (2.5e-3 * u.W, "0.0025[W]"),
        (1e-12 * u.W, "1e-12[W]"),
        (4 * u.mm / u.s, "4[mm/s]"),
        (3, "3"),
        (0.25, "0.25"),
        ("ON", "ON"),
        (True, "1"),
    ],
)
def test_format_tokens(value, text):
    assert protocol.format_token(value) == text


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("12.5[mm]", 12.5 * u.mm),
        ("-3[deg]", -3 * u.deg),
        ("1e-12[W]", 1e-12 * u.W),
        ("4[mm/s]", 4 * u.mm / u.s),
        ("3", 3),
        ("0.25", 0.25),
        ("ON", "ON"),
    ],
)
def test_parse_tokens(text, value):
    parsed = protocol.parse_token(text)
    if isinstance(value, u.Quantity):
        assert parsed.unit == value.unit
        assert parsed.value == pytest.approx(value.value)
    else:
        assert parsed == value
        assert type(parsed) is type(value)


def test_commands_round_trip():
    line = protocol.format_command("MOVE", 12.5 * u.mm)
    assert line == "MOVE 12.5[mm]"
    assert protocol.parse_command(line) == ("MOVE", [12.5 * u.mm])
    assert protocol.parse_command("*IDN?") == ("*IDN?", [])
    assert protocol.parse_command("volt 2 1.5[V]") == ("VOLT", [2, 1.5 * u.V])  # verbs are case-insensitive
    with pytest.raises(ProtocolError):
        protocol.parse_command("")
    with pytest.raises(ProtocolError):
        protocol.parse_command("MOVE 12.5[parsec/fortnight/zz]")


def test_replies():
    assert protocol.parse_reply("OK") == []
    assert protocol.parse_reply("OK 12.5[mm]") == [12.5 * u.mm]
    assert protocol.parse_reply("OK 0 100[mm]") == [0, 100 * u.mm]
    assert protocol.format_reply(12.5 * u.mm, 1) == "OK 12.5[mm] 1"
    with pytest.raises(ProtocolError):
        protocol.parse_reply("HELLO")


def test_errors_map_to_driver_exceptions():
    assert (
        protocol.format_error(200, "position 120 mm outside [0, 100] mm")
        == "ERR 200 position 120 mm outside [0, 100] mm"
    )
    with pytest.raises(LimitError, match="200"):
        protocol.parse_reply("ERR 200 out of range")
    with pytest.raises(DriverStopped):
        protocol.parse_reply("ERR 400 emergency stop latched")
    with pytest.raises(DeviceError) as caught:
        protocol.parse_reply("ERR 100 unknown command FOO")
    assert caught.value.code == 100
    assert set(protocol.ERRORS) >= {100, 101, 102, 200, 300, 301, 400, 500, 501}


def test_identification():
    identity = protocol.parse_identity("LUMOSLABS,LL-LS100,SN0042,1.0")
    assert identity.vendor == "LUMOSLABS"
    assert identity.model == "LL-LS100"
    assert identity.serial == "SN0042"
    assert identity.firmware == "1.0"
    assert protocol.format_identity(identity) == "LUMOSLABS,LL-LS100,SN0042,1.0"


def test_arrays_are_sent_in_base64():
    image = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    tokens = protocol.encode_array(image)
    restored = protocol.decode_array(*tokens)
    np.testing.assert_array_equal(restored, image)
    assert restored.dtype == np.float32
    counts = np.array([[1, 65535], [0, 7]], dtype=np.uint16)
    np.testing.assert_array_equal(protocol.decode_array(*protocol.encode_array(counts)), counts)


def test_the_protocol_is_documented():
    from pathlib import Path

    page = Path(__file__).resolve().parents[3] / "docs" / "hardware" / "lumoslabs_protocol.md"
    text = page.read_text(encoding="utf-8")
    for code in protocol.ERRORS:
        assert f"`{code}`" in text
    for verb in (
        "*IDN?",
        "*RST",
        "STATUS?",
        "BUSY?",
        "STOP",
        "LIMITS?",
        "SIM:POWER",
        "SIM:IMAGE",
        "SIM:FAULT",
    ):
        assert f"`{verb}" in text
