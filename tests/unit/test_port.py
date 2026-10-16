"""Tests for lumos.core.signal and lumos.core.port (step P1.3).

Requirements: REQ-SIG-11 (typed ports; incompatible connections refused) and the concepts of §4 of the specs
(connections go from an output port to an input port).
"""

import pytest

from lumos.core.port import (
    Input,
    Output,
    Port,
    PortCompatibilityError,
    check_compatible,
    compatible,
    ports_of,
)
from lumos.core.signal import DataSignal, OpticalSignal, Signal


class FakeWavefront(OpticalSignal):
    """Stand-in for a concrete optical signal (concrete signals arrive in phase 3)."""


class Splitter:
    in_ = Input(OpticalSignal, doc="Incoming beam")
    out1 = Output(OpticalSignal)
    out2 = Output(OpticalSignal, name="transmitted")


class Source:
    out = Output(FakeWavefront)


class Camera:
    in_ = Input(OpticalSignal, multiple=True)
    data = Output(DataSignal)


class Logger:
    in_ = Input(Signal)


# --- Signals -----------------------------------------------------------------------


def test_signal_hierarchy():
    assert issubclass(OpticalSignal, Signal)
    assert issubclass(DataSignal, Signal)
    assert not issubclass(DataSignal, OpticalSignal)


# --- Port declarations ---------------------------------------------------------------


def test_trailing_underscore_is_stripped_from_port_name():
    assert Splitter.in_.name == "in"
    assert Splitter.out1.name == "out1"


def test_explicit_port_name():
    assert Splitter.out2.name == "transmitted"


def test_port_attributes():
    port = Splitter.in_
    assert isinstance(port, Port)
    assert port.direction == "in"
    assert port.signal_type is OpticalSignal
    assert port.doc == "Incoming beam"
    assert not port.multiple
    assert Splitter.out1.direction == "out"
    assert Camera.in_.multiple


def test_signal_type_must_be_a_signal_class():
    with pytest.raises(TypeError):
        Input(int)
    with pytest.raises(TypeError):
        Output("Wavefront")


def test_ports_of_lists_ports_in_order_with_inheritance():
    class FancySplitter(Splitter):
        monitor = Output(DataSignal)

    assert list(ports_of(FancySplitter)) == ["in", "out1", "transmitted", "monitor"]
    assert ports_of(FancySplitter)["in"] is Splitter.in_


# --- Bound ports -------------------------------------------------------------------------


def test_instance_access_returns_a_bound_reference():
    splitter = Splitter()
    ref = splitter.in_
    assert ref.owner is splitter
    assert ref.port is Splitter.in_
    assert ref.name == "in"
    assert ref.direction == "in"
    assert ref.signal_type is OpticalSignal


def test_bound_references_compare_by_owner_and_port():
    a, b = Splitter(), Splitter()
    assert a.out1 == a.out1
    assert hash(a.out1) == hash(a.out1)
    assert a.out1 != b.out1
    assert a.out1 != a.out2


def test_ports_cannot_be_reassigned():
    with pytest.raises(AttributeError):
        Splitter().in_ = None


# --- Compatibility -------------------------------------------------------------------------


def test_output_of_subtype_feeds_input_of_supertype():
    assert compatible(Source().out, Splitter().in_)


def test_output_of_supertype_does_not_feed_input_of_subtype():
    class Picky:
        in_ = Input(FakeWavefront)

    assert not compatible(Splitter().out1, Picky().in_)


def test_any_signal_input_accepts_everything():
    assert compatible(Camera().data, Logger().in_)
    assert compatible(Source().out, Logger().in_)


def test_data_does_not_feed_optical_input():
    assert not compatible(Camera().data, Splitter().in_)


def test_direction_must_be_output_to_input():
    assert not compatible(Splitter().in_, Camera().in_)
    assert not compatible(Splitter().out1, Source().out)
    assert not compatible(Splitter().in_, Source().out)


def test_compatibility_works_on_unbound_ports():
    assert compatible(Source.out, Splitter.in_)


def test_check_compatible_explains_the_problem():
    with pytest.raises(PortCompatibilityError, match=r"data.*in"):
        check_compatible(Camera().data, Splitter().in_)
    assert issubclass(PortCompatibilityError, TypeError)


def test_check_compatible_accepts_valid_connection():
    check_compatible(Source().out, Splitter().in_)
