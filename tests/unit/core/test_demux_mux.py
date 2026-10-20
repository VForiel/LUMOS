"""Tests for the Demux and Mux elements (step P15.7, ADR-0023 §9).

Specification: docs/specs/pistes_evolution.md §1.1 (decisions of 20/10/2026).

- ``Demux(N)`` splits a bus of N lanes into N ports of one lane each (a one-lane bus keeping the name and
  the position of the lane); ``Mux(N)`` puts N lanes side by side in a bus, in the order of its ports,
  without summing or interfering them; they only slice and concatenate.
- An unconnected input of a ``Mux`` is an empty lane (zero amplitude, same kind of signal as the others),
  reported as a warning by the validation.
- Both are pure elements usable from Python, the service (and so the MCP server) and the web palette; their
  ports depend on ``n_lanes``.

Invariants (exact, no reference value needed): ``Mux ∘ Demux`` is the identity; the power of the output of a
``Mux`` is the sum of the powers of its inputs.
"""

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from lumos import Context
from lumos.core.element import Element
from lumos.elements.buses import Demux, Mux
from lumos.fields import Bus, Grid, ModalField, Wavefront

LAM = [1.55, 1.6] * u.um


def lane(values, sources=None):
    a = np.asarray(values, dtype=complex)
    return ModalField(np.tile(a, (2, 1)), LAM, sources=sources)


def amplitudes(bus):
    return np.asarray(bus.modal.amplitudes)


def test_ports_follow_the_number_of_lanes():
    demux = Demux(n_lanes=3)
    assert [n for n, p in demux.instance_ports().items() if p.direction == "out"] == ["out0", "out1", "out2"]
    mux = Mux(n_lanes=4)
    assert [n for n, p in mux.instance_ports().items() if p.direction == "in"] == ["in0", "in1", "in2", "in3"]
    assert mux.port("out").port.bus
    assert demux.port("in").port.bus
    mux.n_lanes = 2
    assert [n for n, p in mux.instance_ports().items() if p.direction == "in"] == ["in0", "in1"]


def test_the_schema_describes_the_repeated_ports():
    ports = Mux.schema()["x-lumos-ports"]
    assert ports["inputs"][0]["name"] == "in{i}"
    assert ports["inputs"][0]["count"] == "n_lanes"
    assert ports["outputs"][0]["name"] == "out"
    assert ports["outputs"][0]["bus"] is True


def test_demux_gives_one_lane_per_port():
    bus = Bus(
        [lane([1.0]), lane([2.0, 3.0]), lane([4.0])],
        names=["a", "b", "c"],
        positions=[[0, 0], [1, 0], [2, 0]] * u.m,
    )
    outputs = Demux(n_lanes=3).apply(**{"in": bus})
    for k, name in enumerate(("a", "b", "c")):
        part = outputs[f"out{k}"]
        assert isinstance(part, Bus)
        assert part.names == (name,)
        np.testing.assert_array_equal(np.asarray(part[0].amplitudes), np.asarray(bus[k].amplitudes))
        np.testing.assert_allclose(part.positions.to_value(u.m), [[k, 0]])


def test_demux_needs_as_many_lanes_as_ports():
    with pytest.raises(ValueError, match="lanes"):
        Demux(n_lanes=2).apply(**{"in": Bus([lane([1.0]), lane([1.0]), lane([1.0])])})


def test_mux_stacks_lanes_without_mixing_them():
    out = Mux(n_lanes=3).apply(in0=lane([1.0]), in1=lane([2.0, 3.0]), in2=lane([1j]))["out"]
    assert isinstance(out, Bus)
    assert out.packed
    assert out.mode_counts == (1, 2, 1)
    assert out.names == ("T1", "T2", "T3")
    np.testing.assert_array_equal(amplitudes(out)[0], [1.0, 2.0, 3.0, 1j])


def test_mux_keeps_the_names_of_one_lane_buses_and_flattens_buses():
    named = Bus([lane([1.0])], names=["UT1"], positions=[[5, 6]] * u.m)
    pair = Bus([lane([2.0]), lane([3.0])], names=["UT2", "UT3"])
    out = Mux(n_lanes=2).apply(in0=named, in1=pair)["out"]
    assert out.names == ("UT1", "UT2", "UT3")
    np.testing.assert_allclose(out.positions.to_value(u.m)[0], [5, 6])
    assert np.all(np.isnan(out.positions.to_value(u.m)[1:]))


def test_mux_of_lanes_of_different_sources_is_a_list_bus():
    out = Mux(n_lanes=2).apply(in0=lane([1.0], sources="a"), in1=lane([1.0], sources="b"))["out"]
    assert len(out) == 2
    assert not out.packed


def test_an_unconnected_mux_input_is_an_empty_lane():
    out = Mux(n_lanes=3).apply(in0=lane([1.0, 2.0]), in1=None, in2=lane([3.0]))["out"]
    assert out.mode_counts == (2, 1, 1)
    np.testing.assert_array_equal(np.asarray(out[1].amplitudes), 0.0)
    with pytest.raises(ValueError, match="no input"):
        Mux(n_lanes=2).apply(in0=None, in1=None)


@settings(max_examples=25, deadline=None)
@given(counts=st.lists(st.integers(1, 3), min_size=1, max_size=5), seed=st.integers(0, 2**32 - 1))
def test_mux_after_demux_is_the_identity(counts, seed):
    rng = np.random.default_rng(seed)
    bus = Bus([lane(rng.normal(size=n) + 1j * rng.normal(size=n)) for n in counts])
    parts = Demux(n_lanes=len(counts)).apply(**{"in": bus})
    back = Mux(n_lanes=len(counts)).apply(**{f"in{k}": parts[f"out{k}"] for k in range(len(counts))})["out"]
    assert back.names == bus.names
    assert back.mode_counts == bus.mode_counts
    np.testing.assert_array_equal(amplitudes(back), amplitudes(bus))
    total = sum(float(np.sum(lane_.mode_power.value)) for lane_ in bus)
    assert float(np.sum(back.modal.mode_power.value)) == pytest.approx(total, rel=1e-12)


def test_free_space_lanes_round_trip():
    grid = Grid(8, dx=1 * u.mm)
    beams = [Wavefront(np.full((8, 8), v, dtype=complex), grid, 1 * u.um) for v in (1.0, 2.0)]
    bus = Bus(beams)
    parts = Demux(n_lanes=2).apply(**{"in": bus})
    back = Mux(n_lanes=2).apply(in0=parts["out0"], in1=parts["out1"])["out"]
    assert back.packed
    np.testing.assert_array_equal(np.asarray(back.field), np.asarray(bus.field))


def build_context(*, connect_all=True):
    from lumos.elements.detectors import PowerMeter
    from lumos.elements.sources import PointSource
    from lumos.elements.telescopes import TelescopeArray

    ctx = Context()
    ctx.observation.band = (3.5 * u.um, 4.0 * u.um)
    star = ctx.add(PointSource(name="star", flux_density=1 * u.Jy))
    array = ctx.add(TelescopeArray(name="array", frame="sky", positions=[[0, 0], [20, 0], [0, 30]] * u.m))
    demux = ctx.add(Demux(name="demux", n_lanes=3))
    mux = ctx.add(Mux(name="mux", n_lanes=3))
    meter = ctx.add(PowerMeter(name="meter"))
    ctx.connect(star.port("out"), array.port("in"))
    ctx.connect(array.port("out"), demux.port("in"))
    for k in range(3):
        if connect_all or k != 1:
            ctx.connect(demux.port(f"out{k}"), mux.port(f"in{k}"))
    ctx.connect(mux.port("out"), meter.port("in"))
    return ctx


def test_demux_and_mux_in_a_context_and_a_document():
    import lumos.io

    ctx = build_context()
    direct = ctx.run(["array.out", "meter.out"])
    np.testing.assert_allclose(
        direct["meter.out"].data.to_value(u.W),
        [float(np.sum(np.abs(np.asarray(lane_.amplitudes)) ** 2)) for lane_ in direct["array.out"]],
        rtol=1e-6,
    )
    loaded = lumos.io.from_document(lumos.io.to_document(ctx))
    again = loaded.run(["meter.out"])["meter.out"].data.to_value(u.W)
    np.testing.assert_allclose(again, direct["meter.out"].data.to_value(u.W), rtol=1e-12)


def test_an_unconnected_mux_input_is_reported_as_a_warning():
    import lumos.io
    from lumos.service import Service

    ctx = build_context(connect_all=False)
    readings = ctx.run(["meter.out"])["meter.out"].data.value
    assert readings[1] == 0
    service = Service()
    session = service.create_session().id
    summary = service.import_lumos(session, lumos.io.to_document(ctx))
    issues = service.validate_context(session, summary.id)
    assert [i.severity for i in issues if "mux.in1" in i.message] == ["warning"]


def test_the_elements_are_registered():
    from lumos.core.registry import registry

    assert registry.get("lumos.buses.Demux") is Demux
    assert registry.get("lumos.buses.Mux") is Mux
    assert isinstance(Element.from_lumos(Mux(n_lanes=5).to_lumos()), Mux)


def test_removing_lanes_drops_the_connections_of_the_removed_ports():
    ctx = build_context()
    ctx.elements["mux"].n_lanes = 2
    targets = [c.target.name for c in ctx._graph.connections if c.target.owner is ctx.elements["mux"]]
    assert targets == ["in0", "in1"]
    assert len(ctx.run(["meter.out"])["meter.out"].data.value) == 2


# --- Number of lanes drawn for each bus port (the editor draws one pin per lane, grouped by a brace) -------


def test_bus_ports_give_the_number_of_lanes_to_draw():
    from lumos.elements.photonics import MMI, DirectionalCoupler, TransferMatrix, Waveguide, YJunction
    from lumos.elements.telescopes import TelescopeArray

    assert Demux(n_lanes=3).port_lanes("in") == 3
    assert Mux(n_lanes=4).port_lanes("out") == 4
    assert MMI(n_ports=4).port_lanes("in") == 4
    assert MMI(n_ports=4).port_lanes("out") == 4
    assert DirectionalCoupler().port_lanes("in") == 2
    assert YJunction(mode="split").port_lanes("in") == 1
    assert YJunction(mode="split").port_lanes("out") == 2
    matrix = TransferMatrix(amplitudes=np.ones((3, 2)) * 0.4, phases=np.zeros((3, 2)) * u.rad)
    assert (matrix.port_lanes("in"), matrix.port_lanes("out")) == (2, 3)
    assert TelescopeArray(positions=[[0, 0], [1, 0], [2, 0]] * u.m).port_lanes("out") == 3
    assert Waveguide().port_lanes("in") is None  # any number of lanes
    assert Demux(n_lanes=3).port_lanes("out0") is None  # not a bus port


def test_the_service_lists_the_lanes_of_the_bus_ports():
    from lumos.service import Service

    service = Service()
    session = service.create_session().id
    ctx = service.create_context(session).id
    service.add_element(session, ctx, "lumos.buses.Mux", name="mux", params={"n_lanes": 3})
    summary = next(e for e in service.describe_context(session, ctx).elements if e.id == "mux")
    assert summary.inputs == ["in0", "in1", "in2"]
    assert summary.lanes == {"out": 3}
