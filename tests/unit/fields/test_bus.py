"""Tests for lumos.fields.Bus: lanes, implicit packing and source identifiers (step P15.6, ADR-0023).

Specification: docs/specs/pistes_evolution.md §1.2 (decisions of 20/10/2026).

- A bus groups N lanes of the same signal class and behaves as a sequence with lane names (``T1``... by
  default) and optional telescope positions.
- Lanes are packed in one array when they share their metadata (guided: sources, wavelengths, weights,
  power unit; free space: class, metadata and shape), and kept as a list otherwise; both forms behave the
  same and ``Bus.packing_problem`` says why lanes cannot be packed.
- Guided packing concatenates the modes: ``(sources, λ, Σ modes)``; lanes may have different numbers of
  modes.

The invariants (round trips, equality of the packed and list forms) are exact: no reference value is needed.
"""

import importlib.util
import json

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from lumos.core.signal import OpticalSignal
from lumos.fields import Bus, Grid, IncoherentBundle, ModalField, Wavefront

HAS_JAX = importlib.util.find_spec("jax") is not None
LAM = [1.55, 1.6] * u.um


def lane(values, *, sources=None, wavelengths=LAM, power_unit=u.W, labels=None):
    a = np.asarray(values, dtype=complex)
    if a.ndim == 1:
        a = np.tile(a, (len(wavelengths), 1))
    return ModalField(a, wavelengths, sources=sources, power_unit=power_unit, labels=labels)


def wavefront(value, n=8):
    return Wavefront(np.full((n, n), value, dtype=complex), Grid(n, dx=1 * u.mm), 1 * u.um)


def test_a_bus_is_an_optical_signal_and_a_sequence():
    bus = Bus([lane([1.0]), lane([2.0]), lane([3.0])])
    assert isinstance(bus, OpticalSignal)
    assert len(bus) == bus.n_lanes == 3
    assert bus.names == ("T1", "T2", "T3")
    assert bus.lane_type is ModalField
    assert [complex(np.asarray(lane_.amplitudes)[0, 0]) for lane_ in bus] == [1, 2, 3]
    assert isinstance(bus[1], ModalField)
    np.testing.assert_array_equal(np.asarray(bus[-1].amplitudes), np.asarray(lane([3.0]).amplitudes))


def test_lanes_must_be_of_one_signal_class():
    with pytest.raises(TypeError, match="lane"):
        Bus([lane([1.0]), wavefront(1.0)])
    with pytest.raises(ValueError, match="lane"):
        Bus([])


def test_names_and_positions_are_per_lane():
    bus = Bus([lane([1.0]), lane([1.0])], names=["UT1", "UT2"], positions=[[0, 0], [47, 0]] * u.m)
    assert bus.names == ("UT1", "UT2")
    np.testing.assert_allclose(bus.positions.to_value(u.m), [[0, 0], [47, 0]])
    assert Bus([lane([1.0])]).positions is None
    with pytest.raises(ValueError, match="name"):
        Bus([lane([1.0])], names=["a", "b"])
    with pytest.raises(ValueError, match="position"):
        Bus([lane([1.0])], positions=[[0, 0], [1, 1]] * u.m)


# --- Guided packing ------------------------------------------------------------------------------------


def test_guided_lanes_sharing_their_metadata_are_packed():
    bus = Bus([lane([1.0]), lane([2.0, 3.0]), lane([4.0])])
    assert bus.packed
    assert bus.mode_counts == (1, 2, 1)
    modal = bus.modal
    assert modal.shape == (2, 4)  # (λ, Σ modes)
    np.testing.assert_array_equal(np.asarray(modal.amplitudes)[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(np.asarray(bus[1].amplitudes), np.asarray(lane([2.0, 3.0]).amplitudes))


def test_guided_constructor_builds_single_mode_lanes_by_default():
    amplitudes = np.arange(6, dtype=complex).reshape(2, 3)
    bus = Bus.guided(amplitudes, LAM, names=["a", "b", "c"], sources="s1")
    assert bus.packed
    assert len(bus) == 3
    assert bus.mode_counts == (1, 1, 1)
    assert bus[2].sources == "s1"
    np.testing.assert_array_equal(np.asarray(bus[2].amplitudes)[:, 0], [2, 5])
    multimode = Bus.guided(amplitudes, LAM, modes=[2, 1])
    assert multimode.mode_counts == (2, 1)
    with pytest.raises(ValueError, match="modes"):
        Bus.guided(amplitudes, LAM, modes=[2, 2])


@pytest.mark.parametrize(
    ("other", "reason"),
    [
        (lane([1.0], sources="other"), "sources"),
        (lane([1.0], wavelengths=[1.5, 1.6] * u.um), "wavelength"),
        (lane([1.0], power_unit=u.photon / u.s), "power unit"),
        (ModalField(np.ones((3, 2, 1)), LAM), "sources"),
    ],
)
def test_guided_packing_is_refused_when_a_shared_property_differs(other, reason):
    lanes = [lane([1.0]), other]
    assert reason in Bus.packing_problem(lanes)
    bus = Bus(lanes)
    assert not bus.packed
    assert len(bus) == 2
    with pytest.raises(ValueError, match=reason):
        _ = bus.modal


@settings(max_examples=30, deadline=None)
@given(
    counts=st.lists(st.integers(1, 3), min_size=1, max_size=5),
    seed=st.integers(0, 2**32 - 1),
)
def test_packed_and_list_forms_hold_the_same_lanes(counts, seed):
    rng = np.random.default_rng(seed)
    lanes = [lane(rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))) for n in counts]
    packed = Bus(lanes)
    listed = Bus(lanes, pack=False)
    assert packed.packed
    assert not listed.packed
    for a, b, original in zip(packed, listed, lanes, strict=True):
        np.testing.assert_array_equal(np.asarray(a.amplitudes), np.asarray(original.amplitudes))
        np.testing.assert_array_equal(np.asarray(b.amplitudes), np.asarray(original.amplitudes))


# --- Free-space packing --------------------------------------------------------------------------------


def test_free_space_lanes_with_the_same_metadata_are_stacked():
    bus = Bus([wavefront(1.0), wavefront(2.0)])
    assert bus.packed
    assert bus.lane_type is Wavefront
    assert bus.field.shape == (2, 8, 8)
    assert bus[1].grid == wavefront(1.0).grid
    np.testing.assert_array_equal(np.asarray(bus[1].field), np.asarray(wavefront(2.0).field))


def test_free_space_packing_needs_the_same_grid():
    other = Wavefront(np.ones((8, 8), dtype=complex), Grid(8, dx=2 * u.mm), 1 * u.um)
    assert "grid" in Bus.packing_problem([wavefront(1.0), other])
    assert not Bus([wavefront(1.0), other]).packed


def test_bundles_with_different_sources_are_not_packed():
    template = wavefront(1.0)
    a = IncoherentBundle(np.ones((2, 8, 8)), template, sources="list-a")
    b = IncoherentBundle(np.ones((2, 8, 8)), template, sources="list-b")
    assert a.sources == "list-a"
    assert "sources" in Bus.packing_problem([a, b])
    assert Bus([a, IncoherentBundle(np.ones((2, 8, 8)), template, sources="list-a")]).packed


# --- Operations ----------------------------------------------------------------------------------------


def test_select_and_concatenate_are_inverse():
    bus = Bus([lane([1.0]), lane([2.0, 3.0]), lane([4.0])], positions=[[0, 0], [1, 0], [2, 0]] * u.m)
    parts = [bus.select([i]) for i in range(3)]
    assert [p.names for p in parts] == [("T1",), ("T2",), ("T3",)]
    back = Bus.concatenate(parts)
    assert back.packed
    assert back.names == bus.names
    assert back.mode_counts == bus.mode_counts
    np.testing.assert_array_equal(np.asarray(back.modal.amplitudes), np.asarray(bus.modal.amplitudes))
    np.testing.assert_allclose(back.positions.to_value(u.m), bus.positions.to_value(u.m))


def test_map_keeps_the_names_and_positions():
    bus = Bus([lane([1.0]), lane([2.0])], names=["a", "b"], positions=[[0, 0], [1, 0]] * u.m)
    doubled = bus.map(lambda signal: signal * 2)
    assert doubled.names == ("a", "b")
    np.testing.assert_allclose(doubled.positions.to_value(u.m), [[0, 0], [1, 0]])
    np.testing.assert_array_equal(np.asarray(doubled.modal.amplitudes)[0], [2, 4])


@pytest.mark.parametrize("pack", [True, False])
def test_json_round_trip(pack):
    bus = Bus(
        [lane([1.0, 2j], sources="s", labels=["LP01", "LP11"]), lane([0.5], sources="s")],
        names=["x", "y"],
        positions=[[1, 2], [3, 4]] * u.m,
        pack=pack,
    )
    back = Bus.from_json(json.loads(json.dumps(bus.to_json())))
    assert back.packed == bus.packed
    assert back.names == ("x", "y")
    assert back[0].labels == ("LP01", "LP11")
    assert back[1].sources == "s"
    np.testing.assert_allclose(back.positions.to_value(u.m), [[1, 2], [3, 4]])
    for a, b in zip(back, bus, strict=True):
        np.testing.assert_array_equal(np.asarray(a.amplitudes), np.asarray(b.amplitudes))


def test_free_space_json_round_trip():
    bus = Bus([wavefront(1.0), wavefront(2j)])
    back = Bus.from_json(json.loads(json.dumps(bus.to_json())))
    assert back.packed
    assert back.lane_type is Wavefront
    np.testing.assert_array_equal(np.asarray(back.field), np.asarray(bus.field))


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
@pytest.mark.parametrize("pack", [True, False])
def test_a_bus_is_a_jax_pytree(pack):
    import jax

    bus = Bus([lane([1.0]), lane([2.0, 3.0])], names=["a", "b"], pack=pack).to_backend("jax")
    leaves, tree = jax.tree_util.tree_flatten(bus)
    back = jax.tree_util.tree_unflatten(tree, leaves)
    assert back.names == ("a", "b")
    assert back.packed == pack
    doubled = jax.jit(lambda b: b.map(lambda s: s * 2))(bus)
    np.testing.assert_allclose(np.asarray(doubled[1].amplitudes)[0], [4, 6])
