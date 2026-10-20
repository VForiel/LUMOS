# ADR-0023: Buses of lanes, implicit packing and explicit injection of telescope arrays

- **Status**: accepted
- **Date**: 2026-10-20
- **Plan step**: P15.6 (also P15.7, P15.8, P15.9)

## Context

A `ModalField` carried N *channels* on a single port, and its last axis mixed two notions: waveguides separated
in space (the outputs of a `TelescopeArray`, the guides of a chip) and the modes of one guide (the `modes` port of
a `Fiber`). A `TelescopeArray` returned such a field directly, so collecting the light in free space and injecting
it into single-mode guides were one element with an ideal coupling: no coupling loss (an Airy pupil couples at
most about 80 % of its power into a single-mode fibre, Shaklan & Roddier 1988, Appl. Opt. 27, 2334), no antenna
lobe of the fibre (off-axis sources coupled like on-axis ones, which distorts the transmission maps of nullers),
and nothing could be placed on one arm before the injection. The review of 19/10/2026 and its decisions of
20/10 (`docs/specs/pistes_evolution.md`, §1) settle the model; this record fixes its design.

## Decision

1. **A lane is one signal.** A guided lane is a `ModalField` of one guide, of shape `(sources, λ, modes)`
   (`modes = 1` for a single-mode guide); its mode axis never designates different guides any more. A
   free-space lane is a `Wavefront`, a `Wavecube` or an `IncoherentBundle`. `ModalField` counts are renamed
   accordingly: `n_sources` (incoherent inputs, was `n_modes`), `n_modes` (was `n_channels`), `mode_power`
   (was `channel_power`); `labels` name the modes.
2. **A bus** (`lumos.fields.Bus`) groups N lanes of the same signal class, separated in space (guides of a
   chip, arms of an interferometer). It behaves as a sequence (`len(bus)`, `bus[i]`, iteration) and carries
   per-lane metadata: names (`T1`…`Tn` by default) and, when known, the projected position of the telescope
   that feeds the lane (used by `FizeauCombiner`). Spectral components, incoherent components (sources) and
   the modes of one guide are never lanes: separating them is the job of a physical component (prism,
   dichroic, photonic lantern).
3. **Implicit packing.** A bus stores its lanes in a single array when they share their metadata, and as a
   list otherwise; the choice is made when the bus is built (and after every `Mux`), and the interface is the
   same.
   - guided: same sources, wavelengths, spectral weights, power unit and number of sources; the number of
     modes may differ between lanes. The array is `(sources, λ, Σ modes)` with the offset of each lane.
   - free space: same class and identical metadata (grid, sources, wavelengths, weights, polarisation, power
     unit: the auxiliary data of the signal as a JAX pytree) and the same array shape. The array is
     `(lanes, *lane_shape)`.
   `Bus.packing_problem(lanes)` says why lanes cannot be packed; a bus is a JAX pytree in both forms.
4. **Source lists are identified.** `ModalField` and `IncoherentBundle` carry `sources`, an identifier of the
   source list they describe (a digest of directions, wavelengths and fluxes, set by `Telescope` and
   `TelescopeArray`, kept by every element, `None` when unknown). Lanes interfere only if they describe the
   same sources in the same order, so packing requires equal identifiers and a recombiner (any component
   whose matrix mixes lanes) refuses lanes that cannot be packed, with the reason, instead of assuming it.
5. **Ports declare buses.** `Input(..., bus=True)` receives a bus (or a single lane) as it is; `Output(...,
   bus=True)` produces one. The signal type of a port stays the type of its lanes, so the compatibility of two
   ports does not change; the flag tells the editor to draw the port as a bus (P15.7) and is published in
   the JSON schema of the element (`x-lumos-ports[].bus`).
6. **A one-lane component applied to a bus acts on every lane.** `Element.apply` maps an element over the
   lanes when a bus reaches an input that does not declare `bus=True`, as N identical copies of the element:
   every bus input must have the same number of lanes (other inputs are shared), every output becomes a bus
   with the names and positions of the input bus, and a stochastic element draws an independent stream per
   lane (its random key gets the lane index), as N copies would. An element with a data output (camera,
   processing) cannot be mapped: its outputs would not be signals of one lane each; it asks the user to
   recombine or `Demux` the bus first.
7. **Elements migrated.**
   - Photonic components take and return guided buses. `channels` lists **lanes** (all when empty); a
     diagonal component (waveguide, phase shifter) acts on each selected lane; a mixing component applies
     its matrix to the vector of all the modes of its selected lanes, writes the outputs back on the same
     lanes when it has as many outputs as input modes, and otherwise puts single-mode output lanes where its
     first selected lane was.
   - `Fiber` takes one free-space lane; on a bus it is mapped (one fibre per lane) and returns a bus of guided
     lanes on `modes` and a bus of near fields on `out`.
   - `Photodiode` and `PowerMeter` read a lane as one value (the power of all its modes or pixels) and a bus
     as one value per lane (axis `channel`, setting `channel` of the power meter = a lane).
   - `TelescopeArray` (P15.6) returns a packed bus of single-mode guided lanes with the positions of its
     telescopes; P15.8 renames this ideal model `IdealInjectionArray` and makes `TelescopeArray` return a bus
     of `IncoherentBundle` in the pupil plane (one pupil per telescope, the geometric phase of its arm), the
     injection becoming an explicit `Fiber`.
8. **Format `.lumos` 1.3.** Connections are unchanged (a connection carries a whole bus). Version 1.3 records
   that `channels` designate lanes and, from P15.8, that `TelescopeArray` is the free-space array: reading a
   1.0–1.2 document converts it (a `TelescopeArray` becomes an `IdealInjectionArray`, which keeps the results
   identical). Documents in which `channels` addressed the modes of a multimode `Fiber` (no shipped preset
   did) cannot be converted automatically; the converter logs a warning naming the component.
9. **`Demux(N)` and `Mux(N)`** (P15.7) are pure elements of the core: `Demux` splits a bus into N one-lane
   buses (keeping names and positions), `Mux` stacks lanes side by side without mixing them (an unconnected
   input becomes an empty lane, zero amplitude, with a warning at validation). On packed buses they are
   slices and concatenations.

## Alternatives considered

- **Lanes addressed in connections** (`"to": "mmi.in[2]"`, option A of the review): one wire per guide
  without extra blocks, but connections to a lane, assembly of lanes and their validation in the engine,
  and a new syntax in the format. B + C gives the same drawing with two pure elements.
- **Keeping the channel axis and adding a separate mode axis to `ModalField`** (`(sources, λ, lanes,
  modes)`): every lane would need the same number of modes, which a bus of a multimode fibre and
  single-mode guides does not have, and free-space signals would still need another mechanism.
- **Always a list of lanes**: simpler, but a recombiner would copy the lanes at every component and a fibre
  per arm could not be vectorised (P15.3).
- **Checking the sources by their number only**: two telescopes pointed at different source lists of the
  same length would interfere silently.

## Consequences

- Code and tests that built a multi-channel `ModalField` for a chip now build a bus (`Bus.guided`); the
  photonic, fibre and detector tests are rewritten accordingly (`test(P15.6)!`).
- A power meter or photodiode on the `modes` output of a multimode fibre reads the total guided power instead
  of one value per mode.
- `Demux`, `Mux`, the bus rendering, `IdealInjectionArray`, the free-space `TelescopeArray` and
  `FizeauCombiner` build on this model (P15.7 to P15.9); P15.3 vectorises mapped elements on packed buses.
