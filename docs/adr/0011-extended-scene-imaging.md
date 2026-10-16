# ADR-0011: Scenes and the imaging of extended scenes

- **Status**: accepted
- **Date**: 2026-10-17
- **Plan step**: P8.1 and P8.5 (REQ-SCN-1, REQ-SCN-6)

## Context

Scenes mix point sources and diffuse objects in a frame (celestial, terrestrial, laboratory). Diffuse objects
are incoherent: their image is the sum of the images of their points. REQ-SCN-6 asks for three imaging
strategies (convolution, field-dependent PSFs, explicit decomposition), chosen automatically, forcible, and
logged; the convolution must be refused for systems that are not shift-invariant.

## Decision

**Scenes.** A `Scene` is a registered group type (`.lumos` group entries carry `"group_type"`) whose child
`combiner` holds the frame, the pointing or site, the distance (laboratory) and the angular grid of the scene.
Components are connected by the type of their output: `SourceList` to the `sources` output, `RadianceField`
to the `radiance` output, resampled onto the scene grid with flux conservation. Laboratory maps keep their
radiance and get angular pixels `dx / d` (radiance invariance); the curvature of close objects is neglected,
explicit fields (lasers, lamps) cover that case.

**Imaging.** Instruments stay models of objects at infinity: the entrance of an imaging system is an input
accepting a `SourceList`. `image_scene(scene, ImagingSystem(ctx, entrance=..., exit=...), strategy=...)`:

- `"decomposition"`: every pixel becomes a point source of flux `L ΔΩ`; all are computed at once as modes
  of an `IncoherentBundle` (batches of 1024). The element `SceneToSources` does the same inside a graph.
- `"isoplanatic"`: the on-axis PSF per unit flux density is computed by the system; the plate scale is
  measured from the windowed centroids of two off-axis PSFs; the scene is resampled onto an odd-sized
  angular grid whose pixels map onto the detector pixels and convolved with FFTs (`scipy.signal.fftconvolve`).
- `"anisoplanatic"`: PSFs at a grid of field points (snapped to whole detector pixels), recentred, combined
  with bilinear weights (weighted convolutions).
- `"auto"`: probe a point at a quarter of the field: same total power within 1 % and, shifted back by whole
  pixels, the same PSF within 1 % in L1 norm over the common pixels → isoplanatic; otherwise decomposition.
  Forcing `"isoplanatic"` on a system failing the probe raises an error. The choice and its reason are logged
  on `lumos.scenes` and returned in `ImageResult.strategy` and `.reason`.

The image is an `ImageResult` (power per pixel and channel); `to_wavecube()` gives a `Wavecube` of the same
intensities for detectors (its phase is meaningless, it is not to be propagated further).

## Alternatives considered

- A dedicated imaging element holding a reference to the downstream system: it would break the purity of
  transfers (ADR-0002) and the acyclic graph.
- Deciding the strategy from the element types (a coronagraph in the chain → decomposition): forbidden by
  principle 2 (no tests on class names); the numerical probe works for any system, including user elements.
- An analytic plate scale from the focal length: unknown for general systems (relay optics, angular output
  grids); measuring it costs two extra PSFs.

## Consequences

- Extended scenes behind coronagraphs and interferometers cost one mode per non-dark scene pixel; scene grids
  must stay coarse enough (or use `SceneToSources(threshold=...)`).
- The probe can be fooled by a system invariant at the probed point only; users can force a strategy.
- Terrestrial scenes (P16) will reuse the frame mechanism; time-dependent components already work through the
  clock of the context.
