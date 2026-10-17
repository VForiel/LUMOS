# Example conversations

These conversations show how an assistant connected to LUMOS works. The tool calls are those it makes, with
their arguments; the answers of LUMOS are abridged. Your assistant may take other paths to the same
result: what matters is that every number comes from a computation of LUMOS, with its unit.

## What contrast does this coronagraph reach at 3 λ/D?

> **You:** Load the vortex coronagraph preset. What raw contrast does it reach at 3 λ/D?

The assistant loads the preset and reads it:

- `load_preset(name="coronagraph")`: an 8 m telescope observing a star and a companion 10⁻⁴ fainter at
  6 λ/D, a charge-2 vortex coronagraph with a Lyot stop of 0.95 D, and a lens of 100 m focal length forming
  the image on 81 × 81 pixels of 5 µm, at 1.6 µm.
- `describe_context()` gives the values with their units.

It works out the scale of the image: λ/D = 1.6 µm / 8 m = 0.2 µrad (41 mas), that is
f λ/D = 20 µm or 4 pixels in the focal plane, so 3 λ/D lies 12 pixels from the star.

To measure the raw contrast, it switches the companion off, then compares the light left at 3 λ/D with the
peak of the star seen without the coronagraph:

- `set_param(path="planet.flux_density", value="0 Jy")`
- `inspect_port(port="image.out")`: the coronagraphic image of the star alone, with its figure;
- `disconnect(source="telescope.out", target="coronagraph.in")`,
  `disconnect(source="coronagraph.out", target="lens.in")`,
  `connect(source="telescope.out", target="lens.in")`, then `inspect_port(port="image.out")`: the
  point-spread function without coronagraph, whose maximum is the reference peak.

> **Assistant:** At 3 λ/D (124 mas, 12 pixels from the star) the coronagraphic image of the star is several
> orders of magnitude below the peak of the star without coronagraph. This is expected: a charge-2 vortex
> cancels the light of an on-axis star for a circular unobstructed pupil (Mawet et al. 2005), and what remains
> comes from the sampling of the simulation. With a central obstruction, spiders or wavefront errors the
> contrast would degrade; I can add them if you want.

## Build me a bench to measure the MTF of this lens

> **You:** Build me a bench to measure the MTF of a 50 mm f/4 lens at 550 nm with a slanted edge.

- `search_docs(query="slanted edge MTF")` and `list_element_types(query="edge")` find the slanted-edge
  target and the processing element `lumos.processing.MTFMeasurement` (ISO 12233).
- `get_element_schema(type_id="lumos.processing.MTFMeasurement")` gives its settings (oversampling...).
- `create_context(name="MTF bench")`, then `add_element(...)` for the target, the lens (focal length
  `"50 mm"`, aperture `"12.5 mm"`) and a camera with its pixel pitch, and `connect(...)` from one to the next.
- `build_pipeline(source="camera.out", steps=[{"type": "lumos.processing.MTFMeasurement", "name": "mtf"}])`
- `validate_context()` reports no issue; `run(outputs=["mtf.out"])`, then
  `get_result(job=..., port="mtf.out", figure=true)` returns the MTF curve with its figure.

> **Assistant:** The bench is ready: edge target → lens → camera → MTF measurement. The measured MTF follows
> the diffraction limit of an f/4 aperture, with a cut-off at 1/(λN) = 455 cycles/mm; the camera samples up
> to its Nyquist frequency. Do you want the Python script (`export_python`) or the `.lumos` file
> (`export_lumos`)?

## Why is my simulation under-sampled?

> **You:** My PSF looks blocky and aliased. Why?

- `describe_context()` shows the elements and their grids.
- `validate_context()` reports no structural problem.
- `explain_sampling(element="image")` tells how each propagation is computed: the method (chosen by the
  planner or forced by the element), the reason, and the input and output grids.
- `inspect_port(port="image.out")` shows the image with its pixel scale.

> **Assistant:** The output grid of `image` has pixels of 5 µm while f λ/D is 20 µm: 4 pixels per λ/D, so the
> core of the PSF is sampled, but the camera behind it bins them 4 × 4, which leaves 1 pixel per λ/D, below
> the Nyquist limit of 2. Either reduce the binning of the camera (`set_param(path="camera.binning",
> value=1)`) or use a smaller pixel pitch. After the change, `explain_sampling` and the new image confirm the
> sampling.

## Can this instrument detect the companion?

> **You:** Can we detect a planet 10⁻⁵ fainter than its star at 6 λ/D with this set-up?

The prompt `estimate_detection` guides this workflow: `resolve_object` for the properties of a real target,
the instrument from a preset, `run`, then `evaluate_detection(port=..., peak=..., resolution=...)` for the
5σ contrast curve of the residual image (with the small-sample correction of Mawet et al. 2014), compared with
the contrast of the planet at its separation.
