# Studies: sweeps and Monte-Carlo runs

A study runs the context many times: a **sweep** over ranges of some parameters, or a **Monte-Carlo run**
drawing its free parameters, and its noise, again and again. Open it with **Study** in the toolbar; the
**Monte-Carlo tolerancing** preset opens with one ready.

## Randomness and the seed

Every random process of a context (photon and read noise of the detectors, turbulence screens, Monte-Carlo
draws) derives from **one seed**, shown in the inspector of the context (select nothing) under *Randomness*.
Type a seed to reproduce a result, or press the dice to draw a new one. The seed is saved in the `.lumos` file:
the same context with the same seed gives the same results, in the interface, in Python and through an AI
assistant.

## Free parameters

The dice button next to a parameter of the inspector **frees** it: give it a prior, uniform, log-uniform or
normal (Gaussian), with values in the unit of the parameter (or with their own unit, `2 mm`). A free
parameter shows its prior under its value (`∼ N(200 mm, 2 mm)`) and its block carries a small die. *Fix again*
removes the prior. Free parameters are also those that calibrations and inferences fit.

## Sweeps

Choose **Sweep**, then for each parameter its range: from, to, number of values, and *Log* for a geometric
spacing (1, 10, 100...). With several parameters, *every combination* runs the grid of their values, *in
pairs* runs their i-th values together. The dialog counts the points before you start.

By default every point of a sweep has the **same noise** (that of the seed): the points differ by their
parameters only, so that small effects show. Tick *Independent noise per point* to draw a new noise at each
point.

## Monte-Carlo runs

Choose **Monte-Carlo** and the number of draws. Each draw takes the free parameters from their priors and,
by default, its own noise realization: the spread of the results combines the tolerances and the noise.
Without free parameters, the draws repeat the run with independent noise (a noise study).

## Results

The **Study** tab of the bottom panel follows the progress, then shows:

- a **curve** for a sweep (one curve per value of the other parameters), or a **histogram** for a Monte-Carlo
  run with the mean (dashed line) and the standard deviation of the values;
- the **value plotted**: the value of a scalar output (a power meter), or a statistic of an image or a field
  (maximum, mean, minimum, sum);
- a **table** of every point (parameters, noise realization, outputs), downloaded as **CSV**.

The instance limits the number of points of a study (1000 on your machine by default, fewer on a public
instance) and the time of a run.

## The same in Python and for assistants

```python
points = ctx.sweep({"focus.distance": np.linspace(190, 210, 11) * u.mm}, outputs=["camera.out"])
ctx["lens"].set_free("focal_length", True, Normal(200 * u.mm, 2 * u.mm))
draws = ctx.sample(100, outputs=["camera.out"])
```

AI assistants use the MCP tools `set_seed`, `set_free_parameter`, `run_sweep`, `run_monte_carlo` and
`get_study` (see the [reference](../assistant/reference.md)); the HTTP API has `POST /api/contexts/{id}/studies`
and `GET /api/jobs/{job}/study` (see [HTTP API](api.md)).
