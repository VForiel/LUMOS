# Studies: sweeps, Monte-Carlo runs and loops

A study runs the context many times: a **sweep** over ranges of some parameters, a **Monte-Carlo run**
drawing its free parameters, and its noise, again and again, or a **loop** that feeds outputs back to the
context step by step. Open it with **Study** in the toolbar; the **Monte-Carlo tolerancing** and **Adaptive
optics loop** presets open with one ready.

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

Numeric settings have the dice too (the number of pixels of a laser, the magnitude of a star): their prior
has no unit, and an integer setting takes the nearest integer of each value drawn or swept. An array
parameter (the Zernike coefficients of a phase plate, the transmissions of a filter wheel) frees one
component at a time: choose it in the editor of the prior; each free component shows its prior under the
array (`transmissions[2] ∼ N(0.5, 0.05)`).

## Sweeps

Choose **Sweep**, then for each parameter (numeric settings and components of arrays included) its range: from, to, number of values, and *Log* for a geometric
spacing (1, 10, 100...). With several parameters, *every combination* runs the grid of their values, *in
pairs* runs their i-th values together. The dialog counts the points before you start.

By default every point of a sweep has the **same noise** (that of the seed): the points differ by their
parameters only, so that small effects show. Tick *Independent noise per point* to draw a new noise at each
point.

The outputs of the study are those of the context by default. *Show the intermediate outputs* lists every
other output of every block, also those connected to another block: the field before the coronagraph, the
image before the processing.

## Monte-Carlo runs

Choose **Monte-Carlo** and the number of draws. Each draw takes the free parameters from their priors and,
by default, its own noise realization: the spread of the results combines the tolerances and the noise.
Without free parameters, the draws repeat the run with independent noise (a noise study).

## Feedback loops

A loop feeds an output back to a field at the next step: the correction computed by an adaptive optics
controller drives the commands of its deformable mirror. Select the block that computes the correction: the
*Feedback loops* section of the inspector lists its feedbacks and closes a new one (its output, the field it
drives, and the mode: *add* integrates, *set* replaces). A feedback is drawn as a dashed amber curve back to
the block it drives; select it to remove it.

Choose **Loop** in the study dialog, the number of steps and the period (`2 ms`): each step computes the
outputs, applies the feedbacks, then advances the clock by the period. The study tab plots the outputs
against time (the residual of the wavefront sensor falls as the loop closes). The loop keeps its final state,
the commands of the mirror for instance; **Undo** puts back the state before the loop. The *i* button of the
*Feedback loops* section recalls these steps in the inspector.

## Results

The **Study** tab of the bottom panel follows the progress, then shows:

- a **curve** for a sweep (one curve per value of the other parameters) or a loop (against time), or a
  **histogram** for a Monte-Carlo run with the mean (dashed line) and the standard deviation of the values;
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

AI assistants use the MCP tools `set_seed`, `set_free_parameter`, `run_sweep`, `run_monte_carlo`,
`add_feedback`, `run_loop` and `get_study` (see the [reference](../assistant/reference.md)); the HTTP API has
`POST /api/contexts/{id}/studies`, `POST /api/contexts/{id}/loops` and `GET /api/jobs/{job}/study` (see
[HTTP API](api.md)).
