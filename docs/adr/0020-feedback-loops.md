# ADR-0020: Feedback loops as time iterations of the context

- **Status**: accepted
- **Date**: 2026-10-18
- **Plan step**: P14.14

## Context

Adaptive optics, servo loops and iterative calibrations feed a measurement back to an actuator. REQ-CTX-2
keeps the graph of signals acyclic and asks that "physical loops (AO, servo, iterative calibration) are time
iterations driven by the context". Until now LUMOS had the pieces (time-dependent turbulence, a deformable
mirror with a `commands` state, an ideal modal correction inside `Atmosphere`) but no way to close a loop:
a review of the web interface (18/10/2026) asked for an adaptive optics preset with a real feedback loop.

## Decision

1. **Feedbacks** connect an output port to a field of an element: `ctx.feedback("ctrl.commands",
   "dm.commands", mode="add")`. They are not connections of the graph (which stays acyclic): they are
   applied *between* two iterations, so a loop always has at least one frame of delay, as a real loop. Two
   modes: `"set"` (the field takes the value of the port) and `"add"` (the value is added: an integrator whose
   gain is in the element that computes the correction). The value of the port is a `DataProduct` (or a
   quantity) of the unit and shape of the field.
2. **`ctx.loop(steps, period=..., outputs=...)`** runs the context `steps` times: compute the outputs and the
   sources of the feedbacks at time `t_k`, apply the feedbacks, advance the clock by `period`. It returns one
   `LoopStep(step, time, outputs)` per iteration; the states written by the feedbacks and the clock are kept
   (the loop has run), unless `restore=True`.
3. Feedbacks are saved in `.lumos` documents (top-level `"feedbacks": [{"from", "to", "mode"}]`), which makes
   the format **1.1**; LUMOS reads 1.0 and 1.1 documents, a 1.0 reader refuses 1.1 documents instead of
   dropping the loops silently. `ctx.to_python()` writes `ctx.feedback(...)`.
4. Adaptive optics elements: `WavefrontSensor`, an ideal Shack-Hartmann (mean gradient of the optical path
   in square sub-apertures, from the phase differences of neighbouring pixels, so that large wrapped phases
   are measured; plus the RMS residual of the optical path, valid below about λ/4), and `AOController`, an
   integrator `Δc = -g R s` whose reconstructor `R` is the regularised pseudo-inverse of the interaction
   matrix of a deformable mirror of the same geometry (computed analytically from the separable influence
   functions). A source element may check its feedback targets (`check_feedback`): the controller refuses a
   mirror of another geometry.
5. The service runs a loop as a job (`run_loop`, kind `loop`); its steps are read like the points of a study
   (`get_study`, parameters `time`), so that the web interface and assistants plot and export them the same
   way.

## Alternatives considered

- Cycles in the graph with a delay element: the planner, the caching of results and the validation assume an
  acyclic graph (ADR-0006); a cycle would need a fixed-point semantics everywhere for one use.
- Callbacks in Python between runs: not serialisable, not available from the web interface or to assistants.
- A single "AO system" element (sensor, controller and mirror together): simpler for one use, but hides the
  loop and does not extend to servo loops, calibrations or bench actuators.

## Consequences

- Loops are explicit, saved, undoable in the service, and available in Python, in the web interface and to
  assistants.
- Every loop has a delay of one period at least; shorter latencies are modelled by a smaller period.
- The controller repeats the geometry of its mirror (checked when the loop starts). Other controllers (modal,
  predictive) and sensors (pyramid, noise of the sensor) can follow the same pattern.
