# Contexts, graphs and time

A **context** holds the elements of a system, the connections between their ports, a clock and the settings
shared by all elements: operating mode, digital twin, array backend, observation band and random seed.

```python
from astropy import units as u
from lumos import Context

ctx = Context(seed=42)
source = ctx.add(Laser(name="laser", power=1 * u.mW))
source >> Lens(name="lens") >> Camera(name="cam")  # connects unique ports and adds the elements
image = ctx.run(["cam.out"])["cam.out"]
```

## Graph and execution

Elements are connected port to port with `ctx.connect(a.out, b.in_)` or `a >> b` when the ports are unique.
A connection is accepted only if the signal produced by the output is the type expected by the input (or a
subtype), if a single input is not already connected, and if it does not create a cycle: physical loops are
iterations in time, not cycles of signals.

`ctx.run(outputs)` computes only the elements needed by the requested outputs, in topological order. The
outputs of every element are cached; changing a parameter, a state or a connection invalidates that element
and everything downstream, so the next run recomputes only what changed. Elements are reached by name,
`ctx["lens"]`, and ports by path, `"cam.out"`.

## Groups

An `ElementGroup` gathers elements into one element, for instance a photonic chip or a planetary system. It
exposes some internal ports as its own:

```python
from lumos.core.group import ElementGroup

chip = ElementGroup(name="chip")
y, phi = chip.add(YSplitter(name="y"), PhaseShifter(name="phi"))
y.a >> phi
chip.expose("in", y.in_)
chip.expose("out", phi.out)
ctx.add(chip)
ctx["chip.phi"].phase = 0.3 * u.rad  # children are reached by dotted paths
```

Groups nest freely and are saved like elements.

## Time and actions

The clock has an absolute start date (`ctx.time`, an `astropy.time.Time`) and an elapsed time (`ctx.elapsed`).
It advances when the context waits or when an **action** completes. Actions are methods of elements declared
with a duration, fixed or computed from the element properties:

```python
from lumos.core.action import action


class Stage(Element):
    velocity = Param(unit=u.mm / u.s, default=2 * u.mm / u.s)
    position = State(unit=u.mm, default=0 * u.mm)

    @action(duration=lambda self, target: abs(target - self.position) / self.velocity)
    def move(self, target):
        self.position = target


stage = ctx.add(Stage(name="stage"))
stage.move(10 * u.mm)  # the clock advances by 5 s
with ctx.parallel():  # simultaneous actions: the clock advances by the longest one
    stage.move(0 * u.mm)
    ctx.wait(1 * u.s)
```

Elements whose transfer function depends on time set `time_dependent = True` and read `values.time`; they are
recomputed whenever the clock advances.

## Simulation and operation

In **simulation** (the default), an action updates the virtual state of the element, is written to the
journal and advances the clock by its predicted duration. In **operation**, the action is sent as a command to
the driver of the element; the clock advances by the measured duration. With the **digital twin** (on by
default) the simulated model is updated too, and the state measured on the equipment is kept next to it;
without the twin, the state is read back from the equipment. `ctx.simulate(element)` keeps one element
simulated in operation mode, for instance a component without real equipment.

## Journal

`ctx.journal` records every action and wait (element, arguments, mode, predicted and measured durations,
driver) and the messages of LUMOS emitted during runs, such as the propagation methods chosen by the
planner. Each entry records whether the action came from Python, the web interface or an AI assistant. The
same entries are emitted on the `lumos.context` logger; `lumos.core.journal.JsonFormatter` writes them as JSON
lines.

## Sweeps and Monte-Carlo

```python
results = ctx.sweep(
    {"laser.power": [1, 2, 5] * u.mW, "chip.phi.phase": [0, 0.5] * u.rad}, outputs=["cam.out"]
)  # every combination
stage.set_free("velocity", True, Normal(2 * u.mm / u.s, 0.1 * u.mm / u.s))
draws = ctx.sample(100, outputs=["cam.out"])  # free parameters drawn from their priors
```

The fields get their original values back after a sweep. Draws come from a reproducible random stream: every
element has its own stream, `ctx.rng(element)`, derived from the seed of the context.

**Noise of the points.** The streams also depend on a *noise realization*, `ctx.realization` (0 by default):
every realization gives independent streams, reproducible with the same seed. A Monte-Carlo run draws a new
realization at each point (`noise="independent"`, the default of `sample`), so that the spread of its outputs
includes the noise of the detectors and of the turbulence; without free parameters it repeats the run with
independent noise. A sweep keeps the noise of the context at every point (`noise="fixed"`, the default of
`sweep`): its points differ by their parameters only (common random numbers), which makes small differences
visible. Every point tells its realization (`point.realization`), and `progress(done, total)` follows long
studies.

```python
noise_only = ctx.sample(200, outputs=["cam.out"])            # photon noise alone, 200 realizations
fixed = ctx.sample(100, outputs=["cam.out"], noise="fixed")  # priors only, the noise of the context
```

## Feedback loops

The graph of a context has no cycle: a loop (adaptive optics, a servo, an iterative calibration) is a series
of **time iterations** of the context (ADR-0020). A *feedback* sends the value of an output port to a field at
the next iteration; `ctx.loop` runs the iterations, the clock advancing by the period at each one.

```python
ctx.feedback("ctrl.commands", "dm.commands", mode="add")  # "add": integrate; "set": replace
steps = ctx.loop(50, period=2 * u.ms, outputs=["wfs.residual"])
for step in steps:
    print(step.step, step.time, step.outputs["wfs.residual"].data)
```

Iteration `k` computes the outputs at `t0 + k·period`, then applies the feedbacks, then advances the clock: a
correction always acts one period later, like a real loop. The fields driven by the feedbacks and the clock keep
their final values (`restore=True` puts them back). Feedbacks are listed by `ctx.feedbacks`, removed by
`ctx.remove_feedback`, saved in `.lumos` files and written by `ctx.to_python()`. An element may refuse a target:
an `AOController` drives only a mirror of its own geometry.

## Files and scripts

`lumos.io.save(ctx, "bench.lumos")` writes the whole context: settings, clock, elements, groups and
connections; arrays larger than 64 kB go to a `bench.npz` file next to it. `lumos.io.load` reads it back
without importing any module. `ctx.to_python()` writes an equivalent Python script, a readable starting point
for a system built in the web interface or by an assistant. The JSON Schema of the format is published with
the documentation (`lumos-1.1.schema.json`; `lumos-1.0.schema.json` for the documents of LUMOS 0.x without
feedback loops).
