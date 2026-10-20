"""Procedures as the body of studies; independent iterations in parallel (step P15.13, ADR-0024 §7).

Specification: docs/specs/pistes_evolution.md §3, question 2 (decided on 20/10/2026): a procedure may be the
body of a sweep or a Monte-Carlo run, its named results being the outputs of each point; independent
iterations (a "for each" without dependence between iterations, the points of a study on a procedure) run in
parallel on the workers of the service.

Independent references:

- the power read by a power meter is proportional to the power of the laser (linearity);
- each point of a study on a procedure starts from the same context (the procedure of one point does not see
  the actions of another): the clock and the states are those of the context at every point, and the context
  is unchanged afterwards;
- the results do not depend on the number of workers (the noise of each point is its own realization).
"""

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.priors import Uniform
from lumos.core.procedure import Procedure, run_procedure
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.lab import Shutter
from lumos.elements.sources import GaussianLaser

MEASURE = {
    "name": "measure",
    "steps": [
        {"do": "action", "element": "shutter", "action": "open"},
        {"do": "action", "element": "camera", "action": "expose"},
        {"do": "measure", "outputs": ["meter.out"], "as": "reading"},
        {"do": "process", "function": "total", "args": {"product": "$reading"}, "as": "watts"},
        {"do": "measure", "outputs": ["camera.out"], "as": "image"},
        {"do": "process", "function": "total", "args": {"product": "$image"}, "as": "counts"},
    ],
}


def bench(noise=False):
    ctx = Context(seed=11)
    laser = ctx.add(GaussianLaser(name="laser", power=1e-13 * u.W, n_pixels=32, pixel_scale=0.25 * u.mm))
    shutter = ctx.add(Shutter(name="shutter"))
    camera = ctx.add(Camera(name="camera", exposure=0.1 * u.s, noise=noise))
    laser >> shutter >> camera
    ctx.connect(shutter.port("out"), ctx.add(PowerMeter(name="meter")).port("in"))
    ctx.procedures["measure"] = Procedure.from_json(MEASURE)
    return ctx


def test_a_sweep_runs_the_procedure_at_every_point():
    ctx = bench()
    powers = [1e-13, 2e-13, 4e-13] * u.W
    points = ctx.sweep({"laser.power": powers}, procedure="measure")
    watts = np.array([p.outputs["watts"].to_value(u.W) for p in points])
    np.testing.assert_allclose(watts / watts[0], [1, 2, 4], rtol=1e-9)
    assert set(points[0].outputs) == {"reading", "watts", "image", "counts"}
    # every point starts from the context as it is, which is left unchanged
    assert ctx.elapsed == 0 * u.s
    assert ctx["shutter"].is_open == 0
    assert ctx["camera"].frame == 0
    assert ctx["laser"].power == 1e-13 * u.W


def test_the_outputs_of_a_study_on_a_procedure_can_be_chosen():
    points = bench().sweep({"laser.power": [1e-13, 2e-13] * u.W}, procedure="measure", outputs=["watts"])
    assert set(points[0].outputs) == {"watts"}


@pytest.mark.parametrize("kind", ["sweep", "sample"])
def test_the_results_do_not_depend_on_the_number_of_workers(kind):
    def study(workers):
        ctx = bench(noise=True)
        if kind == "sweep":
            return ctx.sweep(
                {"laser.power": [1e-13, 2e-13, 3e-13, 4e-13] * u.W},
                procedure="measure",
                noise="independent",
                workers=workers,
            )
        ctx["laser"].set_free("power", prior=Uniform(1e-13 * u.W, 4e-13 * u.W))
        return ctx.sample(4, procedure="measure", workers=workers)

    alone, together = study(1), study(3)
    for a, b in zip(alone, together, strict=True):
        assert a.params == b.params
        assert a.outputs["counts"] == b.outputs["counts"]
    assert len({p.outputs["counts"].value for p in alone}) == 4  # noise and parameters differ


def test_a_parallel_for_each_gives_the_results_of_a_sequential_one():
    steps = [
        {"do": "action", "element": "shutter", "action": "open"},
        {
            "do": "for_each",
            "values": ["1e-13 W", "2e-13 W", "3e-13 W"],
            "as": "power",
            "steps": [
                {"do": "set", "target": "laser.power", "value": "$power"},
                {"do": "measure", "outputs": ["meter.out"], "as": "readings", "collect": True},
            ],
        },
    ]
    sequential = run_procedure(bench(), Procedure.from_json({"name": "s", "steps": steps}))
    steps[1]["parallel"] = True
    ctx = bench()
    parallel = run_procedure(ctx, Procedure.from_json({"name": "p", "steps": steps}), workers=3)
    a = [r.data.to_value(u.W) for r in sequential["readings"]]
    b = [r.data.to_value(u.W) for r in parallel["readings"]]
    np.testing.assert_allclose(b, a, rtol=1e-12)
    assert ctx["laser"].power == 1e-13 * u.W  # parallel iterations run on copies of the context


def test_parallel_iterations_and_studies_on_procedures_are_simulations():
    ctx = bench()
    ctx.mode = "operation"
    for name in ("laser", "camera", "meter", "shutter"):
        ctx.simulate(ctx[name], False)
    procedure = Procedure.from_json(
        {
            "name": "p",
            "steps": [{"do": "for_each", "values": [1, 2], "as": "x", "parallel": True, "steps": []}],
        }
    )
    assert any("parallel" in problem for problem in procedure.problems(ctx))
    with pytest.raises(ValueError, match="simulation"):
        ctx.sweep({"laser.power": [1e-13] * u.W}, procedure="measure")
