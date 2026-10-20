"""Procedures: declarative sequences of actions, measurements and processing (step P15.11, ADR-0024).

Specification: docs/specs/pistes_evolution.md §3 (decisions of 20/10/2026) and ADR-0024: a procedure is a JSON
list of steps (action, measure, repeat, for_each, process, set, use, wait, if, save, show) run by the context,
identical in simulation and in operation; it is validated before running; named results are its outputs.

Independent references:

- a camera without noise gives ``(η N t + D t) / g + bias`` ADU per pixel (Janesick, *Photon Transfer*):
  a dark frame (shutter closed) is ``D t / g + bias``, so a frame minus the dark is the image of a camera
  without dark current and bias, exactly;
- the mean of N independent frames has a pixel noise divided by √N: for a dark of mean D t electrons and read
  noise σ, the standard deviation of the mean dark over the pixels is ``sqrt(D t + σ²) / (g √N)``;
- the power read by a power meter is proportional to the power of the laser.
"""

import json

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.procedure import MAX_ITERATIONS, Procedure, ProcedureError, run_procedure
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.lab import Shutter
from lumos.elements.sources import GaussianLaser

DARK = 2000.0  # e-/s


def bench(*, noise=False, dark=DARK, bias=100.0, read_noise=0.0):
    ctx = Context(seed=3)
    laser = ctx.add(GaussianLaser(name="laser", power=1e-12 * u.W, n_pixels=32, pixel_scale=0.25 * u.mm))
    shutter = ctx.add(Shutter(name="shutter"))
    camera = ctx.add(
        Camera(
            name="camera",
            exposure=0.5 * u.s,
            dark_current=dark * u.electron / u.s,
            bias=bias * u.adu,
            read_noise=read_noise * u.electron,
            noise=noise,
        )
    )
    laser >> shutter >> camera
    return ctx


DARK_THEN_MEASURE = {
    "name": "dark then measure",
    "description": "a dark from 8 frames shutter closed, then 4 frames shutter open, minus the dark",
    "steps": [
        {"do": "action", "element": "shutter", "action": "close"},
        {
            "do": "repeat",
            "times": 8,
            "steps": [
                {"do": "action", "element": "camera", "action": "expose"},
                {"do": "measure", "outputs": ["camera.out"], "as": "darks", "collect": True},
            ],
        },
        {"do": "process", "function": "mean", "args": {"frames": "$darks"}, "as": "dark"},
        {"do": "action", "element": "shutter", "action": "open"},
        {
            "do": "repeat",
            "times": 4,
            "steps": [
                {"do": "action", "element": "camera", "action": "expose"},
                {"do": "measure", "outputs": ["camera.out"], "as": "images", "collect": True},
            ],
        },
        {"do": "process", "function": "mean", "args": {"frames": "$images"}, "as": "image"},
        {
            "do": "process",
            "function": "subtract_dark",
            "args": {"frame": "$image", "dark": "$dark"},
            "as": "science",
        },
        {"do": "show", "results": ["science"]},
    ],
}


def test_a_dark_then_a_measurement_without_noise_is_exact():
    ctx = bench()
    result = run_procedure(ctx, Procedure.from_json(DARK_THEN_MEASURE))
    assert len(result["darks"]) == 8
    assert len(result["images"]) == 4
    assert result.shown == ["science"]
    reference = bench(dark=0.0, bias=0.0)
    reference["shutter"].open()
    expected = reference.run(["camera.out"])["camera.out"].data.to_value(u.adu)
    np.testing.assert_allclose(result["science"].data.to_value(u.adu), expected, rtol=1e-9, atol=1e-9)
    dark = result["dark"].data.to_value(u.adu)
    np.testing.assert_allclose(dark, DARK * 0.5 / 1.0 + 100.0)  # gain 1 e-/ADU by default


def test_the_mean_of_noisy_darks_follows_the_square_root_law():
    ctx = bench(noise=True, read_noise=5.0)
    n = 16
    procedure = Procedure.from_json(
        {
            "name": "darks",
            "steps": [
                {"do": "action", "element": "shutter", "action": "close"},
                {
                    "do": "repeat",
                    "times": n,
                    "steps": [
                        {"do": "action", "element": "camera", "action": "expose"},
                        {"do": "measure", "outputs": ["camera.out"], "as": "darks", "collect": True},
                    ],
                },
                {"do": "process", "function": "mean", "args": {"frames": "$darks"}, "as": "dark"},
            ],
        }
    )
    dark = run_procedure(ctx, procedure)["dark"].data.to_value(u.adu)
    expected = np.sqrt(DARK * 0.5 + 5.0**2) / np.sqrt(n)
    assert float(np.std(dark)) == pytest.approx(expected, rel=0.15)
    assert float(np.mean(dark)) == pytest.approx(DARK * 0.5 + 100.0, rel=0.01)


def test_for_each_set_and_collect():
    ctx = Context()
    laser = ctx.add(GaussianLaser(name="laser", n_pixels=32, pixel_scale=0.25 * u.mm))
    laser >> ctx.add(PowerMeter(name="meter"))
    procedure = Procedure.from_json(
        {
            "name": "power sweep",
            "steps": [
                {
                    "do": "for_each",
                    "values": ["1 mW", "2 mW", "4 mW"],
                    "as": "power",
                    "steps": [
                        {"do": "set", "target": "laser.power", "value": "$power"},
                        {"do": "measure", "outputs": ["meter.out"], "as": "readings", "collect": True},
                    ],
                }
            ],
        }
    )
    result = run_procedure(ctx, procedure)
    watts = [float(np.sum(r.data.to_value(u.W))) for r in result["readings"]]
    np.testing.assert_allclose(np.array(watts) / watts[0], [1, 2, 4], rtol=1e-6)
    assert ctx["laser"].power == 4 * u.mW  # the procedure leaves the bench as it set it


def test_ranges_and_conditions():
    ctx = Context()
    ctx.add(GaussianLaser(name="laser", n_pixels=32, pixel_scale=0.25 * u.mm)) >> ctx.add(
        PowerMeter(name="meter")
    )
    procedure = Procedure.from_json(
        {
            "name": "threshold",
            "steps": [
                {
                    "do": "for_each",
                    "values": {"start": "1 mW", "stop": "3 mW", "num": 3},
                    "as": "power",
                    "steps": [
                        {"do": "set", "target": "laser.power", "value": "$power"},
                        {"do": "measure", "outputs": ["meter.out"], "as": "reading"},
                        {
                            "do": "process",
                            "function": "total",
                            "args": {"product": "$reading"},
                            "as": "watts",
                        },
                        {
                            "do": "if",
                            "left": "$watts",
                            "op": ">",
                            "right": "1.5 mW",
                            "then": [
                                {"do": "measure", "outputs": ["meter.out"], "as": "high", "collect": True}
                            ],
                            "else": [
                                {"do": "measure", "outputs": ["meter.out"], "as": "low", "collect": True}
                            ],
                        },
                    ],
                }
            ],
        }
    )
    result = run_procedure(ctx, procedure)
    assert (len(result["low"]), len(result["high"])) == (1, 2)


def test_wait_advances_the_clock_and_actions_take_their_time():
    ctx = bench()
    procedure = Procedure.from_json(
        {
            "name": "timing",
            "steps": [
                {"do": "action", "element": "shutter", "action": "close"},
                {"do": "wait", "duration": "2 s"},
                {"do": "action", "element": "camera", "action": "expose"},
            ],
        }
    )
    result = run_procedure(ctx, procedure)
    assert ctx.elapsed.to_value(u.s) == pytest.approx(0.010 + 2 + 0.5)
    assert [step["do"] for step in result.journal] == ["action", "wait", "action"]


def test_use_feeds_an_input_with_a_named_result():
    from lumos.processing.reduction import DarkSubtraction

    ctx = bench()
    subtract = ctx.add(DarkSubtraction(name="subtract"))
    ctx.connect(ctx["camera"].port("out"), subtract.port("in"))
    document = json.loads(json.dumps(DARK_THEN_MEASURE))
    document["steps"] = [
        *document["steps"][:3],
        {"do": "action", "element": "shutter", "action": "open"},
        {"do": "use", "result": "dark", "port": "subtract.dark"},
        {"do": "measure", "outputs": ["subtract.out"], "as": "science"},
    ]
    result = run_procedure(ctx, Procedure.from_json(document))
    reference = bench(dark=0.0, bias=0.0)
    reference["shutter"].open()
    expected = reference.run(["camera.out"])["camera.out"].data.to_value(u.adu)
    np.testing.assert_allclose(result["science"].data.to_value(u.adu), expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"do": "action", "element": "nowhere", "action": "close"}, "nowhere"),
        ({"do": "action", "element": "shutter", "action": "fly"}, "fly"),
        ({"do": "measure", "outputs": ["camera.nothing"], "as": "x"}, "camera.nothing"),
        ({"do": "process", "function": "eval", "args": {}, "as": "x"}, "eval"),
        ({"do": "process", "function": "mean", "args": {"frames": "$undefined"}, "as": "x"}, "undefined"),
        ({"do": "set", "target": "laser.colour", "value": 1}, "colour"),
        ({"do": "repeat", "times": MAX_ITERATIONS + 1, "steps": []}, "iterations"),
        ({"do": "jump"}, "jump"),
    ],
)
def test_problems_are_found_before_running(step, message):
    ctx = bench()
    procedure = Procedure.from_json({"name": "wrong", "steps": [step]})
    problems = procedure.problems(ctx)
    assert any(message in problem for problem in problems), problems
    with pytest.raises(ProcedureError, match=message):
        run_procedure(ctx, procedure)


def test_nested_loops_count_their_iterations():
    inner = {"do": "repeat", "times": 200, "steps": [{"do": "wait", "duration": "1 ms"}]}
    outer = {"do": "repeat", "times": 100, "steps": [inner]}
    problems = Procedure.from_json({"name": "big", "steps": [outer]}).problems(bench())
    assert any("iterations" in p for p in problems)


def test_progress_and_cancellation():
    ctx = bench()
    seen = []

    def cancelled():
        return len(seen) >= 3

    with pytest.raises(ProcedureError, match="cancelled"):
        run_procedure(
            ctx,
            Procedure.from_json(DARK_THEN_MEASURE),
            progress=lambda done, total: seen.append(done),
            cancelled=cancelled,
        )
    assert 3 <= len(seen) < 10


def test_procedures_are_saved_with_their_context_and_alone():
    import lumos.io

    ctx = bench()
    ctx.procedures["dark"] = Procedure.from_json(DARK_THEN_MEASURE)
    document = lumos.io.to_document(ctx)
    assert document["procedures"]["dark"]["steps"][0]["do"] == "action"
    loaded = lumos.io.from_document(json.loads(json.dumps(document)))
    assert loaded.procedures["dark"].to_json() == Procedure.from_json(DARK_THEN_MEASURE).to_json()
    alone = lumos.io.to_document(ctx.procedures["dark"])
    assert alone["kind"] == "procedure"
    assert lumos.io.from_document(alone).to_json() == ctx.procedures["dark"].to_json()


def test_the_python_api():
    ctx = bench()
    ctx.procedures["dark"] = Procedure.from_json(DARK_THEN_MEASURE)
    result = ctx.run_procedure("dark")
    assert "science" in result.results


def test_operation_needs_a_confirmation():
    ctx = bench()
    ctx.mode = "operation"
    ctx.simulate(ctx["laser"])
    ctx.simulate(ctx["camera"])
    procedure = Procedure.from_json(
        {"name": "close", "steps": [{"do": "action", "element": "shutter", "action": "close"}]}
    )
    assert procedure.devices(ctx) == ["shutter"]
    with pytest.raises(ProcedureError, match="confirm"):
        run_procedure(ctx, procedure, confirm=lambda devices: False)
