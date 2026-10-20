"""Procedures exported to Python, and the script mode (step P15.12, ADR-0024 §6).

Requirements (docs/specs/pistes_evolution.md §3): the editor offers "Export to Python" (blocks to script, not
the reverse); the exported script does what the procedure does. The *script mode* is a procedure written as a
Python script; it runs on local and private instances only, never on the public instance; loading a document
holding a script never runs it.

Reference: the exported script and the procedure run on two identical contexts give the same named results
(cameras without noise: exact), since the export is an equivalent program.
"""

import numpy as np
import pytest
from astropy import units as u

from lumos import Context
from lumos.core.procedure import Procedure, ProcedureError, run_procedure
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.lab import Shutter
from lumos.elements.sources import GaussianLaser

DOCUMENT = {
    "name": "dark then measure",
    "steps": [
        {"do": "action", "element": "shutter", "action": "close"},
        {
            "do": "repeat",
            "times": 3,
            "steps": [
                {"do": "action", "element": "camera", "action": "expose"},
                {"do": "measure", "outputs": ["camera.out"], "as": "darks", "collect": True},
            ],
        },
        {"do": "process", "function": "mean", "args": {"frames": "$darks"}, "as": "dark"},
        {"do": "action", "element": "shutter", "action": "open"},
        {"do": "wait", "duration": "0.2 s"},
        {
            "do": "for_each",
            "values": {"start": "1 pW", "stop": "2 pW", "num": 2},
            "as": "power",
            "steps": [
                {"do": "set", "target": "laser.power", "value": "$power"},
                {"do": "action", "element": "camera", "action": "expose"},
                {"do": "measure", "outputs": ["camera.out", "meter.out"], "as": "frames", "collect": True},
            ],
        },
        {"do": "measure", "outputs": ["camera.out"], "as": "image"},
        {
            "do": "process",
            "function": "subtract_dark",
            "args": {"frame": "$image", "dark": "$dark"},
            "as": "science",
        },
        {"do": "process", "function": "total", "args": {"product": "$science"}, "as": "flux"},
        {
            "do": "if",
            "left": "$flux",
            "op": ">",
            "right": "0 adu",
            "then": [{"do": "show", "results": ["science"]}],
            "else": [],
        },
    ],
}


def bench():
    ctx = Context(seed=1)
    laser = ctx.add(GaussianLaser(name="laser", power=1e-12 * u.W, n_pixels=32, pixel_scale=0.25 * u.mm))
    shutter = ctx.add(Shutter(name="shutter"))
    camera = ctx.add(
        Camera(name="camera", exposure=0.5 * u.s, dark_current=1000 * u.electron / u.s, noise=False)
    )
    laser >> shutter >> camera
    ctx.connect(shutter.port("out"), ctx.add(PowerMeter(name="meter")).port("in"))
    return ctx


def test_the_exported_script_does_what_the_procedure_does():
    procedure = Procedure.from_json(DOCUMENT)
    ran = bench()
    expected = run_procedure(ran, procedure)
    script = procedure.to_python()
    assert "ctx[" in script
    assert "def " not in script.split("\n", 1)[0]
    namespace = {"ctx": bench()}
    exec(compile(script, "procedure.py", "exec"), namespace)  # the script exported just above
    results = namespace["results"]
    assert set(results) == set(expected.results)
    np.testing.assert_allclose(results["science"].data.value, expected["science"].data.value, rtol=1e-12)
    assert results["flux"] == expected["flux"]
    assert len(results["frames"]) == 2
    assert namespace["ctx"].elapsed.to_value(u.s) == pytest.approx(ran.elapsed.to_value(u.s))
    assert namespace["ctx"]["laser"].power == ran["laser"].power


def test_a_script_procedure_runs_its_code_in_python():
    procedure = Procedure.from_json(
        {
            "name": "script",
            "script": "ctx['shutter'].open()\nresults['image'] = ctx.run(['camera.out'])['camera.out']\n",
        }
    )
    assert procedure.is_script
    assert procedure.problems(bench()) == []
    result = run_procedure(bench(), procedure)
    assert "image" in result
    assert procedure.to_json()["script"].startswith("ctx['shutter']")


def test_loading_a_script_never_runs_it(tmp_path):
    import lumos.io

    ctx = bench()
    ctx.procedures["s"] = Procedure.from_json({"name": "s", "script": "raise SystemExit('ran')"})
    path = tmp_path / "bench.lumos"
    lumos.io.save(ctx, path)
    loaded = lumos.io.load(path)  # must not raise SystemExit
    assert loaded.procedures["s"].is_script


def test_a_procedure_has_steps_or_a_script_not_both():
    with pytest.raises(ProcedureError, match="script"):
        Procedure.from_json({"name": "both", "steps": [], "script": "pass"})


def test_the_service_runs_scripts_only_where_they_are_enabled(tmp_path):
    from lumos.service import Service, ServiceConfig, ServiceError

    def service(enabled):
        return Service(ServiceConfig(archive_root=str(tmp_path / "archives"), scripts_enabled=enabled))

    script = {"name": "s", "script": "results['two'] = 1 + 1\n"}
    refused = service(False)
    session = refused.create_session().id
    ctx = refused.create_context(session).id
    info = refused.set_procedure(session, ctx, script)
    assert any("script" in p for p in info.problems)
    with pytest.raises(ServiceError, match="script"):
        refused.run_procedure(session, ctx, "s")
    refused.shutdown()

    allowed = service(True)
    session = allowed.create_session().id
    ctx = allowed.create_context(session).id
    allowed.set_procedure(session, ctx, script)
    job = allowed.wait_job(session, allowed.run_procedure(session, ctx, "s").id, timeout=30)
    assert job.status == "done", job.message
    assert allowed.get_result(session, job.id, "two").values == [2]
    allowed.shutdown()


def test_the_public_instance_never_enables_scripts():
    pytest.importorskip("fastapi")
    from lumos.web import WebConfig
    from lumos.web.config import ConfigError

    assert WebConfig().service.scripts_enabled  # local profile
    assert not WebConfig(profile="public", allowed_hosts=["x.org"]).service.scripts_enabled
    private = WebConfig(profile="public", allowed_hosts=["x.org"], anonymous_role=None)
    private.service.scripts_enabled = True
    private.check()  # a private instance (every visitor logs in) may enable them
    open_to_all = WebConfig(profile="public", allowed_hosts=["x.org"], anonymous_role="simulator")
    open_to_all.service.scripts_enabled = True
    with pytest.raises(ConfigError, match="script"):
        open_to_all.check()


def test_the_preset_takes_a_dark_then_measures():
    from lumos.service import Service

    service = Service()
    session = service.create_session().id
    ctx = service.load_preset(session, "dark_then_measure").id
    names = [p.name for p in service.list_procedures(session, ctx)]
    assert "dark then measure" in names
    assert service.list_procedures(session, ctx)[0].problems == []
    service.shutdown()
