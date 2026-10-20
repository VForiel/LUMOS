"""Procedures of a context in its Python export (step P15.18, review of 20/10/2026, ADR-0024 §6).

Requirements: the procedures are saved in the ``.lumos`` file (format 1.3) and, when the context is exported
to Python, included by default, each one a function taking the context and returning its named results; the
export without them is still possible.

Reference: a function of the export run on a context built by the same export gives the named results of the
procedure run by the engine on an identical context (camera without noise: exact), the export being an
equivalent program; the expected power is also the analytic one (an ideal power meter reads the source power).
"""

import numpy as np
import pytest
from astropy import units as u

import lumos.io
from lumos import Context
from lumos.core.procedure import Procedure
from lumos.elements.detectors import Camera, PowerMeter
from lumos.elements.lab import Shutter
from lumos.elements.sources import GaussianLaser
from lumos.io.codegen import to_python

MEASURE = {
    "name": "dark then measure",
    "description": "a dark, then the science frame",
    "steps": [
        {"do": "action", "element": "shutter", "action": "close"},
        {"do": "action", "element": "camera", "action": "expose"},
        {"do": "measure", "outputs": ["camera.out"], "as": "dark"},
        {"do": "action", "element": "shutter", "action": "open"},
        {"do": "action", "element": "camera", "action": "expose"},
        {"do": "measure", "outputs": ["camera.out"], "as": "image"},
        {"do": "measure", "outputs": ["meter.out"], "as": "power"},
    ],
}

SCRIPT = {"name": "2-script", "script": "results['two'] = 1 + 1\n"}


def bench():
    ctx = Context(seed=1, name="bench")
    laser = ctx.add(GaussianLaser(name="laser", power=1e-12 * u.W, n_pixels=32, pixel_scale=0.25 * u.mm))
    shutter = ctx.add(Shutter(name="shutter"))
    camera = ctx.add(Camera(name="camera", exposure=0.5 * u.s, noise=False))
    laser >> shutter >> camera
    ctx.connect(shutter.port("out"), ctx.add(PowerMeter(name="meter")).port("in"))
    ctx.procedures["dark then measure"] = Procedure.from_json(MEASURE)
    ctx.procedures["2-script"] = Procedure.from_json(SCRIPT)
    return ctx


def test_the_procedures_are_functions_of_the_export():
    ctx = bench()
    script = ctx.to_python()
    assert "def dark_then_measure(ctx):" in script
    assert "a dark, then the science frame" in script  # the description is the docstring
    namespace: dict = {"__name__": "export"}
    exec(compile(script, "bench.py", "exec"), namespace)  # the script exported just above
    built = namespace["ctx"]
    results = namespace["dark_then_measure"](built)
    expected = bench().run_procedure("dark then measure")
    assert set(results) == set(expected.results)
    np.testing.assert_allclose(results["image"].data.value, expected["image"].data.value, rtol=1e-12)
    assert float(np.sum(results["power"].data.to_value(u.W))) == pytest.approx(1e-12, rel=1e-9)
    # a script procedure is a function too, under a valid Python name
    functions = [name for name, value in namespace.items() if callable(value) and "script" in name]
    assert len(functions) == 1
    assert namespace[functions[0]](built) == {"two": 2}


def test_the_export_can_leave_the_procedures_out():
    script = to_python(bench(), procedures=False)
    assert "def dark_then_measure" not in script
    assert "Context(" in script


def test_the_procedures_are_saved_in_the_lumos_file(tmp_path):
    path = tmp_path / "bench.lumos"
    lumos.io.save(bench(), path)
    assert sorted(lumos.io.load(path).procedures) == ["2-script", "dark then measure"]


def test_the_service_and_the_web_api_export_the_procedures(tmp_path):
    from lumos.service import Service, ServiceConfig

    service = Service(ServiceConfig(archive_root=str(tmp_path)))
    session = service.create_session().id
    context = service.import_lumos(session, lumos.io.to_document(bench())).id
    assert "def dark_then_measure(ctx):" in service.export_python(session, context)
    assert "def dark_then_measure" not in service.export_python(session, context, procedures=False)
    service.shutdown()

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lumos.web import WebConfig, create_app

    with TestClient(
        create_app(WebConfig(service=ServiceConfig(archive_root=str(tmp_path)))), base_url="http://localhost"
    ) as client:
        client.get("/api/info")
        cid = client.post("/api/contexts/import", json=lumos.io.to_document(bench())).json()["id"]
        assert "def dark_then_measure(ctx):" in client.get(f"/api/contexts/{cid}/export.py").text
        assert (
            "def dark_then_measure" not in client.get(f"/api/contexts/{cid}/export.py?procedures=false").text
        )
