"""Free settings, components of arrays and intermediate outputs in the studies of the service (step P14.17).

Requirements (review of 19/10/2026; ADR-0022): the service frees a numeric setting (``"laser.n_pixels"``) or a
component of an array parameter (``"wheel.transmissions[0]"``) with a prior, describes them in the summary of
the element (``free``), sweeps and draws them; the points of a study give their values (integers for integer
settings). Any output port, also one connected to another block (an intermediate output), can be computed by a
study.

Expected values: a filter wheel on slot 1 transmits ``P × transmissions[0]`` to a power meter (energy
conservation of an ideal filter): each point is checked against the value drawn for this point.
"""

import pytest
from astropy import units as u

from lumos.service import Service, ServiceConfig, ServiceError


@pytest.fixture
def service(tmp_path):
    service = Service(ServiceConfig(archive_root=str(tmp_path / "archives")))
    yield service
    service.shutdown()


@pytest.fixture
def session(service):
    return service.create_session(origin="web").id


def wheel_bench(service, session):
    ctx = service.create_context(session, name="wheel").id
    # pixels of 0.25 mm: the 1 mm beam fits in the smallest grid drawn (32 pixels), the power is all measured
    laser = {"power": "2 mW", "pixel_scale": "0.25 mm"}
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser", params=laser)
    service.add_element(session, ctx, "lumos.lab.FilterWheel", name="wheel")
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "laser.out", "wheel.in")
    service.connect(session, ctx, "wheel.out", "meter.in")
    return ctx


def study(service, session, ctx, **request):
    job = service.run_study(session, ctx, **request)
    job = service.wait_job(session, job.id, timeout=120)
    assert job.status == "done", job.error
    return service.get_study(session, job.id)


def watts(summary):
    return (summary.values[0] * u.Unit(summary.unit)).to_value(u.W)


def test_settings_and_components_are_freed_and_described(service, session):
    ctx = wheel_bench(service, session)
    laser = service.set_free(
        session, ctx, "laser.n_pixels", prior={"kind": "uniform", "low": "31.5", "high": "64.5"}
    )
    assert laser.free["n_pixels"] == {"kind": "uniform", "low": "31.5", "high": "64.5"}
    wheel = service.set_free(
        session, ctx, "wheel.transmissions[0]", prior={"kind": "normal", "mean": "0.5", "std": "0.05"}
    )
    assert wheel.free["transmissions[0]"]["kind"] == "normal"
    with pytest.raises(ServiceError) as error:
        service.set_free(
            session, ctx, "wheel.transmissions[9]", prior={"kind": "uniform", "low": 0, "high": 1}
        )
    assert "transmissions" in error.value.message
    with pytest.raises(ServiceError):
        service.set_free(
            session, ctx, "laser.n_pixels", prior={"kind": "uniform", "low": "1 mm", "high": "2 mm"}
        )
    fixed = service.set_free(session, ctx, "wheel.transmissions[0]", free=False)
    assert "transmissions[0]" not in fixed.free


def test_a_monte_carlo_run_draws_them(service, session):
    ctx = wheel_bench(service, session)
    service.set_free(session, ctx, "laser.n_pixels", prior={"kind": "uniform", "low": "31.5", "high": "64.5"})
    service.set_free(
        session, ctx, "wheel.transmissions[0]", prior={"kind": "normal", "mean": "0.5", "std": "0.05"}
    )
    result = study(service, session, ctx, kind="sample", n=12, outputs=["meter.out"])
    assert result.parameters == ["laser.n_pixels", "wheel.transmissions[0]"]
    for point in result.points:
        pixels = point.params["laser.n_pixels"]
        assert pixels == str(int(pixels))
        assert 32 <= int(pixels) <= 64
        transmission = float(point.params["wheel.transmissions[0]"])  # written with 6 significant digits
        assert watts(point.outputs["meter.out"]) == pytest.approx(2e-3 * transmission, rel=1e-5)


def test_a_sweep_varies_a_component_and_computes_an_intermediate_output(service, session):
    ctx = wheel_bench(service, session)
    result = study(
        service,
        session,
        ctx,
        kind="sweep",
        parameters={"wheel.transmissions[0]": {"start": "0.2", "stop": "0.6", "num": 3}},
        outputs=["wheel.out", "meter.out"],  # wheel.out feeds the meter: an intermediate output
    )
    assert [float(p.params["wheel.transmissions[0]"]) for p in result.points] == pytest.approx(
        [0.2, 0.4, 0.6]
    )
    for point, transmission in zip(result.points, [0.2, 0.4, 0.6], strict=True):
        assert watts(point.outputs["meter.out"]) == pytest.approx(2e-3 * transmission, rel=1e-6)
        assert point.outputs["wheel.out"].kind != ""
