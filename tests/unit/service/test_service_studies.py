"""Studies in the service layer: seed, free parameters, sweeps and Monte-Carlo runs (step P14.13).

Requirements (REQ-CTX-6, REQ-CTX-10, REQ-EL-3, review of 18/10/2026: "define the seed of every random process
and sweep for Monte-Carlo from the visual editor"):

- the seed of a context is read and set like its name; it is saved in the ``.lumos`` document and undone;
- a parameter is marked free with a prior (uniform, log-uniform, normal; values with their units) or fixed
  again; the description of the context shows the free parameters and their priors;
- a study runs in the background like a run: a sweep (values listed, or ``start``/``stop``/``num`` with an
  optional ``log`` spacing; ``grid`` or ``zip``) or a Monte-Carlo run of ``n`` draws; its points give the
  values of the parameters, the realization of the noise and a summary of every output; a CSV table is
  exported;
- the number of points is bounded by a quota; a study where nothing varies is refused with a hint.

Expected values: a lossless laser of 2 mW through a top-hat filter of power transmission ``T`` centred on its
wavelength gives ``2 mW × T`` on a power meter (definition of the power transmission; the Gaussian beam
truncated by the grid loses less than 1e-3 of its power, as in the parity test of P14.11). Monte-Carlo
transmissions drawn from U(0.2, 0.8) have the mean 0.5 and the standard deviation 0.6/√12.
"""

import csv
import io
import json

import numpy as np
import pytest
from astropy import units as u

from lumos.service import Service, ServiceConfig, ServiceError

POWER = 2e-3  # W


@pytest.fixture
def service(tmp_path):
    return Service(ServiceConfig(archive_root=str(tmp_path / "archives")))


@pytest.fixture
def session(service):
    return service.create_session(origin="web").id


def filter_bench(service, session, transmission=1.0):
    ctx = service.create_context(session, name="filter").id
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser", params={"power": "2 mW"})
    service.add_element(
        session,
        ctx,
        "lumos.optics.SpectralFilter",
        name="filt",
        params={"center": "633 nm", "width": "10 nm", "transmission": transmission},
    )
    service.add_element(session, ctx, "lumos.detectors.PowerMeter", name="meter")
    service.connect(session, ctx, "laser.out", "filt.in")
    service.connect(session, ctx, "filt.out", "meter.in")
    return ctx


def watts(summary):
    return (summary.values[0] * u.Unit(summary.unit)).to_value(u.W)


def study(service, session, ctx, **request):
    job = service.run_study(session, ctx, **request)
    assert job.kind == request["kind"]
    job = service.wait_job(session, job.id, timeout=120)
    assert job.status == "done", job.error
    return service.get_study(session, job.id)


# --- Seed ------------------------------------------------------------------------------------------------


def test_the_seed_is_set_saved_and_undone(service, session):
    ctx = filter_bench(service, session)
    first = service.describe_context(session, ctx).seed
    assert isinstance(first, int)
    assert service.set_context(session, ctx, seed=1234).seed == 1234
    assert json.loads(service.export_lumos(session, ctx))["context"]["seed"] == 1234
    assert service.undo(session, ctx).seed == first
    with pytest.raises(ServiceError):
        service.set_context(session, ctx, seed=-3)


# --- Free parameters -------------------------------------------------------------------------------------


def test_a_parameter_is_freed_with_a_prior_and_fixed_again(service, session):
    ctx = filter_bench(service, session)
    service.set_free(session, ctx, "laser.waist", True, {"kind": "normal", "mean": "1 mm", "std": "50 um"})
    service.set_free(session, ctx, "filt.transmission", True, {"kind": "uniform", "low": 0.2, "high": 0.8})
    summary = service.describe_context(session, ctx)
    laser = next(e for e in summary.elements if e.id == "laser")
    filt = next(e for e in summary.elements if e.id == "filt")
    assert laser.free == {"waist": {"kind": "normal", "mean": "1 mm", "std": "0.05 mm"}}
    assert filt.free == {"transmission": {"kind": "uniform", "low": "0.2", "high": "0.8"}}
    document = json.loads(service.export_lumos(session, ctx))
    entry = next(e for e in document["elements"] if e["id"] == "filt")
    assert entry["free"]["transmission"]["prior"]["kind"] == "uniform"

    service.set_free(session, ctx, "laser.waist", False)
    laser = next(e for e in service.describe_context(session, ctx).elements if e.id == "laser")
    assert laser.free == {}
    assert service.undo(session, ctx).elements  # an edit like the others
    laser = next(e for e in service.describe_context(session, ctx).elements if e.id == "laser")
    assert "waist" in laser.free


def test_wrong_priors_are_refused_with_a_hint(service, session):
    ctx = filter_bench(service, session)
    with pytest.raises(ServiceError) as error:
        service.set_free(
            session, ctx, "laser.waist", True, {"kind": "gaussian", "mean": "1 mm", "std": "1 um"}
        )
    assert "normal" in error.value.hint
    with pytest.raises(ServiceError):
        service.set_free(session, ctx, "laser.waist", True, {"kind": "uniform", "low": "1 s", "high": "2 s"})
    with pytest.raises(ServiceError):
        service.set_free(session, ctx, "laser.nothing", True, {"kind": "uniform", "low": 0, "high": 1})


# --- Sweeps ----------------------------------------------------------------------------------------------


def test_a_sweep_of_listed_values(service, session):
    ctx = filter_bench(service, session)
    result = study(
        service,
        session,
        ctx,
        kind="sweep",
        parameters={"filt.transmission": [0.25, 0.5, 1.0]},
        outputs=["meter.out"],
    )
    assert result.kind == "sweep"
    assert result.parameters == ["filt.transmission"]
    assert result.outputs == ["meter.out"]
    assert [p.params["filt.transmission"] for p in result.points] == ["0.25", "0.5", "1"]
    measured = [watts(p.outputs["meter.out"]) for p in result.points]
    assert measured == pytest.approx([POWER * t for t in (0.25, 0.5, 1.0)], rel=1e-3)
    assert {p.realization for p in result.points} == {0}
    # the context is left as it was
    filt = next(e for e in service.describe_context(session, ctx).elements if e.id == "filt")
    assert filt.params["transmission"] == "1"


def test_a_sweep_of_a_range_with_units_and_log_spacing(service, session):
    ctx = filter_bench(service, session)
    result = study(
        service,
        session,
        ctx,
        kind="sweep",
        parameters={
            "filt.transmission": {"start": 0.2, "stop": 1.0, "num": 5},
            "laser.waist": {"start": "1 mm", "stop": "100 mm", "num": 3, "log": True},
        },
        mode="grid",
        outputs=["meter.out"],
    )
    assert len(result.points) == 15
    transmissions = sorted({float(p.params["filt.transmission"]) for p in result.points})
    assert transmissions == pytest.approx(np.linspace(0.2, 1.0, 5))
    waists = sorted({u.Quantity(p.params["laser.waist"]).to_value(u.mm) for p in result.points})
    assert waists == pytest.approx([1.0, 10.0, 100.0])


def test_zip_sweeps_pair_the_values(service, session):
    ctx = filter_bench(service, session)
    result = study(
        service,
        session,
        ctx,
        kind="sweep",
        parameters={"filt.transmission": [0.5, 1.0], "laser.power": ["1 mW", "4 mW"]},
        mode="zip",
        outputs=["meter.out"],
    )
    assert [watts(p.outputs["meter.out"]) for p in result.points] == pytest.approx([0.5e-3, 4e-3], rel=1e-3)


# --- Monte-Carlo -----------------------------------------------------------------------------------------


def test_a_monte_carlo_run_draws_the_free_parameters(service, session):
    ctx = filter_bench(service, session)
    service.set_free(session, ctx, "filt.transmission", True, {"kind": "uniform", "low": 0.2, "high": 0.8})
    service.set_context(session, ctx, seed=2026)
    n = 200
    result = study(service, session, ctx, kind="sample", n=n, outputs=["meter.out"])
    assert result.kind == "sample"
    assert result.seed == 2026
    transmissions = np.array([float(p.params["filt.transmission"]) for p in result.points])
    powers = np.array([watts(p.outputs["meter.out"]) for p in result.points])
    assert len(result.points) == n
    assert np.all((transmissions >= 0.2) & (transmissions <= 0.8))
    np.testing.assert_allclose(powers, POWER * transmissions, rtol=1e-3)
    sigma = 0.6 / np.sqrt(12)
    assert transmissions.mean() == pytest.approx(0.5, abs=4 * sigma / np.sqrt(n))
    assert transmissions.std(ddof=1) == pytest.approx(sigma, rel=0.25)

    again = study(service, session, ctx, kind="sample", n=n, outputs=["meter.out"])
    assert [p.params for p in again.points] == [p.params for p in result.points]  # same seed, same draws
    service.set_context(session, ctx, seed=2027)
    other = study(service, session, ctx, kind="sample", n=n, outputs=["meter.out"])
    assert [p.params for p in other.points] != [p.params for p in result.points]


def test_a_monte_carlo_run_of_the_noise_alone(service, session):
    ctx = service.create_context(session, name="noise").id
    service.add_element(session, ctx, "lumos.sources.GaussianLaser", name="laser", params={"power": "1 nW"})
    service.add_element(session, ctx, "lumos.detectors.Camera", name="camera", params={"noise": True})
    service.connect(session, ctx, "laser.out", "camera.in")
    result = study(service, session, ctx, kind="sample", n=4, outputs=["camera.out"])
    sums = [p.outputs["camera.out"].statistics["sum"] for p in result.points]
    assert [p.realization for p in result.points] == [1, 2, 3, 4]
    assert len(set(sums)) == 4  # independent photon noise
    again = study(service, session, ctx, kind="sample", n=4, outputs=["camera.out"])
    assert [p.outputs["camera.out"].statistics["sum"] for p in again.points] == sums


def test_a_study_where_nothing_varies_is_refused(service, session):
    ctx = filter_bench(service, session)  # no free parameter, no noise
    with pytest.raises(ServiceError) as error:
        service.run_study(session, ctx, kind="sample", n=3, outputs=["meter.out"])
    assert "free" in error.value.hint
    with pytest.raises(ServiceError):
        service.run_study(session, ctx, kind="sweep", parameters={}, outputs=["meter.out"])
    with pytest.raises(ServiceError):
        service.run_study(session, ctx, kind="loop", outputs=["meter.out"])


def test_the_points_of_a_study_are_bounded(tmp_path):
    service = Service(ServiceConfig(max_study_points=10))
    session = service.create_session().id
    ctx = filter_bench(service, session)
    with pytest.raises(ServiceError) as error:
        service.run_study(
            session,
            ctx,
            kind="sweep",
            parameters={"filt.transmission": {"start": 0, "stop": 1, "num": 11}},
            outputs=["meter.out"],
        )
    assert error.value.code == "quota"


def test_a_study_is_exported_as_a_csv_table(service, session):
    ctx = filter_bench(service, session)
    job = service.run_study(
        session, ctx, kind="sweep", parameters={"filt.transmission": [0.5, 1.0]}, outputs=["meter.out"]
    )
    service.wait_job(session, job.id, timeout=60)
    rows = list(csv.reader(io.StringIO(service.study_csv(session, job.id))))
    header, data = rows[0], rows[1:]
    assert header[:3] == ["point", "realization", "filt.transmission"]
    assert header[3] == "meter.out [W]"
    assert len(data) == 2
    assert float(data[1][2]) == 1.0
    assert float(data[1][3]) == pytest.approx(POWER, rel=1e-3)
