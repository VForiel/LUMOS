"""Compute quotas of the service layer (step P14.3, REQ-WEB-10, ADR-0019).

Requirements: a public instance bounds the size of the grids, the memory, the time and the concurrency of the
computations. The quotas are enforced by the service, so that the web interface and the MCP server share
them; a refused computation says which quota and how to stay within it (code ``quota``).

Independent reference for the memory: a complex field sampled on N x N pixels in double precision holds
N² complex128 numbers of 16 bytes (IEEE 754 binary64 real and imaginary parts), whatever else is allocated.
"""

import threading
import time

import pytest

from lumos.core.context import Context
from lumos.service import Service, ServiceConfig, ServiceError

LASER = "lumos.sources.GaussianLaser"
METER = "lumos.detectors.PowerMeter"


def bench(service, session, n_pixels=128):
    ctx = service.create_context(session, name="bench").id
    service.add_element(session, ctx, LASER, name="laser", params={"power": "1 mW"})
    service.set_param(session, ctx, "laser.n_pixels", n_pixels)
    service.add_element(session, ctx, METER, name="meter")
    service.connect(session, ctx, "laser.out", "meter.in")
    return ctx


def refused(call):
    with pytest.raises(ServiceError) as error:
        call()
    assert error.value.code == "quota"
    return error.value


def test_grids_larger_than_the_quota_are_refused_before_computing():
    service = Service(ServiceConfig(max_grid_pixels=256))
    session = service.create_session().id
    ctx = bench(service, session, n_pixels=512)
    error = refused(lambda: service.run(session, ctx, ["meter.out"]))
    assert "laser" in error.message
    assert "512" in error.message
    assert "256" in error.hint
    refused(lambda: service.inspect_port(session, ctx, "laser.out"))
    service.set_param(session, ctx, "laser.n_pixels", 256)
    job = service.wait_job(session, service.run(session, ctx, ["meter.out"]).id, timeout=60)
    assert job.status == "done", job.error
    service.shutdown()


def test_the_memory_of_the_fields_is_bounded():
    n = 256
    one_field_mb = n * n * 16 / 1e6  # one complex128 field of n x n pixels
    tight = Service(ServiceConfig(max_memory_mb=0.9 * one_field_mb))
    session = tight.create_session().id
    ctx = bench(tight, session, n_pixels=n)
    error = refused(lambda: tight.run(session, ctx, ["meter.out"]))
    assert "memory" in error.message
    tight.shutdown()
    roomy = Service(ServiceConfig(max_memory_mb=1000))
    session = roomy.create_session().id
    ctx = bench(roomy, session, n_pixels=n)
    assert roomy.wait_job(session, roomy.run(session, ctx, ["meter.out"]).id, timeout=60).status == "done"
    roomy.shutdown()


def test_the_memory_grows_with_the_number_of_wavelengths():
    n = 128
    one_field_mb = n * n * 16 / 1e6
    service = Service(ServiceConfig(max_memory_mb=4.5 * one_field_mb))
    session = service.create_session().id
    ctx = bench(service, session, n_pixels=n)
    service.set_observation(session, ctx, minimum="1 um", maximum="2 um", n_wavelengths=5)
    error = refused(lambda: service.run(session, ctx, ["meter.out"]))
    assert "wavelength" in error.hint
    service.shutdown()


@pytest.fixture
def slow_runs(monkeypatch):
    """Make every run of a context last at least 1 s."""
    original = Context.run

    def run(self, *args, **kwargs):
        time.sleep(1.0)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Context, "run", run)


def test_runs_at_the_same_time_are_bounded_for_the_whole_instance(slow_runs):
    service = Service(ServiceConfig(max_concurrent_runs=1, max_jobs_per_session=5))
    alice, bob = service.create_session().id, service.create_session().id
    first = service.run(alice, bench(service, alice), ["meter.out"])
    error = refused(lambda: service.run(bob, bench(service, bob), ["meter.out"]))
    assert "busy" in error.message
    assert service.wait_job(alice, first.id, timeout=60).status == "done"
    second = service.run(bob, bench(service, bob), ["meter.out"])
    assert service.wait_job(bob, second.id, timeout=60).status == "done"
    service.shutdown()


def test_inspections_respect_the_time_quota(slow_runs):
    service = Service(ServiceConfig(max_run_seconds=0.2))
    session = service.create_session().id
    ctx = bench(service, session)
    started = time.perf_counter()
    with pytest.raises(ServiceError) as error:
        service.inspect_port(session, ctx, "meter.out")
    assert "0.2" in str(error.value)
    assert time.perf_counter() - started < 30
    service.shutdown()


def test_without_quotas_nothing_is_refused():
    config = ServiceConfig()
    assert config.max_grid_pixels is None
    assert config.max_memory_mb is None
    assert config.max_concurrent_runs is None
    service = Service(config)
    session = service.create_session().id
    ctx = bench(service, session, n_pixels=512)
    assert (
        service.wait_job(session, service.run(session, ctx, ["meter.out"]).id, timeout=120).status == "done"
    )
    service.shutdown()


def test_quotas_apply_to_every_session_whatever_its_origin():
    service = Service(ServiceConfig(max_grid_pixels=64))
    results = {}

    def attempt(origin):
        session = service.create_session(origin=origin).id
        ctx = bench(service, session, n_pixels=128)
        try:
            service.run(session, ctx, ["meter.out"])
            results[origin] = "ran"
        except ServiceError as error:
            results[origin] = error.code

    threads = [threading.Thread(target=attempt, args=(o,)) for o in ("python", "web", "mcp")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {"python": "quota", "web": "quota", "mcp": "quota"}
    service.shutdown()
