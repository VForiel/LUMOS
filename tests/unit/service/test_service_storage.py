"""Storage of the sessions of a public instance (step P14.21, review of 19/10/2026; REQ-WEB-10).

Question of the review: do the generated and archived data accumulate until the disk is full? On a public
instance the results must go when the session ends, a session may not keep more than 1 GB, and the user is
warned when the data of the session grow large or the disk is nearly full.

Requirements, with ``session_storage`` on (default of the public profile):

- the archives of a session are in a directory of its own, listed and read by this session only, and deleted
  when the session closes;
- beyond ``max_session_bytes`` the oldest data of the session are deleted (the newest kept), and the job that
  wrote them says so;
- :meth:`Service.storage` gives the bytes used, the limit and the free disk, with warnings when the session
  uses 80 % of its limit or when the free disk is below twice ``min_free_disk_bytes``; below
  ``min_free_disk_bytes``, archiving is refused (quota);
- only the ``max_jobs_kept`` latest finished jobs keep their results in memory (any profile).

Without ``session_storage`` (local profile), archives stay where they are, as before.
"""

import shutil
import time
from collections import namedtuple
from pathlib import Path

import pytest

from lumos.service import Service, ServiceConfig, ServiceError

LASER = "lumos.sources.GaussianLaser"
METER = "lumos.detectors.PowerMeter"


def bench(service, session):
    cid = service.create_context(session).id
    laser = service.add_element(session, cid, LASER).id
    meter = service.add_element(session, cid, METER).id
    service.connect(session, cid, f"{laser}.out", f"{meter}.in")
    return cid, f"{meter}.out"


def finish(service, session, job):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        info = service.get_job(session, job.id)
        if info.status not in ("queued", "running"):
            assert info.status == "done", info.error
            return info
        time.sleep(0.05)
    pytest.fail("the run did not end")


def size_of(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


@pytest.fixture
def make(tmp_path):
    services = []

    def make(**options):
        service = Service(ServiceConfig(archive_root=str(tmp_path / "archives"), **options))
        services.append(service)
        return service

    yield make
    for service in services:
        service.shutdown()


def test_the_archives_of_a_session_are_its_own_and_go_with_it(make, tmp_path):
    service = make(session_storage=True)
    alice = service.create_session(origin="web").id
    bob = service.create_session(origin="web").id
    cid, port = bench(service, alice)
    info = finish(service, alice, service.run(alice, cid, [port], archive=True))
    path = Path(info.archive)
    assert path.is_dir()

    assert [a.path for a in service.list_archives(session=alice)] == [str(path)]
    assert service.list_archives(session=bob) == []
    assert service.archive_zip(str(path), session=alice).startswith(b"PK")
    with pytest.raises(ServiceError) as refused:
        service.archive_zip(str(path), session=bob)
    assert refused.value.code == "forbidden"
    with pytest.raises(ServiceError):
        service.read_archive_metadata(str(path), session=bob)

    service.close_session(alice)
    assert not path.exists()
    assert not any(p.name.endswith(alice) for p in (tmp_path / "archives").iterdir())


def test_beyond_the_limit_the_oldest_data_of_the_session_are_deleted(make):
    probe = make(session_storage=True)
    session = probe.create_session(origin="web").id
    cid, port = bench(probe, session)
    one = size_of(finish(probe, session, probe.run(session, cid, [port], archive=True)).archive)

    service = make(session_storage=True, max_session_bytes=int(one * 2.5))
    session = service.create_session(origin="web").id
    cid, port = bench(service, session)
    paths = []
    for _ in range(3):
        info = finish(service, session, service.run(session, cid, [port], archive=True))
        paths.append(Path(info.archive))
        assert info.warnings == []
    info = finish(service, session, service.run(session, cid, [port], archive=True))
    paths.append(Path(info.archive))
    assert not paths[0].exists()  # the oldest went
    assert all(p.exists() for p in paths[1:])
    assert any("deleted" in w for w in info.warnings)
    storage = service.storage(session)
    assert storage.used_bytes <= storage.limit_bytes


def test_the_storage_warns_before_the_limit_and_when_the_disk_is_nearly_full(make, monkeypatch):
    probe = make(session_storage=True)
    session = probe.create_session(origin="web").id
    cid, port = bench(probe, session)
    one = size_of(finish(probe, session, probe.run(session, cid, [port], archive=True)).archive)

    service = make(session_storage=True, max_session_bytes=int(one * 1.15), min_free_disk_bytes=10**6)
    session = service.create_session(origin="web").id
    cid, port = bench(service, session)
    assert service.storage(session).warnings == []
    finish(service, session, service.run(session, cid, [port], archive=True))
    storage = service.storage(session)
    assert storage.used_bytes >= 0.8 * storage.limit_bytes
    assert any("%" in w for w in storage.warnings)  # e.g. "the session uses 87 % of its 1 MB"

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**9, 10**9 - 15 * 10**5, 15 * 10**5))
    assert any("disk" in w for w in service.storage(session).warnings)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**9, 10**9 - 10**5, 10**5))
    with pytest.raises(ServiceError) as refused:
        service.run(session, cid, [port], archive=True)
    assert refused.value.code == "quota"
    finish(service, session, service.run(session, cid, [port]))  # without archive, it still runs


def test_only_the_latest_finished_jobs_keep_their_results(make):
    service = make(max_jobs_kept=2)
    session = service.create_session(origin="web").id
    cid, port = bench(service, session)
    jobs = [finish(service, session, service.run(session, cid, [port])) for _ in range(3)]
    with pytest.raises(ServiceError) as gone:
        service.get_job(session, jobs[0].id)
    assert gone.value.code == "not_found"
    for job in jobs[1:]:
        assert service.get_result(session, job.id, port).unit == "W"


def test_without_session_storage_the_archives_stay(make, tmp_path):
    service = make()
    session = service.create_session(origin="web").id
    cid, port = bench(service, session)
    info = finish(service, session, service.run(session, cid, [port], archive=True))
    service.close_session(session)
    assert Path(info.archive).is_dir()
    assert (tmp_path / "archives") in Path(info.archive).parents
