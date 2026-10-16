"""Tests for hardware detection and automatic backend selection (step P2.3).

Requirements: REQ-HPC-2 (hardware report: OS, CPU, RAM, GPU, libraries; automatic backend selection that the
user can override), REQ-RES-2 (the report goes into archive metadata, so it must be YAML-serialisable).
Reference values come from the Python standard library (platform, os, importlib.metadata).
"""

import importlib
import importlib.util
import os
import platform
import subprocess
import sys
import warnings
from importlib import metadata

import pytest
import yaml

import lumos
from lumos import backend
from lumos.backend.hardware import GPUInfo, HardwareReport, hardware_report, select_backend

# --- Report contents --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report():
    return hardware_report(probe_gpus=False)


def test_report_is_exposed_at_package_level():
    assert lumos.hardware_report is hardware_report


def test_os_and_python(report):
    assert report.os == platform.system()
    assert report.machine == platform.machine()
    assert report.python == platform.python_version()


def test_cpu(report):
    assert report.cpu_logical_cores == os.cpu_count()
    assert report.cpu_physical_cores is None or 1 <= report.cpu_physical_cores <= report.cpu_logical_cores
    assert isinstance(report.cpu_model, str)
    assert report.cpu_model


def test_ram(report):
    assert report.ram_total_bytes > 0
    assert 0 <= report.ram_available_bytes <= report.ram_total_bytes


def test_library_versions(report):
    assert report.libraries["numpy"] == metadata.version("numpy")
    assert report.libraries["astropy"] == metadata.version("astropy")
    for name in ("jax", "cupy", "torch", "numba"):
        # the distribution name may differ from the module name (cupy-cuda12x...): use the module version
        expected = None
        if importlib.util.find_spec(name):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # e.g. CuPy warns when CUDA_PATH is not set
                expected = importlib.import_module(name).__version__
        assert report.libraries[name] == expected


def test_gpu_probes_survive_warnings(monkeypatch, caplog):
    """An import-time warning of a GPU library (even turned into an error) must not hide its GPUs."""
    from lumos.backend import hardware

    def probe():
        warnings.warn("CUDA path could not be detected", UserWarning, stacklevel=1)
        return [GPUInfo("jax", "Fake GPU", "cuda", 1)]

    monkeypatch.setattr(hardware, "available", lambda: ("numpy", "jax"))
    monkeypatch.setitem(hardware._PROBES, "jax", probe)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = hardware_report(probe_gpus=True)
    assert [gpu.name for gpu in report.gpus] == ["Fake GPU"]
    assert "CUDA path could not be detected" in caplog.text


def test_backends(report):
    assert report.backends == backend.available()


def test_no_gpu_probing_means_no_gpu_listed(report):
    assert report.gpus == ()


def test_report_without_gpu_probing_does_not_import_heavy_libraries():
    code = (
        "import sys, lumos; lumos.hardware_report(probe_gpus=False); "
        "print(sorted(m for m in ('jax', 'cupy', 'torch') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_gpu_probing_runs_without_error():
    probed = hardware_report(probe_gpus=True)
    assert all(isinstance(gpu, GPUInfo) for gpu in probed.gpus)


def test_report_is_yaml_serialisable(report):
    data = report.to_dict()
    assert yaml.safe_load(yaml.safe_dump(data)) == data
    assert data["os"] == platform.system()


def test_summary_is_readable(report):
    text = report.summary()
    assert platform.system() in text
    assert str(os.cpu_count()) in text


# --- Automatic selection (pure function of a report) ------------------------------------------------


def make_report(backends, gpus=()):
    return HardwareReport(
        os="Linux",
        os_release="6.0",
        os_version="#1",
        machine="x86_64",
        python="3.13.0",
        cpu_model="Test CPU",
        cpu_logical_cores=8,
        cpu_physical_cores=4,
        ram_total_bytes=16 * 2**30,
        ram_available_bytes=8 * 2**30,
        gpus=tuple(gpus),
        libraries={},
        backends=tuple(backends),
    )


def gpu(backend_name, platform_name="cuda"):
    return GPUInfo(backend=backend_name, name="Test GPU", platform=platform_name, memory_bytes=8 * 2**30)


@pytest.mark.parametrize(
    ("backends", "gpus", "expected"),
    [
        (("numpy",), (), "numpy"),
        (("numpy", "jax"), (), "numpy"),  # no GPU: NumPy is the CPU reference
        (("numpy", "jax"), (gpu("jax"),), "jax"),
        (("numpy", "cupy"), (gpu("cupy"),), "cupy"),
        (("numpy", "torch"), (gpu("torch", "metal"),), "torch"),
        (("numpy", "jax", "cupy", "torch"), (gpu("cupy"), gpu("torch"), gpu("jax")), "jax"),  # JAX preferred
        (("numpy", "cupy", "torch"), (gpu("torch"), gpu("cupy")), "cupy"),
    ],
)
def test_selection_policy(backends, gpus, expected):
    assert select_backend(make_report(backends, gpus)) == expected


# --- Configuration ---------------------------------------------------------------------------------


@pytest.fixture
def restore_backend_setting():
    previous = lumos.config.backend
    yield
    lumos.config.backend = previous


def test_default_backend_setting_is_auto():
    code = "import lumos; print(lumos.config.backend)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "auto"


def test_auto_resolves_with_the_selection_policy(restore_backend_setting):
    lumos.config.backend = "auto"
    assert backend.current().name == select_backend(hardware_report(probe_gpus=True))


def test_backend_setting_can_be_forced(restore_backend_setting):
    lumos.config.backend = "numpy"
    assert backend.current().name == "numpy"
    assert backend.get_default() == "numpy"


def test_backend_setting_is_validated(restore_backend_setting):
    with pytest.raises(ValueError):
        lumos.config.backend = "tensorflow"


def test_set_default_and_config_are_the_same_setting(restore_backend_setting):
    backend.set_default("numpy")
    assert lumos.config.backend == "numpy"
    backend.set_default("auto")
    assert lumos.config.backend == "auto"
