"""Hardware detection and automatic backend selection (REQ-HPC-2, ADR-0004).

:func:`hardware_report` describes the machine LUMOS runs on: operating system, CPU, memory, GPUs seen by the
installed backends and versions of the main libraries. The report is stored in archive metadata
(REQ-RES-2) and drives :func:`select_backend`, used when ``lumos.config.backend`` is ``"auto"``.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

from lumos.backend._core import available

__all__ = ["GPUInfo", "HardwareReport", "hardware_report", "select_backend"]

logger = logging.getLogger(__name__)

_LIBRARIES = ("numpy", "scipy", "astropy", "sympy", "matplotlib", "jax", "jaxlib", "cupy", "torch", "numba")
# Preference order among GPU backends (ADR-0004): JAX is the differentiable reference of LUMOS.
_GPU_PREFERENCE = ("jax", "cupy", "torch")


@dataclass(frozen=True)
class GPUInfo:
    """A GPU (or other accelerator) seen by a backend."""

    backend: str
    name: str
    platform: str
    memory_bytes: int | None = None


@dataclass(frozen=True)
class HardwareReport:
    """Description of the machine and software environment."""

    os: str
    os_release: str
    os_version: str
    machine: str
    python: str
    cpu_model: str
    cpu_logical_cores: int
    cpu_physical_cores: int | None
    ram_total_bytes: int
    ram_available_bytes: int
    gpus: tuple[GPUInfo, ...] = ()
    libraries: dict[str, str | None] = field(default_factory=dict)
    backends: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the report as plain data (YAML and JSON serialisable)."""
        data = asdict(self)
        data["gpus"] = [asdict(gpu) for gpu in self.gpus]
        data["backends"] = list(self.backends)
        return data

    def summary(self) -> str:
        """Return a short human-readable description."""
        physical = f"{self.cpu_physical_cores} physical / " if self.cpu_physical_cores else ""
        total_gib, available_gib = self.ram_total_bytes / 2**30, self.ram_available_bytes / 2**30
        lines = [
            f"OS:      {self.os} {self.os_release} ({self.machine}), Python {self.python}",
            f"CPU:     {self.cpu_model}, {physical}{self.cpu_logical_cores} logical cores",
            f"RAM:     {total_gib:.1f} GiB ({available_gib:.1f} GiB available)",
        ]
        if self.gpus:
            for gpu in self.gpus:
                memory = f", {gpu.memory_bytes / 2**30:.1f} GiB" if gpu.memory_bytes else ""
                lines.append(f"GPU:     {gpu.name} ({gpu.platform} via {gpu.backend}{memory})")
        else:
            lines.append("GPU:     none detected")
        lines.append(f"Backends: {', '.join(self.backends)}")
        return "\n".join(lines)


def _cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            if result.stdout.strip():
                return result.stdout.strip()
        elif sys.platform == "win32":  # lets type checkers skip winreg on other platforms
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("could not read the CPU model: %s", error)
    return platform.processor() or platform.machine() or "unknown"


def _library_versions() -> dict[str, str | None]:
    distributions = metadata.packages_distributions()
    versions: dict[str, str | None] = {}
    for module in _LIBRARIES:
        names = distributions.get(module, [])
        versions[module] = metadata.version(names[0]) if names else None
    return versions


def _jax_gpus() -> list[GPUInfo]:
    import jax

    gpus = []
    for device in jax.devices():
        if device.platform == "cpu":
            continue
        stats = device.memory_stats() or {}
        gpus.append(GPUInfo("jax", str(device.device_kind), device.platform, stats.get("bytes_limit")))
    return gpus


def _cupy_gpus() -> list[GPUInfo]:
    import cupy  # pyright: ignore[reportMissingImports]  (optional backend)

    runtime = cupy.cuda.runtime
    gpu_platform = "rocm" if runtime.is_hip else "cuda"
    gpus = []
    for index in range(runtime.getDeviceCount()):
        properties = runtime.getDeviceProperties(index)
        name = properties["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        gpus.append(GPUInfo("cupy", name, gpu_platform, int(properties["totalGlobalMem"])))
    return gpus


def _torch_gpus() -> list[GPUInfo]:
    import torch  # pyright: ignore[reportMissingImports]  (optional backend)

    gpus = []
    if torch.cuda.is_available():
        gpu_platform = "rocm" if torch.version.hip else "cuda"
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(GPUInfo("torch", properties.name, gpu_platform, int(properties.total_memory)))
    if torch.backends.mps.is_available():
        gpus.append(GPUInfo("torch", "Apple GPU", "metal"))
    return gpus


_PROBES = {"jax": _jax_gpus, "cupy": _cupy_gpus, "torch": _torch_gpus}


def hardware_report(*, probe_gpus: bool = True) -> HardwareReport:
    """Describe the machine and the software environment.

    Parameters
    ----------
    probe_gpus
        Whether to ask the installed GPU-capable backends (JAX, CuPy, PyTorch) for their devices. This
        imports those libraries, which can take a few seconds.
    """
    backends = available()
    gpus: list[GPUInfo] = []
    if probe_gpus:
        for name in _GPU_PREFERENCE:
            if name in backends:
                try:
                    gpus.extend(_PROBES[name]())
                except Exception as error:  # a broken GPU stack must not prevent LUMOS from running
                    logger.warning("could not list the %s devices: %s", name, error)
    memory = psutil.virtual_memory()
    return HardwareReport(
        os=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine=platform.machine(),
        python=platform.python_version(),
        cpu_model=_cpu_model(),
        cpu_logical_cores=os.cpu_count() or 1,
        cpu_physical_cores=psutil.cpu_count(logical=False),
        ram_total_bytes=int(memory.total),
        ram_available_bytes=int(memory.available),
        gpus=tuple(gpus),
        libraries=_library_versions(),
        backends=backends,
    )


def select_backend(report: HardwareReport) -> str:
    """Choose the backend to use on the machine described by ``report`` (policy of ADR-0004).

    A GPU backend is chosen when it sees a GPU, JAX first, then CuPy, then PyTorch; otherwise NumPy, the CPU
    reference.
    """
    for name in _GPU_PREFERENCE:
        if name in report.backends and any(gpu.backend == name for gpu in report.gpus):
            return name
    return "numpy"
