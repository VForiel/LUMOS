"""The automatic choice of backend does not import JAX when JAX cannot see an accelerator (step P15.3).

Requirements (ADR-0004): "auto" chooses JAX only when JAX sees a GPU (or another accelerator); otherwise
NumPy. JAX sees an accelerator only through a plugin package (the ``jax_plugins`` namespace of the CUDA,
ROCm and Metal plugins, ``jax_cuda12_plugin``, ``libtpu``; JAX documentation, "Installing JAX"). Without one,
asking JAX for its devices only costs its import (about 2 s, paid by the first run of every process) and
cannot change the choice, so it is skipped; with one, JAX is asked as before.
"""

import importlib.util

from lumos.backend import hardware


def patch_plugins(monkeypatch, installed):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in hardware.JAX_ACCELERATOR_PLUGINS:
            return object() if name in installed else None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_jax_is_not_asked_without_an_accelerator_plugin(monkeypatch):
    patch_plugins(monkeypatch, installed=())
    asked = []
    monkeypatch.setitem(hardware._PROBES, "jax", lambda: asked.append("jax") or [])
    monkeypatch.setattr(hardware, "available", lambda: ("numpy", "jax"))
    report = hardware.hardware_report(probe_gpus=True)
    assert asked == []
    assert hardware.select_backend(report) == "numpy"


def test_jax_is_asked_when_a_plugin_is_installed(monkeypatch):
    patch_plugins(monkeypatch, installed=("jax_cuda12_plugin",))
    gpu = hardware.GPUInfo("jax", "NVIDIA A100", "gpu", 40 * 2**30)
    monkeypatch.setitem(hardware._PROBES, "jax", lambda: [gpu])
    monkeypatch.setattr(hardware, "available", lambda: ("numpy", "jax"))
    report = hardware.hardware_report(probe_gpus=True)
    assert report.gpus == (gpu,)
    assert hardware.select_backend(report) == "jax"
