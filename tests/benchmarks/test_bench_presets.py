"""Benchmarks of the presets, run as the interface runs them (step P15.3, REQ-HPC-6).

Not run by default: ``pytest -m benchmark tests/benchmarks``. Each preset is loaded, then its outputs are
computed once from a cold cache per round.
"""

from pathlib import Path

import pytest

import lumos.io

pytestmark = pytest.mark.benchmark

PRESETS = sorted((Path(__file__).resolve().parents[2] / "examples" / "presets").glob("*.lumos"))


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_preset(benchmark, path):
    def setup():
        ctx = lumos.io.load(path)
        return (ctx,), {}

    def run(ctx):
        return ctx.run(ctx.meta.get("outputs", []))

    benchmark.pedantic(run, setup=setup, rounds=3, iterations=1)
