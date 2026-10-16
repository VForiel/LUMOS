"""JAX shares the GPU instead of reserving most of its memory (step P2.2, REQ-HPC-1, ADR-0008).

By default JAX reserves 75 % of the GPU memory when it creates its GPU client
(``XLA_PYTHON_CLIENT_PREALLOCATE``, JAX documentation "GPU memory allocation"). On a 6 GB card also driving
the display, the test suite then ran out of memory in cuFFT and cuBLAS. LUMOS sets the variable to ``false``
before it creates the JAX backend or lists the JAX devices, unless the user has set it.

Reference: the environment of a fresh Python process, read after LUMOS loaded JAX.
"""

import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")

VARIABLE = "XLA_PYTHON_CLIENT_PREALLOCATE"


def run(code, value=None):
    env = {k: v for k, v in os.environ.items() if k != VARIABLE}
    if value is not None:
        env[VARIABLE] = value
    script = f"import os\n{code}\nprint(os.environ.get({VARIABLE!r}))"
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "code",
    [
        "from lumos import backend; backend.get('jax')",
        "from lumos.backend.hardware import hardware_report; hardware_report(probe_gpus=True)",
    ],
)
def test_lumos_disables_jax_preallocation(code):
    assert run(code) == "false"


def test_user_setting_is_kept():
    assert run("from lumos import backend; backend.get('jax')", value="true") == "true"
