"""Shared pytest configuration."""

import os

from hypothesis import settings

# "dev" keeps local runs fast; CI sets HYPOTHESIS_PROFILE=ci for a deeper search.
# On compiled backends (JAX) the first example of every array shape includes its compilation, which is not
# what the tests measure: no deadline then.
_COMPILED = os.environ.get("LUMOS_TEST_BACKEND", "numpy") not in ("numpy", "cupy")
settings.register_profile("dev", max_examples=50, **({"deadline": None} if _COMPILED else {}))
settings.register_profile("ci", max_examples=300, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_sessionstart(session):
    """Resolve the automatic backend once, before any timed (Hypothesis) test.

    The first ``lumos.backend.current()`` call probes the GPUs (about 1-2 s). Paying it here keeps the
    per-example Hypothesis deadline independent of the order in which tests run.
    """
    from lumos import backend

    backend.current()

    # The tests compare results with NumPy references (np.asarray, np.testing): they run on the NumPy
    # reference backend unless LUMOS_TEST_BACKEND names another one; tests of the automatic choice set "auto".
    import lumos

    lumos.config.backend = os.environ.get("LUMOS_TEST_BACKEND", "numpy")
