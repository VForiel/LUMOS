"""Array backends of LUMOS (REQ-HPC-1).

LUMOS stores its data in arrays of a **backend** and writes numerical code once, against the
`Python Array API standard <https://data-apis.org/array-api/>`_:

- ``numpy``: the CPU reference, always available;
- ``jax``: JIT compilation, vectorisation, automatic differentiation, CPU/GPU/TPU (differentiable reference);
- ``cupy`` and ``torch``: optional GPU backends.

Backends are imported only when first used. The **current** backend is local to the running thread or task
and changed with :func:`use`::

    with lumos.backend.use("jax"):
        ...  # new arrays are created with JAX

.. note::
   Loading the JAX backend enables JAX's 64-bit mode (``jax_enable_x64``), because LUMOS computes in double
   precision by default (REQ-HPC-5).
"""

from lumos.backend._core import (
    Backend,
    BackendUnavailableError,
    available,
    backend_of,
    convert,
    current,
    get,
    get_default,
    known,
    set_default,
    to_numpy,
    use,
)
from lumos.backend._jax_pytrees import install as _install_jax_pytrees
from lumos.backend.jit import jit

_install_jax_pytrees()

__all__ = [
    "Backend",
    "BackendUnavailableError",
    "available",
    "backend_of",
    "convert",
    "current",
    "get",
    "get_default",
    "jit",
    "known",
    "set_default",
    "to_numpy",
    "use",
]
