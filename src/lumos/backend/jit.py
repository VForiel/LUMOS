"""Compilation of numerical functions (REQ-HPC-3).

:func:`jit` compiles a pure function with :func:`jax.jit` when it runs on JAX arrays (or when JAX is the
current backend), and calls it directly otherwise, without importing JAX. The same LUMOS code therefore runs
uncompiled on NumPy and compiled, vectorisable and differentiable on JAX.

Conventions for other compilation techniques
--------------------------------------------
- Prefer vectorised Array API code: it runs on every backend and JAX compiles it.
- Use ``numba`` (``@numba.njit(cache=True)``) only for CPU loops that cannot be vectorised; keep such kernels
  in private ``_kernels`` modules working on NumPy arrays, behind a function that converts backend arrays.
- Write a C++ extension (nanobind) only when profiling shows a need, and record the decision in an ADR.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Sequence
from typing import Any, overload

from lumos.backend._core import current
from lumos.backend._jax_pytrees import register

__all__ = ["jit"]


def _uses_jax(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if current().name == "jax":
        return True
    if "jax" not in sys.modules:
        return False  # no JAX array can exist before JAX is imported
    import jax

    register()
    return any(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves((args, kwargs)))


@overload
def jit[F: Callable[..., Any]](function: F, *, static_argnames: Sequence[str] = ()) -> F: ...
@overload
def jit[F: Callable[..., Any]](
    function: None = None, *, static_argnames: Sequence[str] = ()
) -> Callable[[F], F]: ...


def jit[F: Callable[..., Any]](
    function: F | None = None, *, static_argnames: Sequence[str] = ()
) -> F | Callable[[F], F]:
    """Compile ``function`` with JAX when it is called on JAX arrays.

    Usable as ``@jit`` or ``@jit(static_argnames=("n",))``. Static arguments must be hashable; a new value
    triggers a new compilation (use them for values determining shapes).
    """
    if function is None:
        return lambda f: jit(f, static_argnames=static_argnames)

    compiled: Callable[..., Any] | None = None

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiled
        if not _uses_jax(args, kwargs):
            return function(*args, **kwargs)
        if compiled is None:
            import jax

            register()
            compiled = jax.jit(function, static_argnames=tuple(static_argnames))
        return compiled(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
