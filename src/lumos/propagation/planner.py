"""Automatic choice of the propagation method (REQ-PROP-4, step P4.6).

The planner arrives in step P4.6; until then ``method="auto"`` is refused.
"""

from __future__ import annotations

from typing import Any

from lumos.fields.grid import Grid
from lumos.propagation._common import Method

__all__ = ["choose"]


def choose(signal: Any, z: float | None, grid_out: Grid) -> Method:
    """Return the method the planner selects for propagating ``signal`` by ``z`` onto ``grid_out``."""
    raise NotImplementedError("the propagation planner is not available yet; pass an explicit method")
