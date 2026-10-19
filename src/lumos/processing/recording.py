"""Recorder: a block keeping the signal of a port at every point of a study (step P14.20).

A :class:`Recorder` taps an output (connect the output to its input; the output may feed other blocks too). It
passes its signal unchanged; studies and loops of the service keep the figure of its output at every point
(one out of ``every``), so that the results browser of the web interface shows it point by point, also when
the study is not recorded as a whole.
"""

from __future__ import annotations

from typing import Any

from lumos.core.element import Element, Values
from lumos.core.param import Setting
from lumos.core.port import Input, Output
from lumos.core.registry import register
from lumos.core.signal import Signal

__all__ = ["Recorder"]


@register("lumos.processing.Recorder")
class Recorder(Element):
    """Record the signal of a port at each point of a study or step of a loop (passes it unchanged)."""

    every = Setting(int, default=1, doc="records one point out of every (1: all of them)")
    in_ = Input(Signal)
    out = Output(Signal)

    @staticmethod
    def transfer(values: Values, inputs: Any) -> dict[str, Any]:
        """Return the input signal."""
        return {"out": inputs["in"]}
