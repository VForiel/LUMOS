"""Abstract signal types flowing between elements (REQ-SIG-*).

The hierarchy of signal classes defines which ports can be connected: an output producing a signal of
type ``T`` can feed an input expecting ``S`` when ``T`` is ``S`` or a subclass of ``S``. Concrete signals
(``Wavefront``, ``Wavecube``, ``DataProduct``...) derive from these classes; they arrive in phase 3.
"""

__all__ = ["DataSignal", "OpticalSignal", "Signal"]


class Signal:
    """Anything that flows along a connection. An input of type ``Signal`` accepts every signal."""


class OpticalSignal(Signal):
    """Light: coherent fields, incoherent bundles, radiance fields, modal amplitudes..."""


class DataSignal(Signal):
    """Detected or processed data: images, cubes, time series, scalars..."""
