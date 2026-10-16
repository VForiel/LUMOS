r"""Rectangular sampling grids of the transverse plane (REQ-SIG-1).

A :class:`Grid` describes where the samples of a 2D field are: ``nx × ny`` pixels of steps ``dx, dy`` around a
centre ``(x0, y0)``, in a **spatial** plane (lengths, e.g. a pupil or a detector) or an **angular** plane
(angles, e.g. the sky or the far field).

Conventions (§9 of the specifications)
--------------------------------------
- Arrays are indexed ``[..., y, x]``: the grid shape is ``(ny, nx)``.
- The central pixel is index ``N // 2`` for even and odd ``N``, and the coordinates are

  .. math:: x_i = (i - \lfloor N_x / 2 \rfloor)\,\Delta x + x_0, \qquad
            y_j = (j - \lfloor N_y / 2 \rfloor)\,\Delta y + y_0 .

  This is the ordering of a centred discrete Fourier transform (:func:`lumos.backend.fft.fft2c`), so the
  origin of a field and of its spectrum fall on the same pixel.
- An angular plane maps to the focal plane of a lens of focal length :math:`f` with :math:`x = f\,\theta`.

Coordinates are computed only when asked for, in the current backend and precision, and cached.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Literal, cast

import numpy as np
from astropy import units as u

from lumos.backend._core import get
from lumos.backend.memory import dtypes
from lumos.core.param import Setting
from lumos.units import quantity_from_json, quantity_to_json, to_si

__all__ = ["Grid", "GridSetting", "Plane"]

Plane = Literal["spatial", "angular"]

_SI_UNITS: dict[str, u.UnitBase] = {"spatial": u.m, "angular": u.rad}


def _size(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a boolean")
    try:
        size = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if size < 1:
        raise ValueError(f"{name} must be at least 1, got {size}")
    return size


def _plane_of(value: Any, name: str) -> Plane:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity (a length or an angle), got {value!r}")
    unit = cast(u.UnitBase, value.unit)
    if unit.is_equivalent(u.m):
        return "spatial"
    if unit.is_equivalent(u.rad):
        return "angular"
    raise u.UnitsError(f"{name} must be a length or an angle, got unit '{value.unit}'")


def _scalar(value: u.Quantity, unit: u.UnitBase, name: str) -> float:
    number: Any = value.to_value(unit)
    if np.ndim(number) != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(number)


class Grid:
    """Rectangular sampling of a spatial or angular plane.

    Parameters
    ----------
    nx
        Number of pixels along x (columns).
    ny
        Number of pixels along y (rows); defaults to ``nx``.
    dx
        Step along x: a length (spatial plane) or an angle (angular plane). Its unit is used to display
        every quantity of the grid.
    dy
        Step along y, in the same plane as ``dx``; defaults to ``dx``.
    x0, y0
        Coordinates of the central pixel ``(ny // 2, nx // 2)``; default to zero.

    Raises
    ------
    TypeError
        If a size is not an integer or a step or centre is not a quantity.
    ValueError
        If a size is not positive or a step is not positive and finite.
    astropy.units.UnitsError
        If a step is neither a length nor an angle, or the steps and centre are not in the same plane.

    Examples
    --------
    >>> from astropy import units as u
    >>> from lumos.fields import Grid
    >>> grid = Grid(4, dx=1 * u.mm)
    >>> grid.x
    <Quantity [-2., -1.,  0.,  1.] mm>
    """

    __slots__ = ("_cache", "_display", "_nx", "_ny", "_plane", "_si", "_unit")

    _nx: int
    _ny: int
    _plane: Plane
    _unit: u.UnitBase
    _display: tuple[float, float, float, float]
    _si: tuple[float, float, float, float]
    _cache: dict[tuple[str, str, str], Any]

    def __init__(
        self,
        nx: int,
        ny: int | None = None,
        *,
        dx: u.Quantity,
        dy: u.Quantity | None = None,
        x0: u.Quantity | None = None,
        y0: u.Quantity | None = None,
    ) -> None:
        nx = _size(nx, "nx")
        ny = nx if ny is None else _size(ny, "ny")
        plane = _plane_of(dx, "dx")
        unit: u.UnitBase = dx.unit  # type: ignore[assignment]
        values = {"dx": dx, "dy": dx if dy is None else dy, "x0": x0, "y0": y0}
        display = []
        for name, value in values.items():
            if value is None:
                display.append(0.0)
                continue
            if _plane_of(value, name) != plane:
                raise u.UnitsError(f"{name} ({value.unit}) is not in the {plane} plane of dx ({unit})")
            display.append(_scalar(value, unit, name))
        for name, step in zip(("dx", "dy"), display[:2], strict=True):
            if not (math.isfinite(step) and step > 0):
                raise ValueError(f"{name} must be positive and finite, got {step} {unit}")
        if not all(math.isfinite(offset) for offset in display[2:]):
            raise ValueError("the centre (x0, y0) must be finite")
        scale = float(to_si(1.0 * unit))
        self._set(nx, ny, plane, unit, tuple(display), tuple(value * scale for value in display))

    def _set(
        self,
        nx: int,
        ny: int,
        plane: Plane,
        unit: u.UnitBase,
        display: tuple[float, ...],
        si: tuple[float, ...],
    ) -> None:
        for name, value in (
            ("_nx", nx),
            ("_ny", ny),
            ("_plane", plane),
            ("_unit", unit),
            ("_display", display),
            ("_si", si),
            ("_cache", {}),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_extent(
        cls,
        nx: int,
        ny: int | None = None,
        *,
        width: u.Quantity,
        height: u.Quantity | None = None,
        x0: u.Quantity | None = None,
        y0: u.Quantity | None = None,
    ) -> Grid:
        """Build a grid of ``nx × ny`` pixels covering ``width × height`` (``height`` defaults to width)."""
        nx = _size(nx, "nx")
        ny = nx if ny is None else _size(ny, "ny")
        height = width if height is None else height
        return cls(nx, ny, dx=width / nx, dy=height / ny, x0=x0, y0=y0)

    # --- Immutability and value semantics ---------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: grids are immutable values."""
        raise AttributeError("Grid objects are immutable; use with_center() or build a new grid")

    def _key(self) -> tuple[Any, ...]:
        return (self._nx, self._ny, self._plane, *self._si)

    def __eq__(self, other: object) -> bool:
        """Compare the sampling in SI units (the display unit does not matter)."""
        if not isinstance(other, Grid):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash consistent with equality."""
        return hash(self._key())

    def __repr__(self) -> str:
        """Show the sampling in the display unit."""
        dx, dy, x0, y0 = self._display
        return (
            f"Grid(nx={self._nx}, ny={self._ny}, dx={dx:g} {self._unit}, dy={dy:g} {self._unit}, "
            f"x0={x0:g} {self._unit}, y0={y0:g} {self._unit})"
        )

    # --- Sizes and steps --------------------------------------------------------------------------

    @property
    def nx(self) -> int:
        """Number of pixels along x (columns)."""
        return self._nx

    @property
    def ny(self) -> int:
        """Number of pixels along y (rows)."""
        return self._ny

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape ``(ny, nx)``."""
        return (self._ny, self._nx)

    @property
    def size(self) -> int:
        """Number of pixels."""
        return self._nx * self._ny

    @property
    def plane(self) -> Plane:
        """``"spatial"`` (lengths) or ``"angular"`` (angles)."""
        return self._plane

    @property
    def unit(self) -> u.UnitBase:
        """Unit used to display the quantities of the grid (the unit of ``dx``)."""
        return self._unit

    def _quantity(self, value: Any) -> u.Quantity:
        return u.Quantity(value, self._unit)

    @property
    def dx(self) -> u.Quantity:
        """Step along x."""
        return self._quantity(self._display[0])

    @property
    def dy(self) -> u.Quantity:
        """Step along y."""
        return self._quantity(self._display[1])

    @property
    def x0(self) -> u.Quantity:
        """The x coordinate of the central pixel."""
        return self._quantity(self._display[2])

    @property
    def y0(self) -> u.Quantity:
        """The y coordinate of the central pixel."""
        return self._quantity(self._display[3])

    @property
    def dx_si(self) -> float:
        """Step along x in SI units (m or rad)."""
        return self._si[0]

    @property
    def dy_si(self) -> float:
        """Step along y in SI units (m or rad)."""
        return self._si[1]

    @property
    def x0_si(self) -> float:
        """The x coordinate of the centre in SI units."""
        return self._si[2]

    @property
    def y0_si(self) -> float:
        """The y coordinate of the centre in SI units."""
        return self._si[3]

    @property
    def center_index(self) -> tuple[int, int]:
        """Indices ``(iy, ix) = (ny // 2, nx // 2)`` of the central pixel."""
        return (self._ny // 2, self._nx // 2)

    @property
    def extent(self) -> tuple[u.Quantity, u.Quantity]:
        """Width and height ``(nx·dx, ny·dy)``."""
        dx, dy = self._display[:2]
        return self._quantity(self._nx * dx), self._quantity(self._ny * dy)

    @property
    def bounds(self) -> tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
        """Outer pixel edges ``(xmin, xmax, ymin, ymax)``, e.g. for ``imshow(extent=...)``."""
        dx, dy, x0, y0 = self._display
        cx, cy = self._nx // 2, self._ny // 2
        return (
            self._quantity((-cx - 0.5) * dx + x0),
            self._quantity((self._nx - 1 - cx + 0.5) * dx + x0),
            self._quantity((-cy - 0.5) * dy + y0),
            self._quantity((self._ny - 1 - cy + 0.5) * dy + y0),
        )

    @property
    def pixel_area(self) -> u.Quantity:
        """Area ``dx·dy`` of a pixel (m² for a spatial plane, rad² = sr for an angular plane)."""
        return self.dx * self.dy

    # --- Coordinates ------------------------------------------------------------------------------

    @property
    def x(self) -> u.Quantity:
        """Pixel centres along x, as a NumPy quantity in :attr:`unit`."""
        dx, _, x0, _ = self._display
        return self._quantity((np.arange(self._nx) - self._nx // 2) * dx + x0)

    @property
    def y(self) -> u.Quantity:
        """Pixel centres along y, as a NumPy quantity in :attr:`unit`."""
        _, dy, _, y0 = self._display
        return self._quantity((np.arange(self._ny) - self._ny // 2) * dy + y0)

    def _cached(self, kind: str, backend: str | None, build: Any) -> Any:
        selected = get(backend)
        real, _ = dtypes(backend=selected.name)
        key = (kind, selected.name, str(real))
        if key not in self._cache:
            self._cache[key] = build(selected, real)
        return self._cache[key]

    def axes(self, backend: str | None = None) -> tuple[Any, Any]:
        """Return the 1D SI coordinates ``(x, y)`` of the pixel centres as backend arrays.

        Parameters
        ----------
        backend
            Backend name; defaults to the current backend. The dtype follows ``lumos.config.precision``.
        """

        def build(selected: Any, real: Any) -> tuple[Any, Any]:
            dx, dy, x0, y0 = self._si
            x = (np.arange(self._nx) - self._nx // 2) * dx + x0
            y = (np.arange(self._ny) - self._ny // 2) * dy + y0
            return selected.asarray(x, dtype=real), selected.asarray(y, dtype=real)

        return self._cached("axes", backend, build)

    def coords(self, backend: str | None = None) -> tuple[Any, Any]:
        """Return the 2D SI coordinates ``(X, Y)`` of the pixel centres, each of shape ``(ny, nx)``.

        ``X[j, i] = x_i`` and ``Y[j, i] = y_j``. The arrays are computed on first use and cached per
        backend and precision.
        """

        def build(selected: Any, real: Any) -> tuple[Any, Any]:
            x, y = self.axes(selected.name)
            xp = selected.xp
            return xp.broadcast_to(x[None, :], self.shape), xp.broadcast_to(y[:, None], self.shape)

        return self._cached("coords", backend, build)

    def polar(self, backend: str | None = None) -> tuple[Any, Any]:
        """Return the polar SI coordinates ``(r, θ)`` of the pixel centres, θ measured from +x towards +y."""

        def build(selected: Any, real: Any) -> tuple[Any, Any]:
            xx, yy = self.coords(selected.name)
            xp = selected.xp
            return xp.sqrt(xx**2 + yy**2), xp.atan2(yy, xx)

        return self._cached("polar", backend, build)

    def index(self, x: u.Quantity, y: u.Quantity) -> tuple[Any, Any]:
        r"""Return the fractional pixel indices ``(ix, iy)`` of the point(s) ``(x, y)``.

        This inverts :math:`x_i = (i - N_x // 2)\,\Delta x + x_0`; integer results are pixel centres.
        """
        si = _SI_UNITS[self._plane]
        for name, value in (("x", x), ("y", y)):
            if _plane_of(value, name) != self._plane:
                raise u.UnitsError(f"{name} is not in the {self._plane} plane of the grid")
        dx, dy, x0, y0 = self._si
        x_si: Any = x.to_value(si)
        y_si: Any = y.to_value(si)
        ix = (x_si - x0) / dx + self._nx // 2
        iy = (y_si - y0) / dy + self._ny // 2
        return ix, iy

    # --- Derived grids ------------------------------------------------------------------------------

    def with_center(self, x0: u.Quantity, y0: u.Quantity) -> Grid:
        """Return the same sampling around a new centre ``(x0, y0)``."""
        return Grid(self._nx, self._ny, dx=self.dx, dy=self.dy, x0=x0, y0=y0)

    def _mapped(self, focal_length: u.Quantity, *, to: Plane) -> Grid:
        source: Plane = "angular" if to == "spatial" else "spatial"
        if self._plane != source:
            raise ValueError(f"to_{to}() needs a grid in the {source} plane, this one is {self._plane}")
        f = float(to_si(focal_length, u.m))
        if not (math.isfinite(f) and f > 0):
            raise ValueError("focal_length must be positive and finite")
        factor = f if to == "spatial" else 1.0 / f
        dx, dy, x0, y0 = (value * factor * _SI_UNITS[to] for value in self._si)
        return Grid(self._nx, self._ny, dx=dx, dy=dy, x0=x0, y0=y0)

    def to_spatial(self, focal_length: u.Quantity) -> Grid:
        """Map an angular grid to the focal plane of a lens of focal length ``f``: ``x = f·θ``."""
        return self._mapped(focal_length, to="spatial")

    def to_angular(self, focal_length: u.Quantity) -> Grid:
        """Map a focal-plane grid to angles seen through a lens of focal length ``f``: ``θ = x / f``."""
        return self._mapped(focal_length, to="angular")

    # --- Serialisation ----------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Encode the grid as a strict-JSON-compatible dictionary, keeping the display unit."""
        dx, dy, x0, y0 = (self._quantity(value) for value in self._display)
        return {
            "nx": self._nx,
            "ny": self._ny,
            "dx": quantity_to_json(dx),
            "dy": quantity_to_json(dy),
            "x0": quantity_to_json(x0),
            "y0": quantity_to_json(y0),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Grid:
        """Decode a dictionary produced by :meth:`to_json`."""
        return cls(
            data["nx"],
            data["ny"],
            dx=quantity_from_json(data["dx"]),
            dy=quantity_from_json(data["dy"]),
            x0=quantity_from_json(data["x0"]),
            y0=quantity_from_json(data["y0"]),
        )


class GridSetting(Setting):
    """A setting holding a :class:`Grid` or ``None`` (saved in ``.lumos`` files with :meth:`Grid.to_json`)."""

    def __init__(self, *, default: Grid | None = None, doc: str = "") -> None:
        super().__init__(object, default=default, doc=doc)

    def validate(self, value: Any) -> tuple[Any, Any]:
        """Accept a :class:`Grid` or ``None``."""
        if value is not None and not isinstance(value, Grid):
            raise TypeError(f"{self.name}: expected a lumos.fields.Grid or None, got {type(value).__name__}")
        return value, value

    def encode(self, value: Any) -> Any:
        """Return the JSON form of the grid."""
        return None if value is None else value.to_json()

    def decode(self, data: Any) -> Any:
        """Return the grid of a JSON form."""
        return None if data is None else Grid.from_json(data)
