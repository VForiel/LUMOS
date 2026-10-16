"""Detected and processed data with units, axes, mask and uncertainties (REQ-SIG-9).

A :class:`DataProduct` is what a detector or a processing block produces: an image, a cube, a time series or
scalars. It carries

- ``data``: a NumPy quantity (``adu``, ``electron``, ``photon``, ``W``...);
- ``axes``: one named 1D coordinate quantity per data axis (pixels by default);
- ``mask``: optional booleans, ``True`` for invalid samples;
- ``uncertainty``: optional standard deviations, in the unit of the data;
- ``meta``: acquisition metadata (strings, numbers, booleans, quantities, lists of them).

Sums and differences of independent products add their uncertainties in quadrature, and scaling by a
constant scales the uncertainty by its absolute value (JCGM 100:2008, *GUM*, §5.1).

Files
-----
- FITS (:meth:`DataProduct.to_fits`): the data in the primary HDU with ``BUNIT`` and a linear WCS
  (``CTYPE``/``CUNIT``/``CRPIX``/``CRVAL``/``CDELT``) for every regularly sampled axis, then the extensions
  ``MASK``, ``UNCERT``, ``AXIS<i>`` (exact coordinates) and ``META`` (JSON).
- NPZ (:meth:`DataProduct.to_npz`): plain arrays and a JSON header, readable with ``allow_pickle=False``.
- HDF5 (:meth:`DataProduct.to_hdf5`, needs ``h5py``): datasets and a JSON attribute.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy import units as u
from astropy.io import fits

from lumos.core.signal import DataSignal
from lumos.fields.grid import Grid
from lumos.units import quantity_from_json, quantity_to_json

__all__ = ["DataProduct"]

_QUANTITY_TAG = "__quantity__"


def _meta_to_json(value: Any) -> Any:
    if isinstance(value, u.Quantity):
        return {_QUANTITY_TAG: quantity_to_json(value)}
    if isinstance(value, Mapping):
        return {str(key): _meta_to_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_meta_to_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"metadata values must be JSON-compatible or quantities, got {type(value).__name__}")


def _meta_from_json(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_QUANTITY_TAG}:
            return quantity_from_json(value[_QUANTITY_TAG])
        return {key: _meta_from_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_meta_from_json(item) for item in value]
    return value


def _unit_string(unit: u.UnitBase) -> str:
    try:
        return unit.to_string("fits")
    except (ValueError, u.UnitsError):
        return unit.to_string()


def _parse_unit(text: str) -> u.UnitBase:
    try:
        return cast(u.UnitBase, u.Unit(text, format="fits"))
    except ValueError:
        return cast(u.UnitBase, u.Unit(text))


def _axis_unit(values: u.Quantity) -> str:
    return cast(u.UnitBase, values.unit).to_string()


def _linear_step(values: np.ndarray) -> float | None:
    """Step of a regularly sampled axis, or ``None``."""
    if values.size == 1:
        return 1.0
    steps = np.diff(values)
    if np.allclose(steps, steps[0], rtol=1e-12, atol=0.0) and steps[0] != 0:
        return float(steps[0])
    return None


class DataProduct(DataSignal):
    """Detected or processed data with units, physical axes, mask, uncertainties and metadata.

    Parameters
    ----------
    data
        A quantity (use ``u.dimensionless_unscaled`` for pure numbers).
    axes
        Mapping from axis names to 1D coordinate quantities, one per data axis, in the order of the data
        axes. Defaults to pixel indices named ``axis0``, ``axis1``...
    mask
        Booleans of the shape of ``data``; ``True`` marks invalid samples.
    uncertainty
        Standard deviations, a quantity of the shape of ``data`` convertible to its unit.
    meta
        Acquisition metadata.
    """

    __slots__ = ("_axes", "_data", "_mask", "_meta", "_uncertainty")

    def __init__(
        self,
        data: u.Quantity,
        *,
        axes: Mapping[str, u.Quantity] | None = None,
        mask: Any = None,
        uncertainty: u.Quantity | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(data, u.Quantity):
            raise TypeError(
                "data must be an astropy Quantity (use u.dimensionless_unscaled for pure numbers)"
            )
        values = u.Quantity(np.asarray(data.value), data.unit)
        shape = values.shape
        if axes is None:
            axes = {f"axis{i}": np.arange(n) * u.pix for i, n in enumerate(shape)}
        if len(axes) != len(shape):
            raise ValueError(f"one axis per data dimension is needed: {len(shape)} expected, got {len(axes)}")
        checked: dict[str, u.Quantity] = {}
        for (name, coordinates), n in zip(axes.items(), shape, strict=True):
            if not isinstance(coordinates, u.Quantity):
                raise TypeError(f"axis {name!r} must be a Quantity")
            coordinates = cast(u.Quantity, np.atleast_1d(coordinates))
            if coordinates.shape != (n,):
                raise ValueError(f"axis {name!r} has shape {coordinates.shape}, the data needs ({n},)")
            checked[str(name)] = coordinates
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"the mask shape {mask.shape} differs from the data shape {shape}")
        if uncertainty is not None:
            if not isinstance(uncertainty, u.Quantity):
                raise TypeError("uncertainty must be a Quantity")
            uncertainty = cast(u.Quantity, uncertainty.to(values.unit))
            if uncertainty.shape != shape:
                raise ValueError(
                    f"the uncertainty shape {uncertainty.shape} differs from the data shape {shape}"
                )
        self._data = values
        self._axes = checked
        self._mask = mask
        self._uncertainty = uncertainty
        self._meta = dict(meta or {})
        _meta_to_json(self._meta)  # validate early

    @classmethod
    def from_grid(
        cls,
        data: u.Quantity,
        grid: Grid,
        *,
        leading_axes: Mapping[str, u.Quantity] | None = None,
        **kwargs: Any,
    ) -> DataProduct:
        """Build a product whose last two axes are the ``y`` and ``x`` coordinates of ``grid``."""
        axes = dict(leading_axes or {})
        axes["y"] = grid.y
        axes["x"] = grid.x
        return cls(data, axes=axes, **kwargs)

    # --- Attributes -------------------------------------------------------------------------------

    @property
    def data(self) -> u.Quantity:
        """The data quantity."""
        return self._data

    @property
    def unit(self) -> u.UnitBase:
        """Unit of the data."""
        return cast(u.UnitBase, self._data.unit)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data."""
        return tuple(self._data.shape)

    @property
    def axes(self) -> dict[str, u.Quantity]:
        """Coordinates of each axis (a copy of the mapping)."""
        return dict(self._axes)

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the axes, in data order."""
        return tuple(self._axes)

    @property
    def mask(self) -> np.ndarray | None:
        """Invalid-sample mask, if any."""
        return self._mask

    @property
    def uncertainty(self) -> u.Quantity | None:
        """Standard deviations in the data unit, if any."""
        return self._uncertainty

    @property
    def meta(self) -> dict[str, Any]:
        """Acquisition metadata (a copy)."""
        return dict(self._meta)

    def __repr__(self) -> str:
        """Summarise the product."""
        return f"DataProduct(shape={self.shape}, unit={self.unit}, axes={self.axis_names})"

    # --- Arithmetic -------------------------------------------------------------------------------

    def _same_axes(self, other: DataProduct) -> None:
        if other.axis_names != self.axis_names or any(
            not np.allclose(other._axes[name].to_value(self._axes[name].unit), self._axes[name].value)
            for name in self.axis_names
        ):
            raise ValueError("cannot combine data products with different axes")

    def _combine(self, other: DataProduct, sign: float) -> DataProduct:
        self._same_axes(other)
        data = cast(u.Quantity, self._data + sign * other._data.to(self.unit))
        if self._uncertainty is None and other._uncertainty is None:
            uncertainty = None
        else:
            a = self._uncertainty.value if self._uncertainty is not None else 0.0
            b = other._uncertainty.to_value(self.unit) if other._uncertainty is not None else 0.0
            uncertainty = np.sqrt(np.asarray(a) ** 2 + np.asarray(b) ** 2) * np.ones(self.shape) * self.unit
        masks = [m for m in (self._mask, other._mask) if m is not None]
        mask = np.logical_or.reduce(masks) if masks else None
        return DataProduct(data, axes=self._axes, mask=mask, uncertainty=uncertainty, meta=self._meta)

    def __add__(self, other: Any) -> DataProduct:
        """Sum of independent products (uncertainties in quadrature, masks combined)."""
        if not isinstance(other, DataProduct):
            return NotImplemented
        return self._combine(other, 1.0)

    def __sub__(self, other: Any) -> DataProduct:
        """Difference of independent products (uncertainties in quadrature, masks combined)."""
        if not isinstance(other, DataProduct):
            return NotImplemented
        return self._combine(other, -1.0)

    def __mul__(self, other: Any) -> DataProduct:
        """Product by an exact constant (number, array or quantity, e.g. a gain)."""
        if isinstance(other, DataProduct):
            return NotImplemented
        data = cast(u.Quantity, self._data * other)
        uncertainty = (
            None if self._uncertainty is None else cast(u.Quantity, self._uncertainty * np.abs(other))
        )
        return DataProduct(data, axes=self._axes, mask=self._mask, uncertainty=uncertainty, meta=self._meta)

    __rmul__ = __mul__

    # --- JSON -------------------------------------------------------------------------------------

    def _header(self) -> dict[str, Any]:
        return {
            "unit": self.unit.to_string(),
            "axes": [{"name": name, "unit": _axis_unit(values)} for name, values in self._axes.items()],
            "meta": _meta_to_json(self._meta),
        }

    def to_json(self) -> dict[str, Any]:
        """Encode the product as a strict-JSON-compatible dictionary."""
        return {
            "data": quantity_to_json(self._data),
            "axes": {name: quantity_to_json(values) for name, values in self._axes.items()},
            "mask": None if self._mask is None else quantity_to_json(u.Quantity(self._mask.astype(np.uint8))),
            "uncertainty": None if self._uncertainty is None else quantity_to_json(self._uncertainty),
            "meta": _meta_to_json(self._meta),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> DataProduct:
        """Decode a dictionary produced by :meth:`to_json`."""
        mask = None if data.get("mask") is None else quantity_from_json(data["mask"]).value.astype(bool)
        uncertainty = None if data.get("uncertainty") is None else quantity_from_json(data["uncertainty"])
        return cls(
            quantity_from_json(data["data"]),
            axes={name: quantity_from_json(values) for name, values in data["axes"].items()},
            mask=mask,
            uncertainty=uncertainty,
            meta=_meta_from_json(data.get("meta", {})),
        )

    # --- FITS -------------------------------------------------------------------------------------

    def to_fits(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write a FITS file (see the module documentation for its layout)."""
        primary = fits.PrimaryHDU(np.asarray(self._data.value))
        header = primary.header
        header["BUNIT"] = _unit_string(self.unit)
        ndim = len(self.shape)
        for i, (name, values) in enumerate(self._axes.items()):
            k = ndim - i  # FITS axis 1 is the last NumPy axis
            step = _linear_step(np.asarray(values.value, dtype=float))
            if step is not None:
                header[f"CTYPE{k}"] = name[:68]
                header[f"CUNIT{k}"] = _unit_string(cast(u.UnitBase, values.unit))
                header[f"CRPIX{k}"] = 1.0
                header[f"CRVAL{k}"] = float(values.value[0])
                header[f"CDELT{k}"] = step
        hdus: list[Any] = [primary]
        if self._mask is not None:
            hdus.append(fits.ImageHDU(self._mask.astype(np.uint8), name="MASK"))
        if self._uncertainty is not None:
            unc = fits.ImageHDU(np.asarray(self._uncertainty.value), name="UNCERT")
            unc.header["BUNIT"] = _unit_string(self.unit)
            hdus.append(unc)
        for i, (name, values) in enumerate(self._axes.items()):
            axis = fits.ImageHDU(np.asarray(values.value, dtype=float), name=f"AXIS{i}")
            axis.header["AXNAME"] = name
            axis.header["AXUNIT"] = _axis_unit(values)
            hdus.append(axis)
        meta = json.dumps(self._header()).encode("ascii")
        hdus.append(fits.ImageHDU(np.frombuffer(meta, dtype=np.uint8), name="META"))
        fits.HDUList(hdus).writeto(Path(path), overwrite=overwrite)

    @classmethod
    def from_fits(cls, path: str | Path) -> DataProduct:
        """Read a FITS file written by :meth:`to_fits` (or a plain image with ``BUNIT``)."""
        hdul: Any
        with fits.open(Path(path)) as hdul:
            primary = hdul[0]
            names = {hdu.name for hdu in hdul}
            header_json: dict[str, Any] = {}
            if "META" in names:
                header_json = json.loads(bytes(np.asarray(hdul["META"].data, dtype=np.uint8)).decode("ascii"))
            unit = (
                u.Unit(header_json["unit"])
                if "unit" in header_json
                else _parse_unit(primary.header.get("BUNIT", ""))
            )
            data = u.Quantity(np.array(primary.data), unit)
            axes: dict[str, u.Quantity] | None = None
            if any(name.startswith("AXIS") for name in names):
                axes = {}
                for i in range(data.ndim):
                    hdu = hdul[f"AXIS{i}"]
                    axes[str(hdu.header["AXNAME"])] = u.Quantity(
                        np.array(hdu.data), u.Unit(hdu.header["AXUNIT"])
                    )
            mask = np.array(hdul["MASK"].data).astype(bool) if "MASK" in names else None
            uncertainty = u.Quantity(np.array(hdul["UNCERT"].data), unit) if "UNCERT" in names else None
        return cls(
            data,
            axes=axes,
            mask=mask,
            uncertainty=uncertainty,
            meta=_meta_from_json(header_json.get("meta", {})),
        )

    # --- NPZ and HDF5 -----------------------------------------------------------------------------

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays = {"data": np.asarray(self._data.value)}
        for i, values in enumerate(self._axes.values()):
            arrays[f"axis{i}"] = np.asarray(values.value)
        if self._mask is not None:
            arrays["mask"] = self._mask
        if self._uncertainty is not None:
            arrays["uncertainty"] = np.asarray(self._uncertainty.value)
        return arrays

    @classmethod
    def _from_arrays(cls, arrays: Mapping[str, np.ndarray], header: Mapping[str, Any]) -> DataProduct:
        unit = u.Unit(header["unit"])
        axes = {
            axis["name"]: u.Quantity(np.asarray(arrays[f"axis{i}"]), u.Unit(axis["unit"]))
            for i, axis in enumerate(header["axes"])
        }
        return cls(
            u.Quantity(np.asarray(arrays["data"]), unit),
            axes=axes,
            mask=np.asarray(arrays["mask"]) if "mask" in arrays else None,
            uncertainty=u.Quantity(np.asarray(arrays["uncertainty"]), unit)
            if "uncertainty" in arrays
            else None,
            meta=_meta_from_json(header.get("meta", {})),
        )

    def to_npz(self, path: str | Path) -> None:
        """Write a NumPy ``.npz`` archive of plain arrays and a JSON header (no pickled objects)."""
        arrays: dict[str, Any] = {"header": np.array(json.dumps(self._header())), **self._arrays()}
        np.savez(Path(path), allow_pickle=False, **arrays)

    @classmethod
    def from_npz(cls, path: str | Path) -> DataProduct:
        """Read a ``.npz`` archive written by :meth:`to_npz` (``allow_pickle=False``)."""
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        header = json.loads(str(arrays.pop("header")))
        return cls._from_arrays(arrays, header)

    def to_hdf5(self, path: str | Path) -> None:
        """Write an HDF5 file (needs ``h5py``)."""
        h5py = _require_h5py()
        with h5py.File(Path(path), "w") as f:
            for key, array in self._arrays().items():
                f.create_dataset(key, data=array)
            f.attrs["lumos_header"] = json.dumps(self._header())

    @classmethod
    def from_hdf5(cls, path: str | Path) -> DataProduct:
        """Read an HDF5 file written by :meth:`to_hdf5` (needs ``h5py``)."""
        h5py = _require_h5py()
        with h5py.File(Path(path), "r") as f:
            arrays = {key: np.asarray(f[key]) for key in f}
            header = json.loads(str(f.attrs["lumos_header"]))
        return cls._from_arrays(arrays, header)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError:
        raise ImportError("HDF5 files need h5py: pip install 'lumos-maxima[hdf5]'") from None
    return h5py
