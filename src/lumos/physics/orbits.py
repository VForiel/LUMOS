r"""Keplerian orbits (REQ-SCN-2, REQ-SCN-7).

A bound orbit of semi-major axis :math:`a` and eccentricity :math:`e` around a mass :math:`M` has the period
:math:`P = 2\pi\sqrt{a^3 / GM}` (Kepler's third law). The position at time :math:`t` follows from the mean
anomaly :math:`M = 2\pi (t - t_p)/P`, the eccentric anomaly solving Kepler's equation
:math:`E - e \sin E = M`, the true anomaly :math:`\nu` and the radius :math:`r = a(1 - e\cos E)`
(Murray & Dermott, *Solar System Dynamics*, §2.4).

Orientation: the orbit lies in a plane inclined by :math:`i` about the line of nodes (the x axis rotated by
the longitude of the ascending node :math:`\Omega`), with argument of periapsis :math:`\omega`. The observer
looks down the -z axis (z points towards the observer); :math:`i = 0` is face-on.
"""

from __future__ import annotations

import math

from astropy import units as u

from lumos.units import require_unit, to_si

__all__ = ["kepler_position", "kepler_position_3d", "orbital_period", "orbital_period_si", "solve_kepler"]

_G = 6.67430e-11  # m3 kg-1 s-2, CODATA 2018


def orbital_period_si(a: float, mass: float) -> float:
    """Period in seconds of an orbit of semi-major axis ``a`` (m) around ``mass`` (kg)."""
    return 2 * math.pi * math.sqrt(a**3 / (_G * mass))


def orbital_period(a: u.Quantity, mass: u.Quantity) -> u.Quantity:
    """Period of an orbit of semi-major axis ``a`` around ``mass``."""
    a_si = float(to_si(require_unit(a, u.m, name="a")))
    m_si = float(to_si(require_unit(mass, u.kg, name="mass")))
    return orbital_period_si(a_si, m_si) * u.s


def solve_kepler(mean_anomaly: float, e: float, *, tolerance: float = 1e-14) -> float:
    """Solve Kepler's equation ``E - e sin E = M`` for the eccentric anomaly ``E`` (Newton's method)."""
    if not 0 <= e < 1:
        raise ValueError(f"the eccentricity of a bound orbit is in [0, 1), got {e}")
    m = math.remainder(mean_anomaly, 2 * math.pi)
    eccentric = m if e < 0.8 else math.copysign(math.pi, m)
    for _ in range(100):
        step = (eccentric - e * math.sin(eccentric) - m) / (1 - e * math.cos(eccentric))
        eccentric -= step
        if abs(step) < tolerance:
            break
    return eccentric + (mean_anomaly - m)


def kepler_position_3d(
    t: float,
    *,
    period: float,
    a: float,
    e: float = 0.0,
    inclination: float = 0.0,
    argument_periapsis: float = 0.0,
    longitude_node: float = 0.0,
    time_periapsis: float = 0.0,
) -> tuple[float, float, float]:
    """Return the position ``(x, y, z)`` at time ``t`` (unit of ``a``, radians, z towards the observer)."""
    mean_anomaly = 2 * math.pi * (t - time_periapsis) / period
    eccentric = solve_kepler(mean_anomaly, e)
    nu = 2 * math.atan2(
        math.sqrt(1 + e) * math.sin(eccentric / 2), math.sqrt(1 - e) * math.cos(eccentric / 2)
    )
    r = a * (1 - e * math.cos(eccentric))
    # in the orbital plane, periapsis along x' then rotated by omega
    xo, yo = r * math.cos(nu + argument_periapsis), r * math.sin(nu + argument_periapsis)
    # incline about the line of nodes (x), then rotate the node by Omega about z
    x1, y1, z1 = xo, yo * math.cos(inclination), yo * math.sin(inclination)
    c, s = math.cos(longitude_node), math.sin(longitude_node)
    return c * x1 - s * y1, s * x1 + c * y1, z1


def kepler_position(t: float, **orbit: float) -> tuple[float, float]:
    """Projected position ``(x, y)`` on the sky plane (see :func:`kepler_position_3d`)."""
    x, y, _ = kepler_position_3d(t, **orbit)
    return x, y
