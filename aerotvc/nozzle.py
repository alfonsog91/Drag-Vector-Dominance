"""Choked, quasi-one-dimensional isentropic perfect-gas nozzle model."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _gas(g):
    if not np.isfinite(g) or g <= 1:
        raise ValueError("g must be finite and greater than one")


def temperature_ratio(mach, g=1.4):
    _gas(g)
    _positive(mach, "Mach number")
    return 1 / (1 + (g - 1) * mach**2 / 2)


def pressure_ratio(mach, g=1.4):
    return temperature_ratio(mach, g) ** (g / (g - 1))


def area_ratio(mach, g=1.4):
    _gas(g)
    _positive(mach, "Mach number")
    return ((2 / (g + 1)) * (1 + (g - 1) * mach**2 / 2)) ** (
        (g + 1) / (2 * (g - 1))) / mach


def mach_from_area_ratio(ratio, g=1.4, branch="supersonic"):
    _gas(g)
    if not np.isfinite(ratio) or ratio < 1:
        raise ValueError("area ratio must be finite and at least one")
    if branch not in ("supersonic", "subsonic"):
        raise ValueError("branch must be supersonic or subsonic")
    if ratio == 1:
        return 1.0
    # Solve in log(M) so large ratios and tiny subsonic Mach numbers retain
    # relative accuracy without an arbitrary upper Mach cutoff.
    exponent = (g + 1) / (2 * (g - 1))

    def residual(log_m):
        log_area = (exponent * (np.log(2 / (g + 1))
                    + np.logaddexp(0, np.log((g - 1) / 2) + 2 * log_m))
                    - log_m)
        return log_area - np.log(ratio)

    end = 1.0 if branch == "supersonic" else -1.0
    while residual(end) < 0:
        end *= 2
    lo, hi = sorted((0.0, end))
    return float(np.exp(brentq(residual, lo, hi, xtol=1e-14)))


@dataclass(frozen=True)
class NozzleResult:
    mach: float
    mdot: float
    pe: float
    te: float
    ve: float
    thrust: float
    cf: float
    isp: float
    separation_risk: bool


@dataclass(frozen=True)
class Nozzle:
    A_t: float
    A_exit: float
    g: float = 1.4
    R: float = 287.0

    def __post_init__(self):
        _positive(self.A_t, "throat area")
        _positive(self.A_exit, "exit area")
        _positive(self.R, "gas constant")
        _gas(self.g)
        if self.A_exit < self.A_t:
            raise ValueError("exit area must be at least the throat area")

    def solve(self, p0, T0, pa=0.0):
        _positive(p0, "stagnation pressure")
        _positive(T0, "stagnation temperature")
        if not np.isfinite(pa) or pa < 0:
            raise ValueError("ambient pressure must be finite and nonnegative")
        g = self.g
        mach = mach_from_area_ratio(self.A_exit / self.A_t, g)
        te = T0 * temperature_ratio(mach, g)
        pe = p0 * pressure_ratio(mach, g)
        mdot = self.A_t * p0 * np.sqrt(g / (self.R * T0)) * (
            2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        ve = mach * np.sqrt(g * self.R * te)
        thrust = mdot * ve + (pe - pa) * self.A_exit
        return NozzleResult(mach, mdot, pe, te, ve, thrust,
                            thrust / (p0 * self.A_t), thrust / (mdot * 9.80665),
                            bool(pa > 0 and pe / pa < 0.4))
