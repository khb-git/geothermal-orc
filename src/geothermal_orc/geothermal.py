"""Geothermal resource model: silica chemistry and thermal decline.

Silica
------
Dissolved silica limits how far brine can be cooled before amorphous silica
scales out in heat exchangers and reinjection wells.  Two solubility relations
are used, both of the form ``log10(C) = a/T + b`` with ``C`` in mg/kg (as SiO2)
and ``T`` in kelvin:

* amorphous silica (Fournier & Rowe, 1977): ``log10(C) = -731/T + 4.52``
* quartz: ``log10(C) = -1309/T + 5.19``

In a producing field the deep brine is taken to be in equilibrium with quartz at
reservoir temperature; quartz controls the dissolved load.  At the surface,
amorphous silica controls the scaling limit, so the safe reinjection temperature
is the one at which the (quartz-set) silica load just reaches amorphous
saturation.  The quartz geothermometer of Fournier & Potter (1982, no steam
loss) is provided as the inverse map from silica to reservoir temperature.

Decline
-------
Production temperature decline is modelled as linear or exponential.  Default
annual rates follow Snyder et al. (2017), who found ~0.5 %/yr for binary plants
and ~0.8 %/yr for flash plants from US monthly production reports, with ~90 % of
wells following a linear trend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import brentq

# Snyder et al. (2017) default annual temperature drawdown rates.
DECLINE_RATE_BINARY = 0.005   # 0.5 %/yr
DECLINE_RATE_FLASH = 0.008    # 0.8 %/yr


# --------------------------------------------------------------------------- #
# Silica solubility (mg/kg as SiO2; T in kelvin)
# --------------------------------------------------------------------------- #
def amorphous_silica_solubility(T_K: float) -> float:
    """Amorphous-silica solubility, mg/kg (Fournier & Rowe, 1977)."""
    return 10.0 ** (-731.0 / T_K + 4.52)


def quartz_solubility(T_K: float) -> float:
    """Quartz solubility, mg/kg."""
    return 10.0 ** (-1309.0 / T_K + 5.19)


def quartz_geothermometer(SiO2_mgkg: float) -> float:
    """Reservoir temperature (C) from dissolved silica via the quartz
    (no-steam-loss) geothermometer, Fournier & Potter (1982).

    Valid roughly 25-330 C.
    """
    S = SiO2_mgkg
    return (
        -42.198
        + 0.28831 * S
        - 3.6686e-4 * S ** 2
        + 3.1665e-7 * S ** 3
        + 77.034 * math.log10(S)
    )


def silica_saturation_index(C_mgkg: float, T_K: float) -> float:
    """Amorphous-silica saturation index ``C / C_sat(T)``.

    SI > 1 => supersaturated => scaling risk; SI < 1 => undersaturated.
    """
    return C_mgkg / amorphous_silica_solubility(T_K)


def min_reinjection_temperature(C_mgkg: float, SI_limit: float = 1.0) -> float:
    """Lowest brine temperature (C) before amorphous-silica saturation.

    Solves ``C / C_sat(T) = SI_limit`` for ``T``.  Cooling brine below this
    temperature drives the saturation index above ``SI_limit`` and risks scale.
    """
    target = C_mgkg / SI_limit

    def f(T_K: float) -> float:
        return amorphous_silica_solubility(T_K) - target

    # Solubility rises with T, so bracket from cold to hot.
    T_lo, T_hi = 273.15, 623.15
    if f(T_lo) > 0:          # already soluble even when freezing -> no limit
        return T_lo - 273.15
    if f(T_hi) < 0:          # never soluble in range -> saturated everywhere
        return T_hi - 273.15
    T_root = brentq(f, T_lo, T_hi, xtol=1e-6)
    return T_root - 273.15


# --------------------------------------------------------------------------- #
# Resource
# --------------------------------------------------------------------------- #
@dataclass
class GeothermalResource:
    """A liquid-dominated geothermal resource feeding a binary plant.

    Parameters
    ----------
    T_reservoir_C : float
        Initial production (reservoir) temperature, deg C.
    mass_flow : float
        Brine mass flow, kg/s.
    silica_mgkg : float, optional
        Dissolved silica (mg/kg as SiO2).  If omitted it is set to quartz
        equilibrium at the reservoir temperature.
    decline_rate : float
        Fractional annual temperature decline (default: binary, Snyder 2017).
    decline_mode : {'linear', 'exponential'}
    """

    T_reservoir_C: float
    mass_flow: float
    silica_mgkg: Optional[float] = None
    decline_rate: float = DECLINE_RATE_BINARY
    decline_mode: str = "linear"

    def __post_init__(self) -> None:
        if self.silica_mgkg is None:
            self.silica_mgkg = quartz_solubility(self.T_reservoir_C + 273.15)
        if self.decline_mode not in ("linear", "exponential"):
            raise ValueError("decline_mode must be 'linear' or 'exponential'")

    # ------------------------------------------------------------------ #
    def temperature_at(self, years: float) -> float:
        """Production temperature (C) after ``years`` of operation."""
        T0 = self.T_reservoir_C
        if self.decline_mode == "linear":
            return T0 * (1.0 - self.decline_rate * years)
        return T0 * math.exp(-self.decline_rate * years)

    def min_reinjection_temperature(self, SI_limit: float = 1.0) -> float:
        """Scaling-limited minimum reinjection temperature (C)."""
        return min_reinjection_temperature(self.silica_mgkg, SI_limit)

    def saturation_index_at(self, T_C: float) -> float:
        """Amorphous-silica saturation index if brine is cooled to ``T_C``."""
        return silica_saturation_index(self.silica_mgkg, T_C + 273.15)

    def scaling_safe(self, T_reinjection_C: float, SI_limit: float = 1.0) -> bool:
        """True if reinjecting at ``T_reinjection_C`` avoids amorphous scaling."""
        return self.saturation_index_at(T_reinjection_C) <= SI_limit
