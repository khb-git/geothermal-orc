"""Heat-exchanger analysis: LMTD, counter-flow pinch and T-Q profiles.

The central routine, :func:`counterflow_profile`, discretises a counter-flow
exchanger by *duty* (heat transferred) so that phase change on the working-fluid
side -- preheating, evaporation, optional superheat -- is resolved correctly.
The minimum approach over the profile is the pinch; in subcritical ORC
evaporators it almost always sits at the working-fluid bubble point rather than
at either terminal, which is exactly why a terminal-only ``LMTD`` check is not
sufficient and a swept profile is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from CoolProp.CoolProp import PropsSI


def lmtd(dT1: float, dT2: float) -> float:
    """Log-mean temperature difference for terminal approaches ``dT1``/``dT2``.

    Falls back to the arithmetic mean when the two approaches are nearly equal
    (the LMTD expression is removably singular there).  Raises if either
    approach is non-positive, which signals a temperature cross.
    """
    if dT1 <= 0.0 or dT2 <= 0.0:
        raise ValueError(f"non-positive approach (dT1={dT1:.3f}, dT2={dT2:.3f}): "
                         "temperature cross / infeasible exchanger")
    if math.isclose(dT1, dT2, rel_tol=1e-6):
        return 0.5 * (dT1 + dT2)
    return (dT1 - dT2) / math.log(dT1 / dT2)


@dataclass
class HeatExchangerProfile:
    """Swept counter-flow profile aligned on cumulative duty.

    Attributes
    ----------
    duty : ndarray
        Cumulative duty (W) from the cold-fluid inlet, ascending, length N+1.
    T_hot, T_cold : ndarray
        Hot- and cold-side temperatures (K) at each duty station.
    Q_total : float
        Total exchanger duty (W).
    pinch : float
        Minimum hot-cold approach (K) over the profile.
    pinch_duty : float
        Duty (W) at which the pinch occurs.
    """

    duty: np.ndarray
    T_hot: np.ndarray
    T_cold: np.ndarray
    Q_total: float
    pinch: float
    pinch_duty: float

    @property
    def feasible(self) -> bool:
        """True if the hot stream stays above the cold stream everywhere."""
        return self.pinch > 0.0

    def lmtd(self) -> float:
        """Profile LMTD using the two terminal approaches."""
        return lmtd(self.T_hot[-1] - self.T_cold[-1], self.T_hot[0] - self.T_cold[0])


def counterflow_profile(
    hot_fluid: str,
    m_hot: float,
    P_hot: float,
    T_hot_in: float,
    cold_fluid: str,
    m_cold: float,
    P_cold: float,
    h_cold_in: float,
    h_cold_out: float,
    n: int = 60,
) -> HeatExchangerProfile:
    """Build a counter-flow T-Q profile from stream definitions.

    The cold stream is defined by its enthalpy span ``[h_cold_in, h_cold_out]``
    (so phase change is handled implicitly); the hot stream gives up the matching
    duty.  Temperatures are obtained from CoolProp at constant pressure on each
    side, which captures the curvature of the two-phase region.

    Returns a :class:`HeatExchangerProfile`.  No feasibility exception is raised
    here -- inspect ``.pinch`` / ``.feasible`` -- so the routine can be used
    inside a root-find that brackets infeasible designs.
    """
    if h_cold_out <= h_cold_in:
        raise ValueError("cold stream must gain enthalpy (h_cold_out > h_cold_in)")

    Q_total = m_cold * (h_cold_out - h_cold_in)          # W
    duty = np.linspace(0.0, Q_total, n + 1)              # from cold inlet

    # Cold side: enthalpy rises with cumulative duty.
    h_cold = h_cold_in + duty / m_cold
    T_cold = np.array([PropsSI("T", "P", P_cold, "H", float(h), cold_fluid)
                       for h in h_cold])

    # Hot side (counter-flow): hot enters at the cold-outlet end (duty = Q_total)
    # with T_hot_in, and has surrendered (Q_total - duty) by each station.
    h_hot_in = PropsSI("H", "T", T_hot_in, "P", P_hot, hot_fluid)
    h_hot = h_hot_in - (Q_total - duty) / m_hot
    T_hot = np.array([PropsSI("T", "P", P_hot, "H", float(h), hot_fluid)
                      for h in h_hot])

    approach = T_hot - T_cold
    i_min = int(np.argmin(approach))
    return HeatExchangerProfile(
        duty=duty,
        T_hot=T_hot,
        T_cold=T_cold,
        Q_total=Q_total,
        pinch=float(approach[i_min]),
        pinch_duty=float(duty[i_min]),
    )


def required_area(Q_total: float, U: float, dT_lm: float) -> float:
    """Heat-transfer area (m^2) from ``Q = U A dT_lm``."""
    if dT_lm <= 0.0:
        raise ValueError("non-positive LMTD")
    return Q_total / (U * dT_lm)
