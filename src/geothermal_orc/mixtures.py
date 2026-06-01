"""Zeotropic working-fluid mixtures.

A zeotropic mixture boils and condenses over a **temperature glide** (the bubble
and dew points differ at a given pressure) rather than at a single saturation
temperature.  That glide can be matched to the brine's sensible cooling in the
evaporator and the cooling water's warming in the condenser, shrinking the
average temperature difference — and therefore the exergy destruction — of the
heat exchange.  This is the standard route to higher second-law (utilization)
efficiency from a low-temperature resource.

CoolProp's high-level ``PropsSI`` cannot do P-H / P-S flashes for mixtures, so
this module is built on the flashes that *do* work for mixtures — P-T (single
phase) and P-Q (saturation) — with root-finding for the isentropic pump/turbine
targets.  State numbering matches :mod:`geothermal_orc.cycle`:

    1 condenser outlet (bubble-point liquid)   2 pump outlet
    3 evaporator outlet (dew-point vapour)      4 turbine outlet
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI

from .geothermal import GeothermalResource
from .thermo import specific_flow_exergy, DEAD_STATE_T


def mixture_string(components: Sequence[str], mole_fractions: Sequence[float]) -> str:
    """Build a CoolProp mixture spec, e.g. ``Isobutane[0.7]&Isopentane[0.3]``."""
    if len(components) != len(mole_fractions):
        raise ValueError("components and mole_fractions must be the same length")
    x = np.asarray(mole_fractions, dtype=float)
    if abs(x.sum() - 1.0) > 1e-6 or np.any(x < 0):
        raise ValueError("mole_fractions must be non-negative and sum to 1")
    return "&".join(f"{c}[{xi:.6f}]" for c, xi in zip(components, x))


def bubble_dew(spec: str, P: float) -> Tuple[float, float]:
    """(bubble, dew) temperatures [K] of the mixture at pressure ``P`` [Pa]."""
    return (PropsSI("T", "P", P, "Q", 0.0, spec),
            PropsSI("T", "P", P, "Q", 1.0, spec))


def temperature_glide(spec: str, P: float) -> float:
    """Dew minus bubble temperature [K] at pressure ``P``."""
    Tb, Td = bubble_dew(spec, P)
    return Td - Tb


def _pressure_for_saturation_T(spec: str, T_target: float, Q: float) -> float:
    """Find the pressure at which the mixture's Q-saturation temperature is T.

    Scans a pressure grid (skipping points where the mixture flash fails near
    the critical region), then bisects within the valid, monotonic range."""
    Pcrit = PropsSI("Pcrit", spec)
    grid = np.geomspace(2.0e4, 0.92 * Pcrit, 48)
    Pv: List[float] = []
    Tv: List[float] = []
    for P in grid:
        try:
            Tv.append(PropsSI("T", "P", P, "Q", Q, spec))
            Pv.append(P)
        except Exception:
            continue
    Pv_a, Tv_a = np.array(Pv), np.array(Tv)
    if T_target < Tv_a.min() or T_target > Tv_a.max():
        raise ValueError(
            f"saturation temperature {T_target-273.15:.1f} C is out of the "
            f"workable range for {spec}")
    i = int(np.searchsorted(Tv_a, T_target))
    lo, hi = Pv_a[max(0, i - 1)], Pv_a[min(len(Pv_a) - 1, i)]

    def f(P):
        return PropsSI("T", "P", P, "Q", Q, spec) - T_target

    return float(brentq(f, lo, hi, xtol=1.0, rtol=1e-8))


def _pressure_for_mean_saturation_T(spec: str, T_mean_target: float) -> float:
    """Pressure at which the mean of bubble and dew temperature equals T_mean.

    This is the fair anchor for comparing a gliding mixture against a pure
    fluid that condenses (or boils) at a single temperature: both reject (or
    accept) heat at the same *mean* temperature, so the glide's only effect is
    the improved match to the sensible heat-sink (or source) stream."""
    Pcrit = PropsSI("Pcrit", spec)
    grid = np.geomspace(2.0e4, 0.92 * Pcrit, 48)
    Pv: List[float] = []
    Mv: List[float] = []
    for P in grid:
        try:
            Tb = PropsSI("T", "P", P, "Q", 0.0, spec)
            Td = PropsSI("T", "P", P, "Q", 1.0, spec)
            Mv.append(0.5 * (Tb + Td))
            Pv.append(P)
        except Exception:
            continue
    Pv_a, Mv_a = np.array(Pv), np.array(Mv)
    if T_mean_target < Mv_a.min() or T_mean_target > Mv_a.max():
        raise ValueError(
            f"mean saturation temperature {T_mean_target-273.15:.1f} C is out "
            f"of the workable range for {spec}")
    i = int(np.searchsorted(Mv_a, T_mean_target))
    lo, hi = Pv_a[max(0, i - 1)], Pv_a[min(len(Pv_a) - 1, i)]

    def f(P):
        Tb = PropsSI("T", "P", P, "Q", 0.0, spec)
        Td = PropsSI("T", "P", P, "Q", 1.0, spec)
        return 0.5 * (Tb + Td) - T_mean_target

    return float(brentq(f, lo, hi, xtol=1.0, rtol=1e-8))


def _T_at_Ph(spec: str, P: float, h: float, T_lo: float, T_hi: float) -> float:
    """Single-phase temperature at (P, h) by root-finding on a P-T flash."""
    def f(T):
        return PropsSI("H", "T", T, "P", P, spec) - h
    return float(brentq(f, T_lo, T_hi, xtol=1e-4, rtol=1e-8))


def _s_at_Ph_singlephase(spec: str, P: float, h: float,
                         T_lo: float, T_hi: float) -> float:
    T = _T_at_Ph(spec, P, h, T_lo, T_hi)
    return PropsSI("S", "T", T, "P", P, spec)


@dataclass
class MixtureCycleResult:
    spec: str
    P_evap: float
    P_cond: float
    T_evap_dew: float       # K (turbine inlet)
    T_cond_bubble: float    # K (pump inlet)
    glide_evap: float       # K
    glide_cond: float       # K
    w_pump: float
    w_turbine: float
    w_net: float
    q_in: float
    q_out: float
    eta_th: float
    # resource-coupled (None until solved with a resource)
    m_wf: float = None
    m_brine: float = None
    W_net: float = None
    brine_T_out: float = None
    eta_utilization: float = None
    evaporator_pinch: float = None


class MixtureCycle:
    """A subcritical ORC on a zeotropic mixture, defined by its glide endpoints."""

    def __init__(
        self,
        components: Sequence[str],
        mole_fractions: Sequence[float],
        T_evap_dew_C: float,
        T_cond_mean_C: float = 30.0,
        eta_pump: float = 0.75,
        eta_turbine: float = 0.80,
        brine_fluid: str = "Water",
        P_brine: float = 1.0e6,
        T0: float = DEAD_STATE_T,
    ) -> None:
        self.spec = mixture_string(components, mole_fractions)
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        self.brine_fluid = brine_fluid
        self.P_brine = P_brine
        self.T0 = T0

        self.T_evap_dew = T_evap_dew_C + 273.15
        self.T_cond_mean = T_cond_mean_C + 273.15
        # Turbine inlet is the dew point at P_evap.  The condenser is anchored on
        # its MEAN temperature (fair vs a pure fluid condensing at one T), with
        # the glide free to match the cooling-water warming.
        self.P_evap = _pressure_for_saturation_T(self.spec, self.T_evap_dew, 1.0)
        self.P_cond = _pressure_for_mean_saturation_T(self.spec, self.T_cond_mean)
        self.T_cond_bubble = PropsSI("T", "P", self.P_cond, "Q", 0.0, self.spec)
        if self.P_cond >= self.P_evap:
            raise ValueError("condensing pressure must be below evaporating pressure")

    # ------------------------------------------------------------------ #
    def solve(self) -> MixtureCycleResult:
        spec = self.spec
        # State 1: bubble-point liquid at P_cond.
        h1 = PropsSI("H", "P", self.P_cond, "Q", 0.0, spec)
        s1 = PropsSI("S", "P", self.P_cond, "Q", 0.0, spec)
        # State 3: dew-point vapour at P_evap.
        h3 = PropsSI("H", "P", self.P_evap, "Q", 1.0, spec)
        s3 = PropsSI("S", "P", self.P_evap, "Q", 1.0, spec)
        Tb_evap = PropsSI("T", "P", self.P_evap, "Q", 0.0, spec)
        Td_cond = PropsSI("T", "P", self.P_cond, "Q", 1.0, spec)

        # Pump: isentropic to P_evap (subcooled liquid), then efficiency.
        h2s = self._h_at_Ps_liquid(self.P_evap, s1)
        h2 = h1 + (h2s - h1) / self.eta_pump
        # Turbine: isentropic to P_cond (superheated for a dry mixture), then eff.
        h4s = self._h_at_Ps_vapour(self.P_cond, s3)
        h4 = h3 - self.eta_turbine * (h3 - h4s)

        w_pump = h2 - h1
        w_turbine = h3 - h4
        w_net = w_turbine - w_pump
        q_in = h3 - h2
        q_out = h4 - h1

        return MixtureCycleResult(
            spec=spec, P_evap=self.P_evap, P_cond=self.P_cond,
            T_evap_dew=self.T_evap_dew, T_cond_bubble=self.T_cond_bubble,
            glide_evap=self.T_evap_dew - Tb_evap,
            glide_cond=Td_cond - self.T_cond_bubble,
            w_pump=w_pump, w_turbine=w_turbine, w_net=w_net,
            q_in=q_in, q_out=q_out, eta_th=w_net / q_in,
        )

    # ------- isentropic helpers (root-find since P-S flash is unavailable) --- #
    def _h_at_Ps_liquid(self, P: float, s_target: float) -> float:
        # Subcooled liquid: bubble temperature at P is the upper bound.
        Tb = PropsSI("T", "P", P, "Q", 0.0, self.spec)
        def f(T):
            return PropsSI("S", "T", T, "P", P, self.spec) - s_target
        T = brentq(f, self.T_cond_bubble - 30.0, Tb - 1e-6, xtol=1e-4)
        return PropsSI("H", "T", T, "P", P, self.spec)

    def _h_at_Ps_vapour(self, P: float, s_target: float) -> float:
        # Superheated vapour: dew temperature at P is the lower bound.
        Td = PropsSI("T", "P", P, "Q", 1.0, self.spec)
        def f(T):
            return PropsSI("S", "T", T, "P", P, self.spec) - s_target
        T = brentq(f, Td + 1e-6, Td + 120.0, xtol=1e-4)
        return PropsSI("H", "T", T, "P", P, self.spec)

    # ------------------------------------------------------------------ #
    def _cold_side_profile(self, h2: float, n: int = 60):
        """Evaporator cold-side (T, h) from pump outlet (2) to dew vapour (3).

        Built from P-T flashes in the subcooled-liquid preheat and P-Q flashes
        through the two-phase glide, so the gliding temperature is resolved."""
        spec, P = self.spec, self.P_evap
        h_f = PropsSI("H", "P", P, "Q", 0.0, spec)
        T_f = PropsSI("T", "P", P, "Q", 0.0, spec)
        T2 = _T_at_Ph(spec, P, h2, self.T_cond_bubble - 30.0, T_f - 1e-6)
        n_sub = max(6, n // 3)
        T_sub = np.linspace(T2, T_f, n_sub)
        h_sub = np.array([PropsSI("H", "T", float(T), "P", P, spec) for T in T_sub])
        q = np.linspace(0.0, 1.0, n - n_sub)
        h_tp = np.array([PropsSI("H", "P", P, "Q", float(x), spec) for x in q])
        T_tp = np.array([PropsSI("T", "P", P, "Q", float(x), spec) for x in q])
        h_grid = np.concatenate([h_sub, h_tp])
        T_grid = np.concatenate([T_sub, T_tp])
        order = np.argsort(h_grid)
        return h_grid[order], T_grid[order]

    def solve_with_resource(
        self,
        m_brine: float,
        T_brine_in_C: float,
        pinch_evap: float = 5.0,
        n: int = 60,
    ) -> MixtureCycleResult:
        """Size the mixture cycle to a brine stream (evaporator pinch matched)."""
        base = self.solve()
        spec = self.spec
        h1 = PropsSI("H", "P", self.P_cond, "Q", 0.0, spec)
        h2 = h1 + base.w_pump
        h3 = PropsSI("H", "P", self.P_evap, "Q", 1.0, spec)
        T_brine_in = T_brine_in_C + 273.15

        h_cold_grid_full, T_cold_grid_full = self._cold_side_profile(h2, n=n)
        frac = np.linspace(0.0, 1.0, n + 1)
        h_cold = h2 + frac * (h3 - h2)
        T_cold = np.interp(h_cold, h_cold_grid_full, T_cold_grid_full)
        dh_cold = h3 - h2

        h_hot_in = PropsSI("H", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)

        def pinch_gap(m_wf):
            Q_total = m_wf * dh_cold
            h_hot = h_hot_in - (1.0 - frac) * Q_total / m_brine
            T_hot = np.array([PropsSI("T", "P", self.P_brine, "H", float(h),
                                      self.brine_fluid) for h in h_hot])
            return float(np.min(T_hot - T_cold)) - pinch_evap

        cp_brine = PropsSI("C", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)
        dT_avail = max(T_brine_in - (self.T_evap_dew + pinch_evap), 1.0)
        m_guess = m_brine * cp_brine * dT_avail / max(base.q_in, 1.0)
        m_hi = max(1e-3, 2.0 * m_guess)
        while pinch_gap(m_hi) > 0.0:
            m_hi *= 2.0
            if m_hi > 1e7:
                raise RuntimeError("could not bracket working-fluid flow")
        m_lo = m_hi
        while pinch_gap(m_lo) <= 0.0:
            m_lo *= 0.5
            if m_lo < 1e-9:
                raise RuntimeError("could not bracket working-fluid flow")
        m_wf = brentq(pinch_gap, m_lo, m_hi, xtol=1e-3, rtol=1e-8)

        Q_total = m_wf * dh_cold
        h_hot = h_hot_in - Q_total / m_brine
        brine_T_out = PropsSI("T", "P", self.P_brine, "H", h_hot, self.brine_fluid)

        W_net = m_wf * base.w_net
        s_b_in = PropsSI("S", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)
        e_b_in = specific_flow_exergy(self.brine_fluid, h_hot_in, s_b_in, self.T0)
        eta_util = W_net / (m_brine * e_b_in)

        base.m_wf = m_wf
        base.m_brine = m_brine
        base.W_net = W_net
        base.brine_T_out = brine_T_out
        base.eta_utilization = eta_util
        base.evaporator_pinch = pinch_evap
        return base


def screen_compositions(
    component_light: str,
    component_heavy: str,
    light_fractions: Sequence[float],
    resource: GeothermalResource,
    T_evap_dew_C: float,
    T_cond_mean_C: float = 30.0,
    *,
    pinch_evap: float = 5.0,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
) -> List[MixtureCycleResult]:
    """Sweep mixture composition; pure endpoints are included automatically.

    Returns results (each carries net power and utilization) in input order."""
    out: List[MixtureCycleResult] = []
    for x in light_fractions:
        comps = [component_light, component_heavy]
        fr = [x, 1.0 - x]
        try:
            mc = MixtureCycle(comps, fr, T_evap_dew_C=T_evap_dew_C,
                              T_cond_mean_C=T_cond_mean_C,
                              eta_pump=eta_pump, eta_turbine=eta_turbine)
            res = mc.solve_with_resource(
                m_brine=resource.mass_flow, T_brine_in_C=resource.T_reservoir_C,
                pinch_evap=pinch_evap)
        except Exception:
            res = None
        out.append(res)
    return out
