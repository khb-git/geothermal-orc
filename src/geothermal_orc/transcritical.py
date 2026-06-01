"""Transcritical ORC cycles.

A *transcritical* cycle pumps the working fluid above its critical pressure
before heating, so there is **no evaporation plateau** — the fluid passes
smoothly from liquid-like to gas-like through the pseudo-critical region.  Heat
rejection is still subcritical (the fluid condenses below the critical point).

Because the supercritical heating curve is a gentle S-shape rather than a flat
boiling line, it can track the brine's straight-line sensible cooling far more
closely than a subcritical cycle, whose pinch is pinned at the bubble point.
For ~150 C resources this is the documented route (propane, R-143a, CO2) to
higher specific power.

State numbering matches :mod:`geothermal_orc.cycle`:

    1 condenser outlet (saturated liquid)   2 pump outlet (supercritical)
    3 heater outlet / turbine inlet          4 turbine outlet
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI

from .thermo import (
    State, saturation_pressure, critical_pressure, critical_temperature,
    specific_flow_exergy, DEAD_STATE_T,
)
from .cycle import CycleResult


class TranscriticalCycle:
    """A transcritical ORC: supercritical heat addition, subcritical condensing."""

    def __init__(
        self,
        fluid: str,
        P_high: float,
        T_turb_in_C: float,
        T_cond_C: float,
        eta_pump: float = 0.75,
        eta_turbine: float = 0.80,
        brine_fluid: str = "Water",
        P_brine: float = 1.0e6,
        T0: float = DEAD_STATE_T,
    ) -> None:
        self.fluid = fluid
        self.P_high = P_high
        self.T_turb_in = T_turb_in_C + 273.15
        self.T_cond = T_cond_C + 273.15
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        self.brine_fluid = brine_fluid
        self.P_brine = P_brine
        self.T0 = T0

        Pc = critical_pressure(fluid)
        if P_high <= Pc:
            raise ValueError(
                f"P_high ({P_high/1e5:.1f} bar) must exceed the critical "
                f"pressure ({Pc/1e5:.1f} bar) for a transcritical cycle")
        self.P_cond = saturation_pressure(fluid, self.T_cond)
        if self.T_turb_in <= self.T_cond:
            raise ValueError("turbine-inlet temperature must exceed T_cond")

    # ------------------------------------------------------------------ #
    def _state_points(self) -> Dict[int, State]:
        f = self.fluid
        s1 = State.from_PQ(f, self.P_cond, 0.0)                 # sat liquid
        h2s = PropsSI("H", "P", self.P_high, "S", s1.s, f)
        h2 = s1.h + (h2s - s1.h) / self.eta_pump
        s2 = State.from_Ph(f, self.P_high, h2)                  # supercritical
        s3 = State.from_TP(f, self.T_turb_in, self.P_high)      # turbine inlet
        h4s = PropsSI("H", "P", self.P_cond, "S", s3.s, f)
        h4 = s3.h - self.eta_turbine * (s3.h - h4s)
        s4 = State.from_Ph(f, self.P_cond, h4)
        return {1: s1, 2: s2, 3: s3, 4: s4}

    def solve(self) -> CycleResult:
        st = self._state_points()
        s1, s2, s3, s4 = st[1], st[2], st[3], st[4]
        w_pump = s2.h - s1.h
        w_turb = s3.h - s4.h
        w_net = w_turb - w_pump
        q_in = s3.h - s2.h
        q_out = s4.h - s1.h
        return CycleResult(
            fluid=self.fluid, states=st, w_pump=w_pump, w_turbine=w_turb,
            w_net=w_net, q_in=q_in, q_out=q_out, eta_th=w_net / q_in,
            eta_carnot=1.0 - self.T_cond / self.T_turb_in,
            turbine_exit_quality=s4.Q,
        )

    # ------------------------------------------------------------------ #
    def solve_with_resource(
        self,
        m_brine: float,
        T_brine_in_C: float,
        pinch_heater: float = 5.0,
        cooling_T_in_C: float = 15.0,
        cooling_pinch: float = 5.0,
        cooling_fluid: str = "Water",
        P_cooling: float = 2.0e5,
        n: int = 80,
    ) -> CycleResult:
        """Size the supercritical heater to a brine stream (pinch matched).

        The pinch may sit *inside* the heater, near the pseudo-critical region
        where the working fluid's heat capacity peaks — not at an inlet — so the
        whole profile is checked, not just the terminals."""
        base = self.solve()
        st = base.states
        T_brine_in = T_brine_in_C + 273.15

        # Heater cold side: state 2 (supercritical liquid-like) -> state 3.
        h_cold_in, h_cold_out = st[2].h, st[3].h
        dh_cold = h_cold_out - h_cold_in
        frac = np.linspace(0.0, 1.0, n + 1)
        h_cold_grid = h_cold_in + frac * dh_cold
        T_cold_grid = np.array(
            [PropsSI("T", "P", self.P_high, "H", float(h), self.fluid)
             for h in h_cold_grid])
        h_hot_in = PropsSI("H", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)
        # Floor enthalpy guards the brine flash when the bracket search probes
        # over-large flows (which would otherwise cool the brine below water's
        # valid range); a clamped point reads as a crossed pinch, not a crash.
        h_floor = PropsSI("H", "T", 275.0, "P", self.P_brine, self.brine_fluid)

        def pinch_gap(m_wf: float) -> float:
            Q_total = m_wf * dh_cold
            h_hot = np.maximum(h_hot_in - (1.0 - frac) * Q_total / m_brine, h_floor)
            T_hot = np.array(
                [PropsSI("T", "P", self.P_brine, "H", float(h), self.brine_fluid)
                 for h in h_hot])
            return float(np.min(T_hot - T_cold_grid)) - pinch_heater

        cp_brine = PropsSI("C", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)
        dT_avail = max(T_brine_in - (self.T_turb_in + pinch_heater), 1.0)
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
        h_hot_out = h_hot_in - Q_total / m_brine
        brine_T_out = PropsSI("T", "P", self.P_brine, "H", h_hot_out,
                              self.brine_fluid)

        W_net = m_wf * base.w_net
        Q_in = m_wf * base.q_in
        Q_out = m_wf * base.q_out

        # Cooling water sizing.
        T_cw_in = cooling_T_in_C + 273.15
        T_cw_out = self.T_cond - cooling_pinch
        if T_cw_out <= T_cw_in:
            raise ValueError("cooling water cannot reach the required outlet T")
        h_cw_in = PropsSI("H", "T", T_cw_in, "P", P_cooling, cooling_fluid)
        h_cw_out = PropsSI("H", "T", T_cw_out, "P", P_cooling, cooling_fluid)
        s_cw_in = PropsSI("S", "T", T_cw_in, "P", P_cooling, cooling_fluid)
        s_cw_out = PropsSI("S", "T", T_cw_out, "P", P_cooling, cooling_fluid)
        m_cw = Q_out / (h_cw_out - h_cw_in)

        s_b_in = PropsSI("S", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)
        s_b_out = PropsSI("S", "T", brine_T_out, "P", self.P_brine, self.brine_fluid)
        h_b_in = h_hot_in
        h_b_out = h_hot_out

        T0 = self.T0
        s1, s2, s3, s4 = st[1], st[2], st[3], st[4]
        Edest = {
            "pump":      T0 * m_wf * (s2.s - s1.s),
            "heater":    T0 * (m_wf * (s3.s - s2.s) + m_brine * (s_b_out - s_b_in)),
            "turbine":   T0 * m_wf * (s4.s - s3.s),
            "condenser": T0 * (m_wf * (s1.s - s4.s) + m_cw * (s_cw_out - s_cw_in)),
        }
        Edest_total = sum(Edest.values())

        e_b_in = specific_flow_exergy(self.brine_fluid, h_b_in, s_b_in, T0)
        e_b_out = specific_flow_exergy(self.brine_fluid, h_b_out, s_b_out, T0)
        e_cw_in = specific_flow_exergy(cooling_fluid, h_cw_in, s_cw_in, T0)
        e_cw_out = specific_flow_exergy(cooling_fluid, h_cw_out, s_cw_out, T0)
        E_in = m_brine * e_b_in + m_cw * e_cw_in
        E_out = m_brine * e_b_out + m_cw * e_cw_out
        exergy_resid = (E_in - (W_net + E_out + Edest_total)) / max(abs(E_in), 1.0)
        energy_resid = (Q_in - Q_out - W_net) / max(abs(Q_in), 1.0)
        eta_util = W_net / (m_brine * e_b_in)

        return CycleResult(
            fluid=self.fluid, states=st, w_pump=base.w_pump,
            w_turbine=base.w_turbine, w_net=base.w_net, q_in=base.q_in,
            q_out=base.q_out, eta_th=base.eta_th, eta_carnot=base.eta_carnot,
            turbine_exit_quality=base.turbine_exit_quality,
            m_wf=m_wf, m_brine=m_brine, brine_T_in=T_brine_in,
            brine_T_out=brine_T_out, W_net=W_net, Q_in=Q_in, Q_out=Q_out,
            eta_utilization=eta_util, exergy_in_brine=m_brine * e_b_in,
            exergy_destruction=Edest, energy_balance_residual=energy_resid,
            exergy_balance_residual=exergy_resid, evaporator=None, condenser=None,
        )
