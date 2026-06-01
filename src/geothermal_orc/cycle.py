"""Subcritical binary (Organic Rankine Cycle) solver.

State-point numbering follows DiPippo (2016) for binary plants:

    1  condenser outlet  -> saturated liquid at P_cond (pump inlet)
    2  pump outlet       -> compressed liquid at P_evap
    3  evaporator outlet -> turbine inlet (saturated or superheated vapour)
    4  turbine outlet     -> condenser inlet at P_cond

:meth:`ORCCycle.solve` returns the intensive (per-kg working fluid) cycle.
:meth:`ORCCycle.solve_with_resource` couples the cycle to a brine stream: the
working-fluid mass flow is sized so the evaporator pinch matches the target,
and a full energy and exergy balance is assembled and checked for closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI

from .thermo import (
    State,
    saturation_pressure,
    critical_temperature,
    specific_flow_exergy,
    DEAD_STATE_T,
)
from .heat_exchanger import counterflow_profile, HeatExchangerProfile


@dataclass
class CycleResult:
    """Outcome of a cycle solve.  Rates are ``None`` in per-mass-only mode."""

    fluid: str
    states: Dict[int, State]

    # Intensive (per kg working fluid), J/kg
    w_pump: float
    w_turbine: float
    w_net: float
    q_in: float
    q_out: float
    eta_th: float
    eta_carnot: float
    turbine_exit_quality: float          # Q4, -1 if superheated
    recuperator_duty: float = 0.0        # J/kg recovered internally (0 if none)

    # Extensive (resource-coupled); None otherwise
    m_wf: Optional[float] = None         # kg/s
    m_brine: Optional[float] = None      # kg/s
    brine_T_in: Optional[float] = None   # K
    brine_T_out: Optional[float] = None  # K
    W_net: Optional[float] = None        # W
    Q_in: Optional[float] = None         # W
    Q_out: Optional[float] = None        # W
    eta_utilization: Optional[float] = None       # second-law (DiPippo)
    exergy_in_brine: Optional[float] = None       # W, vs dead state
    exergy_destruction: Optional[Dict[str, float]] = None  # W per component
    energy_balance_residual: Optional[float] = None        # relative
    exergy_balance_residual: Optional[float] = None        # relative
    evaporator: Optional[HeatExchangerProfile] = None
    condenser: Optional[HeatExchangerProfile] = None

    def summary(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            f"ORC ({self.fluid})",
            f"  eta_th        = {self.eta_th*100:6.2f} %",
            f"  eta_carnot    = {self.eta_carnot*100:6.2f} %",
            f"  w_net         = {self.w_net/1e3:6.2f} kJ/kg",
        ]
        if self.W_net is not None:
            lines += [
                f"  m_wf          = {self.m_wf:6.2f} kg/s",
                f"  W_net         = {self.W_net/1e3:6.1f} kW",
                f"  eta_util(2nd) = {self.eta_utilization*100:6.2f} %",
                f"  brine out     = {self.brine_T_out-273.15:6.2f} C",
            ]
        return "\n".join(lines)


class ORCCycle:
    """A subcritical binary ORC, optionally recuperated."""

    def __init__(
        self,
        fluid: str,
        T_evap_C: float,
        T_cond_C: float,
        superheat: float = 0.0,
        eta_pump: float = 0.75,
        eta_turbine: float = 0.80,
        recuperator_effectiveness: float = 0.0,
        dp_evap_frac: float = 0.0,
        dp_cond_frac: float = 0.0,
        brine_fluid: str = "Water",
        P_brine: float = 1.0e6,
        T0: float = DEAD_STATE_T,
    ) -> None:
        self.fluid = fluid
        self.T_evap = T_evap_C + 273.15
        self.T_cond = T_cond_C + 273.15
        self.superheat = superheat
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        if not 0.0 <= recuperator_effectiveness < 1.0:
            raise ValueError("recuperator_effectiveness must be in [0, 1)")
        self.recup_eps = recuperator_effectiveness
        if dp_evap_frac < 0.0 or dp_cond_frac < 0.0:
            raise ValueError("pressure-drop fractions must be non-negative")
        self.dp_evap_frac = dp_evap_frac
        self.dp_cond_frac = dp_cond_frac
        self.brine_fluid = brine_fluid
        self.P_brine = P_brine
        self.T0 = T0

        Tc = critical_temperature(fluid)
        if self.T_evap >= Tc:
            raise ValueError(
                f"T_evap ({T_evap_C:.1f} C) >= critical temperature "
                f"({Tc-273.15:.1f} C); this solver is subcritical only"
            )
        if self.T_cond >= self.T_evap:
            raise ValueError("T_cond must be below T_evap")

        self.P_evap = saturation_pressure(fluid, self.T_evap)
        self.P_cond = saturation_pressure(fluid, self.T_cond)

    # ------------------------------------------------------------------ #
    def _state_points(self) -> Dict[int, State]:
        f = self.fluid
        # Non-isobaric heat exchange: the pump must deliver above the
        # evaporation pressure (high-side drop) and the turbine exhausts above
        # the condensing pressure (low-side drop).  Both cut net power.
        P_pump_out = self.P_evap * (1.0 + self.dp_evap_frac)
        P_turb_out = self.P_cond * (1.0 + self.dp_cond_frac)

        s1 = State.from_PQ(f, self.P_cond, 0.0)                 # sat liquid
        # Pump: isentropic target then efficiency.
        h2s = PropsSI("H", "P", P_pump_out, "S", s1.s, f)
        h2 = s1.h + (h2s - s1.h) / self.eta_pump
        s2 = State.from_Ph(f, P_pump_out, h2)
        # Evaporator outlet: saturated vapour, optionally superheated.
        if self.superheat > 0.0:
            s3 = State.from_TP(f, self.T_evap + self.superheat, self.P_evap)
        else:
            s3 = State.from_PQ(f, self.P_evap, 1.0)
        # Turbine: isentropic target then efficiency (expands to raised back-pressure).
        h4s = PropsSI("H", "P", P_turb_out, "S", s3.s, f)
        h4 = s3.h - self.eta_turbine * (s3.h - h4s)
        s4 = State.from_Ph(f, P_turb_out, h4)
        states = {1: s1, 2: s2, 3: s3, 4: s4}

        # Optional recuperator: turbine exhaust (4) preheats pump outlet (2).
        # Effectiveness-NTU definition Q = eps * Q_max, with Q_max the smaller of
        # what each stream could exchange (hot cooled to the cold inlet T, cold
        # heated to the hot inlet T).  Q <= Q_max guarantees no temperature cross.
        if self.recup_eps > 0.0 and s4.T > s2.T:
            h_hot_to_coldin = PropsSI("H", "T", s2.T, "P", s4.P, f)
            h_cold_to_hotin = PropsSI("H", "T", s4.T, "P", s2.P, f)
            Q_hot_max = s4.h - h_hot_to_coldin     # hot stream cooled to T2
            Q_cold_max = h_cold_to_hotin - s2.h    # cold stream heated to T4
            Q_recup = self.recup_eps * max(0.0, min(Q_hot_max, Q_cold_max))
            if Q_recup > 0.0:
                s2r = State.from_Ph(f, s2.P, s2.h + Q_recup)  # evaporator inlet
                s4r = State.from_Ph(f, s4.P, s4.h - Q_recup)  # condenser inlet
                states[5] = s2r
                states[6] = s4r
        return states

    def solve(self) -> CycleResult:
        """Intensive cycle (per kg working fluid)."""
        st = self._state_points()
        s1, s2, s3, s4 = st[1], st[2], st[3], st[4]

        # With a recuperator the evaporator starts from 2' (=5) and the
        # condenser from 4' (=6); work terms are unchanged.
        h_evap_in = st[5].h if 5 in st else s2.h
        h_cond_in = st[6].h if 6 in st else s4.h
        q_recup = (st[5].h - s2.h) if 5 in st else 0.0

        w_pump = s2.h - s1.h
        w_turb = s3.h - s4.h
        w_net = w_turb - w_pump
        q_in = s3.h - h_evap_in
        q_out = h_cond_in - s1.h
        eta_th = w_net / q_in
        eta_carnot = 1.0 - self.T_cond / self.T_evap

        return CycleResult(
            fluid=self.fluid,
            states=st,
            w_pump=w_pump,
            w_turbine=w_turb,
            w_net=w_net,
            q_in=q_in,
            q_out=q_out,
            eta_th=eta_th,
            eta_carnot=eta_carnot,
            turbine_exit_quality=s4.Q,
            recuperator_duty=q_recup,
        )

    # ------------------------------------------------------------------ #
    def solve_with_resource(
        self,
        m_brine: float,
        T_brine_in_C: float,
        pinch_evap: float = 5.0,
        cooling_T_in_C: float = 15.0,
        cooling_pinch: float = 5.0,
        cooling_fluid: str = "Water",
        P_cooling: float = 2.0e5,
        n: int = 80,
        n_search: Optional[int] = None,
    ) -> CycleResult:
        """Size the cycle to a brine stream and assemble full balances.

        The working-fluid flow ``m_wf`` is found so the evaporator pinch equals
        ``pinch_evap``.  Cooling-water flow is sized so its outlet sits
        ``cooling_pinch`` below the condensing temperature.
        """
        base = self.solve()
        st = base.states
        T_brine_in = T_brine_in_C + 273.15
        q_in = base.q_in

        # Evaporator cold-side enthalpy span.  With a recuperator the brine only
        # supplies heat above the recuperated inlet 2' (=state 5).
        h_cold_in = st[5].h if 5 in st else st[2].h
        h_cold_out = st[3].h
        dh_cold = h_cold_out - h_cold_in

        # --- fast pinch sizing ------------------------------------------- #
        # The cold-side T-Q profile depends only on the (fixed) enthalpy grid,
        # not on the working-fluid flow, so it is computed once.  Each trial
        # flow then needs only the brine-side temperatures.  This is
        # mathematically identical to sweeping ``counterflow_profile`` but
        # avoids recomputing the (expensive) organic-fluid side every
        # root-find iteration.
        ns = n_search if n_search is not None else n
        frac = np.linspace(0.0, 1.0, ns + 1)
        h_cold_grid = h_cold_in + frac * dh_cold
        T_cold_grid = np.array(
            [PropsSI("T", "P", self.P_evap, "H", float(h), self.fluid)
             for h in h_cold_grid]
        )
        h_hot_in = PropsSI("H", "T", T_brine_in, "P", self.P_brine,
                           self.brine_fluid)

        def pinch_gap(m_wf: float) -> float:
            Q_total = m_wf * dh_cold
            h_hot = h_hot_in - (1.0 - frac) * Q_total / m_brine
            T_hot = np.array(
                [PropsSI("T", "P", self.P_brine, "H", float(h), self.brine_fluid)
                 for h in h_hot]
            )
            return float(np.min(T_hot - T_cold_grid)) - pinch_evap

        # ``pinch_gap`` decreases monotonically in ``m_wf`` (more flow -> more
        # duty -> brine cooled further -> smaller approach).  Seed the bracket
        # from a brine-side energy estimate, then expand to be safe.
        cp_brine = PropsSI("C", "T", T_brine_in, "P", self.P_brine,
                           self.brine_fluid)
        dT_avail = max(T_brine_in - (self.T_evap + pinch_evap), 1.0)
        m_guess = m_brine * cp_brine * dT_avail / max(q_in, 1.0)
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

        evap = counterflow_profile(
            hot_fluid=self.brine_fluid, m_hot=m_brine, P_hot=self.P_brine,
            T_hot_in=T_brine_in,
            cold_fluid=self.fluid, m_cold=m_wf, P_cold=self.P_evap,
            h_cold_in=h_cold_in, h_cold_out=h_cold_out, n=n,
        )

        # Rates.
        W_net = m_wf * base.w_net
        Q_in = m_wf * base.q_in
        Q_out = m_wf * base.q_out
        brine_T_out = float(evap.T_hot[0])

        # --- cooling water sizing (condenser) ---------------------------- #
        T_cw_in = cooling_T_in_C + 273.15
        T_cw_out = self.T_cond - cooling_pinch
        if T_cw_out <= T_cw_in:
            raise ValueError("cooling water cannot reach the required outlet "
                             "temperature; raise cooling_pinch or lower inlet T")
        h_cw_in = PropsSI("H", "T", T_cw_in, "P", P_cooling, cooling_fluid)
        h_cw_out = PropsSI("H", "T", T_cw_out, "P", P_cooling, cooling_fluid)
        s_cw_in = PropsSI("S", "T", T_cw_in, "P", P_cooling, cooling_fluid)
        s_cw_out = PropsSI("S", "T", T_cw_out, "P", P_cooling, cooling_fluid)
        m_cw = Q_out / (h_cw_out - h_cw_in)

        # --- brine stream entropies (for exergy) ------------------------- #
        s_b_in = PropsSI("S", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)
        s_b_out = PropsSI("S", "T", brine_T_out, "P", self.P_brine, self.brine_fluid)
        h_b_in = PropsSI("H", "T", T_brine_in, "P", self.P_brine, self.brine_fluid)
        h_b_out = PropsSI("H", "T", brine_T_out, "P", self.P_brine, self.brine_fluid)

        # --- exergy destruction (T0 * entropy generation) --------------- #
        T0 = self.T0
        s1, s2, s3, s4 = st[1], st[2], st[3], st[4]
        # Recuperated stream states feeding the evaporator (2') and condenser (4').
        s_evap_in = st[5].s if 5 in st else s2.s
        s_cond_in = st[6].s if 6 in st else s4.s
        Edest = {
            "pump":       T0 * m_wf * (s2.s - s1.s),
            "evaporator": T0 * (m_wf * (s3.s - s_evap_in) + m_brine * (s_b_out - s_b_in)),
            "turbine":    T0 * m_wf * (s4.s - s3.s),
            "condenser":  T0 * (m_wf * (s1.s - s_cond_in) + m_cw * (s_cw_out - s_cw_in)),
        }
        if 5 in st:
            # Internal recuperator: cold side 2->2' gains entropy, hot side 4->4'
            # loses it; the sum is the irreversibility of the internal exchange.
            Edest["recuperator"] = T0 * m_wf * ((st[5].s - s2.s) + (st[6].s - s4.s))
        Edest_total = sum(Edest.values())

        # --- exergy bookkeeping referenced to the dead state ------------- #
        e_b_in = specific_flow_exergy(self.brine_fluid, h_b_in, s_b_in, T0)
        e_b_out = specific_flow_exergy(self.brine_fluid, h_b_out, s_b_out, T0)
        e_cw_in = specific_flow_exergy(cooling_fluid, h_cw_in, s_cw_in, T0)
        e_cw_out = specific_flow_exergy(cooling_fluid, h_cw_out, s_cw_out, T0)
        E_in = m_brine * e_b_in + m_cw * e_cw_in
        E_out = m_brine * e_b_out + m_cw * e_cw_out
        # Balance: E_in = W_net + E_out + Edest_total
        exergy_resid = (E_in - (W_net + E_out + Edest_total)) / max(abs(E_in), 1.0)

        # --- energy balance closure -------------------------------------- #
        energy_resid = (Q_in - Q_out - W_net) / max(abs(Q_in), 1.0)

        eta_util = W_net / (m_brine * e_b_in)

        return CycleResult(
            fluid=self.fluid,
            states=st,
            w_pump=base.w_pump,
            w_turbine=base.w_turbine,
            w_net=base.w_net,
            q_in=base.q_in,
            q_out=base.q_out,
            eta_th=base.eta_th,
            eta_carnot=base.eta_carnot,
            turbine_exit_quality=base.turbine_exit_quality,
            recuperator_duty=base.recuperator_duty,
            m_wf=m_wf,
            m_brine=m_brine,
            brine_T_in=T_brine_in,
            brine_T_out=brine_T_out,
            W_net=W_net,
            Q_in=Q_in,
            Q_out=Q_out,
            eta_utilization=eta_util,
            exergy_in_brine=m_brine * e_b_in,
            exergy_destruction=Edest,
            energy_balance_residual=energy_resid,
            exergy_balance_residual=exergy_resid,
            evaporator=evap,
            condenser=None,
        )
