"""Plant-boundary model: from cycle (gross) power to deliverable net power.

The :mod:`geothermal_orc.cycle` layer computes the thermodynamic cycle and its
internal net power (turbine minus working-fluid feed pump).  A real binary plant
must additionally spend power on **heat rejection** (air-cooled-condenser fans)
and **brine handling** (production and injection pumping), and its condensing
temperature is set by the **ambient** heat sink rather than chosen freely.  This
module adds that balance-of-plant layer.

Conventions
-----------
* ``W_gross``      turbine (generator) output, W
* ``W_net_cycle``  W_gross minus the working-fluid feed pump, W
* ``W_net_plant``  W_net_cycle minus parasitic loads (fans + brine pumps), W

Parasitic models
----------------
Air-cooled-condenser fan power is built from the air-side energy balance:
``m_air = Q_reject / (cp_air * dT_air)`` and ``P_fan = m_air * dp_fan /
(rho_air * eta_fan)``.  Brine pumping is modelled as a loop pressure rise that
stands in for production lift plus injection: ``P = m_brine * dP /
(rho_brine * eta_pump)``.  Defaults are documented, site-independent starting
points — a real project replaces them with well- and site-specific values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI

from .cycle import ORCCycle, CycleResult
from .geothermal import GeothermalResource
from .heat_exchanger import counterflow_profile, HeatExchangerProfile
from .optimization import optimize_evaporation_temperature

# --- documented default parameters ------------------------------------------ #
AIR_CP = 1005.0           # J/kg-K, dry air
AIR_RHO = 1.2             # kg/m3
ACC_DT_AIR = 14.0         # K, air temperature rise across the condenser bundle
ACC_FAN_DP = 170.0        # Pa, fan static pressure rise (air-cooled bundle)
ACC_FAN_EFF = 0.60        # -, fan + drive + motor efficiency

BRINE_RHO = 950.0         # kg/m3, hot brine (~liquid water, moderate T)
BRINE_LOOP_DP = 1.0e6     # Pa (10 bar): production lift + injection, placeholder
BRINE_PUMP_EFF = 0.70     # -

CONDENSER_APPROACH = 20.0  # K, condensing temp above ambient dry-bulb (air-cooled)


# --- parasitic-load models -------------------------------------------------- #
def air_cooled_fan_power(
    Q_reject: float,
    dT_air: float = ACC_DT_AIR,
    dp_fan: float = ACC_FAN_DP,
    eta_fan: float = ACC_FAN_EFF,
    cp_air: float = AIR_CP,
    rho_air: float = AIR_RHO,
) -> float:
    """Electrical fan power [W] to reject ``Q_reject`` [W] in an air-cooled condenser."""
    if Q_reject <= 0.0:
        return 0.0
    m_air = Q_reject / (cp_air * dT_air)        # kg/s of air
    return m_air * dp_fan / (rho_air * eta_fan)


def brine_pump_power(
    m_brine: float,
    dP: float = BRINE_LOOP_DP,
    eta_pump: float = BRINE_PUMP_EFF,
    rho_brine: float = BRINE_RHO,
) -> float:
    """Production + injection pumping power [W] for ``m_brine`` [kg/s]."""
    if m_brine <= 0.0:
        return 0.0
    return m_brine * dP / (rho_brine * eta_pump)


def condensing_temperature(ambient_C: float,
                           approach: float = CONDENSER_APPROACH) -> float:
    """Air-cooled condensing temperature [C] = ambient dry-bulb + approach (ITD)."""
    return ambient_C + approach


# --- results ---------------------------------------------------------------- #
@dataclass
class PlantResult:
    """A full plant operating point: cycle plus balance-of-plant parasitics."""

    fluid: str
    ambient_C: float
    T_cond_C: float
    T_evap_C: float
    W_gross: float          # turbine output, W
    W_pump_wf: float        # working-fluid feed pump, W
    P_fan: float            # air-cooled-condenser fans, W
    P_brine_pump: float     # brine production + injection, W
    W_net_plant: float      # W (can be <= 0 at hostile ambient)
    net_gross_ratio: float  # W_net_plant / W_gross
    feasible: bool
    cycle: Optional[CycleResult] = None

    @property
    def W_net_cycle(self) -> float:
        return self.W_gross - self.W_pump_wf

    @property
    def parasitic_fraction(self) -> float:
        """Parasitics as a fraction of gross output."""
        if self.W_gross <= 0.0:
            return 0.0
        return (self.P_fan + self.P_brine_pump) / self.W_gross


@dataclass
class SeasonalResult:
    """Net plant power across an ambient profile, with annual roll-up."""

    ambient_C: np.ndarray
    W_net_plant: np.ndarray        # W, per profile entry
    annual_energy_MWh: float
    capacity_factor: float         # mean / peak across the profile
    rated_W: float                 # peak net plant power across the profile
    details: List[Optional[PlantResult]]


# --- plant evaluation ------------------------------------------------------- #
def evaluate_plant(
    fluid: str,
    resource: GeothermalResource,
    ambient_C: float,
    *,
    T_cond_C: Optional[float] = None,
    condenser_approach: float = CONDENSER_APPROACH,
    T_evap_C: Optional[float] = None,
    pinch_evap: float = 5.0,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
    dT_air: float = ACC_DT_AIR,
    dp_fan: float = ACC_FAN_DP,
    eta_fan: float = ACC_FAN_EFF,
    brine_dP: float = BRINE_LOOP_DP,
    brine_eta: float = BRINE_PUMP_EFF,
    brine_rho: float = BRINE_RHO,
) -> PlantResult:
    """Evaluate one plant operating point.

    The condensing temperature defaults to ``ambient_C + condenser_approach``
    (air-cooled).  If ``T_evap_C`` is omitted, the evaporation temperature is
    optimized for net cycle power at the resulting condensing temperature;
    otherwise the supplied value is used (fixed design — see off-design use).
    """
    Tc = T_cond_C if T_cond_C is not None else condensing_temperature(
        ambient_C, condenser_approach)

    if T_evap_C is None:
        opt = optimize_evaporation_temperature(
            fluid, resource, T_cond_C=Tc, pinch_evap=pinch_evap,
            eta_pump=eta_pump, eta_turbine=eta_turbine)
        if not opt.feasible:
            return PlantResult(fluid, ambient_C, Tc, float("nan"),
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, None)
        Te = opt.T_evap_opt_C
    else:
        Te = T_evap_C

    try:
        cyc = ORCCycle(fluid, T_evap_C=Te, T_cond_C=Tc,
                       eta_pump=eta_pump, eta_turbine=eta_turbine)
        res = cyc.solve_with_resource(
            m_brine=resource.mass_flow, T_brine_in_C=resource.T_reservoir_C,
            pinch_evap=pinch_evap)
    except Exception:
        return PlantResult(fluid, ambient_C, Tc, Te,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, None)

    W_gross = res.w_turbine * res.m_wf
    W_pump_wf = res.w_pump * res.m_wf
    P_fan = air_cooled_fan_power(res.Q_out, dT_air, dp_fan, eta_fan)
    P_brine = brine_pump_power(resource.mass_flow, brine_dP, brine_eta, brine_rho)
    W_net_plant = res.W_net - P_fan - P_brine
    ngr = W_net_plant / W_gross if W_gross > 0.0 else 0.0

    return PlantResult(
        fluid=fluid, ambient_C=ambient_C, T_cond_C=Tc, T_evap_C=Te,
        W_gross=W_gross, W_pump_wf=W_pump_wf, P_fan=P_fan,
        P_brine_pump=P_brine, W_net_plant=W_net_plant, net_gross_ratio=ngr,
        feasible=True, cycle=res,
    )


def seasonal_performance(
    fluid: str,
    resource: GeothermalResource,
    ambient_profile_C: Sequence[float],
    *,
    condenser_approach: float = CONDENSER_APPROACH,
    fixed_T_evap_C: Optional[float] = None,
    **plant_kwargs,
) -> SeasonalResult:
    """Net plant power across an ambient profile (e.g. 12 monthly means).

    With ``fixed_T_evap_C`` the same design runs at every ambient (closer to a
    fixed installed plant); otherwise the plant is re-optimized at each ambient.
    Annual energy assumes each profile entry covers an equal share of 8760 h;
    the capacity factor is the mean-to-peak ratio across the profile.
    """
    amb = np.asarray(ambient_profile_C, dtype=float)
    nets: List[float] = []
    details: List[Optional[PlantResult]] = []
    for a in amb:
        pr = evaluate_plant(fluid, resource, float(a),
                            condenser_approach=condenser_approach,
                            T_evap_C=fixed_T_evap_C, **plant_kwargs)
        details.append(pr)
        nets.append(pr.W_net_plant if pr.feasible else 0.0)

    nets_arr = np.array(nets)
    hours = 8760.0 / len(nets_arr) if len(nets_arr) else 0.0
    annual_MWh = float(np.sum(np.clip(nets_arr, 0.0, None)) * hours / 1e6)
    rated = float(np.max(nets_arr)) if len(nets_arr) else 0.0
    cap = (annual_MWh * 1e6) / (rated * 8760.0) if rated > 0.0 else 0.0

    return SeasonalResult(
        ambient_C=amb, W_net_plant=nets_arr, annual_energy_MWh=annual_MWh,
        capacity_factor=cap, rated_W=rated, details=details,
    )


# --- off-design operation (fixed installed hardware) ------------------------ #
# As a geothermal resource cools, a *built* plant does not get re-optimized — it
# runs off-design on fixed hardware.  We model two fixed constraints:
#   * the turbine swallowing capacity, via a Stodola "ellipse" (cone) law
#       m_wf = K * sqrt( (P_evap^2 - P_cond^2) / T_in ),
#     with K calibrated at the design point; and
#   * the evaporator conductance UA, held at its design value.
# The condensing pressure is set by the (fixed) ambient.  These two relations,
# solved together, fix the operating point at any cooler brine temperature.

BRINE_FLUID = "Water"
BRINE_PRESSURE = 1.0e6   # Pa, matches the cycle solver's brine pressure

PARTLOAD_BETA = 0.8      # curvature of the part-load turbine-efficiency penalty


def part_load_turbine_efficiency(
    m_ratio: float,
    eta_design: float,
    beta: float = PARTLOAD_BETA,
    floor: float = 0.5,
) -> float:
    """Off-design isentropic efficiency of a fixed turbine at part load.

    A reduced quadratic penalty in the flow ratio ``m_ratio = m / m_design``:
    ``eta = eta_design * (1 - beta*(1 - m_ratio)**2)``, equal to the design
    value at ``m_ratio = 1`` and falling away on either side.  Real part-load
    curves are turbine-specific; this captures the direction and rough
    magnitude (a few to ~10 efficiency points lost at 60-70% load)."""
    factor = 1.0 - beta * (1.0 - m_ratio) ** 2
    return eta_design * min(1.0, max(floor, factor))


def profile_UA(profile: HeatExchangerProfile) -> float:
    """Conductance UA [W/K] implied by a counterflow T-Q profile.

    Sums the per-segment ``dQ / LMTD``.  Returns ``inf`` if the streams cross
    (a temperature pinch of zero needs infinite area)."""
    duty = np.asarray(profile.duty, dtype=float)
    dT = np.asarray(profile.T_hot, dtype=float) - np.asarray(profile.T_cold, dtype=float)
    if np.any(dT <= 1e-9):
        return float("inf")
    dq = np.diff(duty)
    a, b = dT[:-1], dT[1:]
    lm = np.where(np.abs(a - b) < 1e-9, 0.5 * (a + b), (a - b) / np.log(a / b))
    return float(np.sum(dq / lm))


@dataclass
class DesignPoint:
    """Fixed-hardware descriptors captured at the year-0 design."""

    fluid: str
    T_cond_C: float
    P_cond: float
    eta_pump: float
    eta_turbine: float
    m_brine: float
    UA_evap: float       # W/K
    stodola_K: float
    T_evap_C: float      # design evaporation temperature
    m_wf: float          # design working-fluid flow


def design_plant(
    fluid: str,
    resource: GeothermalResource,
    ambient_C: float,
    *,
    condenser_approach: float = CONDENSER_APPROACH,
    pinch_evap: float = 5.0,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
):
    """Optimize a year-0 design and capture its fixed-hardware descriptors."""
    Tc = condensing_temperature(ambient_C, condenser_approach)
    opt = optimize_evaporation_temperature(
        fluid, resource, T_cond_C=Tc, pinch_evap=pinch_evap,
        eta_pump=eta_pump, eta_turbine=eta_turbine)
    Te = opt.T_evap_opt_C
    res = ORCCycle(fluid, T_evap_C=Te, T_cond_C=Tc, eta_pump=eta_pump,
                   eta_turbine=eta_turbine).solve_with_resource(
        m_brine=resource.mass_flow, T_brine_in_C=resource.T_reservoir_C,
        pinch_evap=pinch_evap)

    UA = profile_UA(res.evaporator)
    P_evap_d = PropsSI("P", "T", Te + 273.15, "Q", 1, fluid)
    P_cond_d = PropsSI("P", "T", Tc + 273.15, "Q", 1, fluid)
    K = res.m_wf / np.sqrt(max(P_evap_d ** 2 - P_cond_d ** 2, 1.0) / (Te + 273.15))

    design = DesignPoint(fluid, Tc, P_cond_d, eta_pump, eta_turbine,
                         resource.mass_flow, UA, float(K), Te, res.m_wf)
    return design, res


def off_design_operation(
    design: DesignPoint,
    T_brine_in_C: float,
    *,
    pinch_floor: float = 0.4,
    dT_air: float = ACC_DT_AIR,
    dp_fan: float = ACC_FAN_DP,
    eta_fan: float = ACC_FAN_EFF,
    brine_dP: float = BRINE_LOOP_DP,
    brine_eta: float = BRINE_PUMP_EFF,
    brine_rho: float = BRINE_RHO,
) -> PlantResult:
    """Operate the fixed-hardware plant at a (cooler) brine temperature."""
    fluid, Tc, P_cond = design.fluid, design.T_cond_C, design.P_cond
    T_brine_in = T_brine_in_C + 273.15

    def m_stodola(Te_C):
        P_evap = PropsSI("P", "T", Te_C + 273.15, "Q", 1, fluid)
        val = P_evap ** 2 - P_cond ** 2
        if val <= 0.0:
            return 0.0
        return design.stodola_K * np.sqrt(val / (Te_C + 273.15))

    def ua_residual(Te_C):
        m_wf = m_stodola(Te_C)
        if m_wf <= 0.0:
            return -design.UA_evap
        try:
            base = ORCCycle(fluid, T_evap_C=Te_C, T_cond_C=Tc,
                            eta_pump=design.eta_pump,
                            eta_turbine=design.eta_turbine).solve()
            st = base.states
            prof = counterflow_profile(
                hot_fluid=BRINE_FLUID, m_hot=design.m_brine, P_hot=BRINE_PRESSURE,
                T_hot_in=T_brine_in, cold_fluid=fluid, m_cold=m_wf,
                P_cold=st[2].P, h_cold_in=st[2].h, h_cold_out=st[3].h, n=60)
            return min(profile_UA(prof), 1e12) - design.UA_evap
        except Exception:
            return 1e12 - design.UA_evap   # treat as too-tight / infeasible-high

    Tcrit_C = PropsSI("Tcrit", fluid) - 273.15
    lo = Tc + 2.0
    hi = min(T_brine_in_C - pinch_floor, Tcrit_C - 2.0)

    def infeasible():
        return PlantResult(fluid, float("nan"), Tc, float("nan"),
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, None)

    if hi <= lo:
        return infeasible()
    try:
        flo, fhi = ua_residual(lo), ua_residual(hi)
    except Exception:
        return infeasible()
    if not (np.isfinite(flo) and flo < 0.0 < fhi):
        return infeasible()

    Te = float(brentq(ua_residual, lo, hi, xtol=1e-2))
    m_wf = m_stodola(Te)
    # Fixed turbine runs off-design at part load -> degraded isentropic efficiency.
    m_ratio = m_wf / design.m_wf if design.m_wf > 0 else 1.0
    eta_t_od = part_load_turbine_efficiency(m_ratio, design.eta_turbine)
    base = ORCCycle(fluid, T_evap_C=Te, T_cond_C=Tc, eta_pump=design.eta_pump,
                    eta_turbine=eta_t_od).solve()

    W_gross = base.w_turbine * m_wf
    W_pump_wf = base.w_pump * m_wf
    Q_out = base.q_out * m_wf
    P_fan = air_cooled_fan_power(Q_out, dT_air, dp_fan, eta_fan)
    P_brine = brine_pump_power(design.m_brine, brine_dP, brine_eta, brine_rho)
    W_net_plant = (W_gross - W_pump_wf) - P_fan - P_brine
    ngr = W_net_plant / W_gross if W_gross > 0.0 else 0.0

    return PlantResult(
        fluid=fluid, ambient_C=float("nan"), T_cond_C=Tc, T_evap_C=Te,
        W_gross=W_gross, W_pump_wf=W_pump_wf, P_fan=P_fan, P_brine_pump=P_brine,
        W_net_plant=W_net_plant, net_gross_ratio=ngr,
        feasible=(W_net_plant > 0.0), cycle=None,
    )


def decline_curves(
    fluid: str,
    resource: GeothermalResource,
    years: Sequence[float],
    ambient_C: float,
    *,
    decline_rate: float = 0.005,
    decline_mode: str = "linear",
    condenser_approach: float = CONDENSER_APPROACH,
    pinch_evap: float = 5.0,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
):
    """Net plant power over field life: re-optimized envelope vs fixed plant.

    Returns ``(design, T_brine_C, W_reoptimized, W_fixed)``.  ``W_reoptimized``
    rebuilds the optimal plant at each year (optimistic upper bound);
    ``W_fixed`` runs the year-0 hardware off-design (the realistic case)."""
    declining = GeothermalResource(
        T_reservoir_C=resource.T_reservoir_C, mass_flow=resource.mass_flow,
        decline_rate=decline_rate, decline_mode=decline_mode)
    design, _ = design_plant(
        fluid, declining, ambient_C, condenser_approach=condenser_approach,
        pinch_evap=pinch_evap, eta_pump=eta_pump, eta_turbine=eta_turbine)

    Tb, reopt, fixed = [], [], []
    for y in years:
        T_y = declining.temperature_at(y)
        Tb.append(T_y)
        res_y = GeothermalResource(T_reservoir_C=T_y, mass_flow=resource.mass_flow)
        pr = evaluate_plant(fluid, res_y, ambient_C,
                            condenser_approach=condenser_approach,
                            pinch_evap=pinch_evap, eta_pump=eta_pump,
                            eta_turbine=eta_turbine)
        reopt.append(pr.W_net_plant if pr.feasible else 0.0)
        od = off_design_operation(design, T_y, eta_fan=ACC_FAN_EFF)
        fixed.append(od.W_net_plant if od.feasible else 0.0)

    return design, np.array(Tb), np.array(reopt), np.array(fixed)


# --- pinch / area / power trade-off ----------------------------------------- #
def pinch_area_tradeoff(
    fluid: str,
    resource: GeothermalResource,
    T_evap_C: float,
    pinches_C: Sequence[float],
    *,
    T_cond_C: float = 30.0,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
    dp_evap_frac: float = 0.0,
    dp_cond_frac: float = 0.0,
):
    """Trade-off between evaporator pinch, conductance, and net power.

    At fixed evaporation temperature, a tighter pinch lets the brine be cooled
    closer to the working fluid — more flow, more power — but needs more heat-
    exchanger area (a larger UA).  Returns ``(pinches, W_net, UA_evap)`` so the
    diminishing-returns 'knee' of power versus UA can be plotted."""
    W, UA = [], []
    for p in pinches_C:
        res = ORCCycle(
            fluid, T_evap_C=T_evap_C, T_cond_C=T_cond_C, eta_pump=eta_pump,
            eta_turbine=eta_turbine, dp_evap_frac=dp_evap_frac,
            dp_cond_frac=dp_cond_frac,
        ).solve_with_resource(
            m_brine=resource.mass_flow, T_brine_in_C=resource.T_reservoir_C,
            pinch_evap=p)
        W.append(res.W_net)
        UA.append(profile_UA(res.evaporator))
    return np.asarray(pinches_C, dtype=float), np.array(W), np.array(UA)
