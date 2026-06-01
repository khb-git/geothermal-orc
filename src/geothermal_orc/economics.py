"""Techno-economics: component costs and levelized cost of electricity.

Turns *net power* into *cost per megawatt-hour*, so the design objective can
shift from "most megawatts" to "cheapest MWh".  Capital costs use the **module
costing** method of Turton et al. (the standard in ORC techno-economics): a
purchased-equipment cost from a log-quadratic size correlation, scaled by a
bare-module factor and updated with the Chemical Engineering Plant Cost Index
(CEPCI).  Geothermal projects are dominated by **well** costs, which are added
on top of the surface plant, and the LCOE folds in O&M and financing via a
capital-recovery factor.

Cost coefficients are from Turton et al., *Analysis, Synthesis and Design of
Chemical Processes* (USD 2001); material/pressure factors are held at a carbon-
steel, low-pressure baseline and exposed for the user to refine.  Well counts,
well cost, O&M, discount rate and life are documented, adjustable assumptions —
a real project replaces them with site data.  The model is validated by landing
the LCOE in the published range for medium-temperature binary plants
(~$80-150/MWh; cf. Lazard, IRENA).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .geothermal import GeothermalResource
from .plant import evaluate_plant, profile_UA, PlantResult, ACC_DT_AIR

# --- Turton module-costing coefficients (USD 2001) -------------------------- #
# log10(Cp0) = K1 + K2 log10(A) + K3 (log10 A)^2 ; bare-module F_BM = B1 + B2.
# Capacity A: heat exchangers and air coolers in m^2; pump and turbine in kW.
_TURTON: Dict[str, dict] = {
    "hx":      {"K": (4.3247, -0.3030, 0.1634), "F_BM": 1.63 + 1.66},   # shell&tube
    "acc":     {"K": (4.0336, 0.2341, 0.0497), "F_BM": 0.96 + 1.21},    # air cooler
    "pump":    {"K": (3.3892, 0.0536, 0.1538), "F_BM": 1.89 + 1.35},    # centrifugal
    "turbine": {"K": (2.7051, 1.4398, -0.1776), "F_BM": 3.5},           # axial
}


@dataclass
class EconomicAssumptions:
    """Documented, adjustable cost and finance assumptions."""

    cepci_ref: float = 397.0      # CEPCI 2001 (Turton basis)
    cepci_now: float = 800.0      # ~2024 CEPCI
    # Overall heat-transfer coefficients, W/m^2-K (service-typical).
    U_evap: float = 900.0         # brine / boiling organic
    U_cond: float = 450.0         # air-cooled condenser (bare-tube basis)
    U_recup: float = 600.0        # vapour / liquid organic
    # Surface-plant indirects (engineering, installation beyond bare module,
    # contingency) and geothermal well costs.
    indirect_factor: float = 1.4
    well_cost: float = 5.0e6      # USD per well (production or injection)
    n_wells: float = 3.0
    # Operation & finance.
    om_frac_capex: float = 0.03   # annual O&M as a fraction of surface CAPEX
    discount_rate: float = 0.08
    plant_life_yr: float = 30.0
    capacity_factor: float = 0.85


@dataclass
class EconomicResult:
    fluid: str
    capex_total: float            # USD
    capex_surface: float          # USD (ORC island + indirects)
    capex_wells: float            # USD
    component_costs: Dict[str, float]
    areas: Dict[str, float]       # m^2
    annual_energy_MWh: float
    annual_cost: float            # USD/yr (CRF*CAPEX + O&M)
    crf: float
    lcoe: float                   # USD/MWh
    W_net_plant: float            # W
    specific_capex: float         # USD/kW net


# --- costing primitives ----------------------------------------------------- #
def _purchased_cost(component: str, capacity: float) -> float:
    k1, k2, k3 = _TURTON[component]["K"]
    la = math.log10(max(capacity, 1e-6))
    return 10.0 ** (k1 + k2 * la + k3 * la * la)        # USD 2001


def bare_module_cost(component: str, capacity: float,
                     assume: EconomicAssumptions) -> float:
    """Installed (bare-module) cost in present USD, CEPCI-updated."""
    cp0 = _purchased_cost(component, capacity)
    f_bm = _TURTON[component]["F_BM"]
    return cp0 * f_bm * (assume.cepci_now / assume.cepci_ref)


def capital_recovery_factor(rate: float, years: float) -> float:
    """Annualizes a capital sum: CRF = i(1+i)^n / ((1+i)^n - 1)."""
    if rate <= 0.0:
        return 1.0 / years
    g = (1.0 + rate) ** years
    return rate * g / (g - 1.0)


def _lmtd(dt1: float, dt2: float) -> float:
    if dt1 <= 0.0 or dt2 <= 0.0:
        return float("nan")
    if abs(dt1 - dt2) < 1e-9:
        return 0.5 * (dt1 + dt2)
    return (dt1 - dt2) / math.log(dt1 / dt2)


# --- plant CAPEX and LCOE --------------------------------------------------- #
def plant_capex(pr: PlantResult, assume: EconomicAssumptions = EconomicAssumptions()):
    """Component areas, bare-module costs, and total CAPEX for a plant point."""
    cyc = pr.cycle
    # Evaporator area from its conductance.
    UA_evap = profile_UA(cyc.evaporator)
    A_evap = UA_evap / assume.U_evap
    # Condenser (air-cooled): UA from duty and the air-side LMTD.
    dt1 = pr.T_cond_C - pr.ambient_C
    dt2 = pr.T_cond_C - (pr.ambient_C + ACC_DT_AIR)
    UA_cond = cyc.Q_out / _lmtd(dt1, dt2)
    A_cond = UA_cond / assume.U_cond

    areas = {"evaporator": A_evap, "condenser": A_cond}
    comp = {
        "evaporator": bare_module_cost("hx", A_evap, assume),
        "condenser":  bare_module_cost("acc", A_cond, assume),
        "pump":       bare_module_cost("pump", pr.W_pump_wf / 1e3, assume),
        "turbine":    bare_module_cost("turbine", pr.W_gross / 1e3, assume),
    }
    # Optional recuperator surface (only when one is in use).
    if getattr(cyc, "recuperator_duty", 0.0) and cyc.m_wf:
        Q_recup = cyc.recuperator_duty * cyc.m_wf
        UA_recup = Q_recup / 10.0          # ~10 K mean approach, representative
        A_recup = UA_recup / assume.U_recup
        areas["recuperator"] = A_recup
        comp["recuperator"] = bare_module_cost("hx", A_recup, assume)

    capex_surface = sum(comp.values()) * assume.indirect_factor
    capex_wells = assume.n_wells * assume.well_cost
    capex_total = capex_surface + capex_wells
    return capex_total, capex_surface, capex_wells, comp, areas


def levelized_cost(
    fluid: str,
    resource: GeothermalResource,
    ambient_C: float,
    assume: EconomicAssumptions = EconomicAssumptions(),
    *,
    T_evap_C: Optional[float] = None,
    pinch_evap: float = 5.0,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
) -> EconomicResult:
    """Full techno-economic evaluation -> LCOE for a design point."""
    pr = evaluate_plant(fluid, resource, ambient_C, T_evap_C=T_evap_C,
                        pinch_evap=pinch_evap, eta_pump=eta_pump,
                        eta_turbine=eta_turbine)
    if not pr.feasible or pr.W_net_plant <= 0.0:
        raise ValueError("infeasible or non-positive net power; cannot cost it")

    capex_total, capex_surface, capex_wells, comp, areas = plant_capex(pr, assume)

    energy_MWh = pr.W_net_plant * assume.capacity_factor * 8760.0 / 1e6
    crf = capital_recovery_factor(assume.discount_rate, assume.plant_life_yr)
    annual_cost = crf * capex_total + assume.om_frac_capex * capex_surface
    lcoe = annual_cost / energy_MWh

    return EconomicResult(
        fluid=fluid, capex_total=capex_total, capex_surface=capex_surface,
        capex_wells=capex_wells, component_costs=comp, areas=areas,
        annual_energy_MWh=energy_MWh, annual_cost=annual_cost, crf=crf,
        lcoe=lcoe, W_net_plant=pr.W_net_plant,
        specific_capex=capex_total / (pr.W_net_plant / 1e3),
    )


# --- design economics: cost-optimal vs power-optimal, sensitivity ----------- #
import dataclasses as _dc
from typing import Sequence as _Seq


def pinch_lcoe_tradeoff(
    fluid: str,
    resource: GeothermalResource,
    ambient_C: float,
    pinches_C: _Seq[float],
    assume: EconomicAssumptions = EconomicAssumptions(),
    *,
    T_evap_C: Optional[float] = None,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
):
    """Net power and LCOE versus evaporator pinch.

    Power rises as the pinch tightens (more heat recovered); LCOE need not,
    because tighter pinch also means more heat-exchanger area.  Whether the
    cost-optimal pinch is wider than the power-optimal one depends on how much
    of the CAPEX is the (fixed) wells versus the (pinch-sensitive) surface
    plant.  Returns ``(pinches, W_net, lcoe)``."""
    W, L = [], []
    for p in pinches_C:
        try:
            res = levelized_cost(fluid, resource, ambient_C, assume,
                                 T_evap_C=T_evap_C, pinch_evap=p,
                                 eta_pump=eta_pump, eta_turbine=eta_turbine)
            W.append(res.W_net_plant)
            L.append(res.lcoe)
        except Exception:
            W.append(float("nan"))
            L.append(float("nan"))
    return np.asarray(pinches_C, dtype=float), np.array(W), np.array(L)


def lcoe_sensitivity(
    fluid: str,
    resource: GeothermalResource,
    ambient_C: float,
    parameter: str,
    values: _Seq[float],
    base: EconomicAssumptions = EconomicAssumptions(),
    *,
    T_evap_C: Optional[float] = None,
    pinch_evap: float = 5.0,
):
    """LCOE as one economic assumption is varied over ``values``.

    ``parameter`` is any field of :class:`EconomicAssumptions` (e.g.
    ``well_cost``, ``discount_rate``, ``capacity_factor``)."""
    out = []
    for v in values:
        assume = _dc.replace(base, **{parameter: v})
        out.append(levelized_cost(fluid, resource, ambient_C, assume,
                                  T_evap_C=T_evap_C, pinch_evap=pinch_evap).lcoe)
    return np.array(out)
