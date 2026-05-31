"""Cycle optimisation and multi-fluid screening.

:func:`optimize_evaporation_temperature` maximises net power over the
evaporation temperature for a fixed fluid and resource, honouring three
constraints commonly binding in geothermal binary design:

* subcritical operation (``T_evap`` below the critical temperature),
* a feasible evaporator pinch (working fluid must stay below the brine),
* the silica scaling limit on brine reinjection temperature.

Infeasible evaporation temperatures are penalised so a bounded scalar optimiser
converges onto the feasible interior optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from scipy.optimize import minimize_scalar

from .cycle import ORCCycle, CycleResult
from .thermo import critical_temperature
from .geothermal import GeothermalResource


@dataclass
class OptimizationResult:
    fluid: str
    T_evap_opt_C: float
    W_net_opt: float                  # W
    result: CycleResult
    feasible: bool
    note: str = ""


def _evaluate(
    fluid: str,
    T_evap_C: float,
    resource: GeothermalResource,
    T_cond_C: float,
    superheat: float,
    eta_pump: float,
    eta_turbine: float,
    pinch_evap: float,
    cooling_T_in_C: float,
    cooling_pinch: float,
    SI_limit: float,
    n: int = 80,
    n_search: Optional[int] = None,
) -> Optional[CycleResult]:
    """Return a resource-coupled result, or ``None`` if infeasible."""
    try:
        cyc = ORCCycle(
            fluid=fluid, T_evap_C=T_evap_C, T_cond_C=T_cond_C,
            superheat=superheat, eta_pump=eta_pump, eta_turbine=eta_turbine,
        )
        res = cyc.solve_with_resource(
            m_brine=resource.mass_flow,
            T_brine_in_C=resource.T_reservoir_C,
            pinch_evap=pinch_evap,
            cooling_T_in_C=cooling_T_in_C,
            cooling_pinch=cooling_pinch,
            n=n,
            n_search=n_search,
        )
    except (ValueError, RuntimeError):
        return None
    if res.evaporator is not None and not res.evaporator.feasible:
        return None
    # Silica scaling constraint on the brine leaving the evaporator.
    T_reinject_C = res.brine_T_out - 273.15
    if not resource.scaling_safe(T_reinject_C, SI_limit):
        return None
    return res


def optimize_evaporation_temperature(
    fluid: str,
    resource: GeothermalResource,
    T_cond_C: float = 30.0,
    superheat: float = 0.0,
    eta_pump: float = 0.75,
    eta_turbine: float = 0.80,
    pinch_evap: float = 5.0,
    cooling_T_in_C: float = 15.0,
    cooling_pinch: float = 5.0,
    SI_limit: float = 1.0,
    T_evap_bounds_C: Optional[tuple] = None,
) -> OptimizationResult:
    """Find the evaporation temperature that maximises net power."""
    Tc_C = critical_temperature(fluid) - 273.15
    if T_evap_bounds_C is None:
        lo = T_cond_C + 5.0
        hi = min(Tc_C - 2.0, resource.T_reservoir_C - pinch_evap - 1.0)
        T_evap_bounds_C = (lo, hi)
    lo, hi = T_evap_bounds_C
    if hi <= lo:
        return OptimizationResult(fluid, float("nan"), float("-inf"),
                                  None, False, "empty feasible interval")

    worst = -1e30

    # A coarse profile is enough to locate the optimum; the winning point is
    # re-solved at full resolution below.
    N_SEARCH_PROFILE = 60
    N_SEARCH_GRID = 30

    def neg_power(T_evap_C: float) -> float:
        res = _evaluate(fluid, T_evap_C, resource, T_cond_C, superheat,
                        eta_pump, eta_turbine, pinch_evap,
                        cooling_T_in_C, cooling_pinch, SI_limit,
                        n=N_SEARCH_PROFILE, n_search=N_SEARCH_GRID)
        if res is None:
            return -worst        # large positive => penalised
        return -res.W_net

    opt = minimize_scalar(neg_power, bounds=(lo, hi), method="bounded",
                          options={"xatol": 0.05})
    T_opt = float(opt.x)
    best = _evaluate(fluid, T_opt, resource, T_cond_C, superheat,
                     eta_pump, eta_turbine, pinch_evap,
                     cooling_T_in_C, cooling_pinch, SI_limit)
    if best is None:
        return OptimizationResult(fluid, T_opt, float("-inf"), None, False,
                                  "no feasible evaporation temperature found")
    return OptimizationResult(fluid, T_opt, best.W_net, best, True)


def screen_fluids(
    fluids: List[str],
    resource: GeothermalResource,
    **kwargs,
) -> List[OptimizationResult]:
    """Optimise every fluid in ``fluids`` and rank by net power (descending)."""
    results = [optimize_evaporation_temperature(f, resource, **kwargs)
               for f in fluids]
    results.sort(key=lambda r: (r.feasible, r.W_net_opt), reverse=True)
    return results
