"""Project finance and risk: from levelized cost to an investment decision.

Tier 3 answered "what does a megawatt-hour cost?".  This layer answers "is the
project worth building, and how risky is that answer?".  It wraps the
:class:`~geothermal_orc.economics.EconomicResult` in a discounted-cash-flow
model — revenue at a power-purchase-agreement (PPA) price, escalating O&M,
optional debt financing, tax and straight-line depreciation — and reports the
net present value (NPV), internal rate of return (IRR), payback period, and the
break-even PPA.  A Monte-Carlo layer propagates uncertainty in the handful of
inputs that actually move the result (PPA price, well cost, capacity factor,
cost of capital) into distributions of LCOE and NPV.

Internal consistency: under all-equity, no-tax, no-escalation assumptions with a
discount rate equal to the one behind the Tier 3 capital-recovery factor, the
break-even PPA equals the Tier 3 LCOE *exactly* — the LCOE is, by construction,
the price at which NPV is zero.  The :func:`project_cashflow` model reproduces
that identity before it generalizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from .economics import EconomicResult, EconomicAssumptions, capital_recovery_factor


@dataclass
class ProjectAssumptions:
    """Revenue, financing, and tax assumptions for the cash-flow model."""

    ppa_price: float = 90.0        # USD/MWh, power-purchase price (year 1)
    ppa_escalation: float = 0.0    # annual price escalation (fraction)
    om_escalation: float = 0.02    # annual O&M escalation (fraction)
    debt_fraction: float = 0.0     # 0 = all equity
    debt_rate: float = 0.06        # loan interest rate
    debt_term_yr: float = 15.0
    tax_rate: float = 0.0          # 0 = pre-tax
    depreciation_yr: float = 0.0   # 0 = no depreciation; else straight-line years
    discount_rate: float = 0.08    # equity / project discount rate
    plant_life_yr: float = 30.0


@dataclass
class ProjectResult:
    npv: float
    irr: float
    payback_yr: float
    breakeven_ppa: float
    lcoe: float
    cashflow: np.ndarray           # equity cash flow, year 0..life
    years: np.ndarray


# --- cash-flow primitives --------------------------------------------------- #
def _amortization(principal: float, rate: float, term: int):
    """Level-payment loan schedule -> (interest[t], principal[t]) per year."""
    if principal <= 0.0 or term <= 0:
        return np.zeros(0), np.zeros(0)
    if rate == 0.0:
        pay = principal / term
        prin = np.full(term, pay)
        return np.zeros(term), prin
    pay = principal * rate / (1.0 - (1.0 + rate) ** (-term))
    interest, principal_pay, bal = [], [], principal
    for _ in range(term):
        i = bal * rate
        p = pay - i
        bal -= p
        interest.append(i)
        principal_pay.append(p)
    return np.array(interest), np.array(principal_pay)


def _om_year1(econ: EconomicResult) -> float:
    """Recover the year-1 O&M cost embedded in the Tier 3 annualized cost."""
    return econ.annual_cost - econ.crf * econ.capex_total


def _build_cashflow(econ: EconomicResult, proj: ProjectAssumptions,
                    ppa_price: float) -> np.ndarray:
    """Equity cash flow array (year 0..life) at a given PPA price."""
    life = int(round(proj.plant_life_yr))
    capex = econ.capex_total
    energy = econ.annual_energy_MWh
    om0 = _om_year1(econ)
    debt = capex * proj.debt_fraction
    equity0 = capex - debt
    interest, principal = _amortization(debt, proj.debt_rate,
                                        int(round(proj.debt_term_yr)))

    cfs = np.zeros(life + 1)
    cfs[0] = -equity0
    for t in range(1, life + 1):
        rev = energy * ppa_price * (1.0 + proj.ppa_escalation) ** (t - 1)
        om = om0 * (1.0 + proj.om_escalation) ** (t - 1)
        dep = (capex / proj.depreciation_yr
               if proj.depreciation_yr > 0 and t <= proj.depreciation_yr else 0.0)
        i_t = interest[t - 1] if t - 1 < len(interest) else 0.0
        p_t = principal[t - 1] if t - 1 < len(principal) else 0.0
        ebt = rev - om - dep - i_t
        tax = max(ebt, 0.0) * proj.tax_rate
        cfs[t] = rev - om - i_t - p_t - tax
    return cfs


def _npv(rate: float, cfs: np.ndarray) -> float:
    t = np.arange(len(cfs))
    return float(np.sum(cfs / (1.0 + rate) ** t))


def _irr(cfs: np.ndarray) -> float:
    if not (np.any(cfs > 0) and np.any(cfs < 0)):
        return float("nan")
    f = lambda r: _npv(r, cfs)
    try:
        if f(-0.5) * f(2.0) > 0:
            return float("nan")
        return float(brentq(f, -0.5, 2.0, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def _payback(cfs: np.ndarray) -> float:
    cum = np.cumsum(cfs)
    for t in range(1, len(cum)):
        if cum[t] >= 0:
            prev = cum[t - 1]
            return (t - 1) + (-prev) / (cum[t] - prev)   # linear interpolation
    return float("nan")


# --- public API ------------------------------------------------------------- #
def breakeven_ppa(econ: EconomicResult, proj: ProjectAssumptions) -> float:
    """PPA price at which NPV = 0."""
    f = lambda p: _npv(proj.discount_rate, _build_cashflow(econ, proj, p))
    return float(brentq(f, 0.0, 5000.0, maxiter=200))


def project_cashflow(econ: EconomicResult,
                     proj: ProjectAssumptions = ProjectAssumptions()) -> ProjectResult:
    """Full DCF: NPV, IRR, payback, and break-even PPA for a project."""
    cfs = _build_cashflow(econ, proj, proj.ppa_price)
    return ProjectResult(
        npv=_npv(proj.discount_rate, cfs),
        irr=_irr(cfs),
        payback_yr=_payback(cfs),
        breakeven_ppa=breakeven_ppa(econ, proj),
        lcoe=econ.lcoe,
        cashflow=cfs,
        years=np.arange(len(cfs)),
    )


def _sample(rng: np.random.RandomState, spec: Tuple) -> float:
    kind = spec[0]
    if kind == "normal":
        return rng.normal(spec[1], spec[2])
    if kind == "uniform":
        return rng.uniform(spec[1], spec[2])
    if kind == "triangular":
        return rng.triangular(spec[1], spec[2], spec[3])
    raise ValueError(f"unknown distribution {kind!r}")


def monte_carlo(
    econ: EconomicResult,
    proj: ProjectAssumptions = ProjectAssumptions(),
    assume: EconomicAssumptions = EconomicAssumptions(),
    distributions: Optional[Dict[str, Tuple]] = None,
    n: int = 2000,
    seed: int = 0,
) -> dict:
    """Propagate input uncertainty into LCOE and NPV distributions.

    Holds the *plant design* fixed (net power, component areas) and samples the
    economic/financial drivers that dominate the result.  ``distributions`` maps
    any of ``ppa_price, well_cost, capacity_factor, discount_rate, om_frac_capex``
    to a tuple ``("normal", mean, sd)``, ``("uniform", lo, hi)`` or
    ``("triangular", lo, mode, hi)``.  Returns sampled arrays, percentiles, and
    the probability that NPV > 0."""
    rng = np.random.RandomState(seed)
    if distributions is None:
        distributions = {
            "ppa_price":       ("triangular", 0.8 * proj.ppa_price,
                                proj.ppa_price, 1.4 * proj.ppa_price),
            "well_cost":       ("normal", assume.well_cost, 0.25 * assume.well_cost),
            "capacity_factor": ("triangular", 0.70, assume.capacity_factor, 0.95),
            "discount_rate":   ("normal", assume.discount_rate, 0.02),
        }

    surface = econ.capex_surface
    life = proj.plant_life_yr
    lcoe = np.empty(n)
    npv = np.empty(n)

    for k in range(n):
        well = max(1e5, _sample(rng, distributions["well_cost"])
                   if "well_cost" in distributions else assume.well_cost)
        cf = float(np.clip(_sample(rng, distributions["capacity_factor"])
                   if "capacity_factor" in distributions else assume.capacity_factor,
                   0.30, 0.98))
        disc = max(0.01, _sample(rng, distributions["discount_rate"])
                   if "discount_rate" in distributions else assume.discount_rate)
        ppa = max(0.0, _sample(rng, distributions["ppa_price"])
                  if "ppa_price" in distributions else proj.ppa_price)
        om_frac = (_sample(rng, distributions["om_frac_capex"])
                   if "om_frac_capex" in distributions else assume.om_frac_capex)

        capex_total = surface + assume.n_wells * well
        energy = econ.W_net_plant * cf * 8760.0 / 1e6
        crf = capital_recovery_factor(disc, life)
        om0 = om_frac * surface
        lcoe[k] = (crf * capex_total + om0) / energy

        # Lightweight EconomicResult for the DCF (only fields _build_cashflow uses).
        e = EconomicResult(
            fluid=econ.fluid, capex_total=capex_total, capex_surface=surface,
            capex_wells=capex_total - surface, component_costs=econ.component_costs,
            areas=econ.areas, annual_energy_MWh=energy,
            annual_cost=crf * capex_total + om0, crf=crf, lcoe=lcoe[k],
            W_net_plant=econ.W_net_plant, specific_capex=capex_total / (econ.W_net_plant / 1e3))
        p = ProjectAssumptions(**{**proj.__dict__, "ppa_price": ppa, "discount_rate": disc})
        npv[k] = _npv(disc, _build_cashflow(e, p, ppa))

    pct = lambda a: {p: float(np.percentile(a, p)) for p in (10, 50, 90)}
    return {
        "lcoe": lcoe, "npv": npv,
        "lcoe_percentiles": pct(lcoe), "npv_percentiles": pct(npv),
        "p_npv_positive": float(np.mean(npv > 0)), "n": n,
    }
