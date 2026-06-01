"""Tests for the project-finance module (Tier 4)."""
import numpy as np
import pytest

from geothermal_orc import (
    GeothermalResource, levelized_cost,
    ProjectAssumptions, project_cashflow, breakeven_ppa, monte_carlo,
)
from geothermal_orc.economics import EconomicAssumptions


@pytest.fixture(scope="module")
def econ():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    return levelized_cost("Isobutane", r, ambient_C=10.0, T_evap_C=95.0)


def test_breakeven_equals_lcoe(econ):
    # Under matching assumptions, break-even PPA == Tier 3 LCOE by construction.
    proj = ProjectAssumptions(ppa_price=econ.lcoe, om_escalation=0.0)
    assert breakeven_ppa(econ, proj) == pytest.approx(econ.lcoe, rel=1e-3)
    assert project_cashflow(econ, proj).npv == pytest.approx(0.0, abs=1.0)


def test_irr_equals_discount_at_breakeven(econ):
    # If NPV = 0 at the discount rate, the IRR is that discount rate.
    proj = ProjectAssumptions(ppa_price=econ.lcoe, om_escalation=0.0,
                              discount_rate=0.08)
    assert project_cashflow(econ, proj).irr == pytest.approx(0.08, abs=2e-3)


def test_npv_monotonic_in_price(econ):
    npvs = [project_cashflow(econ, ProjectAssumptions(ppa_price=p)).npv
            for p in (80, 100, 120, 140)]
    assert all(npvs[i] < npvs[i + 1] for i in range(len(npvs) - 1))
    # Profitable above LCOE, unprofitable below.
    assert project_cashflow(econ, ProjectAssumptions(ppa_price=econ.lcoe + 30)).npv > 0
    assert project_cashflow(econ, ProjectAssumptions(ppa_price=econ.lcoe - 30)).npv < 0


def test_leverage_raises_equity_irr(econ):
    # Debt cheaper than the unlevered return lifts the equity IRR.
    base = project_cashflow(econ, ProjectAssumptions(ppa_price=120.0))
    levered = project_cashflow(econ, ProjectAssumptions(
        ppa_price=120.0, debt_fraction=0.6, debt_rate=0.06))
    assert levered.irr > base.irr
    assert 0 < base.payback_yr < 30


def test_tax_reduces_npv(econ):
    pretax = project_cashflow(econ, ProjectAssumptions(ppa_price=120.0))
    aftertax = project_cashflow(econ, ProjectAssumptions(
        ppa_price=120.0, tax_rate=0.25, depreciation_yr=10.0))
    assert aftertax.npv < pretax.npv


def test_monte_carlo_distribution(econ):
    mc = monte_carlo(econ, ProjectAssumptions(ppa_price=110.0), n=1500, seed=1)
    assert len(mc["lcoe"]) == len(mc["npv"]) == 1500
    lp = mc["lcoe_percentiles"]
    assert lp[10] < lp[50] < lp[90]
    np_ = mc["npv_percentiles"]
    assert np_[10] < np_[50] < np_[90]
    assert 0.0 <= mc["p_npv_positive"] <= 1.0
    # Median LCOE tracks the deterministic value (well cost ~ symmetric).
    assert lp[50] == pytest.approx(econ.lcoe, rel=0.10)
