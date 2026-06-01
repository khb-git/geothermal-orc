"""Tests for the techno-economics module (Tier 3)."""
import pytest

from geothermal_orc import (
    GeothermalResource, EconomicAssumptions,
    bare_module_cost, capital_recovery_factor, levelized_cost,
)


def test_capital_recovery_factor():
    # 8% over 30 years is the standard ~0.0888.
    assert capital_recovery_factor(0.08, 30) == pytest.approx(0.0888, abs=1e-3)
    # Zero-rate limit is straight-line 1/n.
    assert capital_recovery_factor(0.0, 25) == pytest.approx(1.0 / 25)


def test_bare_module_cost_scales_and_updates():
    a = EconomicAssumptions()
    small = bare_module_cost("hx", 100.0, a)
    big = bare_module_cost("hx", 1000.0, a)
    assert 0 < small < big                      # larger area costs more
    # CEPCI scaling is linear in the index ratio.
    a2 = EconomicAssumptions(cepci_now=2 * a.cepci_now)
    assert bare_module_cost("hx", 100.0, a2) == pytest.approx(2 * small, rel=1e-9)


@pytest.fixture(scope="module")
def base():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    return levelized_cost("Isobutane", r, ambient_C=10.0, T_evap_C=95.0)


def test_lcoe_in_published_range(base):
    # Medium-temperature geothermal binary: ~$80-150/MWh (Lazard, IRENA).
    assert 60.0 < base.lcoe < 150.0
    assert base.annual_energy_MWh > 0.0


def test_capex_structure(base):
    # Wells dominate geothermal capital cost.
    assert base.capex_wells / base.capex_total > 0.4
    assert all(c > 0 for c in base.component_costs.values())
    assert base.capex_total == pytest.approx(
        base.capex_surface + base.capex_wells, rel=1e-9)
    # Specific cost in a sane band for binary plants.
    assert 3000.0 < base.specific_capex < 12000.0


def test_lcoe_rises_with_costlier_capital():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    cheap = levelized_cost("Isobutane", r, ambient_C=10.0, T_evap_C=95.0,
                           assume=EconomicAssumptions(discount_rate=0.06))
    dear = levelized_cost("Isobutane", r, ambient_C=10.0, T_evap_C=95.0,
                          assume=EconomicAssumptions(discount_rate=0.12))
    assert dear.lcoe > cheap.lcoe


def test_lower_capacity_factor_raises_lcoe():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    hi = levelized_cost("Isobutane", r, ambient_C=10.0, T_evap_C=95.0,
                        assume=EconomicAssumptions(capacity_factor=0.90))
    lo = levelized_cost("Isobutane", r, ambient_C=10.0, T_evap_C=95.0,
                        assume=EconomicAssumptions(capacity_factor=0.60))
    assert lo.lcoe > hi.lcoe


# --- cost-optimal vs power-optimal, sensitivity (Tier 3 items 3-4) ---------- #
import numpy as np
from geothermal_orc import pinch_lcoe_tradeoff, lcoe_sensitivity


@pytest.fixture(scope="module")
def resource():
    return GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)


def test_power_rises_as_pinch_tightens(resource):
    pin = [2, 4, 6, 9, 12]
    p, W, L = pinch_lcoe_tradeoff("Isobutane", resource, 10.0, pin, T_evap_C=95.0)
    assert len(p) == len(W) == len(L) == 5
    assert all(W[i] > W[i + 1] for i in range(len(W) - 1))   # monotonic in pinch


def test_cost_optimal_pinch_depends_on_cost_structure(resource):
    pin = [2, 3, 5, 7, 9, 12]
    # Well-dominated CAPEX: the fixed wells reward maximum energy -> tight pinch.
    _, W, L = pinch_lcoe_tradeoff("Isobutane", resource, 10.0, pin, T_evap_C=95.0)
    assert int(np.argmin(L)) == int(np.argmax(W))            # cost-opt == power-opt
    # Cheap wells: surface (area) cost bites, so the cost-optimal pinch widens.
    cheap = EconomicAssumptions(well_cost=1.5e6, n_wells=2)
    _, Wc, Lc = pinch_lcoe_tradeoff("Isobutane", resource, 10.0, pin,
                                    cheap, T_evap_C=95.0)
    assert int(np.argmin(Lc)) > int(np.argmax(Wc))           # wider than power-opt


def test_sensitivity_monotonicity(resource):
    well = lcoe_sensitivity("Isobutane", resource, 10.0, "well_cost",
                            [3e6, 5e6, 7e6], T_evap_C=95.0)
    assert well[0] < well[1] < well[2]
    disc = lcoe_sensitivity("Isobutane", resource, 10.0, "discount_rate",
                            [0.05, 0.08, 0.12], T_evap_C=95.0)
    assert disc[0] < disc[1] < disc[2]
    cf = lcoe_sensitivity("Isobutane", resource, 10.0, "capacity_factor",
                          [0.6, 0.75, 0.9], T_evap_C=95.0)
    assert cf[0] > cf[1] > cf[2]                              # higher CF -> lower LCOE
