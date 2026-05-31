"""Benchmarks against the published literature.

These tests pin the model's headline numbers to values and rankings reported in
the open ORC / geothermal literature, converting "plausibility" into checks a
reviewer can trace:

* Subcritical ORC thermal efficiency at the canonical 100 C / 30 C condition is
  ~10-12% with realistic component efficiencies (Saleh et al., Energy 32 (2007)
  1210-1221; Su et al., Energy 142 (2018) — "around 10%").  Ideal (isentropic)
  efficiency is ~13-16% for these fluids.
* Working-fluid selection: for a ~150 C resource the best subcritical binary
  fluids are the light alkanes (isobutane / propane) ahead of higher-critical
  fluids — Augustine et al. (NREL) found isobutane optimal for 140-170 C; Astolfi
  / specific-power studies find propane strong near 150 C.
* Plant-level second-law (utilization) efficiency for a ~150 C binary resource
  lands in DiPippo's reported range (~30-55%).

These are bounds and consensus rankings, not a single proprietary heat-and-mass
balance; reproducing a specific named plant's full balance needs proprietary
data and is left as future work.
"""
import pytest

from geothermal_orc import (
    ORCCycle, GeothermalResource, screen_fluids, evaluate_plant,
)

CANONICAL = dict(T_evap_C=95.0, T_cond_C=30.0)   # subcritical, below all Tcrit


@pytest.mark.parametrize("fluid", ["Isobutane", "n-Pentane", "R245fa"])
def test_real_eta_th_matches_published_subcritical_range(fluid):
    real = ORCCycle(fluid, eta_pump=0.75, eta_turbine=0.80, **CANONICAL).solve()
    # Published subcritical ORC thermal efficiency is "around 10%".
    assert 0.09 < real.eta_th < 0.14


@pytest.mark.parametrize("fluid", ["Isobutane", "n-Pentane", "R245fa"])
def test_ideal_eta_th_matches_published_range(fluid):
    ideal = ORCCycle(fluid, eta_pump=1.0, eta_turbine=1.0, **CANONICAL).solve()
    assert 0.12 < ideal.eta_th < 0.17
    # An ideal (isentropic) cycle should sit well below the temperature-Carnot
    # bound but capture most of it, since its only irreversibility is the
    # non-isothermal heat exchange.
    assert 0.6 < ideal.eta_th / ideal.eta_carnot < 0.95


@pytest.fixture(scope="module")
def ranking_150C():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    return screen_fluids(
        ["Propane", "Isobutane", "n-Butane", "Isopentane", "n-Pentane", "R245fa"],
        r, T_cond_C=30.0)


def test_light_alkanes_top_the_150C_screen(ranking_150C):
    # Consensus: isobutane / propane lead at ~150 C (Augustine NREL; Astolfi).
    top2 = {r.fluid for r in ranking_150C[:2]}
    assert {"Propane", "Isobutane"} == top2
    # ...and they should beat the higher-critical pentanes/R245fa.
    by_fluid = {r.fluid: r.W_net_opt for r in ranking_150C}
    assert by_fluid["Isobutane"] > by_fluid["n-Pentane"]
    assert by_fluid["Isobutane"] > by_fluid["R245fa"]


def test_plant_utilization_in_dipippo_range():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    pr = evaluate_plant("Isobutane", r, ambient_C=10.0, T_evap_C=95.0)
    util = pr.cycle.eta_utilization
    # DiPippo reports second-law (utilization) efficiencies ~30-55% for
    # moderate-temperature binary plants.
    assert 0.30 < util < 0.55
