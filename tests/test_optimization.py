"""Tests for evaporation-temperature optimisation and fluid screening."""

import pytest

from geothermal_orc.geothermal import GeothermalResource
from geothermal_orc.optimization import (
    optimize_evaporation_temperature,
    screen_fluids,
)
from geothermal_orc.cycle import ORCCycle


@pytest.fixture
def resource():
    return GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)


def test_optimizer_returns_feasible(resource):
    opt = optimize_evaporation_temperature("Isobutane", resource, T_cond_C=30.0)
    assert opt.feasible
    assert opt.W_net_opt > 0.0


def test_optimum_within_bounds(resource):
    opt = optimize_evaporation_temperature("Isobutane", resource, T_cond_C=30.0)
    Tc_C = ORCCycle("Isobutane", T_evap_C=50, T_cond_C=30).T_evap  # placeholder
    assert 35.0 < opt.T_evap_opt_C < 150.0


def test_optimum_beats_arbitrary_interior_point(resource):
    opt = optimize_evaporation_temperature("Isobutane", resource, T_cond_C=30.0)
    # A non-optimal evaporation temperature should not exceed the optimum.
    cyc = ORCCycle("Isobutane", T_evap_C=opt.T_evap_opt_C - 20.0, T_cond_C=30.0)
    other = cyc.solve_with_resource(m_brine=100.0, T_brine_in_C=150.0, pinch_evap=5.0)
    assert opt.W_net_opt >= other.W_net - 1e-6


def test_optimum_result_is_consistent(resource):
    opt = optimize_evaporation_temperature("Isobutane", resource, T_cond_C=30.0)
    assert opt.result is not None
    assert opt.result.W_net == pytest.approx(opt.W_net_opt, rel=1e-6)


def test_silica_constraint_enforced():
    # A very silica-rich resource forces a high reinjection floor, which the
    # optimiser must respect (brine cannot be cooled below the scaling limit).
    rich = GeothermalResource(T_reservoir_C=250.0, mass_flow=100.0)
    opt = optimize_evaporation_temperature("Isopentane", rich, T_cond_C=30.0)
    if opt.feasible:
        T_reinject_C = opt.result.brine_T_out - 273.15
        assert rich.scaling_safe(T_reinject_C)


def test_screen_fluids_ranks_by_power(resource):
    results = screen_fluids(
        ["Isobutane", "n-Pentane", "R245fa", "Isopentane"],
        resource, T_cond_C=30.0,
    )
    feasible = [r for r in results if r.feasible]
    powers = [r.W_net_opt for r in feasible]
    assert powers == sorted(powers, reverse=True)
    assert len(feasible) >= 1


def test_screen_fluids_returns_one_per_input(resource):
    fluids = ["Isobutane", "n-Pentane", "R245fa"]
    results = screen_fluids(fluids, resource, T_cond_C=30.0)
    assert {r.fluid for r in results} == set(fluids)
