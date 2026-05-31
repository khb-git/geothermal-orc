"""Tests for the binary ORC cycle solver.

The backbone here is conservation: energy and exergy balances must close to
numerical precision for any valid case.  These are deterministic, model-agnostic
checks that catch sign errors, mislabelled states and bad bookkeeping.
"""

import pytest

from geothermal_orc.cycle import ORCCycle


def make_cycle(fluid="Isobutane", T_evap_C=120.0, T_cond_C=30.0, **kw):
    return ORCCycle(fluid, T_evap_C=T_evap_C, T_cond_C=T_cond_C, **kw)


def test_energy_closure_per_mass():
    res = make_cycle().solve()
    # q_in - q_out - w_net = 0 per unit mass.
    assert (res.q_in - res.q_out - res.w_net) == pytest.approx(0.0, abs=1e-6)


def test_thermal_efficiency_definition():
    res = make_cycle().solve()
    assert res.eta_th == pytest.approx(res.w_net / res.q_in, rel=1e-12)


def test_efficiency_below_carnot():
    res = make_cycle().solve()
    assert 0.0 < res.eta_th < res.eta_carnot


def test_carnot_from_reservoir_temperatures():
    cyc = make_cycle(T_evap_C=120.0, T_cond_C=30.0)
    res = cyc.solve()
    assert res.eta_carnot == pytest.approx(1.0 - 303.15 / 393.15, rel=1e-9)


def test_dry_fluid_superheated_turbine_exit():
    # Isobutane is a dry fluid: expansion ends superheated (Q == -1).
    res = make_cycle("Isobutane").solve()
    assert res.turbine_exit_quality == -1.0


def test_wet_fluid_two_phase_turbine_exit():
    # R152a is wet: expansion from saturated vapour ends inside the dome.
    res = make_cycle("R152a", T_evap_C=90.0, T_cond_C=30.0).solve()
    assert 0.0 <= res.turbine_exit_quality <= 1.0


def test_supercritical_T_evap_raises():
    with pytest.raises(ValueError):
        # Propane Tcrit ~ 96.7 C; ask for 100 C evaporation.
        ORCCycle("Propane", T_evap_C=100.0, T_cond_C=30.0)


def test_cond_above_evap_raises():
    with pytest.raises(ValueError):
        ORCCycle("Isobutane", T_evap_C=40.0, T_cond_C=50.0)


def test_higher_turbine_efficiency_increases_work():
    lo = make_cycle(eta_turbine=0.70).solve()
    hi = make_cycle(eta_turbine=0.85).solve()
    assert hi.w_net > lo.w_net
    assert hi.eta_th > lo.eta_th


def test_superheat_changes_state_three_temperature():
    base = make_cycle(superheat=0.0).solve()
    sup = make_cycle(superheat=15.0).solve()
    assert sup.states[3].T == pytest.approx(base.states[3].T + 15.0, abs=0.5)


# ----------------------- resource-coupled tests ------------------------- #
def make_resource_result(**kw):
    cyc = make_cycle(**kw)
    return cyc.solve_with_resource(m_brine=100.0, T_brine_in_C=150.0,
                                   pinch_evap=5.0)


def test_resource_energy_balance_closes():
    res = make_resource_result()
    assert abs(res.energy_balance_residual) < 1e-9


def test_resource_exergy_balance_closes():
    # E_in = W_net + E_out + sum(E_destroyed) to numerical precision.
    res = make_resource_result()
    assert abs(res.exergy_balance_residual) < 1e-6


def test_evaporator_pinch_matches_target():
    res = make_resource_result()
    assert res.evaporator.pinch == pytest.approx(5.0, abs=1e-3)


def test_brine_cooled_but_above_zero():
    res = make_resource_result()
    assert res.brine_T_in > res.brine_T_out > 273.15


def test_utilization_efficiency_between_zero_and_one():
    res = make_resource_result()
    assert 0.0 < res.eta_utilization < 1.0


def test_net_power_positive_and_scales_with_brine():
    cyc = make_cycle()
    small = cyc.solve_with_resource(m_brine=50.0, T_brine_in_C=150.0, pinch_evap=5.0)
    large = cyc.solve_with_resource(m_brine=150.0, T_brine_in_C=150.0, pinch_evap=5.0)
    assert small.W_net > 0
    assert large.W_net == pytest.approx(3.0 * small.W_net, rel=1e-3)


def test_exergy_destruction_all_nonnegative():
    res = make_resource_result()
    for component, value in res.exergy_destruction.items():
        assert value >= -1e-6, f"{component} destroyed negative exergy"


def test_evaporator_dominates_exergy_destruction():
    # In geothermal binary plants the brine-side temperature mismatch makes the
    # evaporator the largest single exergy sink.
    res = make_resource_result()
    ed = res.exergy_destruction
    assert ed["evaporator"] == max(ed.values())


def test_W_net_equals_m_wf_times_w_net():
    res = make_resource_result()
    assert res.W_net == pytest.approx(res.m_wf * res.w_net, rel=1e-12)
