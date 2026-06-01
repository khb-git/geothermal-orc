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


# --- recuperator (Tier 2) --------------------------------------------------- #
import pytest as _pytest
from geothermal_orc import GeothermalResource as _Resource


def _recup(fluid, eps):
    return ORCCycle(fluid, T_evap_C=120.0, T_cond_C=30.0,
                    recuperator_effectiveness=eps)


def test_recuperator_effectiveness_validation():
    with _pytest.raises(ValueError):
        ORCCycle("n-Pentane", T_evap_C=120.0, T_cond_C=30.0,
                 recuperator_effectiveness=-0.1)
    with _pytest.raises(ValueError):
        ORCCycle("n-Pentane", T_evap_C=120.0, T_cond_C=30.0,
                 recuperator_effectiveness=1.0)


def test_recuperator_absent_when_eps_zero():
    base = _recup("n-Pentane", 0.0).solve()
    assert base.recuperator_duty == 0.0
    assert 5 not in base.states and 6 not in base.states


def test_recuperator_raises_cycle_efficiency():
    etas = [_recup("n-Pentane", e).solve().eta_th for e in (0.0, 0.4, 0.8)]
    assert etas[0] < etas[1] < etas[2]


def test_recuperator_leaves_work_unchanged_but_cuts_heat():
    base = _recup("n-Pentane", 0.0).solve()
    rec = _recup("n-Pentane", 0.7).solve()
    # Turbine and pump work are untouched by an internal exchanger.
    assert rec.w_net == _pytest.approx(base.w_net, rel=1e-9)
    # Recovered heat reduces both the evaporator duty and the rejected heat.
    assert rec.recuperator_duty > 0.0
    assert rec.q_in < base.q_in
    assert rec.q_out < base.q_out


def test_recuperator_no_temperature_cross():
    rec = _recup("n-Pentane", 0.8).solve()
    s2, s4, s2r, s4r = rec.states[2], rec.states[4], rec.states[5], rec.states[6]
    assert s4r.T >= s2.T - 1e-6      # hot outlet not below cold inlet
    assert s2r.T <= s4.T + 1e-6      # cold outlet not above hot inlet


def test_recuperator_balances_close_on_resource():
    c = _recup("n-Pentane", 0.8)
    res = c.solve_with_resource(m_brine=100.0, T_brine_in_C=150.0, pinch_evap=5.0)
    assert abs(res.energy_balance_residual) < 1e-9
    assert abs(res.exergy_balance_residual) < 1e-6
    assert "recuperator" in res.exergy_destruction


def test_recuperator_raises_brine_reinjection_temperature():
    # The geothermal nuance: internal recovery means the brine gives up less
    # heat, so it is reinjected hotter (worse resource utilization).
    plain = _recup("n-Pentane", 0.0).solve_with_resource(
        m_brine=100.0, T_brine_in_C=150.0, pinch_evap=5.0)
    recup = _recup("n-Pentane", 0.8).solve_with_resource(
        m_brine=100.0, T_brine_in_C=150.0, pinch_evap=5.0)
    assert recup.brine_T_out > plain.brine_T_out + 1.0


# --- pressure drops (Tier 2) ------------------------------------------------ #
def test_pressure_drop_validation():
    with _pytest.raises(ValueError):
        ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0, dp_evap_frac=-0.01)
    with _pytest.raises(ValueError):
        ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0, dp_cond_frac=-0.01)


def test_zero_pressure_drop_is_backward_compatible():
    a = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0).solve()
    b = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0,
                 dp_evap_frac=0.0, dp_cond_frac=0.0).solve()
    assert a.w_net == _pytest.approx(b.w_net, rel=1e-12)


def test_pressure_drops_cost_net_power():
    base = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0).solve()
    drop = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0,
                    dp_evap_frac=0.05, dp_cond_frac=0.10).solve()
    assert drop.w_pump > base.w_pump          # pump must lift higher
    assert drop.w_turbine < base.w_turbine    # raised back-pressure
    assert drop.w_net < base.w_net
    assert drop.eta_th < base.eta_th


def test_pressure_drop_balances_close_on_resource():
    res = ORCCycle("Isobutane", T_evap_C=120.0, T_cond_C=30.0,
                   dp_evap_frac=0.05, dp_cond_frac=0.10).solve_with_resource(
        m_brine=100.0, T_brine_in_C=150.0, pinch_evap=5.0)
    assert abs(res.energy_balance_residual) < 1e-9
    assert abs(res.exergy_balance_residual) < 1e-6
