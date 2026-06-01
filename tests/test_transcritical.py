"""Tests for transcritical ORC cycles (Tier 2)."""
import pytest

from geothermal_orc import (
    TranscriticalCycle, GeothermalResource, optimize_evaporation_temperature,
)
from CoolProp.CoolProp import PropsSI

PCRIT_PROPANE = PropsSI("Pcrit", "Propane")


def test_requires_supercritical_pressure():
    with pytest.raises(ValueError):
        TranscriticalCycle("Propane", P_high=0.9 * PCRIT_PROPANE,
                           T_turb_in_C=140.0, T_cond_C=30.0)


def test_turbine_inlet_above_condensing():
    with pytest.raises(ValueError):
        TranscriticalCycle("Propane", P_high=75e5, T_turb_in_C=20.0, T_cond_C=30.0)


def test_solve_per_kg():
    c = TranscriticalCycle("Propane", P_high=75e5, T_turb_in_C=140.0, T_cond_C=30.0)
    r = c.solve()
    assert r.w_net > 0.0 and r.q_in > 0.0
    assert 0.05 < r.eta_th < 0.20
    # State 2 (pump outlet) is supercritical.
    assert r.states[2].P > PCRIT_PROPANE


def test_resource_balances_close():
    c = TranscriticalCycle("Propane", P_high=75e5, T_turb_in_C=140.0, T_cond_C=30.0)
    res = c.solve_with_resource(m_brine=100.0, T_brine_in_C=150.0, pinch_heater=5.0)
    assert res.W_net > 0.0
    assert abs(res.energy_balance_residual) < 1e-9
    assert abs(res.exergy_balance_residual) < 1e-6
    assert res.brine_T_out < 150.0 + 273.15
    assert 0.3 < res.eta_utilization < 0.6


def test_transcritical_beats_subcritical_at_150C():
    # Subcritical propane is capped near its 96.7 C critical temperature; the
    # transcritical cycle reaches a much hotter turbine inlet and matches the
    # brine slope, so it out-produces the subcritical optimum at 150 C.
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    sub = optimize_evaporation_temperature("Propane", r, T_cond_C=30.0)
    trans = TranscriticalCycle("Propane", P_high=75e5, T_turb_in_C=140.0,
                               T_cond_C=30.0).solve_with_resource(
        m_brine=100.0, T_brine_in_C=150.0, pinch_heater=5.0)
    assert trans.W_net > sub.W_net_opt
    assert trans.eta_utilization > sub.result.eta_utilization
