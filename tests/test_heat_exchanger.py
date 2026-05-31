"""Tests for LMTD and counter-flow pinch analysis."""

import math

import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI

from geothermal_orc.heat_exchanger import (
    lmtd,
    counterflow_profile,
    required_area,
)


def test_lmtd_known_value():
    # dT1=10, dT2=20 -> (20-10)/ln(20/10) = 10/ln2 = 14.4270
    assert lmtd(10.0, 20.0) == pytest.approx(10.0 / math.log(2.0), rel=1e-9)


def test_lmtd_equal_approaches_is_arithmetic_mean():
    assert lmtd(8.0, 8.0) == pytest.approx(8.0, rel=1e-9)


def test_lmtd_near_equal_no_blowup():
    val = lmtd(8.0, 8.0 + 1e-7)
    assert val == pytest.approx(8.0, rel=1e-5)


def test_lmtd_temperature_cross_raises():
    with pytest.raises(ValueError):
        lmtd(-2.0, 10.0)


def test_required_area_basic():
    # Q = U A dT_lm  ->  A = Q/(U dT_lm)
    assert required_area(1.0e6, 800.0, 12.5) == pytest.approx(1.0e6 / (800.0 * 12.5))


def _evap_profile(m_wf=70.0, n=60):
    fluid = "Isobutane"
    P_evap = PropsSI("P", "T", 393.15, "Q", 0, fluid)        # T_evap = 120 C
    h_in = PropsSI("H", "P", P_evap, "T", 320.0, fluid)      # subcooled liquid
    h_out = PropsSI("H", "P", P_evap, "Q", 1, fluid)         # sat vapour
    return counterflow_profile(
        hot_fluid="Water", m_hot=100.0, P_hot=1.0e6, T_hot_in=423.15,
        cold_fluid=fluid, m_cold=m_wf, P_cold=P_evap,
        h_cold_in=h_in, h_cold_out=h_out, n=n,
    )


def test_profile_duty_conservation():
    prof = _evap_profile()
    expected = prof.duty[-1]
    # Hot side gives up exactly the same total duty it receives.
    h_hot_in = PropsSI("H", "T", 423.15, "P", 1.0e6, "Water")
    h_hot_out = PropsSI("H", "T", float(prof.T_hot[0]), "P", 1.0e6, "Water")
    Q_hot = 100.0 * (h_hot_in - h_hot_out)
    assert Q_hot == pytest.approx(expected, rel=1e-6)


def test_profile_pinch_is_minimum_approach():
    prof = _evap_profile()
    approach = prof.T_hot - prof.T_cold
    assert prof.pinch == pytest.approx(float(np.min(approach)), abs=1e-9)


def test_profile_pinch_below_terminal_approaches():
    # Classic ORC result: the evaporator pinch is interior, not terminal.
    prof = _evap_profile()
    term_hot = prof.T_hot[-1] - prof.T_cold[-1]
    term_cold = prof.T_hot[0] - prof.T_cold[0]
    assert prof.pinch <= min(term_hot, term_cold) + 1e-9


def test_more_working_fluid_reduces_pinch():
    # Increasing the cold-side flow draws more heat and tightens the pinch.
    p_small = _evap_profile(m_wf=50.0)
    p_large = _evap_profile(m_wf=90.0)
    assert p_large.pinch < p_small.pinch


def test_profile_feasible_flag():
    prof = _evap_profile(m_wf=50.0)
    assert prof.feasible is (prof.pinch > 0.0)


def test_profile_temperatures_monotonic():
    prof = _evap_profile()
    # Both streams rise with cumulative duty (cold inlet -> cold outlet).
    assert np.all(np.diff(prof.T_cold) >= -1e-6)
    assert np.all(np.diff(prof.T_hot) >= -1e-6)


def test_counterflow_requires_enthalpy_gain():
    with pytest.raises(ValueError):
        counterflow_profile(
            hot_fluid="Water", m_hot=100.0, P_hot=1.0e6, T_hot_in=423.15,
            cold_fluid="Isobutane", m_cold=70.0, P_cold=1.0e6,
            h_cold_in=5.0e5, h_cold_out=4.0e5,  # decreasing -> invalid
        )
