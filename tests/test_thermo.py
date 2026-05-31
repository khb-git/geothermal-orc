"""Tests for the CoolProp-backed thermodynamic state layer."""

import math

import pytest

from geothermal_orc.thermo import (
    State,
    specific_flow_exergy,
    saturation_pressure,
    saturation_temperature,
    critical_temperature,
    critical_pressure,
    DEAD_STATE_T,
    DEAD_STATE_P,
)


def test_water_saturation_pressure_at_100C():
    # Boiling point of water at ~1 atm: published value 101.325 kPa.
    P = saturation_pressure("Water", 373.15)
    assert P == pytest.approx(101_325.0, rel=0.01)


def test_saturation_pressure_temperature_roundtrip():
    P = saturation_pressure("Isobutane", 350.0)
    T = saturation_temperature("Isobutane", P)
    assert T == pytest.approx(350.0, abs=1e-3)


def test_state_from_TP_recovers_inputs():
    s = State.from_TP("R245fa", 320.0, 3.0e5)
    assert s.T == pytest.approx(320.0, abs=1e-6)
    assert s.P == pytest.approx(3.0e5, abs=1e-3)


def test_state_property_self_consistency_Ph():
    # Build from (T,P), then rebuild from (P,h): must land on same state.
    a = State.from_TP("n-Pentane", 360.0, 2.0e5)
    b = State.from_Ph("n-Pentane", a.P, a.h)
    assert b.T == pytest.approx(a.T, abs=1e-4)
    assert b.s == pytest.approx(a.s, rel=1e-6)


def test_state_property_self_consistency_Ps():
    a = State.from_TP("n-Pentane", 360.0, 2.0e5)
    b = State.from_Ps("n-Pentane", a.P, a.s)
    assert b.h == pytest.approx(a.h, rel=1e-6)


def test_saturated_liquid_quality_zero():
    s = State.from_PQ("Isobutane", 4.0e5, 0.0)
    assert s.Q == pytest.approx(0.0, abs=1e-9)
    assert s.is_two_phase


def test_single_phase_quality_is_minus_one():
    s = State.from_TP("Isobutane", 320.0, 30.0e5)  # compressed liquid
    assert s.Q == -1.0
    assert not s.is_two_phase


def test_flow_exergy_zero_at_dead_state():
    # A state evaluated at the dead state has zero flow exergy by definition.
    s = State.from_TP("Water", DEAD_STATE_T, DEAD_STATE_P)
    assert s.flow_exergy(T0=DEAD_STATE_T) == pytest.approx(0.0, abs=1.0)


def test_flow_exergy_positive_above_dead_state():
    s = State.from_TP("Water", 423.15, 1.0e6)  # 150 C brine
    assert s.flow_exergy(T0=DEAD_STATE_T) > 0.0


def test_specific_flow_exergy_matches_method():
    s = State.from_TP("Water", 423.15, 1.0e6)
    e_func = specific_flow_exergy("Water", s.h, s.s, T0=DEAD_STATE_T)
    e_meth = s.flow_exergy(T0=DEAD_STATE_T)
    assert e_func == pytest.approx(e_meth, rel=1e-9)


def test_critical_properties_isobutane():
    # CoolProp reference values for isobutane.
    assert critical_temperature("Isobutane") == pytest.approx(407.81, abs=0.5)
    assert critical_pressure("Isobutane") == pytest.approx(36.29e5, rel=0.01)


def test_celsius_property():
    s = State.from_TP("Water", 373.15, 2.0e5)
    assert s.T_celsius == pytest.approx(100.0, abs=1e-6)


def test_frozen_state_is_immutable():
    s = State.from_TP("Water", 373.15, 2.0e5)
    with pytest.raises(Exception):
        s.T = 400.0  # dataclass(frozen=True)
