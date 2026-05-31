"""Tests for silica solubility, scaling limits and thermal decline."""

import math

import pytest

from geothermal_orc.geothermal import (
    amorphous_silica_solubility,
    quartz_solubility,
    quartz_geothermometer,
    silica_saturation_index,
    min_reinjection_temperature,
    GeothermalResource,
    DECLINE_RATE_BINARY,
    DECLINE_RATE_FLASH,
)


# --- silica solubility validated against literature values --------------- #
def test_amorphous_silica_at_25C():
    # Fournier & Rowe (1977): ~115-120 mg/kg at 25 C.
    assert amorphous_silica_solubility(298.15) == pytest.approx(117.0, abs=5.0)


def test_amorphous_silica_at_100C():
    # ~370 mg/kg near the boiling point.
    assert amorphous_silica_solubility(373.15) == pytest.approx(365.0, abs=15.0)


def test_quartz_at_100C():
    # Quartz solubility ~48-52 mg/kg at 100 C.
    assert quartz_solubility(373.15) == pytest.approx(48.0, abs=4.0)


def test_quartz_at_150C():
    # ~125 mg/kg at 150 C.
    assert quartz_solubility(423.15) == pytest.approx(125.0, abs=6.0)


def test_amorphous_more_soluble_than_quartz():
    for T in (300.0, 350.0, 400.0, 450.0):
        assert amorphous_silica_solubility(T) > quartz_solubility(T)


def test_solubility_increases_with_temperature():
    Ts = [300.0, 350.0, 400.0, 450.0]
    vals = [quartz_solubility(T) for T in Ts]
    assert vals == sorted(vals)


def test_quartz_geothermometer_roundtrip():
    # Solubility at 200 C, then invert with the geothermometer.
    C = quartz_solubility(473.15)
    assert quartz_geothermometer(C) == pytest.approx(200.0, abs=3.0)


def test_saturation_index_definition():
    C = 200.0
    si = silica_saturation_index(C, 350.0)
    assert si == pytest.approx(C / amorphous_silica_solubility(350.0), rel=1e-12)


def test_saturation_index_above_one_is_supersaturated():
    # High silica load cooled to low T -> supersaturated.
    assert silica_saturation_index(400.0, 320.0) > 1.0


def test_min_reinjection_temperature_consistency():
    # At the reported minimum reinjection T, SI must equal 1 (saturation).
    C = 300.0
    T_min_C = min_reinjection_temperature(C, SI_limit=1.0)
    si = silica_saturation_index(C, T_min_C + 273.15)
    assert si == pytest.approx(1.0, abs=1e-3)


def test_higher_silica_raises_reinjection_limit():
    low = min_reinjection_temperature(150.0)
    high = min_reinjection_temperature(500.0)
    assert high > low


# --- resource & decline -------------------------------------------------- #
def test_resource_defaults_to_quartz_equilibrium_silica():
    r = GeothermalResource(T_reservoir_C=200.0, mass_flow=100.0)
    assert r.silica_mgkg == pytest.approx(quartz_solubility(473.15), rel=1e-9)


def test_linear_decline_default_rate():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)
    assert r.decline_rate == DECLINE_RATE_BINARY
    # 150 C at 0.5 %/yr for 10 yr -> 150*(1-0.05) = 142.5 C.
    assert r.temperature_at(10.0) == pytest.approx(142.5, abs=1e-6)


def test_exponential_decline():
    r = GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0,
                           decline_mode="exponential", decline_rate=0.01)
    assert r.temperature_at(5.0) == pytest.approx(150.0 * math.exp(-0.05), rel=1e-9)


def test_flash_rate_steeper_than_binary():
    assert DECLINE_RATE_FLASH > DECLINE_RATE_BINARY


def test_decline_at_time_zero_is_initial():
    r = GeothermalResource(T_reservoir_C=180.0, mass_flow=80.0)
    assert r.temperature_at(0.0) == pytest.approx(180.0, rel=1e-12)


def test_invalid_decline_mode_raises():
    with pytest.raises(ValueError):
        GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0,
                           decline_mode="quadratic")


def test_scaling_safe_check():
    r = GeothermalResource(T_reservoir_C=250.0, mass_flow=100.0)
    T_min = r.min_reinjection_temperature()
    # Just above the limit is safe; just below is not.
    assert r.scaling_safe(T_min + 2.0) is True
    assert r.scaling_safe(T_min - 2.0) is False
