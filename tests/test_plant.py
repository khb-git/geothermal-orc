"""Tests for the plant-boundary layer: parasitics, ambient condensing, season."""
import numpy as np
import pytest

from geothermal_orc import (
    GeothermalResource,
    evaluate_plant,
    seasonal_performance,
    air_cooled_fan_power,
    brine_pump_power,
    condensing_temperature,
)


@pytest.fixture(scope="module")
def resource():
    return GeothermalResource(T_reservoir_C=150.0, mass_flow=100.0)


# --- parasitic-load primitives --------------------------------------------- #
def test_fan_power_zero_for_no_rejection():
    assert air_cooled_fan_power(0.0) == 0.0
    assert air_cooled_fan_power(-5.0) == 0.0


def test_fan_power_scales_linearly_with_duty():
    p1 = air_cooled_fan_power(10e6)
    p2 = air_cooled_fan_power(20e6)
    assert p1 > 0.0
    assert p2 == pytest.approx(2.0 * p1, rel=1e-9)


def test_fan_power_drops_with_more_air_rise():
    # A larger allowed air temperature rise needs less air, hence less fan power.
    assert air_cooled_fan_power(10e6, dT_air=20.0) < air_cooled_fan_power(10e6, dT_air=10.0)


def test_brine_pump_power_scales_with_flow():
    assert brine_pump_power(0.0) == 0.0
    assert brine_pump_power(200.0) == pytest.approx(2.0 * brine_pump_power(100.0), rel=1e-9)


def test_condensing_temperature_is_ambient_plus_approach():
    assert condensing_temperature(15.0, approach=20.0) == pytest.approx(35.0)


# --- plant evaluation ------------------------------------------------------- #
def test_parasitics_reduce_power_in_order(resource):
    pr = evaluate_plant("Isobutane", resource, ambient_C=10.0, T_evap_C=95.0)
    assert pr.feasible
    # gross > cycle-net (pump) > plant-net (pump + parasitics)
    assert pr.W_gross > pr.W_net_cycle > pr.W_net_plant
    assert pr.P_fan > 0.0 and pr.P_brine_pump > 0.0


def test_net_gross_ratio_in_plausible_band(resource):
    pr = evaluate_plant("Isobutane", resource, ambient_C=10.0, T_evap_C=95.0)
    # Air-cooled binary plants typically deliver 70-92% of gross.
    assert 0.6 < pr.net_gross_ratio < 0.95
    assert 0.0 < pr.parasitic_fraction < 0.4


def test_fan_power_is_a_meaningful_share_of_gross(resource):
    pr = evaluate_plant("Isobutane", resource, ambient_C=10.0, T_evap_C=95.0)
    assert pr.P_fan / pr.W_gross > 0.03


def test_cycle_net_matches_underlying_solve(resource):
    pr = evaluate_plant("Isobutane", resource, ambient_C=10.0, T_evap_C=95.0)
    assert pr.W_net_cycle == pytest.approx(pr.cycle.W_net, rel=1e-9)


def test_default_condensing_follows_ambient(resource):
    pr = evaluate_plant("Isobutane", resource, ambient_C=12.0,
                        condenser_approach=20.0, T_evap_C=95.0)
    assert pr.T_cond_C == pytest.approx(32.0)


def test_hotter_ambient_lowers_net_power(resource):
    powers = [
        evaluate_plant("Isobutane", resource, ambient_C=a, T_evap_C=95.0).W_net_plant
        for a in (5.0, 15.0, 25.0, 35.0)
    ]
    # Strictly decreasing: a warmer heat sink both lowers cycle power and
    # raises fan duty.
    assert all(powers[i] > powers[i + 1] for i in range(len(powers) - 1))


# --- seasonal roll-up ------------------------------------------------------- #
def test_seasonal_constant_ambient_gives_full_capacity_factor(resource):
    flat = [15.0] * 4
    s = seasonal_performance("Isobutane", resource, flat,
                             fixed_T_evap_C=95.0)
    assert s.capacity_factor == pytest.approx(1.0, abs=1e-6)
    assert s.annual_energy_MWh > 0.0


def test_seasonal_varying_ambient_below_full_capacity(resource):
    profile = [5.0, 15.0, 25.0, 35.0]
    s = seasonal_performance("Isobutane", resource, profile,
                             fixed_T_evap_C=95.0)
    assert 0.0 < s.capacity_factor < 1.0
    # annual energy equals mean net power times 8760 h (clipped at zero)
    mean_net = float(np.mean(np.clip(s.W_net_plant, 0.0, None)))
    assert s.annual_energy_MWh == pytest.approx(mean_net * 8760.0 / 1e6, rel=1e-6)
    assert s.rated_W == pytest.approx(float(np.max(s.W_net_plant)), rel=1e-9)


# --- off-design operation --------------------------------------------------- #
import types
from geothermal_orc import (
    design_plant, off_design_operation, decline_curves, profile_UA,
    part_load_turbine_efficiency,
)


def test_part_load_efficiency_peaks_at_design():
    assert part_load_turbine_efficiency(1.0, 0.80) == pytest.approx(0.80)
    assert part_load_turbine_efficiency(0.7, 0.80) < 0.80
    assert part_load_turbine_efficiency(0.1, 0.80) >= 0.5 * 0.80  # floor honoured


def test_profile_UA_finite_and_infinite():
    ok = types.SimpleNamespace(
        duty=np.array([0.0, 1.0, 2.0]),
        T_hot=np.array([310.0, 320.0, 330.0]),
        T_cold=np.array([300.0, 305.0, 310.0]))
    val = profile_UA(ok)
    assert np.isfinite(val) and val > 0.0
    crossed = types.SimpleNamespace(
        duty=np.array([0.0, 1.0]),
        T_hot=np.array([300.0, 300.0]),
        T_cold=np.array([300.0, 305.0]))   # zero / negative approach
    assert profile_UA(crossed) == float("inf")


@pytest.fixture(scope="module")
def design(resource):
    d, res = design_plant("Isobutane", resource, ambient_C=10.0)
    return d, res


def test_design_point_descriptors_positive(design):
    d, _ = design
    assert d.UA_evap > 0.0 and d.stodola_K > 0.0 and d.m_wf > 0.0


def test_offdesign_reproduces_design_at_design_brine(design, resource):
    d, _ = design
    od = off_design_operation(d, resource.T_reservoir_C)
    assert od.feasible
    assert od.T_evap_C == pytest.approx(d.T_evap_C, abs=1.0)


def test_offdesign_power_falls_with_cooler_brine(design, resource):
    d, _ = design
    hot = off_design_operation(d, resource.T_reservoir_C)
    cool = off_design_operation(d, resource.T_reservoir_C - 20.0)
    assert hot.feasible and cool.feasible
    assert cool.W_net_plant < hot.W_net_plant
    assert cool.T_evap_C < hot.T_evap_C


def test_decline_curves_decline_monotonically(resource):
    years = [0, 15]
    _, Tb, reopt, fixed = decline_curves(
        "Isobutane", resource, years, ambient_C=10.0, decline_rate=0.005)
    assert len(Tb) == len(reopt) == len(fixed) == 2
    assert reopt[0] > reopt[1] > 0.0
    assert fixed[0] > fixed[1] > 0.0
    # at year 0 the fixed plant is, by construction, the design point
    assert fixed[0] == pytest.approx(reopt[0], rel=0.03)


# --- pinch / area / power trade-off (Tier 2) -------------------------------- #
from geothermal_orc import pinch_area_tradeoff


def test_pinch_area_tradeoff_monotonic(resource):
    pinches = [3.0, 6.0, 9.0, 12.0]
    p, W, UA = pinch_area_tradeoff("Isobutane", resource, T_evap_C=120.0,
                                   pinches_C=pinches)
    assert len(p) == len(W) == len(UA) == 4
    # Tighter pinch -> more power but more conductance (area).
    assert W[0] > W[1] > W[2] > W[3]
    assert UA[0] > UA[1] > UA[2] > UA[3]
