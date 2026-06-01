"""Tests for zeotropic mixture cycles (Tier 2)."""
import pytest

from geothermal_orc import (
    GeothermalResource, ORCCycle,
    MixtureCycle, mixture_string, bubble_dew, temperature_glide,
    screen_compositions,
)

MIX = (["Isobutane", "Isopentane"], [0.7, 0.3])


def test_mixture_string_format_and_validation():
    assert mixture_string(["Isobutane", "Isopentane"], [0.7, 0.3]) == \
        "Isobutane[0.700000]&Isopentane[0.300000]"
    with pytest.raises(ValueError):
        mixture_string(["Isobutane", "Isopentane"], [0.7, 0.7])   # sum != 1
    with pytest.raises(ValueError):
        mixture_string(["Isobutane"], [0.5, 0.5])                 # length mismatch


def test_glide_is_positive_and_shrinks_with_pressure():
    spec = mixture_string(*MIX)
    g_lo = temperature_glide(spec, 3.0e5)
    g_hi = temperature_glide(spec, 2.0e6)
    assert g_lo > 0.0 and g_hi > 0.0
    assert g_lo > g_hi          # glide narrows toward the critical region
    Tb, Td = bubble_dew(spec, 1.0e6)
    assert Td > Tb


def test_mixture_cycle_solves_with_glide():
    mc = MixtureCycle(*MIX, T_evap_dew_C=110.0, T_cond_mean_C=30.0)
    r = mc.solve()
    assert r.w_net > 0.0
    assert 0.05 < r.eta_th < 0.20
    assert r.glide_evap > 0.0 and r.glide_cond > 0.0
    assert mc.P_cond < mc.P_evap


def test_mixture_resource_coupling():
    mc = MixtureCycle(*MIX, T_evap_dew_C=105.0, T_cond_mean_C=30.0)
    res = mc.solve_with_resource(m_brine=100.0, T_brine_in_C=120.0, pinch_evap=5.0)
    assert res.W_net > 0.0
    assert res.m_wf > 0.0
    assert 0.2 < res.eta_utilization < 0.6
    # The brine must leave cooler than it entered.
    assert res.brine_T_out < 120.0 + 273.15


def test_invalid_cycle_pressures_rejected():
    # A dew evaporation temperature below the mean condensing temperature is
    # unphysical (condensing pressure would exceed evaporating pressure).
    with pytest.raises(ValueError):
        MixtureCycle(*MIX, T_evap_dew_C=25.0, T_cond_mean_C=30.0)


def _best_Wnet_pure(fluid, Tb, T_evaps):
    best = 0.0
    for Te in T_evaps:
        try:
            r = ORCCycle(fluid, T_evap_C=Te, T_cond_C=30.0).solve_with_resource(
                m_brine=100.0, T_brine_in_C=Tb, pinch_evap=5.0)
            best = max(best, r.W_net)
        except Exception:
            pass
    return best


def _best_Wnet_mix(fracs, Tb, T_evaps):
    best = 0.0
    for Te in T_evaps:
        try:
            r = MixtureCycle(["Isobutane", "Isopentane"], fracs,
                             T_evap_dew_C=Te, T_cond_mean_C=30.0).solve_with_resource(
                m_brine=100.0, T_brine_in_C=Tb, pinch_evap=5.0)
            best = max(best, r.W_net)
        except Exception:
            pass
    return best


def test_mixture_beats_pure_for_low_enthalpy_resource():
    # Literature consensus (Heberle, Chys): for low-enthalpy resources the
    # gliding mixture matches the sensible streams better and out-produces the
    # best pure fluid.  At 120 C the advantage is a few percent.
    T_evaps = [76, 80, 84, 88]
    pure = _best_Wnet_pure("Isobutane", 120.0, T_evaps)
    mix = _best_Wnet_mix([0.7, 0.3], 120.0, T_evaps)
    assert mix > pure


def test_screen_compositions_runs():
    r = GeothermalResource(T_reservoir_C=120.0, mass_flow=100.0)
    results = screen_compositions("Isobutane", "Isopentane", [0.4, 0.6, 0.8],
                                  r, T_evap_dew_C=84.0)
    assert len(results) == 3
    assert all(res is not None and res.W_net > 0.0 for res in results)
