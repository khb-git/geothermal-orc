"""Tests for the working-fluid library and screening logic."""

import pytest

from geothermal_orc.fluids import (
    LIBRARY,
    Fluid,
    get_fluid,
    screen,
    classify_slope,
)


def test_library_nonempty_and_indexed_by_name():
    assert len(LIBRARY) >= 10
    for name, fl in LIBRARY.items():
        assert fl.name == name
        assert isinstance(fl, Fluid)


def test_get_fluid_by_coolprop_name():
    f = get_fluid("Isobutane")
    assert f.name == "Isobutane"


def test_get_fluid_by_ashrae_id():
    f = get_fluid("R600a")            # isobutane's ASHRAE designation
    assert f.name == "Isobutane"


def test_get_fluid_unknown_raises():
    with pytest.raises(KeyError):
        get_fluid("Unobtanium")


@pytest.mark.parametrize("fluid,expected", [
    ("Water", "wet"),
    ("Ammonia", "wet"),
    ("Propane", "wet"),
    ("Isobutane", "dry"),
    ("n-Pentane", "dry"),
    ("Isopentane", "dry"),
    ("R245fa", "dry"),
    ("R1234yf", "isentropic"),
    ("R1234ze(E)", "isentropic"),
])
def test_slope_classification_matches_literature(fluid, expected):
    assert classify_slope(fluid) == expected


def test_fluid_critical_temperature_populated():
    f = get_fluid("R245fa")
    assert f.Tcrit_C == pytest.approx(153.9, abs=1.0)


def test_low_gwp_flag():
    assert get_fluid("R1234yf").low_gwp is True
    assert get_fluid("R134a").low_gwp is False


def test_screen_by_gwp():
    out = screen(max_gwp=150)
    assert all(f.gwp100 <= 150 for f in out)
    # R134a (1430) and R245fa (1030) must be excluded.
    names = {f.name for f in out}
    assert "R134a" not in names
    assert "R245fa" not in names


def test_screen_by_safety_class():
    out = screen(allowed_safety=["A1", "B1"])
    assert all(f.safety in ("A1", "B1") for f in out)


def test_screen_by_critical_temperature_window():
    out = screen(min_Tcrit_C=120, max_Tcrit_C=200)
    assert all(120 <= f.Tcrit_C <= 200 for f in out)


def test_screen_by_slope_type():
    out = screen(slope_types=["dry"])
    assert all(f.slope_type == "dry" for f in out)
    assert len(out) > 0


def test_screen_sorted_by_critical_temperature():
    out = screen()
    tcs = [f.Tcrit for f in out]
    assert tcs == sorted(tcs)


def test_all_library_fluids_have_zero_odp():
    # The library is deliberately limited to non-ozone-depleting fluids.
    assert all(f.odp == 0.0 for f in LIBRARY.values())
