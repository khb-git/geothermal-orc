"""Curated working-fluid library for low-temperature ORC screening.

Each :class:`Fluid` carries the CoolProp name plus the screening attributes used
in the working-fluid selection literature (Saleh et al., 2007; Bao & Zhao,
2013):

* critical temperature/pressure and normal boiling point (from CoolProp),
* saturated-vapour-curve slope type -- ``dry`` / ``wet`` / ``isentropic`` --
  computed from CoolProp rather than tabulated, so it is internally consistent,
* environmental and safety descriptors: ODP, 100-yr GWP and ASHRAE 34 safety
  class.

Environmental values are public reference figures (ODP from the Montreal
Protocol annexes; GWP100 from IPCC AR4/AR5; safety class from ASHRAE 34) and are
indicative -- different assessment years differ, especially for the HFO fluids
whose GWP is reported as "<1".  They are provided to support *relative*
screening, not regulatory reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from CoolProp.CoolProp import PropsSI

# Slope band (J/kg/K^2) within which a fluid is treated as isentropic.
ISENTROPIC_BAND = 0.10


def _sat_vapor_entropy_slope(fluid: str, reduced_T: float = 0.70) -> float:
    """ds/dT along the saturated-vapour line at ``reduced_T * Tcrit`` (J/kg/K^2).

    The sign of this slope is the standard discriminator between expander
    behaviours: positive => dry (expansion ends superheated), negative => wet,
    near-zero => isentropic.
    """
    Tc = PropsSI("Tcrit", fluid)
    Tt = PropsSI("Ttriple", fluid)
    T = reduced_T * Tc
    if T <= Tt:
        T = Tt + 0.1 * (Tc - Tt)
    dT = 0.5
    s_lo = PropsSI("S", "T", T - dT, "Q", 1, fluid)
    s_hi = PropsSI("S", "T", T + dT, "Q", 1, fluid)
    return (s_hi - s_lo) / (2.0 * dT)


def classify_slope(fluid: str, band: float = ISENTROPIC_BAND) -> str:
    """Return ``'dry'``, ``'wet'`` or ``'isentropic'`` for ``fluid``."""
    xi = _sat_vapor_entropy_slope(fluid)
    if xi > band:
        return "dry"
    if xi < -band:
        return "wet"
    return "isentropic"


@dataclass
class Fluid:
    """A candidate ORC working fluid and its screening descriptors."""

    name: str                 # CoolProp fluid name
    ashrae: str               # e.g. "R600a", "" if none
    family: str               # "alkane", "HFC", "HFO", "natural", ...
    odp: float                # ozone depletion potential
    gwp100: float             # 100-yr global warming potential
    safety: str               # ASHRAE 34 safety class, e.g. "A3"
    display_name: Optional[str] = None

    # Filled in lazily from CoolProp.
    Tcrit: float = field(init=False)        # K
    Pcrit: float = field(init=False)        # Pa
    Tnbp: float = field(init=False)         # K, normal boiling point
    molar_mass: float = field(init=False)   # kg/mol
    slope_type: str = field(init=False)     # dry/wet/isentropic

    def __post_init__(self) -> None:
        self.Tcrit = PropsSI("Tcrit", self.name)
        self.Pcrit = PropsSI("Pcrit", self.name)
        try:
            self.Tnbp = PropsSI("T", "P", 101_325.0, "Q", 0, self.name)
        except Exception:
            self.Tnbp = float("nan")
        self.molar_mass = PropsSI("molarmass", self.name)
        self.slope_type = classify_slope(self.name)
        if self.display_name is None:
            self.display_name = self.name

    @property
    def Tcrit_C(self) -> float:
        return self.Tcrit - 273.15

    @property
    def low_gwp(self) -> bool:
        return self.gwp100 < 150.0  # common regulatory screening threshold


# --------------------------------------------------------------------------- #
# The library.  ASHRAE/ODP/GWP are public reference values (see module docstring)
# --------------------------------------------------------------------------- #
_FLUID_SPECS = [
    # name            ashrae   family     odp   gwp100  safety
    ("Propane",      "R290",  "alkane",  0.0,     3.0,  "A3"),
    ("Isobutane",    "R600a", "alkane",  0.0,     3.0,  "A3"),
    ("n-Butane",     "R600",  "alkane",  0.0,     4.0,  "A3"),
    ("Isopentane",   "R601a", "alkane",  0.0,     5.0,  "A3"),
    ("n-Pentane",    "R601",  "alkane",  0.0,     5.0,  "A3"),
    ("Cyclopentane", "",      "alkane",  0.0,    11.0,  "A3"),
    ("R134a",        "R134a", "HFC",     0.0,  1430.0,  "A1"),
    ("R152a",        "R152a", "HFC",     0.0,   124.0,  "A2"),
    ("R245fa",       "R245fa","HFC",     0.0,  1030.0,  "B1"),
    ("R227ea",       "R227ea","HFC",     0.0,  3220.0,  "A1"),
    ("R236fa",       "R236fa","HFC",     0.0,  9810.0,  "A1"),
    ("R1234yf",      "R1234yf","HFO",    0.0,     4.0,  "A2L"),
    ("R1234ze(E)",   "R1234ze","HFO",    0.0,     7.0,  "A2L"),
    ("Ammonia",      "R717",  "natural", 0.0,     0.0,  "B2L"),
]

LIBRARY: Dict[str, Fluid] = {}
for _spec in _FLUID_SPECS:
    _f = Fluid(*_spec)
    LIBRARY[_f.name] = _f
del _spec, _f


def get_fluid(name: str) -> Fluid:
    """Return the library :class:`Fluid` for ``name`` (CoolProp or ASHRAE id)."""
    if name in LIBRARY:
        return LIBRARY[name]
    for fluid in LIBRARY.values():
        if fluid.ashrae and fluid.ashrae.lower() == name.lower():
            return fluid
    raise KeyError(f"{name!r} not in fluid library; available: {sorted(LIBRARY)}")


def screen(
    max_gwp: Optional[float] = None,
    max_odp: Optional[float] = None,
    allowed_safety: Optional[List[str]] = None,
    min_Tcrit_C: Optional[float] = None,
    max_Tcrit_C: Optional[float] = None,
    slope_types: Optional[List[str]] = None,
) -> List[Fluid]:
    """Filter the library by environmental and thermodynamic criteria.

    All criteria are optional and combined with logical AND.  Returns fluids
    sorted by critical temperature (ascending).
    """
    out = []
    for f in LIBRARY.values():
        if max_gwp is not None and f.gwp100 > max_gwp:
            continue
        if max_odp is not None and f.odp > max_odp:
            continue
        if allowed_safety is not None and f.safety not in allowed_safety:
            continue
        if min_Tcrit_C is not None and f.Tcrit_C < min_Tcrit_C:
            continue
        if max_Tcrit_C is not None and f.Tcrit_C > max_Tcrit_C:
            continue
        if slope_types is not None and f.slope_type not in slope_types:
            continue
        out.append(f)
    return sorted(out, key=lambda x: x.Tcrit)
