"""Thermodynamic state objects backed by CoolProp.

A :class:`State` fixes a pure-fluid equilibrium state from any two independent
intensive properties and exposes the rest (``T, P, h, s, rho, Q``).  All units
are SI: temperature in kelvin, pressure in pascal, specific enthalpy in J/kg,
specific entropy in J/(kg.K), density in kg/m^3.

Flow exergy is evaluated relative to a configurable dead state
``(DEAD_STATE_T, DEAD_STATE_P)``; the default is 25 C and 1 atm, the convention
used throughout DiPippo (2016).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from CoolProp.CoolProp import PropsSI

# Dead-state (environment) reference, DiPippo convention.
DEAD_STATE_T = 298.15          # K  (25 C)
DEAD_STATE_P = 101_325.0       # Pa (1 atm)

# Map of accepted constructor property names to CoolProp keys.
_CP_KEY = {"T": "T", "P": "P", "H": "H", "S": "S", "Q": "Q", "D": "D"}


def _props(output: str, fluid: str, n1: str, v1: float, n2: str, v2: float) -> float:
    """Thin wrapper around :func:`CoolProp.CoolProp.PropsSI` with clear errors."""
    try:
        return PropsSI(output, n1, v1, n2, v2, fluid)
    except Exception as exc:  # pragma: no cover - re-raised with context
        raise ValueError(
            f"CoolProp failed for {output} of {fluid} at {n1}={v1}, {n2}={v2}: {exc}"
        ) from exc


@dataclass(frozen=True)
class State:
    """An equilibrium thermodynamic state of a single working fluid.

    Construct with one of the ``from_*`` class methods.  Instances are frozen so
    that a state, once created, is a trustworthy record of a cycle point.
    """

    fluid: str
    T: float          # K
    P: float          # Pa
    h: float          # J/kg
    s: float          # J/(kg.K)
    rho: float        # kg/m^3
    Q: float          # vapour quality in [0,1]; -1 if single phase

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def _build(cls, fluid: str, n1: str, v1: float, n2: str, v2: float) -> "State":
        k1, k2 = _CP_KEY[n1], _CP_KEY[n2]
        T = _props("T", fluid, k1, v1, k2, v2)
        P = _props("P", fluid, k1, v1, k2, v2)
        h = _props("H", fluid, k1, v1, k2, v2)
        s = _props("S", fluid, k1, v1, k2, v2)
        rho = _props("D", fluid, k1, v1, k2, v2)
        # Quality is only meaningful inside the dome; CoolProp returns values
        # outside [0,1] (e.g. -1 or 1e3) for single-phase points.
        try:
            q = _props("Q", fluid, k1, v1, k2, v2)
            if q < 0.0 or q > 1.0:
                q = -1.0
        except Exception:
            q = -1.0
        return cls(fluid=fluid, T=T, P=P, h=h, s=s, rho=rho, Q=q)

    @classmethod
    def from_TP(cls, fluid: str, T: float, P: float) -> "State":
        return cls._build(fluid, "T", T, "P", P)

    @classmethod
    def from_PQ(cls, fluid: str, P: float, Q: float) -> "State":
        return cls._build(fluid, "P", P, "Q", Q)

    @classmethod
    def from_TQ(cls, fluid: str, T: float, Q: float) -> "State":
        return cls._build(fluid, "T", T, "Q", Q)

    @classmethod
    def from_Ph(cls, fluid: str, P: float, h: float) -> "State":
        return cls._build(fluid, "P", P, "H", h)

    @classmethod
    def from_Ps(cls, fluid: str, P: float, s: float) -> "State":
        return cls._build(fluid, "P", P, "S", s)

    @classmethod
    def from_hs(cls, fluid: str, h: float, s: float) -> "State":
        return cls._build(fluid, "H", h, "S", s)

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def T_celsius(self) -> float:
        return self.T - 273.15

    @property
    def is_two_phase(self) -> bool:
        return 0.0 <= self.Q <= 1.0

    def flow_exergy(
        self,
        T0: float = DEAD_STATE_T,
        h0: Optional[float] = None,
        s0: Optional[float] = None,
        P0: float = DEAD_STATE_P,
    ) -> float:
        """Specific flow (physical) exergy, J/kg.

        ``e = (h - h0) - T0 (s - s0)``.  If the dead-state enthalpy/entropy are
        not supplied they are evaluated for *this* fluid at ``(T0, P0)``.
        """
        if h0 is None or s0 is None:
            h0 = _props("H", self.fluid, "T", T0, "P", P0)
            s0 = _props("S", self.fluid, "T", T0, "P", P0)
        return (self.h - h0) - T0 * (self.s - s0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        phase = f"Q={self.Q:.3f}" if self.is_two_phase else "1-phase"
        return (
            f"State({self.fluid}: T={self.T_celsius:.2f}C, "
            f"P={self.P/1e5:.3f}bar, h={self.h/1e3:.2f}kJ/kg, {phase})"
        )


def specific_flow_exergy(
    fluid: str,
    h: float,
    s: float,
    T0: float = DEAD_STATE_T,
    P0: float = DEAD_STATE_P,
) -> float:
    """Specific flow exergy for arbitrary ``(h, s)`` of ``fluid``.

    Useful for streams represented by raw enthalpy/entropy rather than a
    :class:`State` object (e.g. brine in a heat-exchanger profile).
    """
    h0 = _props("H", fluid, "T", T0, "P", P0)
    s0 = _props("S", fluid, "T", T0, "P", P0)
    return (h - h0) - T0 * (s - s0)


def saturation_pressure(fluid: str, T: float) -> float:
    """Saturation pressure (Pa) at temperature ``T`` (K)."""
    return _props("P", fluid, "T", T, "Q", 0.0)


def saturation_temperature(fluid: str, P: float) -> float:
    """Saturation temperature (K) at pressure ``P`` (Pa)."""
    return _props("T", fluid, "P", P, "Q", 0.0)


def critical_temperature(fluid: str) -> float:
    """Critical temperature (K)."""
    return PropsSI("Tcrit", fluid)


def critical_pressure(fluid: str) -> float:
    """Critical pressure (Pa)."""
    return PropsSI("Pcrit", fluid)
