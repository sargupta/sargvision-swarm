"""Environment physics — ISA density, momentum-theory hover power, wind, altitude.

Anchors (cross-checked vs research_2026_05/46_aerodynamics_physics.md):
  - ISA density ρ(h) = ρ₀ (1 − 0.0065 h/T₀)^4.26, ρ₀ = 1.225 kg/m³, T₀ = 288.15 K
  - momentum-theory hover power P = W^1.5 / (√(2 ρ A) · FoM)
  - at Leh (~3500 m) density ≈ 0.86 kg/m³ → hover power ↑ ~20%
  - at Siachen (~5400 m) density ≈ 0.70 → hover power ↑ ~33%

These make high-altitude (Ladakh) operations cost what they actually cost — a
Ladakh-tuned strategy that ignores density derate over-promises endurance.
"""

from __future__ import annotations

import numpy as np

_RHO0 = 1.225  # kg/m³ sea-level ISA density
_T0 = 288.15  # K sea-level ISA temperature
_G = 9.80665  # m/s²


def isa_density(altitude_m: float, temp_offset_c: float = 0.0) -> float:
    """ISA air density (kg/m³) at altitude, with optional temperature offset.

    temp_offset_c shifts the sea-level temperature (hot day → lower density).
    """
    t0 = _T0 + temp_offset_c
    lapse = 1.0 - 0.0065 * altitude_m / t0
    if lapse <= 0:
        return 0.0
    return float(_RHO0 * lapse**4.26)


def hover_power_w(
    mass_kg: float,
    rotor_area_m2: float,
    *,
    altitude_m: float = 0.0,
    temp_offset_c: float = 0.0,
    figure_of_merit: float = 0.65,
) -> float:
    """Ideal-plus-FoM hover power (electrical watts) via momentum theory.

    P = W^1.5 / (√(2 ρ A) · FoM), W = m·g. rotor_area_m2 is the TOTAL disk area
    (sum over all rotors).
    """
    rho = isa_density(altitude_m, temp_offset_c)
    if rho <= 0 or rotor_area_m2 <= 0:
        return float("inf")
    weight_n = mass_kg * _G
    p_ideal = weight_n**1.5 / np.sqrt(2.0 * rho * rotor_area_m2)
    return float(p_ideal / max(figure_of_merit, 1e-3))


def altitude_derate_factor(altitude_m: float, temp_offset_c: float = 0.0) -> float:
    """Hover-power multiplier vs sea level: (ρ₀/ρ)^0.5 (>= 1).

    At Leh (~3500 m) ≈ 1.19; at Siachen (~5400 m) ≈ 1.32.
    """
    rho = isa_density(altitude_m, temp_offset_c)
    if rho <= 0:
        return float("inf")
    return float(np.sqrt(_RHO0 / rho))


def wind_tolerated(max_airspeed_mps: float, wind_mps: float, *, margin: float = 1.0 / 3.0) -> bool:
    """Heuristic: a platform tolerates sustained wind up to ~1/3 of max airspeed.

    margin is the usable fraction of max airspeed available to fight wind while
    retaining control authority (default 1/3, conservative).
    """
    return wind_mps <= margin * max_airspeed_mps
