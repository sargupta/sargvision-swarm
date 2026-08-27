"""Battery + endurance model — pack-level energy, cold + altitude derate.

Anchors (cross-checked vs research_2026_05/47 + 90_energy_2030_2046.md):
  - Discipline: CELL vs PACK. Drones fly on PACK Wh/kg (~60-70% of cell).
  - 2024 commodity baseline: LiPo ~150 Wh/kg pack; Li-ion ~180.
  - 2026 upside: imported Li-metal (Sion) ~240 Wh/kg pack (supply-risk).
  - Solid-state ships ~2029-2031 and its win is volumetric/safety, NOT
    gravimetric (QSE-5 cell 301 < Sion 500) — modelled at ~250 pack.
  - Ladakh compound derate: cold + altitude-cooling tax ≈ 30-35% of spec.

Endurance is energy / power; power comes from environment.hover_power_w, so a
Ladakh mission honestly costs ~2/3 of its sea-level endurance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Chemistry(Enum):
    LIPO_2024 = "lipo_2024"
    LI_ION_2024 = "li_ion_2024"
    LI_METAL_2026 = "li_metal_2026"  # imported, supply-risk
    SOLID_STATE_2030 = "solid_state_2030"


#: Pack-level Wh/kg by chemistry (NOT cell — already de-rated for BMS+casing).
CHEMISTRY_PRESETS: dict[Chemistry, float] = {
    Chemistry.LIPO_2024: 150.0,
    Chemistry.LI_ION_2024: 180.0,
    Chemistry.LI_METAL_2026: 240.0,
    Chemistry.SOLID_STATE_2030: 250.0,
}


def _cold_derate(temp_c: float) -> float:
    """Usable-capacity fraction at temperature.

    ~1.0 at 25°C, falling below 0°C; ~0.5 at -10°C (LiPo cold-loss). Above 45°C
    a mild discount for accelerated degradation + cooling tax.
    """
    if temp_c >= 25.0:
        return 1.0 if temp_c <= 45.0 else 0.92
    if temp_c <= -10.0:
        return 0.5
    # linear 25°C(1.0) → -10°C(0.5)
    return 0.5 + 0.5 * (temp_c + 10.0) / 35.0


@dataclass
class BatteryModel:
    """A drone battery pack.

    Parameters
    ----------
    mass_kg : pack mass.
    chemistry : Chemistry preset (sets pack Wh/kg).
    usable_fraction : depth-of-discharge limit (default 0.85 — never run flat).
    """

    mass_kg: float
    chemistry: Chemistry = Chemistry.LIPO_2024
    usable_fraction: float = 0.85

    @property
    def nominal_wh(self) -> float:
        return self.mass_kg * CHEMISTRY_PRESETS[self.chemistry]

    def usable_wh(self, temp_c: float = 25.0) -> float:
        """Usable energy after depth-of-discharge + cold derate."""
        return self.nominal_wh * self.usable_fraction * _cold_derate(temp_c)

    def endurance_min(self, avg_power_w: float, temp_c: float = 25.0) -> float:
        """Flight time (minutes) at a given average electrical power draw."""
        if avg_power_w <= 0:
            return float("inf")
        return float(self.usable_wh(temp_c) / avg_power_w * 60.0)

    def ladakh_derate_vs_sealevel(
        self, sea_level_power_w: float, altitude_power_w: float, temp_c: float = -10.0
    ) -> float:
        """Endurance fraction at a Ladakh-style mission vs sea-level baseline.

        Combines cold-capacity loss and the higher hover power at altitude.
        """
        base = self.endurance_min(sea_level_power_w, temp_c=25.0)
        cold = self.endurance_min(altitude_power_w, temp_c=temp_c)
        if base <= 0:
            return 0.0
        return float(cold / base)
