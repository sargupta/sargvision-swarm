"""Mapping from sim local meters → lat/lon for the console map.

Scenario X (IAF Counter-Swarm) anchors near a notional Air Force base. We pick
an LAC-adjacent location so the Bhuvan / CARTO Dark Matter tiles show actual
Indian terrain.
"""

from __future__ import annotations

import math

# Default scene anchor — Leh, Ladakh (close to LAC; useful for demo theatre).
DEFAULT_ANCHOR_LAT = 34.1526
DEFAULT_ANCHOR_LON = 77.5770

# How many *world* meters one degree latitude / longitude is at the anchor.
def meters_per_deg(lat: float) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111_132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111_412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lat, m_per_deg_lon


def local_to_geo(
    x_m: float,
    y_m: float,
    anchor_lat: float = DEFAULT_ANCHOR_LAT,
    anchor_lon: float = DEFAULT_ANCHOR_LON,
) -> tuple[float, float]:
    """Convert sim x (east, meters) + y (north, meters) → (lon, lat)."""
    m_lat, m_lon = meters_per_deg(anchor_lat)
    lat = anchor_lat + (y_m / m_lat)
    lon = anchor_lon + (x_m / m_lon)
    return lon, lat
