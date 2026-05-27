"""Bhuvan tile service mock — ISRO geospatial tile fetcher.

Bhuvan (https://bhuvan.nrsc.gov.in) is ISRO's National Geoportal — Indian
Cartosat-3 + Resourcesat + Sentinel-2 tile mosaic with multispectral +
DEM + SAR base layers.

For SARGVISION CFM, Bhuvan provides:
  - terrain DEM for line-of-sight + masking calculations
  - vegetation + structure classification for clutter modelling
  - base imagery for situational awareness on the operator console
  - administrative + land-use overlays

This mock returns synthetic tile metadata + DEM elevation arrays for any
requested geographic bounding box. Production integration replaces the
synthetic data with actual Bhuvan WMS / WMTS calls.

Reference: ISRO Bhuvan API documentation (https://bhuvan.nrsc.gov.in/api/).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class TileRequest:
    """Bhuvan tile request bounding box + layer specification.

    Attributes
    ----------
    lat_min, lat_max : float
        Latitude bounds (degrees).
    lon_min, lon_max : float
        Longitude bounds (degrees).
    layer : str
        One of "dem", "satellite", "landuse", "admin".
    resolution_m : float
        Target ground sample distance in metres.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    layer: Literal["dem", "satellite", "landuse", "admin"] = "dem"
    resolution_m: float = 30.0


@dataclass(frozen=True)
class TileResponse:
    """Tile response from Bhuvan service.

    Attributes
    ----------
    request : TileRequest
        The original request.
    data : np.ndarray
        For DEM: 2D float array of elevations (metres above MSL).
        For satellite: 3D (H, W, 3) RGB uint8 array.
        For landuse/admin: 2D int16 class labels.
    bounds_m : tuple of (min_x, min_y, max_x, max_y)
        Local-tangent bounds in metres (origin at tile centre).
    resolution_m : float
        Actual ground sample distance.
    layer : str
        Layer name (echoes request).
    is_synthetic : bool
        True when this is a mock response (False reserved for future real API).
    """

    request: TileRequest
    data: np.ndarray
    bounds_m: tuple[float, float, float, float]
    resolution_m: float
    layer: str
    is_synthetic: bool = True


@dataclass
class BhuvanTileService:
    """Mock Bhuvan tile service.

    Generates synthetic terrain consistent with the requested layer + bounds.
    For DEM, uses a deterministic seeded noise field that resembles typical
    Indian terrain (plains + hills + ridges) without referencing specific
    real geography.

    Parameters
    ----------
    rng_seed : int
        Seed for deterministic synthetic terrain.
    elevation_min_m : float
        Minimum simulated elevation (metres above MSL).
    elevation_max_m : float
        Maximum simulated elevation.
    """

    rng_seed: int = 0
    elevation_min_m: float = 200.0
    elevation_max_m: float = 1800.0

    def fetch(self, req: TileRequest) -> TileResponse:
        """Fetch a synthetic tile matching the request."""
        # Compute pixel grid size from bounding box + resolution
        lat_span = max(0.001, req.lat_max - req.lat_min)
        lon_span = max(0.001, req.lon_max - req.lon_min)
        # ~111 km per degree latitude; longitude scaled by cos(lat)
        lat_center = 0.5 * (req.lat_min + req.lat_max)
        m_per_deg_lat = 111_000.0
        m_per_deg_lon = 111_000.0 * float(np.cos(np.deg2rad(lat_center)))
        height_m = lat_span * m_per_deg_lat
        width_m = lon_span * m_per_deg_lon
        n_rows = max(2, int(height_m / req.resolution_m))
        n_cols = max(2, int(width_m / req.resolution_m))
        # cap mock tile size to keep tests fast
        n_rows = min(n_rows, 512)
        n_cols = min(n_cols, 512)

        rng = np.random.default_rng(self.rng_seed + hash((req.lat_min, req.lon_min)) % 2**32)

        if req.layer == "dem":
            # 3-octave fractal-ish noise for synthetic terrain
            grid = np.zeros((n_rows, n_cols), dtype=np.float64)
            for octave, weight in [(1, 1.0), (4, 0.5), (16, 0.25)]:
                low_rows = max(2, n_rows // octave)
                low_cols = max(2, n_cols // octave)
                low = rng.standard_normal((low_rows, low_cols))
                up = _bilinear_upsample(low, n_rows, n_cols)
                grid = grid + weight * up
            grid_min, grid_max = float(grid.min()), float(grid.max())
            if grid_max > grid_min:
                grid = (grid - grid_min) / (grid_max - grid_min)
            data = self.elevation_min_m + grid * (self.elevation_max_m - self.elevation_min_m)
        elif req.layer == "satellite":
            data = rng.integers(40, 200, size=(n_rows, n_cols, 3), dtype=np.uint8)
        elif req.layer == "landuse":
            # 0=water, 1=urban, 2=agriculture, 3=forest, 4=barren
            data = rng.integers(0, 5, size=(n_rows, n_cols), dtype=np.int16)
        elif req.layer == "admin":
            data = rng.integers(0, 3, size=(n_rows, n_cols), dtype=np.int16)
        else:  # pragma: no cover — typing should prevent this
            raise ValueError(f"unknown layer: {req.layer}")

        bounds_m = (
            -width_m / 2.0,
            -height_m / 2.0,
            width_m / 2.0,
            height_m / 2.0,
        )
        return TileResponse(
            request=req,
            data=data,
            bounds_m=bounds_m,
            resolution_m=req.resolution_m,
            layer=req.layer,
            is_synthetic=True,
        )


def _bilinear_upsample(low: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Cheap bilinear resample of a small array to a larger grid."""
    rows_in, cols_in = low.shape
    if rows_in == out_rows and cols_in == out_cols:
        return low
    row_idx = np.linspace(0, rows_in - 1, out_rows)
    col_idx = np.linspace(0, cols_in - 1, out_cols)
    r0 = np.floor(row_idx).astype(int)
    r1 = np.clip(r0 + 1, 0, rows_in - 1)
    c0 = np.floor(col_idx).astype(int)
    c1 = np.clip(c0 + 1, 0, cols_in - 1)
    rw = row_idx - r0
    cw = col_idx - c0
    rw = rw[:, None]
    cw = cw[None, :]
    a = low[r0[:, None], c0[None, :]]
    b = low[r0[:, None], c1[None, :]]
    c = low[r1[:, None], c0[None, :]]
    d = low[r1[:, None], c1[None, :]]
    top = a * (1 - cw) + b * cw
    bot = c * (1 - cw) + d * cw
    return top * (1 - rw) + bot * rw
