"""Tests for the Bhuvan tile service mock."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.cfm.sovereign.bhuvan import (
    BhuvanTileService,
    TileRequest,
)


def test_dem_tile_returns_elevation_array():
    svc = BhuvanTileService(rng_seed=0, elevation_min_m=300.0, elevation_max_m=1500.0)
    req = TileRequest(
        lat_min=28.5,
        lat_max=28.6,
        lon_min=77.1,
        lon_max=77.2,
        layer="dem",
        resolution_m=100.0,
    )
    resp = svc.fetch(req)
    assert resp.layer == "dem"
    assert resp.data.ndim == 2
    assert resp.data.dtype == np.float64
    assert resp.is_synthetic is True
    # elevations within configured band (small slack for noise edge)
    assert resp.data.min() >= 300.0 - 1.0
    assert resp.data.max() <= 1500.0 + 1.0


def test_satellite_tile_returns_rgb():
    svc = BhuvanTileService()
    req = TileRequest(lat_min=12.9, lat_max=13.0, lon_min=77.5, lon_max=77.6, layer="satellite")
    resp = svc.fetch(req)
    assert resp.data.ndim == 3
    assert resp.data.shape[-1] == 3
    assert resp.data.dtype == np.uint8


def test_landuse_tile_returns_class_labels():
    svc = BhuvanTileService()
    req = TileRequest(lat_min=22.0, lat_max=22.05, lon_min=88.3, lon_max=88.35, layer="landuse")
    resp = svc.fetch(req)
    assert resp.data.ndim == 2
    assert resp.data.dtype == np.int16
    assert resp.data.min() >= 0
    assert resp.data.max() < 5  # 5 landuse classes
