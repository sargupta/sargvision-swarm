"""Riemannian metric + grid-Dijkstra geodesic — CHANAKYA's planner backbone.

Equip ℝ² (cruise-altitude plane) with the metric

    g_ij(x) = δ_ij · (1 + β Φ(x))^γ,   β, γ > 0

where Φ is the threat field (see `core.threat_field`). The conformal
factor inflates distance near hostile defense assets so the geodesic
between (start, target) bends around them.

A 2D grid Dijkstra delivers a globally-optimal-on-the-grid geodesic in
O(M log M) for M cells. M=10⁴ resolves a 100×100 grid in ~25 ms with
scipy.sparse.csgraph.

Output is a sequence of (x, y, z) waypoints the swarm can follow, with the
discrete shortest-path cost serving as the discretised action functional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sargvision_swarm.core.threat_field import DefenseAsset, threat_field


@dataclass
class MetricParams:
    beta: float = 1.5
    gamma: float = 2.0


def conformal_factor(threat: np.ndarray, params: MetricParams | None = None) -> np.ndarray:
    """sqrt(g) = (1 + β Φ)^(γ/2) — the multiplier on Euclidean distance."""
    p = params or MetricParams()
    return np.power(1.0 + p.beta * threat, p.gamma / 2.0)


@dataclass
class Grid2D:
    """Uniform 2D grid: x ∈ [x_min, x_max], y ∈ [y_min, y_max], altitude=z_fixed."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    z_fixed: float = 5.0

    def cell_centres(self) -> np.ndarray:
        """(nx*ny, 3) array of grid-cell centres in world coords."""
        xs = np.linspace(self.x_min, self.x_max, self.nx)
        ys = np.linspace(self.y_min, self.y_max, self.ny)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        gz = np.full_like(gx, self.z_fixed)
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / max(self.nx - 1, 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / max(self.ny - 1, 1)

    def world_to_index(self, world: np.ndarray) -> int:
        """Snap a world coord to the nearest grid cell (flat index)."""
        ix = int(np.clip(round((world[0] - self.x_min) / self.dx), 0, self.nx - 1))
        iy = int(np.clip(round((world[1] - self.y_min) / self.dy), 0, self.ny - 1))
        return ix * self.ny + iy

    def index_to_world(self, idx: int) -> np.ndarray:
        ix, iy = divmod(idx, self.ny)
        x = self.x_min + ix * self.dx
        y = self.y_min + iy * self.dy
        return np.array([x, y, self.z_fixed])


def build_grid_graph(
    grid: Grid2D,
    assets: list[DefenseAsset],
    metric: MetricParams | None = None,
) -> csr_matrix:
    """Sparse CSR adjacency for the 8-connected grid weighted by metric.

    Edge weight between adjacent cells = (Δs) · avg(sqrt(g(p₁)), sqrt(g(p₂))).
    The averaging matches the trapezoid quadrature of the path-length integral.
    """
    centres = grid.cell_centres()
    phi = threat_field(centres, assets)
    sqrt_g = conformal_factor(phi, metric)
    nx, ny = grid.nx, grid.ny
    n_cells = nx * ny

    rows = []
    cols = []
    data = []
    # 8-connected neighbourhood
    nbr_offsets = [
        (-1, 0, grid.dx),
        (1, 0, grid.dx),
        (0, -1, grid.dy),
        (0, 1, grid.dy),
        (-1, -1, np.hypot(grid.dx, grid.dy)),
        (-1, 1, np.hypot(grid.dx, grid.dy)),
        (1, -1, np.hypot(grid.dx, grid.dy)),
        (1, 1, np.hypot(grid.dx, grid.dy)),
    ]
    for ix in range(nx):
        for iy in range(ny):
            here = ix * ny + iy
            for dix, diy, ds in nbr_offsets:
                jx, jy = ix + dix, iy + diy
                if not (0 <= jx < nx and 0 <= jy < ny):
                    continue
                there = jx * ny + jy
                w = ds * 0.5 * (sqrt_g[here] + sqrt_g[there])
                rows.append(here)
                cols.append(there)
                data.append(w)
    return csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))


def geodesic_path(
    start: np.ndarray,
    target: np.ndarray,
    grid: Grid2D,
    assets: list[DefenseAsset],
    metric: MetricParams | None = None,
) -> tuple[np.ndarray, float]:
    """Plan a geodesic from `start` to `target` on the grid.

    Returns (waypoints, action_cost):
      waypoints : (K, 3) array — visited grid cells from start to target.
      action_cost : sum of grid-edge weights = discretised ∫ sqrt(g) ds.

    Falls back to a straight-line if Dijkstra finds no path (closed grid).
    """
    graph = build_grid_graph(grid, assets, metric)
    s_idx = grid.world_to_index(start)
    t_idx = grid.world_to_index(target)
    distances, predecessors = dijkstra(
        graph,
        indices=s_idx,
        return_predecessors=True,
        directed=False,
    )
    if not np.isfinite(distances[t_idx]):
        return np.stack([start, target]), float(np.linalg.norm(target - start))
    # Reconstruct path back-to-front.
    path_idx = [t_idx]
    cur = t_idx
    guard = 0
    while cur != s_idx and guard < grid.nx * grid.ny:
        cur = int(predecessors[cur])
        if cur < 0:
            break
        path_idx.append(cur)
        guard += 1
    path_idx.reverse()
    waypoints = np.stack([grid.index_to_world(i) for i in path_idx])
    return waypoints, float(distances[t_idx])


def straight_line_action(
    start: np.ndarray,
    target: np.ndarray,
    assets: list[DefenseAsset],
    metric: MetricParams | None = None,
    n_samples: int = 80,
) -> float:
    """Discretised ∫₀¹ sqrt(g(γ_straight(t))) |γ'| dt for the straight-line path.

    Reference baseline: lets tests assert geodesic_cost < straight_line_cost
    when a defense asset sits between start and target.
    """
    ts = np.linspace(0, 1, n_samples)
    pts = start[None, :] * (1 - ts[:, None]) + target[None, :] * ts[:, None]
    phi = threat_field(pts, assets)
    sg = conformal_factor(phi, metric)
    ds = float(np.linalg.norm(target - start)) / (n_samples - 1)
    # trapezoid
    return float(0.5 * ds * (sg[0] + 2 * sg[1:-1].sum() + sg[-1]))
