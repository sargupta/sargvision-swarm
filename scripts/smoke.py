"""Tiny smoke test — runs the full stack headlessly. Useful in CI."""

from __future__ import annotations

import sys

import numpy as np

from sargvision_swarm.demo.runner import rollout


def main() -> int:
    print("Running smoke rollout (20 drones × flock × 100 steps)…")
    result = rollout(n_drones=20, scenario="flock", steps=100, seed=0, snapshot_every=20)
    final = result.states[-1]
    n_frames = len(result.states)
    if not np.isfinite(final.positions).all():
        print("FAIL: non-finite positions in final state")
        return 2
    print(f"OK: {n_frames} frames, final t={final.t:.2f}s, plan={result.plan_rationale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
