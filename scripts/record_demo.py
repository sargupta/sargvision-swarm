"""Record a LiveSession to a static JSON file for the console's demo-replay mode.

Runs the simulation headless, captures build_frame() output every step, and
writes a compact JSON array of frames. The console bundles this file and plays
it back when no live WebSocket backend is reachable — making the static
Cloudflare Pages deployment a self-contained demo with zero backend.

One file is recorded per scenario; the console picks the file matching the
mission the operator selects, so mission switching works with no backend.

Usage:
    python scripts/record_demo.py --scenario border_strike --frames 300 \
        --out ../sargvision-console/public/demo/session.border_strike.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sargvision_swarm.demo.live_session import LiveSession
from sargvision_swarm.server.bridge import build_frame

# Full float repr costs ~2x the file size for no visible fidelity. Degrees need
# 6 dp (~0.1 m); everything else (metres, m/s, probabilities) needs 3.
_DEGREE_KEYS = {"lon", "lat", "heading_deg"}


def _compact(obj: object, key: str | None = None) -> object:
    """Round every float in a frame so the JSON stays small enough to bundle."""
    if isinstance(obj, float):
        return round(obj, 6 if key in _DEGREE_KEYS else 3)
    if isinstance(obj, dict):
        return {k: _compact(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_compact(v, key) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="border_strike")
    ap.add_argument("--n-drones", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--comm-range", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=600, help="frames to record (10 Hz)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--no-compact",
        dest="compact",
        action="store_false",
        help="keep full float precision (roughly doubles the file size)",
    )
    args = ap.parse_args()

    session = LiveSession(
        n_drones=args.n_drones,
        scenario=args.scenario,
        seed=args.seed,
        comm_range_m=args.comm_range,
    )

    frames: list[dict] = []
    for i in range(args.frames):
        session.step()
        if i % args.stride == 0:
            frame = build_frame(session)
            frames.append(_compact(frame) if args.compact else frame)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "scenario": args.scenario,
            "n_drones": args.n_drones,
            "hz": 10.0,
            "frame_count": len(frames),
            "note": "Recorded demo session for backend-less replay.",
        },
        "frames": frames,
    }
    with out.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = out.stat().st_size / 1024.0
    print(
        f"Recorded {len(frames)} frames ({args.scenario}) → {out} ({size_kb:.1f} KB)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
