# SARGVISION Swarm

[![CI](https://github.com/sargupta/sargvision-swarm/actions/workflows/ci.yml/badge.svg)](https://github.com/sargupta/sargvision-swarm/actions/workflows/ci.yml)
[![Deploy](https://github.com/sargupta/sargvision-swarm/actions/workflows/deploy.yml/badge.svg)](https://github.com/sargupta/sargvision-swarm/actions/workflows/deploy.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

> *Adversarial-aware autonomous swarm command for the Indian defence stack.*
> The brain that decides which of N friendly drones engages which of M hostile contacts,
> while some friendlies may be RF-spoofed and the comms graph is being jammed.

## What this is

`sargvision-swarm` is the orchestration backend for **SARGVISION Swarm**, an adversarial-aware
autonomous command system targeting iDEX ADITI 2.0 PS-11 (₹25 cr Counter-Swarm), DISC-14
(multi-corridor LAC ISR), IAF S-UMS-MR (800 systems × ≥20 drones each), and the Indian Army's
May-2026 Sovereign Swarm tender (decentralised autonomy with joint IP ownership).

The system implements seven Sanskrit-named primitives composed end-to-end:

| Primitive       | Math                                                    | Solves                                       |
|-----------------|---------------------------------------------------------|----------------------------------------------|
| **SHIELD**      | Sheaf-Laplacian loyalty + PageRank trust + Bayesian threat | Compromised friendly drone detection      |
| **VAJRA**       | Voronoi hysteresis + Fiedler λ₂ alarm + tropical attention | Continuous interceptor reassignment under EW |
| **PRAJNA**      | Bayesian threat posterior + cost-of-shot gate           | Skip decoys to save munitions                |
| **SABHA**       | Practical Byzantine Fault Tolerance, K=7, f=2           | Distributed fire authorisation (no SPOF)     |
| **CHANAKYA**    | Riemannian geodesic on threat field                     | Ingress through hostile IADS (SEAD)          |
| **SHESHNAG**    | Couzin-Krause + SIR contagion                           | Adversarial swarm psyops                     |
| **MAYA**        | Replicator dynamics + Wasserstein DRO                   | Strategic posture solver                     |

## Quick start

```bash
# Install (uses uv)
uv pip install -e .

# Run the demonstrator (24 drones, Operation Trishul scenario)
SARGVISION_SCENARIO=border_strike python -m uvicorn sargvision_swarm.server.bridge:app \
    --host 127.0.0.1 --port 8765

# In another terminal — pair with the console:
#   https://github.com/sargupta/sargvision-console
```

## Scenarios

| Scenario id      | Name                                  | iDEX/DISC alignment       |
|------------------|---------------------------------------|---------------------------|
| `border_strike`  | **Operation Trishul** (90s scripted)  | ADITI 2.0 PS-11           |
| `coverage`       | IAF Counter-Swarm                     | ADITI 2.0 PS-11           |
| `migration`      | Governed Migration · Leh → LAC        | DISC-14 PS-16             |
| `formation_v`    | Army LAC Persistent ISR               | DISC-14 PS-21             |
| `flock`          | Navy Carrier Defense Mesh             | DISC-14 PS-32             |
| `sead_ingress`   | SEAD ingress through IADS             | (DRDO TDF candidate)      |
| `hover`          | Idle hold                             | —                         |

Switch scenarios at runtime:

```bash
curl -X POST "http://127.0.0.1:8765/scenario/border_strike?n=24&seed=42"
```

## Architecture

```
SwarmMap (Next.js console)
        │ WebSocket @ 10 Hz (msgpack)
        ▼
FastAPI bridge (server/bridge.py)
        │
        ▼
LiveSession  (demo/live_session.py)
   ├── SHIELD    (orchestrator/shield.py)        — trust + threat posterior
   ├── VAJRA     (orchestrator/vajra.py)         — Voronoi-auction assignment
   ├── ED-CBBA   (orchestrator/ed_cbba.py)       — event-gated consensus auction
   ├── SABHA     (orchestrator/swarm_raft.py)    — Byzantine ROE consensus
   ├── CHANAKYA  (orchestrator/chanakya.py)      — Riemannian geodesic planner
   ├── SHESHNAG  (orchestrator/sheshnag.py)      — adversarial psyops
   ├── MAYA      (orchestrator/maya.py)          — strategic posture
   └── Scenarios (sim/border_strike.py, sim/migration_zones.py, sim/hostiles.py, ...)
```

## Development

```bash
# Run tests
uv run pytest tests -v

# Lint + format
uv run ruff check src tests
uv run ruff format src tests

# Type check
uv run mypy src/sargvision_swarm --ignore-missing-imports
```

## Deployment

Production runs on Fly.io (Mumbai region) via `fly deploy` — see `fly.toml` + `Dockerfile`.
GitHub Actions auto-deploy on push to `main` via `.github/workflows/deploy.yml`.

- **Live bridge:** https://sargvision-swarm-bridge.fly.dev
- **Health check:** https://sargvision-swarm-bridge.fly.dev/healthz
- **WebSocket:** wss://sargvision-swarm-bridge.fly.dev/swarm

## Companion repos

- [`sargvision-console`](https://github.com/sargupta/sargvision-console) — Next.js operator UI
- [`SARGVISION_Docs`](https://github.com/sargupta/SARGVISION_Docs) — DPR, technical briefs (private)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
Copyright 2026 SARGVISION Intelligence Private Limited.
