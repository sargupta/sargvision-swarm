# sargvision-swarm

**SARGVISION AI CoE drone swarm — Phase 0 reference implementation.**

Pure-Python reflex layer (Boids · Olfati-Saber · BVC), in-memory swarm bus (Zenoh-shaped API), pluggable per-drone LLM agent (mock / Ollama / Anthropic), LangGraph-ready mission planner, and a Gradio sandbox you can run on a MacBook with zero hardware.

This repo is the **macOS-runnable subset** of the architecture defined in `~/Documents/AI_Workspace/drone_swarm_research/00_MASTER.md`. Linux-only and hardware-bound pieces (Gazebo Harmonic, PX4 SITL, ROS 2 Jazzy, Crazyswarm2, Zenoh+rmw_zenoh) are stubbed with in-memory equivalents — same API shape — to be swapped on a Linux dev box.

## Quickstart (60 seconds, macOS)

```bash
cd ~/Documents/GitHub/sargvision-swarm
uv sync
uv run swarm demo            # Gradio web app → http://127.0.0.1:7860
```

Headless variant:

```bash
uv run swarm sim --n 30 --scenario flock --steps 200 --snapshot out.png
```

## What's in this repo

```
src/sargvision_swarm/
  core/             reflex layer + shared data types
    state.py            DroneState, SwarmState, Role
    boids.py            Reynolds Boids (vectorized)
    olfati_saber.py     Algorithm 3 — α/β/γ-agents
    bvc.py              Buffered Voronoi Cells collision filter
    reflex.py           composer (algo + BVC safety wrap)
  sim/              double-integrator quadrotor sim (NumPy)
  comms/            in-memory pubsub w/ Zenoh-shaped topic API
  agents/           per-drone LLM agent (mock / Anthropic / Ollama)
  orchestrator/     mission planner (deterministic; LangGraph hooks)
  viz/              Plotly 3D + matplotlib snapshot
  demo/             Gradio sandbox + headless rollout runner
  cli.py            `swarm demo`, `swarm sim`, `swarm info`
```

## Scenarios in the sandbox

| Scenario | Algorithm | What you see |
|---|---|---|
| `flock` | Boids + BVC | Cohesive flock, no goal — emergent group motion |
| `formation_v` | Olfati-Saber + BVC | V-shape converges from random spawn |
| `coverage` | Olfati-Saber + BVC | Drones disperse to per-drone goal cells |
| `hover` | direct PID | Everyone holds station at (0, 0, 5) |

## Pluggable LLM agent

The mock backend ships by default. Switch via env var:

```bash
# Anthropic (needs ANTHROPIC_API_KEY)
export SARGVISION_LLM_BACKEND=anthropic
uv sync --extra llm
uv run swarm demo

# Local Ollama (needs `ollama` running at localhost:11434)
brew install ollama && ollama serve
ollama pull qwen2.5:3b-instruct-q4_K_M
export SARGVISION_LLM_BACKEND=ollama
```

LLM output is a strict-JSON **intent** — `hold_formation` / `advance_to_goal` / `yield_to_neighbor` / `rotate_role` / `report_health`. The reflex layer is what closes the fast control loop. Brooks-subsumption discipline.

## What this repo deliberately doesn't do (yet)

| Deferred | Why | Where it goes |
|---|---|---|
| Gazebo Harmonic + PX4 SITL multi-vehicle scene | Linux-only practical | `scripts/gazebo_sitl.sh` (Phase 1) |
| ROS 2 Jazzy + uXRCE-DDS | Linux-only | TODO |
| Crazyswarm2 + real Crazyflie | Linux + hardware | Phase 1 |
| MARL training (Isaac Sim / OmniDrones / VMAS) | needs RTX GPU | Phase 2 |
| Real Zenoh + rmw_zenoh | better on Linux | swap `InMemoryBus` |
| Doodle Labs / 5G slice integration | needs hardware | Phase 2 |
| Skybrush light-show pipeline | needs ArduCopter + RTK | Phase 3 |

All deferred work has detailed plans in `~/Documents/AI_Workspace/drone_swarm_research/`.

## Architecture (whole-system)

See [00_MASTER.md](file://~/Documents/AI_Workspace/drone_swarm_research/00_MASTER.md). The seven layers, top → bottom:

```
L6  Mission orchestration   Google ADK 2.0 + AgentScope + LangGraph
L5  Per-drone cognition     Qwen 2.5 3B Q4 / Gemma 3 4B  via Ollama  ← this repo (mock)
L4  Perception              VINS-Fusion + Swarm-SLAM + YOLOv11n
L3  Skill layer (MARL)      MAPPO + GNN/attention, CTDE, CBF filter
L2  Reflex                  BVC + GCBF+ / Boids / Olfati-Saber / ED-CBBA / SwarmRaft  ← this repo
L1  Comms                   Zenoh + MAVLink v2 + gRPC + ZeroTier      ← this repo (in-memory)
L0  Hardware                Pixhawk 6X + Jetson Orin / Crazyflie 2.1+
```

## Hard constraints (do not violate)

- End-to-end MARL for 100 outdoor drones in 2026 = **NO-GO**. Use hybrid.
- Real-time dense 3D SLAM at 100 drones = **unsolved**. Hierarchical 10×10 squads.
- India deployment: **865-867 MHz LoRa only** (NOT 868/915).
- PX4 (BSD-3) preferred over ArduPilot (GPLv3) for defense IP retention.
- Build in India — DGFT 2022 ban on imported CBU/CKD drones.

## Roadmap

- **Phase 0 (this repo, macOS):** reflex layer + sim + mock cognition + Gradio demo.
- **Phase 1 (Linux box):** Gazebo Harmonic + PX4 SITL + ROS 2 Jazzy + Zenoh; 20-50 Crazyflie indoor lab.
- **Phase 2 (Linux + GPU):** Isaac Sim + OmniDrones MARL training; 10 Pixhawk 6X outdoor pilot; iDEX submission.
- **Phase 3 (scale):** 50-100 drone outdoor fleet; flagship demo (Bharat Drone Mahotsav 2026 or IAF MBC-3); IAF SUMS RFP response.
- **Phase 4 (productize):** iDEX SPRINT/Prime; benchmark publication; SaaS orchestration platform.

## Develop

```bash
make install-dev      # dev tools (pytest, ruff, mypy)
make test             # full test suite
make lint             # ruff
make format           # ruff format
make snapshot         # writes out.png of final-frame swarm
```

## License

Apache-2.0.
