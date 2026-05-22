# sargvision-swarm — architecture

The 7-layer reference architecture from `00_MASTER.md`, mapped to this repo's code.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  L6  MISSION ORCHESTRATION    src/sargvision_swarm/orchestrator/         │
│      MissionPlanner (deterministic + LangGraph-ready)                    │
├──────────────────────────────────────────────────────────────────────────┤
│  L5  PER-DRONE COGNITION      src/sargvision_swarm/agents/                │
│      DroneAgent + pluggable backend (Mock / Anthropic / Ollama)          │
├──────────────────────────────────────────────────────────────────────────┤
│  L4  PERCEPTION               [TODO Phase 1 — VINS-Fusion + Swarm-SLAM]  │
├──────────────────────────────────────────────────────────────────────────┤
│  L3  SKILL LAYER (MARL)       [TODO Phase 2 — OmniDrones policies]       │
├──────────────────────────────────────────────────────────────────────────┤
│  L2  REFLEX LAYER             src/sargvision_swarm/core/                  │
│      Boids + Olfati-Saber + BVC + compose_reflex                         │
├──────────────────────────────────────────────────────────────────────────┤
│  L1  COMMS                    src/sargvision_swarm/comms/                 │
│      InMemoryBus + Channels (Zenoh-shaped)                               │
├──────────────────────────────────────────────────────────────────────────┤
│  L0  HARDWARE                 src/sargvision_swarm/sim/                   │
│      SimpleSim (double-integrator)                                       │
│      [TODO Phase 1 — Pixhawk 6X + Jetson Orin via uXRCE-DDS]             │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data flow per cycle

```
                          ┌─────────────────────┐
   NL goal ─→ Planner ─→  │  MissionPlan        │
                          │  scenario, algo,    │
                          │  goal, bundles      │
                          └─────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
            ┌──────────┐      ┌──────────┐      ┌──────────┐
            │ Agent 0  │      │ Agent 1  │ ...  │ Agent N  │
            │ observe  │      │ observe  │      │ observe  │
            │ decide   │      │ decide   │      │ decide   │
            └────┬─────┘      └────┬─────┘      └────┬─────┘
                 │ intent          │ intent          │ intent
                 └────────┬────────┴────────┬────────┘
                          ▼                 ▼
                    ┌──────────────────────────┐
                    │ Reflex (Boids/OS + BVC)  │
                    │ velocity command (N,3)   │
                    └─────────┬────────────────┘
                              ▼
                    ┌──────────────────────────┐
                    │ Sim step                  │
                    │ → new SwarmState          │
                    └─────────┬────────────────┘
                              ▼
                          (next cycle)
```

## Why the reflex layer dominates the fast loop

The LLM agent emits a high-level **intent** at slow cadence (1–5 Hz). The reflex layer turns positions, velocities, and intent into safe velocity commands at fast cadence (20 Hz here, 200 Hz on real hardware).

This is Brooks subsumption applied to drone swarms. The MASTER doc explains why end-to-end neural policy is a 2026 NO-GO for 100-drone outdoor fleets.

## Swap-in plan for Linux

| In-memory stub | Real | Effort |
|---|---|---|
| `InMemoryBus` | Zenoh `Session` | API shape already matches; ~1 file |
| `SimpleSim` | Gazebo Harmonic + PX4 SITL + uXRCE-DDS | Phase 1 |
| `MockBackend` | Qwen 2.5 3B via Ollama on Jetson | already supported, just install |
| `MissionPlanner._decide` | full LangGraph DAG with checkpoints | Phase 1 |
