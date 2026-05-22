"""Gradio Sandbox v2 — comms-first.

Tabs:
1. Swarm    — 3D positions + comm-range topology side-by-side
2. Wire log — live A2A / Zenoh / MAVLink / BFT / gRPC message stream
3. Decisions — per-drone intents + SwarmRaft BFT votes + ED-CBBA bids
4. Protocols — A2A / MCP / MAVLink / Zenoh / Brooks-subsumption explainer
5. Challenges — what others have hit + how this design mitigates
"""

from __future__ import annotations

import time

from sargvision_swarm.demo.runner import RolloutResult, rollout
from sargvision_swarm.viz import (
    bandwidth_figure,
    protocol_breakdown_pie,
    swarm_to_plotly_figure,
    topology_figure,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _swarm_at(result: RolloutResult, idx: int):
    return result.states[min(idx, len(result.states) - 1)]


def _3d_figure(result: RolloutResult, idx: int, scenario: str):
    s = _swarm_at(result, idx)
    title = f"3D · {scenario} · N={s.n} · t={s.t:.1f}s"
    return swarm_to_plotly_figure(s, title=title)


def _topo_figure(result: RolloutResult, idx: int):
    s = _swarm_at(result, idx)
    return topology_figure(s, result.comm_model, title="Comm topology")


def _message_table(result: RolloutResult, max_rows: int = 200, proto_filter: str = "ALL"):
    rows = result.message_log.recent(k=max_rows * 2)
    if proto_filter != "ALL":
        rows = [m for m in rows if m.protocol.value == proto_filter]
    rows = rows[-max_rows:]
    return [
        [
            f"{m.t:.2f}",
            m.protocol.value,
            m.src,
            "*" if m.dst is None else m.dst,
            m.topic,
            m.bytes_size,
            _payload_summary(m),
        ]
        for m in rows
    ]


def _payload_summary(m) -> str:
    p = m.payload
    if "method" in p:                # A2A
        return f"method={p['method']} params={p.get('params', {})}"
    if "intent" in p:
        return f"intent={p['intent']}"
    if "decision" in p:
        return f"vote={p['decision']} proposal={p['proposal']}"
    if "bid_score" in p:
        return f"bid task={p.get('task_id')} score={p.get('bid_score', 0):.2f}"
    if "pos" in p:
        pos = p["pos"]
        return f"pose=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
    if "battery" in p:
        return f"hb battery={p['battery']:.2f}"
    return str(p)[:80]


def _intents_table(result: RolloutResult):
    return [[did, intent] for did, intent in sorted(result.intents_by_drone.items())]


def _bft_table(result: RolloutResult):
    return [
        [
            ev["t"],
            ev["proposal"],
            "PASS" if ev["passed"] else "FAIL",
            ev.get("yes", 0),
            ev.get("no", 0),
            str(ev.get("byzantine", [])),
        ]
        for ev in result.bft_events
    ]


def _cbba_table(result: RolloutResult):
    rows = []
    for ev in result.cbba_events[-30:]:
        for task, drone in ev["assignment"].items():
            rows.append([ev["t"], ev["n_bids"], task, drone])
    return rows


# ── Tab content (markdown) ─────────────────────────────────────────────


PROTOCOLS_MD = """
# Protocols on the swarm wire

Real deployment uses **five protocols**, layered:

| Layer | Protocol | What it carries | Wire format |
|---|---|---|---|
| Cognition (LLM ↔ LLM) | **gRPC** | Per-drone intent → ground orchestrator | protobuf over HTTP/2 |
| Agent ↔ agent | **A2A** (Agent2Agent) | Capability discovery + intent share + yield negotiation | **JSON-RPC 2.0 over HTTP+SSE** |
| Agent ↔ tool | **MCP** (Model Context Protocol) | Per-agent tool access (camera, IMU, GPS) | JSON-RPC over stdio / HTTP |
| Robotics middleware | **Zenoh** (or DDS) | Pose, intent gossip, ED-CBBA bids, BFT votes | binary, gossip-based |
| Autopilot bus | **MAVLink v2** (signed) | Heartbeat, position telem, command | binary, signed |

## A2A (the one you asked about)

Google open standard, announced **April 2025**, governed by Linux Foundation, Apache-2.0.

- **Transport:** HTTPS + SSE (server-sent events for streaming)
- **Envelope:** JSON-RPC 2.0
- **Discovery:** "Agent Cards" advertise capabilities + transports
- **Patterns:** synchronous request/response, streaming (SSE), async push notifications
- **Complements MCP:** A2A is agent-to-agent; MCP is agent-to-tool

In **this demo**: `agents/peer_dialogue.py` emits `A2AMessage` envelopes (proper JSON-RPC 2.0 shape) over the in-memory bus. Drop-in a real A2A SDK on Linux and the methods (`share.intent`, `negotiate.yield`, `claim.task`, `share.health`) keep their signatures.

## Why we publish positions only (not velocities)

Velocity leaks intent and doubles bandwidth. BVC needs only positions to compute collision-safe cells. Standard discipline in swarm robotics.

## SwarmRaft (BFT mission state)

A K=7 committee replicates mission state. Tolerates ⌊(K-1)/2⌋ = **3 Byzantine drones** (incl. GNSS-spoofed). Irreversible actions (engage, RTL, abort) require **⅔ quorum**.

## ED-CBBA (event-driven task allocation)

Vanilla CBBA re-bids every cycle → radio chatter explodes at 50+ drones.
**ED-CBBA** re-bids only when a neighbor's known winning bid changes. **~52% less traffic** (arXiv 2509.06481).

## Brooks subsumption discipline

LLM emits **slow-loop intent**. Reflex layer (Boids / Olfati-Saber / BVC) closes the **fast control loop**. No LLM call ever blocks an actuator command. Mandatory because cloud LLM RTT is 50-500 ms and collision avoidance must be < 200 ms.

## Sources

- [A2A specification](https://a2a-protocol.org/latest/specification/)
- [A2A GitHub](https://github.com/a2aproject/A2A)
- [Google announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [Anthropic MCP](https://modelcontextprotocol.io/)
- [PX4 v1.17 in-tree Zenoh](https://docs.px4.io/main/en/middleware/zenoh.html)
- [ED-CBBA paper](https://arxiv.org/abs/2509.06481)
- See also `~/Documents/AI_Workspace/drone_swarm_research/04_comms.md`.
"""


CHALLENGES_MD = """
# What others hit in the field — and how this design mitigates

## 1. Latency >120 ms in complex environments

Field tests of UAV swarms in cluttered terrain regularly see end-to-end latency exceeding 120 ms — enough to break collision-avoidance loops if those loops route through the cloud.

**Mitigation in this design:**
- Collision (BVC + GCBF+) runs **on-drone**, no network dependency.
- LLM intent runs at slow cadence (1-5 Hz) — never on the fast loop.
- See Brooks-subsumption discipline.

## 2. Bandwidth saturation

At 50 drones streaming 1080p video = ~25 Gbps aggregate. Infeasible without 5G slice + dedicated radios.

**Mitigation:**
- Drones stream **2-10 Mbps of semantic + key-frame**, never raw HD.
- Pose broadcasts at 10 Hz = ~32 B/msg = 320 B/s per drone = 32 kB/s for 100 drones — trivial.
- ED-CBBA cuts task-bidding chatter ~52%.
- This demo's bandwidth panel shows live bytes/sec per protocol.

## 3. Multicast storms (DDS on Wi-Fi)

ROS 2 default Fast/Cyclone DDS uses multicast discovery; collapses at ~20 nodes on Wi-Fi.

**Mitigation:** **Zenoh + rmw_zenoh** uses gossip discovery — scales to 100+ on lossy mesh. MDPI 2025 50-UAV paper validated. PX4 v1.17 ships Zenoh in-tree.

## 4. Authentication + key handover

Without per-drone identity, a swarm is one captured drone away from total compromise.

**Mitigation (planned for Phase 2):**
- Per-drone X.509 cert from CoE PKI.
- MAVLink v2 signed messages.
- A2A Agent Cards carry cryptographic identity.
- (arXiv 2201.05657 covers the threat model.)

## 5. GPS-denied: 2-5 cm/min positioning drift

In urban canyon / indoor / jamming scenarios, accumulated VIO drift compounds. Coordination decisions become meaningless after ~5 minutes.

**Mitigation:**
- **Swarm-SLAM (MIT)** for collaborative loop closure across the squad.
- **UWB (NoopLoop / Decawave)** for relative pose between drones.
- Vásárhelyi 2018 tuning works on relative spacing, not absolute coords.

## 6. Byzantine fault / GNSS spoofing

A spoofed drone votes wrong. Without BFT, single bad actor corrupts the swarm.

**Mitigation:** **SwarmRaft K=7 BFT committee**, ⅔ quorum on irreversible decisions. Tolerates 3 Byzantine drones in a 7-member committee. (Demo runs an ED-CBBA-class spoof test at step 100.)

## 7. India regulatory traps

- DGFT 2022: drones in CBU/CKD form **prohibited** to import. Components free. **Must build in India.**
- NPNT mandatory for sales post Jan 2024.
- WPC bands: 2.4 / 5.8 GHz + **865-867 MHz LoRa only**. **NOT 868 / 915 MHz** — both licensed.
- WPC ETA per radio SKU mandatory.

## 8. Sim2real cliff

End-to-end MARL policies trained in Isaac Sim fail outdoors at scale. **No published 100-drone outdoor neural deployment exists in 2026.**

**Mitigation:** Hybrid stack — classical reflex + RL **sub-task** policies + LLM cognition. AttentionSwarm (Mar 2025) is the SOTA indoor benchmark; outdoor frontier is ~10 drones with learned coordination.

## Sources / further reading

- [Authentication and handover for drone swarms (arXiv 2201.05657)](https://arxiv.org/pdf/2201.05657)
- [Drone swarm coordination (Meegle, 2024-2025 survey)](https://www.meegle.com/en_us/topics/autonomous-drones/drone-swarm-coordination)
- [PRC concepts for UAV swarms in future warfare (CNA, 2025)](https://www.cna.org/reports/2025/07/PRC-Concepts-for-UAV-Swarms-in-Future-Warfare.pdf)
- [Mission-critical UAV swarm coordination — integrated ROS+LoRa](https://www.sciencedirect.com/science/article/abs/pii/S0140366424002494)
- This repo's master synthesis: `~/Documents/AI_Workspace/drone_swarm_research/00_MASTER.md`.
"""


# ── Build app ───────────────────────────────────────────────────────────


def build_app():
    import gradio as gr

    PROTOCOL_OPTIONS = ["ALL", "A2A", "Zenoh", "MAVLink", "BFT", "gRPC", "MCP", "DDS"]

    with gr.Blocks(title="SARGVISION Swarm — Comms Sandbox", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # SARGVISION Swarm — Communication Sandbox
            **What you're watching:** real A2A / Zenoh / MAVLink / BFT / gRPC traffic between drones.
            All 5 protocols carry their own message types, byte sizes, packet-loss models, comm-range topology.
            Brooks-subsumption discipline: LLM emits **intent**, reflex layer closes the **fast loop**.
            """
        )

        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                n_drones = gr.Slider(5, 100, value=30, step=1, label="N drones")
                scenario = gr.Dropdown(
                    choices=["flock", "formation_v", "coverage", "hover"],
                    value="coverage",
                    label="Scenario",
                )
                steps = gr.Slider(60, 500, value=200, step=10, label="Sim steps (× 50 ms)")
                comm_range = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Comm range (m)")
                seed = gr.Number(value=42, label="Seed", precision=0)
                run_btn = gr.Button("Run rollout", variant="primary")
                rationale = gr.Markdown()

            with gr.Column(scale=3):
                with gr.Tabs():
                    # ── Tab 1: Swarm + Topology ────────────────────────
                    with gr.Tab("1. Swarm + Topology"):
                        step_slider = gr.Slider(0, 49, value=0, step=1, label="Replay frame")
                        with gr.Row():
                            fig_3d = gr.Plot(label="3D positions")
                            fig_topo = gr.Plot(label="Comm topology")

                    # ── Tab 2: Wire log ────────────────────────────────
                    with gr.Tab("2. Wire log"):
                        with gr.Row():
                            proto_filter = gr.Dropdown(
                                choices=PROTOCOL_OPTIONS, value="ALL", label="Filter by protocol"
                            )
                            n_rows = gr.Slider(20, 500, value=150, step=10, label="Rows")
                            refresh_btn = gr.Button("Refresh log", size="sm")
                        msg_table = gr.Dataframe(
                            headers=["t (s)", "proto", "src", "dst", "topic", "bytes", "payload"],
                            datatype=["number", "str", "number", "str", "str", "number", "str"],
                            interactive=False,
                            wrap=True,
                            label="Last messages",
                        )
                        with gr.Row():
                            fig_bw = gr.Plot(label="Bandwidth (last 5 s)")
                            fig_pie = gr.Plot(label="Protocol mix")

                    # ── Tab 3: Decisions ───────────────────────────────
                    with gr.Tab("3. Decisions + BFT + CBBA"):
                        gr.Markdown("### Per-drone current intent (LLM-driven)")
                        intent_table = gr.Dataframe(
                            headers=["drone", "intent"],
                            datatype=["number", "str"],
                            interactive=False,
                        )
                        gr.Markdown("### SwarmRaft K=7 BFT votes (mission-state quorum)")
                        bft_table = gr.Dataframe(
                            headers=["t (s)", "proposal", "result", "yes", "no", "byzantine"],
                            datatype=["number", "str", "str", "number", "number", "str"],
                            interactive=False,
                        )
                        gr.Markdown("### ED-CBBA task-bidding (event-driven)")
                        cbba_table = gr.Dataframe(
                            headers=["t (s)", "n_bids", "task", "assigned_drone"],
                            datatype=["number", "number", "str", "number"],
                            interactive=False,
                        )

                    # ── Tab 4: Protocols ───────────────────────────────
                    with gr.Tab("4. Protocols"):
                        gr.Markdown(PROTOCOLS_MD)

                    # ── Tab 5: Challenges ──────────────────────────────
                    with gr.Tab("5. Challenges"):
                        gr.Markdown(CHALLENGES_MD)

        # ── Callbacks ──────────────────────────────────────────────────

        def _run(n, sc, st, cr, sd):
            t0 = time.time()
            result = rollout(
                n_drones=int(n),
                scenario=sc,
                steps=int(st),
                seed=int(sd),
                comm_range_m=float(cr),
                snapshot_every=4,
            )
            elapsed = time.time() - t0
            n_frames = len(result.states)
            now = result.states[-1].t
            return (
                result,
                _3d_figure(result, 0, sc),
                _topo_figure(result, 0),
                gr.Slider(minimum=0, maximum=max(0, n_frames - 1), value=0, step=1),
                _message_table(result, max_rows=150, proto_filter="ALL"),
                bandwidth_figure(result.bandwidth.rates_by_protocol(now), title="Bytes / sec by protocol"),
                protocol_breakdown_pie(result.message_log.by_protocol()),
                _intents_table(result),
                _bft_table(result),
                _cbba_table(result),
                (
                    f"**Plan:** {result.plan_rationale}\n\n"
                    f"*{n_frames} frames · {len(result.message_log)} wire messages · {elapsed:.2f} s wall.*"
                ),
            )

        def _scrub(result, idx, sc):
            if result is None:
                return None, None
            return _3d_figure(result, int(idx), sc), _topo_figure(result, int(idx))

        def _refresh_log(result, rows, pf):
            if result is None:
                return [], None, None
            now = result.states[-1].t
            return (
                _message_table(result, max_rows=int(rows), proto_filter=pf),
                bandwidth_figure(result.bandwidth.rates_by_protocol(now), title="Bytes / sec by protocol"),
                protocol_breakdown_pie(result.message_log.by_protocol()),
            )

        run_btn.click(
            _run,
            inputs=[n_drones, scenario, steps, comm_range, seed],
            outputs=[
                state,
                fig_3d,
                fig_topo,
                step_slider,
                msg_table,
                fig_bw,
                fig_pie,
                intent_table,
                bft_table,
                cbba_table,
                rationale,
            ],
        )
        step_slider.change(_scrub, inputs=[state, step_slider, scenario], outputs=[fig_3d, fig_topo])
        refresh_btn.click(
            _refresh_log,
            inputs=[state, n_rows, proto_filter],
            outputs=[msg_table, fig_bw, fig_pie],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", inbrowser=False, share=False)


if __name__ == "__main__":
    main()
