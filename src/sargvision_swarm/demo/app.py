"""Gradio Sandbox v3 — LIVE WATCH tab as primary view.

Streaming generator pushes one rendered frame per swarm tick. You watch the
drones move + messages flash + decisions fire in real time.
"""

from __future__ import annotations

import time

from sargvision_swarm.demo.live_session import LiveSession
from sargvision_swarm.demo.runner import RolloutResult, rollout
from sargvision_swarm.viz import (
    bandwidth_figure,
    protocol_breakdown_pie,
    render_frame,
    swarm_to_plotly_figure,
    to_numpy,
    topology_figure,
)


# ── Live-tab generator ────────────────────────────────────────────────


def _live_generator(n_drones, scenario, steps, comm_range, seed, fps):
    """Yield (image, event_log, status) tuples each tick."""
    session = LiveSession(
        n_drones=int(n_drones),
        scenario=scenario,
        seed=int(seed),
        comm_range_m=float(comm_range),
    )
    target_dt = 1.0 / max(1.0, float(fps))
    started = time.time()

    # First frame so the UI shows something immediately
    img = render_frame(
        swarm=session.swarm,
        trails=session.trails,
        recent_msgs=session.recent_for_render(),
        floating_events=session.floating_for_render(),
        comm_adjacency=session.comm_adjacency(),
        intents=session.intents,
        stats=session.render_stats(),
    )
    yield to_numpy(img), "Starting…\n", "starting"

    last_yield = time.time()
    for _ in range(int(steps)):
        session.step()
        # throttle to fps
        now = time.time()
        sleep_for = max(0.0, target_dt - (now - last_yield))
        if sleep_for:
            time.sleep(sleep_for)

        stats = session.render_stats()
        img = render_frame(
            swarm=session.swarm,
            trails=session.trails,
            recent_msgs=session.recent_for_render(),
            floating_events=session.floating_for_render(),
            comm_adjacency=session.comm_adjacency(),
            intents=session.intents,
            stats=stats,
            bft_flash_voters=set() if not session.bft_flash else session.bft_flash,
            cbba_flash_cells=session.cbba_flash or None,
        )
        log_text = "\n".join(list(session.event_log)[-25:][::-1])
        elapsed = time.time() - started
        status = (
            f"step {session.step_i}/{int(steps)} · t={stats['t']:.2f}s · "
            f"msgs={stats['total_msgs']} ({stats['msgs_per_s']:.0f}/s) · "
            f"wall={elapsed:.1f}s"
        )
        yield to_numpy(img), log_text, status
        last_yield = time.time()


# ── Static-rollout helpers (other tabs) ───────────────────────────────


def _swarm_at(result: RolloutResult, idx: int):
    return result.states[min(idx, len(result.states) - 1)]


def _3d_figure(result: RolloutResult, idx: int, scenario: str):
    s = _swarm_at(result, idx)
    return swarm_to_plotly_figure(s, title=f"3D · {scenario} · N={s.n} · t={s.t:.1f}s")


def _topo_figure(result: RolloutResult, idx: int):
    s = _swarm_at(result, idx)
    return topology_figure(s, result.comm_model, title="Comm topology")


def _message_table(result, max_rows=200, proto_filter="ALL"):
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


def _payload_summary(m):
    p = m.payload
    if "method" in p:
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


def _intents_table(result):
    return [[did, intent] for did, intent in sorted(result.intents_by_drone.items())]


def _bft_table(result):
    return [
        [
            ev["t"], ev["proposal"], "PASS" if ev["passed"] else "FAIL",
            ev.get("yes", 0), ev.get("no", 0), str(ev.get("byzantine", [])),
        ]
        for ev in result.bft_events
    ]


def _cbba_table(result):
    rows = []
    for ev in result.cbba_events[-30:]:
        for task, drone in ev["assignment"].items():
            rows.append([ev["t"], ev["n_bids"], task, drone])
    return rows


# ── Markdown blocks ───────────────────────────────────────────────────


WATCH_INTRO_MD = """
## 🎬 Live Watch — drones engaging in real time

Hit **Play** below. You'll see:

- **Drones** as colored circles (worker · scout · relay · leader).
- **Trails** behind each drone showing recent motion.
- **Comm-range edges** (faint green) — who can hear whom right now.
- **Message arrows** flashing across links — color-keyed by protocol (purple = A2A, blue = Zenoh pose, green = MAVLink heartbeat, orange = gRPC cognition, red = BFT vote).
- **HOLD / ADV / YLD / ROT** labels under each drone = LLM-emitted intent.
- **BFT flash** — yellow halo on the 7 committee members when they vote.
- **CBBA cells** — gold squares painted on the map when a drone claims a task.
- **Event ticker** scrolls every yield negotiation, every BFT result, every CBBA claim.

This is the **same engine** that powers the static tabs — same Brooks-subsumption,
same A2A / Zenoh / MAVLink / BFT / gRPC traffic. The only difference is you watch
it happen one tick at a time.
"""


PROTOCOLS_MD = """
# Protocols on the swarm wire

| Layer | Protocol | What it carries |
|---|---|---|
| Cognition (LLM ↔ LLM) | **gRPC** | Per-drone intent → ground orchestrator |
| Agent ↔ agent | **A2A** | JSON-RPC 2.0 over HTTP+SSE (Google / Linux Foundation, Apr 2025, Apache-2.0) |
| Agent ↔ tool | **MCP** | per-agent tool access |
| Robotics middleware | **Zenoh / DDS** | pose gossip + ED-CBBA bids |
| Autopilot bus | **MAVLink v2 signed** | heartbeat + telem |
| Mission state | **BFT** | SwarmRaft K=7 quorum votes |

See `docs/PROTOCOLS.md` for full spec recap.
"""


CHALLENGES_MD = """
# Real-world swarm-comm challenges → mitigations in this design

- **>120 ms latency** → collision avoidance runs on-drone, never cloud.
- **Bandwidth saturation** → semantic + key-frame uplink (2-10 Mbps/drone), not raw HD.
- **DDS multicast storms** → Zenoh gossip discovery scales 100+ on Wi-Fi.
- **GNSS spoofing** → SwarmRaft K=7 BFT, ⅔ quorum on irreversible.
- **GPS-denied drift** → Swarm-SLAM + UWB; Vásárhelyi tuning is relative.
- **India regs** → 865-867 MHz LoRa only, DGFT 2022 ban on imported CBU.

See `docs/CHALLENGES.md` for full reference list + citations.
"""


# ── Build app ─────────────────────────────────────────────────────────


def build_app():
    import gradio as gr

    PROTOCOL_OPTIONS = ["ALL", "A2A", "Zenoh", "MAVLink", "BFT", "gRPC", "MCP", "DDS"]

    with gr.Blocks(title="SARGVISION Swarm — Live", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🛸 SARGVISION Swarm — Communication Sandbox\n"
            "**5 protocols, real envelopes, watched live.** Brooks-subsumption: LLM = slow intent, reflex = fast control."
        )

        with gr.Tabs():
            # ────────────────── 🎬 LIVE WATCH (default tab) ──────────────────
            with gr.Tab("🎬 Live Watch"):
                gr.Markdown(WATCH_INTRO_MD)
                with gr.Row():
                    with gr.Column(scale=1):
                        live_n = gr.Slider(5, 60, value=24, step=1, label="N drones")
                        live_scenario = gr.Dropdown(
                            choices=["coverage", "flock", "formation_v", "hover"],
                            value="coverage",
                            label="Scenario",
                        )
                        live_steps = gr.Slider(60, 400, value=180, step=10, label="Sim ticks")
                        live_range = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Comm range (m)")
                        live_fps = gr.Slider(2.0, 30.0, value=10.0, step=1.0, label="Frame rate (fps)")
                        live_seed = gr.Number(value=42, label="Seed", precision=0)
                        live_play = gr.Button("▶ Play live", variant="primary", size="lg")
                        live_status = gr.Markdown("Press Play.")
                    with gr.Column(scale=3):
                        live_canvas = gr.Image(
                            label="Swarm — live",
                            interactive=False,
                            show_label=False,
                            show_download_button=False,
                        )
                        live_log = gr.Textbox(
                            label="Event ticker (newest at top)",
                            value="",
                            lines=12,
                            max_lines=12,
                            interactive=False,
                        )

                live_play.click(
                    _live_generator,
                    inputs=[live_n, live_scenario, live_steps, live_range, live_seed, live_fps],
                    outputs=[live_canvas, live_log, live_status],
                )

            # ────────────────── 📊 Static analysis tabs ───────────────────────
            with gr.Tab("📊 Static rollout"):
                state = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=1):
                        n_drones = gr.Slider(5, 100, value=30, step=1, label="N drones")
                        scenario = gr.Dropdown(
                            choices=["flock", "formation_v", "coverage", "hover"],
                            value="coverage",
                            label="Scenario",
                        )
                        steps = gr.Slider(60, 500, value=200, step=10, label="Sim steps")
                        comm_range = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Comm range (m)")
                        seed = gr.Number(value=42, label="Seed", precision=0)
                        run_btn = gr.Button("Run rollout", variant="secondary")
                        rationale = gr.Markdown()
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.Tab("Swarm + Topology"):
                                step_slider = gr.Slider(0, 49, value=0, step=1, label="Replay frame")
                                with gr.Row():
                                    fig_3d = gr.Plot(label="3D positions")
                                    fig_topo = gr.Plot(label="Comm topology")
                            with gr.Tab("Wire log"):
                                with gr.Row():
                                    proto_filter = gr.Dropdown(choices=PROTOCOL_OPTIONS, value="ALL", label="Filter")
                                    n_rows = gr.Slider(20, 500, value=150, step=10, label="Rows")
                                    refresh_btn = gr.Button("Refresh", size="sm")
                                msg_table = gr.Dataframe(
                                    headers=["t (s)", "proto", "src", "dst", "topic", "bytes", "payload"],
                                    datatype=["number", "str", "number", "str", "str", "number", "str"],
                                    interactive=False,
                                    wrap=True,
                                    label="Messages",
                                )
                                with gr.Row():
                                    fig_bw = gr.Plot(label="Bandwidth")
                                    fig_pie = gr.Plot(label="Protocol mix")
                            with gr.Tab("Decisions"):
                                gr.Markdown("### Per-drone intent")
                                intent_table = gr.Dataframe(
                                    headers=["drone", "intent"], datatype=["number", "str"], interactive=False
                                )
                                gr.Markdown("### SwarmRaft K=7 BFT votes")
                                bft_table = gr.Dataframe(
                                    headers=["t (s)", "proposal", "result", "yes", "no", "byzantine"],
                                    datatype=["number", "str", "str", "number", "number", "str"],
                                    interactive=False,
                                )
                                gr.Markdown("### ED-CBBA bidding")
                                cbba_table = gr.Dataframe(
                                    headers=["t (s)", "n_bids", "task", "assigned_drone"],
                                    datatype=["number", "number", "str", "number"],
                                    interactive=False,
                                )

                def _run(n, sc, st, cr, sd):
                    t0 = time.time()
                    result = rollout(
                        n_drones=int(n), scenario=sc, steps=int(st), seed=int(sd),
                        comm_range_m=float(cr), snapshot_every=4,
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
                        f"**Plan:** {result.plan_rationale}\n\n*{n_frames} frames · {len(result.message_log)} wire messages · {elapsed:.2f} s wall.*",
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
                    outputs=[state, fig_3d, fig_topo, step_slider, msg_table, fig_bw, fig_pie, intent_table, bft_table, cbba_table, rationale],
                )
                step_slider.change(_scrub, inputs=[state, step_slider, scenario], outputs=[fig_3d, fig_topo])
                refresh_btn.click(_refresh_log, inputs=[state, n_rows, proto_filter], outputs=[msg_table, fig_bw, fig_pie])

            # ────────────────── 📖 Reference tabs ──────────────────────────
            with gr.Tab("📖 Protocols"):
                gr.Markdown(PROTOCOLS_MD)
            with gr.Tab("⚠ Challenges"):
                gr.Markdown(CHALLENGES_MD)

    return demo


def main() -> None:
    app = build_app()
    app.queue()  # required for streaming generators
    app.launch(server_name="127.0.0.1", inbrowser=False, share=False)


if __name__ == "__main__":
    main()
