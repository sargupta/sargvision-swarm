"""Gradio app — 'SARGVISION Swarm Sandbox'.

Run: `swarm-demo`  (or `python -m sargvision_swarm.demo.app`)
"""

from __future__ import annotations

import time

import numpy as np

from sargvision_swarm.core.state import SwarmState
from sargvision_swarm.demo.runner import RolloutResult, rollout


def _swarm_at(result: RolloutResult, idx: int) -> SwarmState:
    return result.states[min(idx, len(result.states) - 1)]


def _build_figure(result: RolloutResult, idx: int, scenario: str):
    from sargvision_swarm.viz import swarm_to_plotly_figure

    s = _swarm_at(result, idx)
    title = f"SARGVISION Swarm · scenario={scenario} · N={s.n} · t={s.t:.1f}s · step={idx}"
    return swarm_to_plotly_figure(s, title=title)


def build_app():
    import gradio as gr

    with gr.Blocks(title="SARGVISION Swarm Sandbox", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # SARGVISION Swarm Sandbox
            Boids · Olfati-Saber · BVC collision-avoidance · pluggable LLM agents.
            **Phase 0 deliverable** — pure-Python reflex layer on macOS. Linux box adds Gazebo+PX4+ROS 2.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                n_drones = gr.Slider(5, 100, value=30, step=1, label="N drones")
                scenario = gr.Dropdown(
                    choices=["flock", "formation_v", "coverage", "hover"],
                    value="flock",
                    label="Scenario",
                )
                steps = gr.Slider(50, 500, value=200, step=10, label="Sim steps (× 50 ms)")
                seed = gr.Number(value=42, label="Seed", precision=0)
                run_btn = gr.Button("Run rollout", variant="primary")
                rationale = gr.Markdown(label="Plan rationale")
            with gr.Column(scale=2):
                fig = gr.Plot(label="Swarm 3D")
                step_slider = gr.Slider(0, 49, value=0, step=1, label="Replay frame", interactive=True)

        # Cache rollout result inside a Gradio state slot
        state = gr.State(value=None)

        def _run(n: int, sc: str, st: int, sd: float):
            t0 = time.time()
            result = rollout(n_drones=int(n), scenario=sc, steps=int(st), seed=int(sd))
            elapsed = time.time() - t0
            n_frames = len(result.states)
            return (
                result,
                _build_figure(result, 0, sc),
                gr.Slider(minimum=0, maximum=max(0, n_frames - 1), value=0, step=1),
                f"**Plan:** {result.plan_rationale}\n\n*Rollout: {n_frames} frames · {elapsed:.2f} s wall.*",
            )

        def _scrub(result: RolloutResult | None, idx: int, sc: str):
            if result is None:
                return None
            return _build_figure(result, int(idx), sc)

        run_btn.click(
            _run,
            inputs=[n_drones, scenario, steps, seed],
            outputs=[state, fig, step_slider, rationale],
        )
        step_slider.change(_scrub, inputs=[state, step_slider, scenario], outputs=[fig])

        gr.Markdown(
            """
            ---
            ### Deferred to Linux box (Phase 1+)
            - Gazebo Harmonic + PX4 SITL multi-drone scene.
            - ROS 2 Jazzy + uXRCE-DDS bridge.
            - Crazyswarm2 indoor real-fleet integration.
            - Zenoh + rmw_zenoh replacing the in-memory bus.

            See `~/Documents/AI_Workspace/drone_swarm_research/00_MASTER.md`.
            """
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", inbrowser=False, share=False)


if __name__ == "__main__":
    main()
