"""Top-level CLI. `swarm --help`."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="SARGVISION swarm dev CLI")
console = Console()


@app.command()
def demo() -> None:
    """Launch the Gradio sandbox."""
    from sargvision_swarm.demo.app import main

    console.print("[bold cyan]Launching SARGVISION Swarm Sandbox…[/]")
    main()


@app.command()
def sim(
    n: int = typer.Option(30, help="Number of drones"),
    scenario: str = typer.Option("flock", help="flock | formation_v | coverage | hover"),
    steps: int = typer.Option(200, help="Sim steps"),
    seed: int = typer.Option(42, help="RNG seed"),
    snapshot: str | None = typer.Option(None, help="Path to write final-frame PNG"),
) -> None:
    """Run a scenario headlessly."""
    from sargvision_swarm.demo.runner import rollout
    from sargvision_swarm.viz import swarm_to_mpl_snapshot

    console.print(f"[bold]Rollout[/] N={n} scenario={scenario} steps={steps}")
    result = rollout(n_drones=n, scenario=scenario, steps=steps, seed=seed)
    final = result.states[-1]
    console.print(f"Final t={final.t:.2f}s, frames={len(result.states)}")
    console.print(f"Plan rationale: {result.plan_rationale}")
    if snapshot:
        png = swarm_to_mpl_snapshot(final, title=f"{scenario} · t={final.t:.1f}s")
        from pathlib import Path
        Path(snapshot).write_bytes(png)
        console.print(f"Wrote snapshot → {snapshot}")


@app.command()
def info() -> None:
    """Print version + capability matrix."""
    from sargvision_swarm import __version__

    console.print(f"[bold]sargvision-swarm[/] {__version__}")
    console.print("Reflex: Boids, Olfati-Saber, BVC")
    console.print("Sim: pure-Python double integrator (gym-pybullet-drones optional)")
    console.print("Comms: in-memory pubsub (Zenoh-shaped API)")
    console.print("Agents: pluggable LLM backend (mock / anthropic / ollama)")
    console.print("Orchestrator: deterministic + LangGraph-ready")


if __name__ == "__main__":
    app()
