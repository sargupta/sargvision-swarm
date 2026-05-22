"""Plot a swarm state as 3D Plotly figure or matplotlib snapshot."""

from __future__ import annotations

import io

import numpy as np

from sargvision_swarm.core.state import Role, SwarmState


_ROLE_COLOR = {
    Role.WORKER: "#3b82f6",   # blue
    Role.SCOUT: "#10b981",    # green
    Role.RELAY: "#f59e0b",    # amber
    Role.LEADER: "#ef4444",   # red
}


def swarm_to_plotly_figure(swarm: SwarmState, title: str = ""):
    """Return a Plotly Figure with drones as 3D scatter + velocity arrows."""
    import plotly.graph_objects as go

    positions = swarm.positions
    velocities = swarm.velocities
    colors = [_ROLE_COLOR.get(d.role, "#3b82f6") for d in swarm.drones]
    ids = [d.id for d in swarm.drones]

    drone_trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers+text",
        marker=dict(size=5, color=colors, line=dict(width=0)),
        text=[str(i) for i in ids],
        textposition="top center",
        textfont=dict(size=9, color="#555"),
        name="drones",
    )

    # Velocity vectors as line segments
    arrow_x: list[float | None] = []
    arrow_y: list[float | None] = []
    arrow_z: list[float | None] = []
    for p, v in zip(positions, velocities):
        arrow_x.extend([p[0], p[0] + v[0] * 0.5, None])
        arrow_y.extend([p[1], p[1] + v[1] * 0.5, None])
        arrow_z.extend([p[2], p[2] + v[2] * 0.5, None])

    arrow_trace = go.Scatter3d(
        x=arrow_x,
        y=arrow_y,
        z=arrow_z,
        mode="lines",
        line=dict(width=2, color="#94a3b8"),
        name="velocity",
        showlegend=False,
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-30, 30], title="x (m)"),
            yaxis=dict(range=[-30, 30], title="y (m)"),
            zaxis=dict(range=[0, 15], title="z (m)"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        showlegend=False,
    )
    return go.Figure(data=[drone_trace, arrow_trace], layout=layout)


def swarm_to_mpl_snapshot(swarm: SwarmState, title: str = "") -> bytes:
    """Render a matplotlib 3D PNG to bytes — used in headless contexts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    positions = swarm.positions
    velocities = swarm.velocities
    colors = [_ROLE_COLOR.get(d.role, "#3b82f6") for d in swarm.drones]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=18)
    for p, v in zip(positions, velocities):
        ax.plot([p[0], p[0] + v[0] * 0.5], [p[1], p[1] + v[1] * 0.5], [p[2], p[2] + v[2] * 0.5], c="#94a3b8", lw=0.8)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
