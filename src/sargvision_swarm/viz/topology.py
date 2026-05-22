"""Topology viz — drones + comm-range edges + per-link signal-strength colors."""

from __future__ import annotations

import numpy as np

from sargvision_swarm.comms.topology import CommModel
from sargvision_swarm.core.state import Role, SwarmState


_ROLE_COLOR = {
    Role.WORKER: "#3b82f6",
    Role.SCOUT: "#10b981",
    Role.RELAY: "#f59e0b",
    Role.LEADER: "#ef4444",
}


def topology_figure(swarm: SwarmState, comm: CommModel, title: str = ""):
    """Plotly 3D scatter + edges between drones in comm range."""
    import plotly.graph_objects as go

    positions = swarm.positions
    n = swarm.n
    strengths = comm.signal_strength(positions)
    adjacency = comm.adjacency(positions)

    # Edges (i < j to dedupe)
    edge_x, edge_y, edge_z = [], [], []
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                edge_x.extend([positions[i, 0], positions[j, 0], None])
                edge_y.extend([positions[i, 1], positions[j, 1], None])
                edge_z.extend([positions[i, 2], positions[j, 2], None])
                edge_colors.append(strengths[i, j])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2, color="rgba(16, 185, 129, 0.35)"),
        hoverinfo="skip",
        showlegend=False,
        name="comm-links",
    )

    drone_trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers+text",
        marker=dict(
            size=7,
            color=[_ROLE_COLOR.get(d.role, "#3b82f6") for d in swarm.drones],
            line=dict(width=1, color="#1e293b"),
        ),
        text=[str(d.id) for d in swarm.drones],
        textposition="top center",
        textfont=dict(size=9, color="#475569"),
        hovertemplate=(
            "drone %{text}<br>x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<extra></extra>"
        ),
        name="drones",
    )

    n_edges = sum(1 for _ in edge_colors)
    layout = go.Layout(
        title=f"{title} · drones={n} · comm-links={n_edges} · range={comm.range_m}m",
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
    return go.Figure(data=[edge_trace, drone_trace], layout=layout)


def bandwidth_figure(rates: dict[str, dict[str, float]], title: str = ""):
    """Horizontal bar chart of bytes/s per protocol."""
    import plotly.graph_objects as go

    if not rates:
        return go.Figure(layout=go.Layout(title="(no traffic yet)"))
    protos = list(rates.keys())
    bytes_per_s = [rates[p]["bytes_per_s"] for p in protos]
    msgs_per_s = [rates[p]["msgs_per_s"] for p in protos]

    fig = go.Figure(
        data=[
            go.Bar(
                x=bytes_per_s,
                y=protos,
                orientation="h",
                marker=dict(color="#3b82f6"),
                text=[f"{b:.0f} B/s · {m:.1f} msg/s" for b, m in zip(bytes_per_s, msgs_per_s)],
                textposition="outside",
            )
        ],
        layout=go.Layout(
            title=title,
            xaxis=dict(title="bytes / sec (5s window)"),
            yaxis=dict(title="protocol"),
            margin=dict(l=80, r=10, t=40, b=40),
            height=260,
        ),
    )
    return fig


def protocol_breakdown_pie(counts: dict[str, int]):
    """Pie chart of message counts by protocol."""
    import plotly.graph_objects as go

    if not counts:
        return go.Figure(layout=go.Layout(title="(no messages)"))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.5,
                marker=dict(colors=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]),
            )
        ],
        layout=go.Layout(
            title="Messages by protocol",
            height=260,
            margin=dict(l=10, r=10, t=40, b=10),
        ),
    )
    return fig
