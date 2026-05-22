"""FastAPI WebSocket bridge — streams LiveSession frames to the Next.js console.

  uv run swarm-bridge
  # listens on ws://127.0.0.1:8765/swarm

Frame format (msgpack-encoded, see frontend src/lib/types.ts → SwarmFrame):

  {
    t, step, scenario,
    drones: [{ id, lon, lat, alt_m, vel_ms, heading_deg, battery, healthy,
               role, intent, affiliation, platform }],
    edges:  [{ src, dst, strength }],
    recent_messages: [{ t, src, dst, protocol, topic, bytes, summary }],
    bft_events: [...],
    cbba_events: [...],
    stats: { total_msgs, msgs_per_s, by_protocol }
  }
"""

from __future__ import annotations

import asyncio
import math
import os
from contextlib import asynccontextmanager
from typing import Any

import msgpack
import numpy as np
import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sargvision_swarm.comms.protocols import WireMessage
from sargvision_swarm.demo.live_session import LiveSession
from sargvision_swarm.server.geo import DEFAULT_ANCHOR_LAT, DEFAULT_ANCHOR_LON, local_to_geo

log = structlog.get_logger()


# ── Frame builder ──────────────────────────────────────────────────────


def _heading_deg(vx: float, vy: float) -> float:
    return (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0


def _classify_intent(intent: str) -> str:
    return intent.replace("_", " ").upper()


def _affiliation_for_drone(drone) -> str:  # noqa: ANN001
    # Phase A — all SARGVISION drones friendly. Hostile lane comes in Phase C.
    return "friend"


def _platform_for_drone(drone) -> str:  # noqa: ANN001
    # Phase A — single platform. Real fleets will mix ALFA-S / Sheshnaag / Tapas.
    return "ALFA-S"


def build_frame(session: LiveSession) -> dict[str, Any]:
    drones = []
    for d in session.swarm.drones:
        lon, lat = local_to_geo(float(d.pos[0]), float(d.pos[1]))
        vx, vy = float(d.vel[0]), float(d.vel[1])
        drones.append(
            {
                "id": int(d.id),
                "lon": lon,
                "lat": lat,
                "alt_m": float(d.pos[2]),
                "vel_ms": float(math.hypot(vx, vy)),
                "heading_deg": _heading_deg(vx, vy),
                "battery": float(d.battery),
                "healthy": bool(d.healthy),
                "role": d.role.value,
                "intent": session.intents.get(int(d.id), "hold_formation"),
                "affiliation": _affiliation_for_drone(d),
                "platform": _platform_for_drone(d),
            }
        )

    # Comm-range edges
    adj = session.comm_adjacency()
    strengths = session.comm.signal_strength(session.swarm.positions)
    edges = []
    n = session.swarm.n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append(
                    {
                        "src": int(session.swarm.drones[i].id),
                        "dst": int(session.swarm.drones[j].id),
                        "strength": float(strengths[i, j]),
                    }
                )

    # Recent messages (last 60)
    recent = list(session.message_log.recent(k=60))
    recent_payload = [
        {
            "t": float(m.t),
            "src": int(m.src),
            "dst": (None if m.dst is None else int(m.dst)),
            "protocol": m.protocol.value,
            "topic": str(m.topic),
            "bytes": int(m.bytes_size),
            "summary": _summarise_payload(m),
        }
        for m in recent
    ]

    stats_raw = session.render_stats()
    stats = {
        "total_msgs": int(stats_raw.get("total_msgs", 0)),
        "msgs_per_s": float(stats_raw.get("msgs_per_s", 0.0)),
        "by_protocol": {k: int(v) for k, v in stats_raw.get("by_proto", {}).items()},
    }

    # BFT + CBBA history surfaced from session
    bft_events = [_bft_payload(ev) for ev in getattr(session, "bft_history", [])]
    cbba_events = [
        {
            "t": float(ev["t"]),
            "task_id": str(ev["task_id"]),
            "bidder_id": int(ev["bidder_id"]),
            "bid_score": float(ev["bid_score"]),
        }
        for ev in getattr(session, "cbba_history", [])
    ]

    return {
        "t": float(session.swarm.t),
        "step": int(session.step_i),
        "scenario": session.scenario,
        "drones": drones,
        "edges": edges,
        "recent_messages": recent_payload,
        "bft_events": bft_events,
        "cbba_events": cbba_events,
        "stats": stats,
    }


def _summarise_payload(m: WireMessage) -> str:
    p = m.payload
    if isinstance(p, dict):
        if "method" in p:
            return f"{p['method']} → {p.get('params', {}).get('to', '?')}"
        if "intent" in p:
            return f"intent={p['intent']}"
        if "decision" in p:
            return f"vote {p['decision']} on {p['proposal']}"
        if "bid_score" in p:
            return f"bid {p.get('task_id')} score={p.get('bid_score', 0):.2f}"
        if "pos" in p:
            pos = p["pos"]
            return f"pose ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})"
        if "battery" in p:
            return f"hb {p['battery']:.2f}"
    return str(p)[:64]


def _bft_payload(ev: dict) -> dict:
    return {
        "t": float(ev.get("t", 0.0)),
        "proposal": str(ev.get("proposal", "")),
        "passed": bool(ev.get("passed", False)),
        "yes": int(ev.get("yes", 0)),
        "no": int(ev.get("no", 0)),
        "voters": list(ev.get("voters", [])),
        "byzantine": list(ev.get("byzantine", [])),
    }


# ── App + session orchestration ─────────────────────────────────────────


class SwarmService:
    """Single LiveSession running in an asyncio loop, fanned to N WS clients."""

    def __init__(self) -> None:
        self.session: LiveSession | None = None
        self.subscribers: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(
        self,
        n_drones: int = 24,
        scenario: str = "coverage",
        seed: int = 42,
        comm_range_m: float = 15.0,
        hz: float = 10.0,
    ) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                self._stop.set()
                await self._task
            self._stop = asyncio.Event()
            self.session = LiveSession(
                n_drones=n_drones,
                scenario=scenario,
                seed=seed,
                comm_range_m=comm_range_m,
            )
            self._task = asyncio.create_task(self._run_loop(hz))

    async def _run_loop(self, hz: float) -> None:
        dt = 1.0 / max(1.0, hz)
        assert self.session is not None
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        try:
            while not self._stop.is_set():
                self.session.step()
                frame = build_frame(self.session)
                packed = msgpack.packb(frame, use_bin_type=True)
                await self._broadcast(packed)
                next_t += dt
                sleep_for = max(0.0, next_t - loop.time())
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    next_t = loop.time()
        except asyncio.CancelledError:
            return

    async def _broadcast(self, packed: bytes) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.subscribers):
            try:
                await ws.send_bytes(packed)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subscribers.discard(ws)

    async def subscribe(self, ws: WebSocket) -> None:
        self.subscribers.add(ws)

    async def unsubscribe(self, ws: WebSocket) -> None:
        self.subscribers.discard(ws)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task


service = SwarmService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Auto-start a Counter-Swarm scenario on boot so the console has something live.
    n = int(os.getenv("SWARM_N", "24"))
    sc = os.getenv("SWARM_SCENARIO", "coverage")
    await service.start(n_drones=n, scenario=sc, seed=42, comm_range_m=18.0, hz=10.0)
    log.info("swarm-bridge.started", n_drones=n, scenario=sc)
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(lifespan=lifespan, title="SARGVISION Swarm Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "SARGVISION Swarm Bridge",
        "ws": "/swarm",
        "anchor": [DEFAULT_ANCHOR_LON, DEFAULT_ANCHOR_LAT],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/scenario/{name}")
async def set_scenario(name: str, n: int = 24, seed: int = 42) -> dict:
    await service.start(n_drones=n, scenario=name, seed=seed, comm_range_m=18.0, hz=10.0)
    return {"started": name, "n": n, "seed": seed}


@app.websocket("/swarm")
async def swarm_ws(ws: WebSocket) -> None:
    await ws.accept()
    await service.subscribe(ws)
    try:
        while True:
            # We don't expect inbound traffic in Phase A. Drain to keep socket alive.
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await service.unsubscribe(ws)


def main() -> None:
    host = os.getenv("SWARM_BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("SWARM_BRIDGE_PORT", "8765"))
    uvicorn.run(
        "sargvision_swarm.server.bridge:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
