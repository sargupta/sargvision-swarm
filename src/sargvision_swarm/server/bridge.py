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
    # Map role → platform — looks like a real heterogeneous ORBAT.
    name = drone.role.value if hasattr(drone.role, "value") else str(drone.role)
    return {
        "leader": "Sheshnaag-150 (Command)",
        "scout": "ALFA-S Scout",
        "relay": "Tapas-BH Relay",
        "worker": "ALFA-S Striker",
    }.get(name, "ALFA-S Striker")


def build_frame(session: LiveSession) -> dict[str, Any]:
    hostiles = []
    if getattr(session, "hostile_fleet", None) is not None:
        from sargvision_swarm.server.geo import local_to_geo as _local_to_geo
        for h in session.hostile_fleet.hostiles:
            lon, lat = _local_to_geo(float(h.pos[0]), float(h.pos[1]))
            posterior = None
            threat_label = None
            shield_state_pre = getattr(session, "shield_state", None)
            if shield_state_pre is not None:
                post = shield_state_pre.posteriors.get(int(h.id))
                if post is not None:
                    posterior = [float(x) for x in post.tolist()]
                    from sargvision_swarm.orchestrator.shield import (
                        THREAT_CLASSES as _TC,
                    )
                    threat_label = str(_TC[int(post.argmax())])
            hostiles.append(
                {
                    "id": int(h.id),
                    "callsign": h.callsign,
                    "lon": lon,
                    "lat": lat,
                    "alt_m": float(h.pos[2]),
                    "alive": bool(h.alive),
                    "bearing_deg": float(h.spawn_bearing_deg),
                    "intent": h.intent_label,
                    "assigned_to": h.assigned_to,
                    "threat_class": threat_label,
                    "posterior": posterior,
                }
            )

    # ── SHIELD state surfaced per-drone ──
    shield_state = getattr(session, "shield_state", None)
    shield_params = getattr(session, "shield_params", None)
    spoofed_ids = set(getattr(session, "spoofed_ids", set()) or set())
    kill_switched_ids = set(getattr(session, "shield_kill_switched", set()) or set())
    trust_kill_threshold = (
        float(shield_params.trust_kill_threshold) if shield_params is not None else 0.25
    )
    have_shield = (
        shield_state is not None
        and getattr(shield_state, "loyalty", None) is not None
        and shield_state.loyalty.size == session.swarm.n
    )

    drones = []
    intercept = getattr(session, "intercept_assignment", {}) or {}
    task_map = getattr(session, "current_task", {}) or {}
    for d in session.swarm.drones:
        lon, lat = local_to_geo(float(d.pos[0]), float(d.pos[1]))
        vx, vy = float(d.vel[0]), float(d.vel[1])
        loyalty = float(shield_state.loyalty[int(d.id)]) if have_shield else 1.0
        trust = float(shield_state.trust[int(d.id)]) if have_shield else 1.0
        spoofed = int(d.id) in spoofed_ids
        kill_switched = int(d.id) in kill_switched_ids
        if kill_switched:
            shield_class = "kill_switched"
        elif spoofed:
            shield_class = "hijacked"
        elif trust < 0.55:
            shield_class = "suspect"
        else:
            shield_class = "loyal"
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
                "task": task_map.get(int(d.id), "STATION HOLD"),
                "intercept_target": intercept.get(int(d.id)),
                "loyalty": loyalty,
                "trust": trust,
                "shield_class": shield_class,
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

    threat = None
    if getattr(session, "hostile_fleet", None) is not None:
        threat = {
            "total": int(session.hostile_fleet.total),
            "remaining": int(session.hostile_fleet.remaining),
            "neutralized": int(session.hostile_fleet.neutralized),
        }
    kill_events = []
    now = float(session.swarm.t)
    for k in list(getattr(session, "kill_events", []) or []):
        # Only surface kills from the last 4s so the flash sprite fades naturally.
        if now - float(k["t"]) <= 4.0:
            from sargvision_swarm.server.geo import local_to_geo as _l2g
            klon, klat = _l2g(float(k["pos"][0]), float(k["pos"][1]))
            kill_events.append(
                {
                    "t": float(k["t"]),
                    "killer_id": int(k["killer_id"]),
                    "callsign": str(k["callsign"]),
                    "lon": klon,
                    "lat": klat,
                    "alt_m": float(k["pos"][2]),
                }
            )
    # ── VAJRA aggregate telemetry (Voronoi hysteresis + algebraic connectivity) ──
    vajra_state = getattr(session, "vajra_state", None)
    vajra_params = getattr(session, "vajra_params", None)
    vajra_summary: dict[str, Any] | None = None
    if vajra_state is not None:
        owners = getattr(getattr(vajra_state, "voronoi", None), "owner", {}) or {}
        vajra_summary = {
            "lambda2": float(getattr(vajra_state, "lambda2", 0.0)),
            "n_components": int(getattr(vajra_state, "n_components", 1)),
            "fragmented": bool(getattr(session, "fragmentation_alarmed", False)),
            "fragmentation_threshold": float(
                getattr(vajra_params, "fragmentation_threshold", 1e-3)
                if vajra_params is not None
                else 1e-3
            ),
            "jamming_factor": float(
                getattr(vajra_params, "jamming_factor", 0.0) if vajra_params is not None else 0.0
            ),
            "voronoi_owners": {int(k): int(v) for k, v in owners.items()},
            "handover_count": len(getattr(vajra_state, "handover_events", []) or []),
            "n_friendlies": int(session.swarm.n),
            "n_hostiles_alive": int(threat["remaining"]) if threat else 0,
        }

    # ── GOVERNED MIGRATION zones + hazards + occupancy ──
    migration_field = getattr(session, "migration_field", None)
    migration_summary: dict[str, Any] | None = None
    if migration_field is not None:
        occ = migration_field.occupancy()
        from sargvision_swarm.server.geo import local_to_geo as _mig_l2g
        zones_payload = []
        for z in migration_field.zones:
            zlon, zlat = _mig_l2g(float(z.center[0]), float(z.center[1]))
            zones_payload.append(
                {
                    "id": z.id,
                    "name": z.name,
                    "kind": z.kind,
                    "lon": zlon,
                    "lat": zlat,
                    "alt_m": float(z.center[2]),
                    "radius_m": float(z.radius_m * 80.0),  # geo-scaled metres
                    "capacity": int(z.capacity),
                    "occupancy": int(occ.get(z.id, 0)),
                    "color": z.color_hex,
                }
            )
        hazards_payload = []
        for h in migration_field.hazards:
            hlon, hlat = _mig_l2g(float(h.center[0]), float(h.center[1]))
            hazards_payload.append(
                {
                    "id": h.id,
                    "name": h.name,
                    "lon": hlon,
                    "lat": hlat,
                    "alt_m": float(h.center[2]),
                    "radius_m": float(h.radius_m * 80.0),
                    "severity": float(h.severity),
                    "pulse_phase": float(h.pulse_phase),
                }
            )
        # Drone trails — convert sim coords to lon/lat per point
        trails_payload: list[dict] = []
        now_t = float(session.swarm.t)
        for did, pts in migration_field.trails.items():
            if not pts or len(pts) < 2:
                continue
            geo_pts = []
            for (sx, sy) in pts[-40:]:  # cap to last 40 to keep payload small
                glon, glat = _mig_l2g(float(sx), float(sy))
                geo_pts.append([glon, glat])
            trails_payload.append({"id": int(did), "path": geo_pts})

        throughput = migration_field.throughput_per_min(now_t, window_s=60.0)
        migration_summary = {
            "zones": zones_payload,
            "hazards": hazards_payload,
            "trails": trails_payload,
            "throughput_per_min": throughput,
            "completed_loops": int(migration_field.completed_loops),
            "violations": int(migration_field.violations),
            "collisions": int(migration_field.collisions),
            "yields": int(migration_field.yields),
            "assignments": {
                str(k): v for k, v in migration_field.assignment.items()
            },
        }

    # ── SHIELD aggregate summary surfaced to console ──
    loyal_n = sum(1 for d in drones if d["shield_class"] == "loyal")
    suspect_n = sum(1 for d in drones if d["shield_class"] == "suspect")
    hijacked_n = sum(1 for d in drones if d["shield_class"] == "hijacked")
    kill_switched_n = sum(1 for d in drones if d["shield_class"] == "kill_switched")
    threat_mix = {"decoy": 0, "kinetic": 0, "nuisance": 0, "unknown": 0}
    for h in hostiles:
        cls = h.get("threat_class") or "unknown"
        threat_mix[cls] = threat_mix.get(cls, 0) + 1
    shield_summary = {
        "loyal": loyal_n,
        "suspect": suspect_n,
        "hijacked": hijacked_n,
        "kill_switched": kill_switched_n,
        "decoys_skipped": int(getattr(session, "shield_decoy_skipped", 0)),
        "trust_kill_threshold": trust_kill_threshold,
        "threat_mix": threat_mix,
        "hijack_active": bool(getattr(session, "hijack_active", False)),
    }

    return {
        "t": float(session.swarm.t),
        "step": int(session.step_i),
        "scenario": session.scenario,
        "drones": drones,
        "hostiles": hostiles,
        "threat": threat,
        "edges": edges,
        "recent_messages": recent_payload,
        "bft_events": bft_events,
        "cbba_events": cbba_events,
        "kill_events": kill_events,
        "shield": shield_summary,
        "vajra": vajra_summary,
        "migration": migration_summary,
        "stats": stats,
        "flags": {
            "jamming": bool(getattr(session, "jamming", False)),
            "gnss_denied": bool(getattr(session, "gnss_denied", False)),
            "hijack_active": bool(getattr(session, "hijack_active", False)),
        },
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


@app.post("/jam")
async def toggle_jam() -> dict:
    sess = service.session
    if sess is None:
        return {"ok": False, "reason": "no active session"}
    sess.jamming = not sess.jamming
    # Halve comm range when jamming, restore when off.
    if sess.jamming:
        sess.comm.range_m = max(6.0, sess.comm.range_m * 0.45)
    else:
        sess.comm.range_m = 18.0
    return {"ok": True, "jamming": sess.jamming, "range_m": sess.comm.range_m}


@app.post("/gnss/toggle")
async def toggle_gnss() -> dict:
    sess = service.session
    if sess is None:
        return {"ok": False, "reason": "no active session"}
    sess.gnss_denied = not sess.gnss_denied
    return {"ok": True, "gnss_denied": sess.gnss_denied}


@app.post("/hijack/toggle")
async def toggle_hijack() -> dict:
    """SHIELD demo: inject sensor-spoofed friendlies. PageRank trust collapses,
    sheaf loyalty drops, kill-switch fires below threshold."""
    sess = service.session
    if sess is None:
        return {"ok": False, "reason": "no active session"}
    sess.hijack_active = not sess.hijack_active
    return {"ok": True, "hijack_active": sess.hijack_active}


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
