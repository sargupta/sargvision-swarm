"""Border-Strike scenario — multi-axis hostile drone attack on HVT targets.

Spec (Operation Trishul):
  - Two named High-Value Targets (HVTs) on the map:
      * LEH AIRBASE   — military, southwest of swarm
      * KARU POWER ST — energy infra, southeast of swarm
  - Two hostile waves spawned from the north LoC line:
      * Axis A:  6 Shahed-style hostiles → LEH_AB
      * Axis B:  4 Shahed-style hostiles → KARU_PS
  - HVT damage is tracked: a hostile within `impact_radius_m` of its assigned
    HVT scores a hit; HVT status flips PROTECTED → UNDER_ATTACK → STRUCK.
  - 90-second scripted phase machine runs alongside, surfaced to the console
    via the bridge so the operator gets a narrative caption.

Coordinate frame is sim-local meters; bridge re-projects to lat/lon for the
console using `sargvision_swarm.server.geo.local_to_geo`. The LoC line is
sent as static geo data so the frontend can draw a red dashed border.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

HVTStatus = Literal["PROTECTED", "UNDER_ATTACK", "STRUCK"]


@dataclass
class HVT:
    """A high-value target the swarm must defend."""

    id: str
    name: str
    kind: Literal["military", "energy", "command"]
    pos: np.ndarray  # (3,) sim-local meters
    impact_radius_m: float = 2.0
    health: float = 1.0
    status: HVTStatus = "PROTECTED"
    last_threat_t: float = -1e9
    hits_taken: int = 0


@dataclass
class Phase:
    """One step of the scripted demo narrative."""

    name: str
    caption: str
    duration_s: float
    color: str = "friend"  # friend | warn | hostile | ok


# 90-second scripted timeline. Each phase auto-advances at sim-time `started_t
# + duration_s`. Captions are designed to read like a live ATC + AEW narration.
DEFAULT_PHASES: list[Phase] = [
    Phase("PEACETIME", "Defensive patrol · 24 ALFA-S over Leh sector", 8.0, "friend"),
    Phase("DETECTION", "DRISHTI · 6 inbound from N · 4 inbound from E", 12.0, "warn"),
    Phase("CLASSIFICATION", "PRAJNA · classifying RCS / RF / jerk · decoys filtered", 12.0, "warn"),
    Phase("AUCTION", "YAJNA · ED-CBBA assigning interceptors · VAJRA load-balancing", 12.0, "warn"),
    Phase("ROE AUTHORISE", "SABHA · BFT vote 7/7 · ENGAGE authorised", 8.0, "warn"),
    Phase("ENGAGEMENT", "VAJRA · kinetic intercept in progress · HVTs holding", 28.0, "hostile"),
    Phase("POSTMORTEM", "MISSION COMPLETE · HVTs PROTECTED · ₹- cr saved", 10.0, "ok"),
]


@dataclass
class BorderStrikeField:
    """Container for HVTs, LoC line, and phase machine for Operation Trishul.

    The hostile fleet is owned by LiveSession (re-uses the standard HostileFleet
    plumbing); this class only owns the static defense geometry + phase clock.
    """

    hvts: list[HVT] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=lambda: list(DEFAULT_PHASES))
    phase_idx: int = 0
    phase_started_t: float = 0.0
    inr_cr_saved: float = 0.0  # value-of-asset narrative
    # LoC line in sim coords — a polyline drawn east-west across the north.
    # Sent to the console once; rendered as a red dashed PathLayer.
    loc_line: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (-26.0, 18.0),
            (-12.0, 18.5),
            (0.0, 18.0),
            (12.0, 18.5),
            (26.0, 18.2),
        ]
    )

    @classmethod
    def build_default(cls) -> BorderStrikeField:
        """Standard Trishul layout: LEH_AB to the SW, KARU_PS to the SE."""
        return cls(
            hvts=[
                HVT(
                    id="LEH_AB",
                    name="LEH AIRBASE",
                    kind="military",
                    pos=np.array([-6.0, -14.0, 0.0], dtype=float),
                    impact_radius_m=2.5,
                ),
                HVT(
                    id="KARU_PS",
                    name="KARU POWER STN",
                    kind="energy",
                    pos=np.array([12.0, -16.0, 0.0], dtype=float),
                    impact_radius_m=2.2,
                ),
                HVT(
                    id="DBO_FWD",
                    name="DBO FWD POST",
                    kind="command",
                    pos=np.array([-18.0, 8.0, 0.0], dtype=float),
                    impact_radius_m=1.8,
                ),
            ],
        )

    # ── phase machine ──────────────────────────────────────────────────────
    def current_phase(self) -> Phase:
        return self.phases[self.phase_idx]

    def step_phase(
        self,
        t: float,
        *,
        events: dict | None = None,
    ) -> bool:
        """Advance phase pointer. Event-coupled when `events` is provided so
        narrative tracks reality (otherwise a fast intercept finishes kills
        while the wall-clock script is still at CLASSIFICATION).

        `events` keys (all optional bools / counters):
          first_detection      — any hostile has been observed
          posterior_settled    — Bayesian classifier has settled on all contacts
          first_assignment     — VAJRA has assigned at least one interceptor
          bft_authorized       — SABHA has passed at least one engage decree
          first_kill           — at least one hostile neutralised
          all_kinetics_dead    — every kinetic-class hostile KIA
          hvt_all_protected    — every HVT status == PROTECTED
          hvt_all_struck       — every HVT status == STRUCK

        Returns True on phase transition.
        """
        ev = events or {}
        advanced_any = False

        # Loop: allow up to 2 phase advances per tick so the script can catch up
        # if multiple triggers have already fired (fast intercept beats the
        # narrative). Min-dwell of 1.2s ensures each phase is visible at least
        # briefly so the operator can read the caption.
        for _ in range(2):
            if self.phase_idx >= len(self.phases) - 1:
                break
            ph = self.current_phase()
            elapsed = t - self.phase_started_t
            min_dwell = 1.2  # seconds — enough to read the caption

            triggered = False
            if ph.name == "PEACETIME" and ev.get("first_detection"):
                triggered = True
            elif ph.name == "DETECTION" and ev.get("posterior_settled"):
                triggered = True
            elif ph.name == "CLASSIFICATION" and (
                ev.get("first_assignment") or ev.get("bft_authorized")
            ):
                triggered = True
            elif ph.name == "AUCTION" and ev.get("bft_authorized"):
                triggered = True
            elif ph.name == "ROE AUTHORISE" and (
                ev.get("first_kill") or ev.get("all_kinetics_dead")
            ):
                triggered = True
            elif ph.name == "ENGAGEMENT" and (
                ev.get("all_kinetics_dead") or ev.get("hvt_all_struck")
            ):
                triggered = True

            # Honour min-dwell except for the terminal phases when the kill
            # chain is over — POSTMORTEM should appear promptly.
            ready = triggered and (elapsed >= min_dwell or ev.get("all_kinetics_dead"))

            if ready or elapsed >= ph.duration_s:
                self.phase_idx += 1
                self.phase_started_t = t
                advanced_any = True
                continue  # try to advance again in case another event already fired
            break
        return advanced_any

    def reset_phase(self, t: float = 0.0) -> None:
        self.phase_idx = 0
        self.phase_started_t = t

    def phase_serialize(self, t: float) -> dict:
        ph = self.current_phase()
        return {
            "idx": int(self.phase_idx),
            "of": int(len(self.phases)),
            "name": ph.name,
            "caption": ph.caption,
            "color": ph.color,
            "elapsed_s": float(t - self.phase_started_t),
            "duration_s": float(ph.duration_s),
            "progress": float(min(1.0, max(0.0, (t - self.phase_started_t) / ph.duration_s))),
        }

    # ── HVT damage tracking ───────────────────────────────────────────────
    def evaluate_hvt_damage(self, hostiles: list, t: float) -> list[dict]:
        """Walk live hostiles, check distance to assigned HVT, register hits.

        Returns a list of `{hvt_id, t, callsign}` impact events for the event
        log. Mutates HVT.status / health / hits_taken in place.
        """
        events: list[dict] = []
        for h in hostiles:
            if not getattr(h, "alive", False):
                continue
            target_id = getattr(h, "target_hvt", None)
            if target_id is None:
                continue
            hvt = self.hvt_by_id(target_id)
            if hvt is None:
                continue
            d = float(np.linalg.norm(h.pos[:2] - hvt.pos[:2]))
            if d <= hvt.impact_radius_m:
                # Impact: kill the hostile, damage the HVT.
                h.alive = False
                hvt.hits_taken += 1
                hvt.last_threat_t = t
                hvt.health = max(0.0, hvt.health - 0.34)
                hvt.status = "STRUCK" if hvt.health <= 0.01 else "UNDER_ATTACK"
                events.append(
                    {"hvt_id": hvt.id, "t": float(t), "callsign": getattr(h, "callsign", "HST")}
                )
            elif d <= hvt.impact_radius_m * 3.5:
                # Threat ring — flip UNDER_ATTACK if still PROTECTED
                if hvt.status == "PROTECTED":
                    hvt.status = "UNDER_ATTACK"
                    hvt.last_threat_t = t

        # Recovery: if HVT hasn't been threatened for 6s and is not STRUCK,
        # downgrade back to PROTECTED.
        for hvt in self.hvts:
            if hvt.status == "UNDER_ATTACK" and t - hvt.last_threat_t > 6.0 and hvt.health > 0.05:
                hvt.status = "PROTECTED"
        return events

    def hvt_by_id(self, id_: str) -> HVT | None:
        for h in self.hvts:
            if h.id == id_:
                return h
        return None

    def all_hvts_struck(self) -> bool:
        return all(h.status == "STRUCK" for h in self.hvts)

    def all_hvts_protected(self) -> bool:
        return all(h.status == "PROTECTED" for h in self.hvts)


def spawn_border_strike_hostiles(
    fleet,  # HostileFleet
    field_: BorderStrikeField,
    seed: int = 7,
) -> None:
    """Spawn two hostile waves with explicit per-hostile HVT assignments.

    Axis A (6 hostiles): LoC north → LEH_AB
    Axis B (4 hostiles): far east → KARU_PS
    Axis C (2 hostiles, optional sneak): far west → DBO_FWD

    Uses the existing HostileFleet structure but overrides the spawn loop.
    Each hostile gets a `target_hvt` attribute (added dynamically) so the
    LiveSession step loop knows where to vector it.
    """
    rng = random.Random(seed)
    fleet.reset()

    leh = field_.hvt_by_id("LEH_AB")
    karu = field_.hvt_by_id("KARU_PS")
    dbo = field_.hvt_by_id("DBO_FWD")

    # Axis A — 6 hostiles spaced across the LoC line north of Leh.
    if leh is not None:
        for i, x_offset in enumerate([-9.0, -5.5, -2.5, 0.5, 4.0, 7.5]):
            spawn = np.array([x_offset, 22.0, 6.0], dtype=float)
            _spawn_at(
                fleet, spawn, leh.pos, rng=rng, target_id="LEH_AB", callsign=f"AXA-{i + 1:03d}"
            )

    # Axis B — 4 hostiles from the east edge.
    if karu is not None:
        for i, y_offset in enumerate([-8.0, -12.0, -16.0, -20.0]):
            spawn = np.array([24.0, y_offset, 6.0], dtype=float)
            _spawn_at(
                fleet, spawn, karu.pos, rng=rng, target_id="KARU_PS", callsign=f"AXB-{i + 1:03d}"
            )

    # Axis C — 2 sneak hostiles from the west, vectored at DBO forward post.
    if dbo is not None:
        for i, y_offset in enumerate([12.0, 4.0]):
            spawn = np.array([-26.0, y_offset, 6.0], dtype=float)
            _spawn_at(
                fleet, spawn, dbo.pos, rng=rng, target_id="DBO_FWD", callsign=f"AXC-{i + 1:03d}"
            )


def _spawn_at(
    fleet, spawn_pos: np.ndarray, target_pos: np.ndarray, *, rng, target_id: str, callsign: str
) -> None:
    """Append one hostile to `fleet.hostiles` aimed at target_pos."""
    from sargvision_swarm.sim.hostiles import CLASS_SIGNATURE, Hostile

    pos = spawn_pos.copy()
    pos[2] = 6.0 + rng.uniform(-1.0, 1.0)
    to_target = target_pos - pos
    to_target[2] = 0.0
    norm = float(np.linalg.norm(to_target))
    vel = (to_target / max(norm, 1e-6)) * fleet.cruise_speed_ms

    # 80% kinetic, 15% decoy, 5% nuisance — bias toward the kill-chain story.
    r = rng.random()
    if r < 0.80:
        tcls = "kinetic"
    elif r < 0.95:
        tcls = "decoy"
    else:
        tcls = "nuisance"
    sig = CLASS_SIGNATURE[tcls]
    bearing = math.atan2(vel[1], vel[0])

    h = Hostile(
        id=fleet._next_id,
        pos=pos.astype(float),
        vel=vel.astype(float),
        spawn_bearing_deg=math.degrees(bearing) % 360.0,
        callsign=callsign,
        threat_class=tcls,
        rcs=float(np.clip(sig["rcs"] + rng.gauss(0, 0.05), 0.0, 1.0)),
        rf_emit=float(np.clip(sig["rf_emit"] + rng.gauss(0, 0.05), 0.0, 1.0)),
        traj_jerk=sig["jerk_rate"] * 0.5,
    )
    # Attach the HVT assignment dynamically so the existing Hostile dataclass
    # doesn't need a schema bump.
    h.target_hvt = target_id
    fleet.hostiles.append(h)
    fleet._next_id += 1
    fleet.spawned += 1


def vector_hostiles_at_hvts(hostiles: list, field_: BorderStrikeField, dt: float) -> None:
    """Continuously re-aim live hostiles at their assigned HVT each tick.

    Without this, hostiles fly past their target if they were spawned with a
    velocity that doesn't pass exactly through it. Called once per LiveSession
    step before the standard fleet.step() integration.
    """
    for h in hostiles:
        if not getattr(h, "alive", False):
            continue
        target_id = getattr(h, "target_hvt", None)
        if target_id is None:
            continue
        hvt = field_.hvt_by_id(target_id)
        if hvt is None or hvt.status == "STRUCK":
            continue
        to_target = hvt.pos - h.pos
        to_target[2] = 0.0
        norm = float(np.linalg.norm(to_target))
        if norm < 1e-3:
            continue
        speed = float(np.linalg.norm(h.vel))
        speed = max(0.4, min(speed, 1.6))
        h.vel = (to_target / norm) * speed
