"""LiveSession — one-tick-at-a-time swarm runner for streaming demos.

Encapsulates the same physics + protocols as `runner.rollout()` but exposes
`step()` so the Gradio app can render a frame between ticks.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sargvision_swarm.agents.backends import MockBackend
from sargvision_swarm.agents.peer_dialogue import PeerDialogue
from sargvision_swarm.comms import (
    BandwidthTracker,
    Channels,
    CommModel,
    InMemoryBus,
    MessageLog,
    Protocol,
    WireMessage,
)
from sargvision_swarm.comms.protocols import IntentPayload
from sargvision_swarm.core import ReflexParams, SwarmState, compose_reflex
from sargvision_swarm.core.bvc import bvc_safe_velocity
from sargvision_swarm.orchestrator import EDCBBA, MissionGoal, MissionPlanner, SwarmRaft, Task
from sargvision_swarm.orchestrator.shield import (
    ShieldParams,
    ShieldState,
    expected_damage,
    shield_priorities,
    threat_class,
)
from sargvision_swarm.orchestrator.chanakya import (
    ChanakyaParams,
    ChanakyaState,
    chanakya_plan_swarm,
    desired_velocity as chanakya_desired_velocity,
    plan_summary as chanakya_summary,
)
from sargvision_swarm.orchestrator.maya import (
    MayaParams,
    MayaState,
    maya_tick,
    posture_dict,
)
from sargvision_swarm.orchestrator.sheshnag import (
    SheshnagParams,
    SheshnagState,
    sheshnag_tick,
)
from sargvision_swarm.orchestrator.vajra import (
    VajraParams,
    VajraState,
    vajra_assign,
)
from sargvision_swarm.sim import SimConfig, SimpleSim
from sargvision_swarm.sim.defense_field import DefenseField
from sargvision_swarm.sim.hostiles import HostileFleet


def _default_task_for_role(role) -> str:
    name = role.value if hasattr(role, "value") else str(role)
    return {
        "leader": "COMMAND ELEMENT",
        "scout": "PATROL PERIMETER",
        "relay": "RELAY LINK",
        "worker": "STATION HOLD",
    }.get(name, "STATION HOLD")


def _role_orbit_offset(t: float, drone_idx: int, role) -> np.ndarray:
    """Per-role time-varying offset added to the drone's base slot so the
    swarm VISIBLY orbits / patrols on the map even when idle.

      LEADER  — small command orbit (r 1.5 m, period 28 s)
      SCOUT   — racetrack (Lissajous-style 4.5 × 2 m, period 40 s)
      RELAY   — mid-ring orbit (r 2.5 m, period 32 s)
      WORKER  — tight cell orbit (r 1.2 m, period 18 s)

    Each drone phase-shifts by its index so the swarm doesn't move in lockstep.
    Returns a (3,) numpy vector with z = 0.
    """
    name = role.value if hasattr(role, "value") else str(role)
    phase = float(drone_idx)
    if name == "leader":
        r, w = 1.5, 2 * math.pi / 28.0
        return np.array([r * math.cos(w * t), r * math.sin(w * t), 0.0])
    if name == "scout":
        w = 2 * math.pi / 40.0
        ang = w * t + phase
        return np.array([4.5 * math.cos(ang), 2.0 * math.sin(ang * 2), 0.0])
    if name == "relay":
        r, w = 2.5, 2 * math.pi / 32.0
        ang = w * t + phase
        return np.array([r * math.cos(ang), r * math.sin(ang), 0.0])
    # worker — tight cell orbit
    r, w = 1.2, 2 * math.pi / 18.0
    ang = w * t + phase
    return np.array([r * math.cos(ang), r * math.sin(ang), 0.0])
from sargvision_swarm.viz.live_frame import FloatingEvent, RecentMessage, TrailHistory


@dataclass
class TickResult:
    """What changed during the latest tick — fed to the renderer."""

    new_messages: list[WireMessage] = field(default_factory=list)
    floating_events: list[FloatingEvent] = field(default_factory=list)
    bft_flash_voters: set[int] = field(default_factory=set)
    cbba_flash_cells: dict[str, tuple[float, float]] = field(default_factory=dict)
    event_log_lines: list[str] = field(default_factory=list)


class LiveSession:
    """One swarm scenario as a step-able generator."""

    def __init__(
        self,
        n_drones: int = 30,
        scenario: str = "coverage",
        seed: int | None = 42,
        comm_range_m: float = 15.0,
        intent_refresh_every: int = 10,
    ) -> None:
        self.n_drones = n_drones
        self.scenario = scenario
        self.seed = seed
        self.intent_refresh_every = intent_refresh_every
        self._rng = random.Random(seed)

        planner = MissionPlanner()
        self.plan = planner.plan(MissionGoal(goal_text="live", n_drones=n_drones, scenario=scenario))
        self.swarm = SwarmState.random_init(n_drones, seed=seed)
        self.sim = SimpleSim(SimConfig(dt=0.05), seed=seed)
        self.bus = InMemoryBus()
        self.comm = CommModel(range_m=comm_range_m, rng_seed=seed or 0)
        self.peer = PeerDialogue(drones=self.swarm.drones, comm=self.comm, rng_seed=seed or 0)
        self.backend = MockBackend(seed=seed)
        self.raft = SwarmRaft(rng_seed=seed or 0)
        self.intents: dict[int, str] = {d.id: "hold_formation" for d in self.swarm.drones}

        self.cbba_tasks: list[Task] = []
        if scenario == "coverage":
            for i, b in enumerate(self.plan.bundles):
                if b.goal_pos:
                    self.cbba_tasks.append(Task(id=f"cell-{i:02d}", pos=b.goal_pos))
        self.cbba = EDCBBA(drones_pos=self.swarm.positions, tasks=self.cbba_tasks)

        self.message_log = MessageLog(capacity=20000)
        self.bandwidth = BandwidthTracker(window_s=5.0)
        self.trails = TrailHistory(capacity=12)
        self.recent_msgs: deque[RecentMessage] = deque(maxlen=200)
        self.floating: deque[FloatingEvent] = deque(maxlen=40)
        self.event_log: deque[str] = deque(maxlen=300)
        self.bft_flash: set[int] = set()
        self.bft_flash_ttl = 0
        self.cbba_flash: dict[str, tuple[float, float]] = {}
        self.cbba_flash_ttl = 0
        self.bft_count = 0
        self.cbba_count = 0
        # Cross-step history surfaces in the frame for the console.
        self.bft_history: list[dict] = []
        self.cbba_history: list[dict] = []
        self.step_i = 0
        self.goal_pos = np.array(self.plan.goal_pos)

        # Hostile stream only active in counter-swarm scenario.
        self.hostile_fleet: HostileFleet | None = None
        if scenario == "coverage":
            self.hostile_fleet = HostileFleet(
                spawn_count=12,
                spawn_radius_m=40.0,
                cruise_speed_ms=1.0,
                engagement_radius_m=2.8,
                seed=(seed or 0) + 1,
            )
            self.hostile_fleet.spawn_initial(center=self.swarm.positions.mean(axis=0))
            self._hostile_first_contact_fired = False

        # CHANAKYA SEAD-ingress scenario — friendlies cross a hostile IADS.
        self.defense_field: DefenseField | None = None
        self.chanakya_state: ChanakyaState | None = None
        self.chanakya_params: ChanakyaParams | None = None
        self.chanakya_targets: np.ndarray | None = None
        self.chanakya_kills: int = 0
        self.chanakya_arrivals: int = 0
        if scenario == "sead_ingress":
            # Friendlies spawn on the south edge, targets on the north edge,
            # with a 4-radar IADS ring guarding the centre.
            n = n_drones
            ys = np.full(n, -32.0) + np.linspace(-3.0, 3.0, n)
            xs = np.linspace(-12.0, 12.0, n)
            zs = np.full(n, 6.0)
            for i, d in enumerate(self.swarm.drones):
                d.pos = np.array([xs[i], ys[i], zs[i]], dtype=float)
                d.vel = np.zeros(3)
            self.defense_field = DefenseField(seed=(seed or 0) + 11)
            self.defense_field.spawn_iads_layout(
                centre=np.array([0.0, 0.0, 4.0]),
                ring_radius=14.0,
                n_radars=4,
                engagement_radius=5.5,
            )
            self.chanakya_state = ChanakyaState()
            self.chanakya_params = ChanakyaParams()
            self.chanakya_targets = np.stack([
                np.array([xs[i], 32.0, 6.0], dtype=float) for i in range(n)
            ])

        # Role distribution — make this look like a real ORBAT.
        # 1 LEADER + 4 SCOUTS + 4 RELAYS + rest STRIKERS (WORKER).
        from sargvision_swarm.core.state import Role as _Role

        for i, d in enumerate(self.swarm.drones):
            if i == 0:
                d.role = _Role.LEADER
            elif i in (1, 2, 3, 4):
                d.role = _Role.SCOUT
            elif i in (5, 6, 7, 8):
                d.role = _Role.RELAY
            else:
                d.role = _Role.WORKER

        # Per-drone current task label surfaced to the console.
        self.current_task: dict[int, str] = {
            d.id: _default_task_for_role(d.role) for d in self.swarm.drones
        }
        # Interceptor → hostile_id assignment.
        self.intercept_assignment: dict[int, int] = {}
        # Recent kill events (last ~6s of explosions for the console flash layer).
        self.kill_events: deque[dict] = deque(maxlen=60)

        # Live operational flags toggleable from the console.
        self.jamming: bool = False
        self.gnss_denied: bool = False
        self.hijack_active: bool = False  # SHIELD demo: hijack 1–2 friendlies

        # SHIELD state + params (built once, mutated in place across ticks)
        self.shield_state = ShieldState()
        self.shield_state.init(n_drones)
        self.shield_params = ShieldParams()
        # IDs of friendlies whose sensor stream is currently spoofed (for SHIELD demo).
        self.spoofed_ids: set[int] = set()
        # Recent SHIELD events surfaced to event log.
        self.shield_decoy_skipped: int = 0
        self.shield_kill_switched: set[int] = set()

        # VAJRA state + params (Voronoi hysteresis, tropical attention, fragmentation).
        self.vajra_state = VajraState()
        self.vajra_params = VajraParams()
        self.fragmentation_alarmed: bool = False

        # MAYA — strategic posture solver (recomputes every ~30 s sim-time).
        self.maya_state = MayaState()
        self.maya_params = MayaParams()
        self.maya_last_posture_log: float = -1e9

        # SHESHNAG — offensive psyops layer. Authorisation gated through SwarmRaft.
        self.sheshnag_state = SheshnagState()
        self.sheshnag_params = SheshnagParams()
        # Per-hostile panic state lives ON the hostile dataclass; this is the
        # working vector used by sheshnag_tick each call.
        self.panic_vector: np.ndarray = np.zeros(0)
        # Console toggle: when ON, swarm proposes a BFT vote authorising
        # SHESHNAG broadcasts; without authorisation, only SIR decay runs.
        self.sheshnag_armed: bool = False
        self.sheshnag_bft_voted: bool = False

    # ── Public API ─────────────────────────────────────────────────────

    def step(self) -> TickResult:
        """Advance the swarm one tick. Returns events for the renderer."""
        result = TickResult()
        positions = self.swarm.positions
        velocities = self.swarm.velocities

        # ── Reflex ──
        if self.plan.scenario == "formation_v":
            per_goal = np.array(
                [b.goal_pos if b.goal_pos else self.goal_pos.tolist() for b in self.plan.bundles]
            )
            slot_pull = (per_goal - positions) * 1.4 - velocities * 0.6
            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=0.9, dt=0.1)
        elif self.plan.scenario == "coverage":
            per_goal = np.array(
                [b.goal_pos if b.goal_pos else self.goal_pos.tolist() for b in self.plan.bundles]
            )
            # ── Per-role ORBIT offset so drones VISIBLY fly patrol patterns ──
            sim_t = float(self.swarm.t)
            assigned = set(self.intercept_assignment.keys())
            for i, d in enumerate(self.swarm.drones):
                if i in assigned:
                    continue
                if d.battery < 0.20:
                    # RTB: pull toward base origin instead of orbiting
                    per_goal[i] = np.array([0.0, 0.0, 5.5])
                    continue
                per_goal[i] = per_goal[i] + _role_orbit_offset(sim_t, i, d.role)
            # If a drone is assigned to intercept a hostile, override its slot.
            if self.hostile_fleet is not None:
                hostile_by_id = {h.id: h for h in self.hostile_fleet.hostiles}
                for friendly_id, hostile_id in list(self.intercept_assignment.items()):
                    h = hostile_by_id.get(hostile_id)
                    if h is None or not h.alive:
                        self.intercept_assignment.pop(friendly_id, None)
                        self.current_task[friendly_id] = _default_task_for_role(
                            self.swarm.drones[friendly_id].role
                        )
                        continue
                    per_goal[friendly_id] = h.pos
            # Stronger pull for interceptors — they break formation aggressively.
            pull_gain = np.full(positions.shape[0], 1.2)
            for friendly_id in self.intercept_assignment.keys():
                if 0 <= friendly_id < pull_gain.size:
                    pull_gain[friendly_id] = 3.0
            slot_pull = (per_goal - positions) * pull_gain[:, None] - velocities * 0.5
            v_cmd = bvc_safe_velocity(positions, slot_pull, safety_radius=1.0, dt=0.1)
            # ── Battery drain + RTB intent on low battery ──
            for i, d in enumerate(self.swarm.drones):
                drain = 0.0006 + (0.0014 if i in assigned else 0.0)
                d.battery = max(0.0, d.battery - drain)
                if d.battery < 0.20 and not self.current_task.get(i, "").startswith("RTB"):
                    self.current_task[i] = "RTB · LOW BATTERY"
                    self.intents[i] = "rtb"
        elif self.plan.scenario == "sead_ingress" and self.chanakya_state is not None:
            # CHANAKYA: plan geodesics across the threat manifold on first call
            # (or whenever the defense field is dirty), then follow waypoints.
            if self.defense_field is not None and (
                not self.chanakya_state.plans or self.defense_field.consume_dirty()
            ):
                chanakya_plan_swarm(
                    swarm_positions=positions,
                    targets=self.chanakya_targets,
                    defense_field=self.defense_field.active,
                    state=self.chanakya_state,
                    params=self.chanakya_params,
                )
            v_cmd = np.zeros_like(positions)
            for i in range(positions.shape[0]):
                v_cmd[i] = chanakya_desired_velocity(
                    drone_idx=i,
                    drone_pos=positions[i],
                    state=self.chanakya_state,
                    params=self.chanakya_params,
                    cruise_speed=1.4,
                )
            v_cmd = bvc_safe_velocity(positions, v_cmd, safety_radius=1.0, dt=0.1)
        elif self.plan.scenario == "hover":
            v_cmd = (self.goal_pos - positions) * 1.0 - velocities * 0.8
        else:
            v_cmd = compose_reflex(positions, velocities, algorithm=self.plan.algorithm, goal_pos=self.goal_pos)

        self.sim.step(self.swarm, v_cmd)
        t = self.swarm.t

        # Update trails
        for d in self.swarm.drones:
            self.trails.push(d.id, float(d.pos[0]), float(d.pos[1]))

        # ── CHANAKYA scenario bookkeeping ─────────────────────────
        if self.defense_field is not None and self.chanakya_state is not None:
            hit_mask = self.defense_field.kill_radius_check(self.swarm.positions)
            new_kills = 0
            for i, hit in enumerate(hit_mask):
                d = self.swarm.drones[i]
                if hit and d.healthy:
                    d.healthy = False
                    new_kills += 1
            if new_kills:
                self.chanakya_kills += new_kills
                result.event_log_lines.append(
                    f"t={t:.2f}s  💀 CHANAKYA — {new_kills} drone(s) lost inside SAM radius"
                )
            # Arrivals = drones whose waypoint queue is exhausted.
            for i in range(self.swarm.n):
                plan = self.chanakya_state.plans.get(i)
                if plan is None:
                    continue
                k = self.chanakya_state.next_waypoint_idx.get(i, 0)
                if k >= plan.waypoints.shape[0]:
                    if not getattr(self.swarm.drones[i], "_chanakya_arrived", False):
                        self.swarm.drones[i]._chanakya_arrived = True
                        self.chanakya_arrivals += 1
                        result.event_log_lines.append(
                            f"t={t:.2f}s  ✅ CHANAKYA DRN-{self.swarm.drones[i].id:03d} reached target"
                        )

        # ── Hostile fleet ───────────────────────────────────────────
        if self.hostile_fleet is not None:
            tick = self.hostile_fleet.step(
                dt=self.sim.cfg.dt,
                friendly_positions=self.swarm.positions,
                center=self.swarm.positions.mean(axis=0),
            )

            # ── SHIELD trust-weighted Bayesian engagement assignment ──
            # Hijack injection (demo toggle): corrupt 1-2 friendly sensor streams
            # so the sheaf-Laplacian residual rises and PageRank trust collapses.
            if self.hijack_active and not self.spoofed_ids:
                self.spoofed_ids = set(
                    self._rng.sample(range(self.n_drones), k=min(2, self.n_drones))
                )
                result.event_log_lines.append(
                    f"t={t:.2f}s  ⚠  HIJACK INJECT — friendlies {sorted(self.spoofed_ids)} sensor-spoofed"
                )
            elif not self.hijack_active and self.spoofed_ids:
                self.spoofed_ids = set()

            friendly_roles = [d.role.value for d in self.swarm.drones]
            adj = self.comm_adjacency()
            # ── SHIELD computes the trust-weighted Bayesian priority matrix ──
            priority_E, friendly_ids, hostile_ids_active = shield_priorities(
                friendly_positions=self.swarm.positions,
                friendly_roles=friendly_roles,
                hostiles=self.hostile_fleet.hostiles,
                adjacency=adj,
                state=self.shield_state,
                params=self.shield_params,
                spoofed_ids=self.spoofed_ids if self.spoofed_ids else None,
            )
            # ── VAJRA resolves with Voronoi hysteresis + tropical attention ──
            hostile_pos_active = np.array(
                [
                    next(h.pos for h in self.hostile_fleet.hostiles if h.id == hid)
                    for hid in hostile_ids_active
                ]
            ) if hostile_ids_active else np.zeros((0, 3))
            # Jamming flag in the console flows directly into HSL-CC concurrency cap.
            self.vajra_params.jamming_factor = 0.5 if self.jamming else 0.0
            shield_new = vajra_assign(
                priority_matrix=priority_E,
                friendly_positions=self.swarm.positions,
                friendly_ids=friendly_ids,
                hostile_positions=hostile_pos_active,
                hostile_ids=hostile_ids_active,
                adjacency=adj,
                state=self.vajra_state,
                params=self.vajra_params,
                already_assigned=self.intercept_assignment,
            )

            # Fragmentation alarm — only emit on edge.
            if (
                self.vajra_state.n_components > 1
                and not self.fragmentation_alarmed
            ):
                self.fragmentation_alarmed = True
                result.event_log_lines.append(
                    f"t={t:.2f}s  ⚠  VAJRA FRAGMENTATION — comm graph split into "
                    f"{self.vajra_state.n_components} components (λ₂={self.vajra_state.lambda2:.3f})"
                )
            elif (
                self.vajra_state.n_components == 1
                and self.fragmentation_alarmed
            ):
                self.fragmentation_alarmed = False
                result.event_log_lines.append(
                    f"t={t:.2f}s  ✓  VAJRA comm graph reconnected (λ₂={self.vajra_state.lambda2:.3f})"
                )

            # Surface newly kill-switched drones (trust < threshold).
            killed_now = {
                i for i, T in enumerate(self.shield_state.trust)
                if T < self.shield_params.trust_kill_threshold
            }
            newly_killed = killed_now - self.shield_kill_switched
            for kid in sorted(newly_killed):
                result.event_log_lines.append(
                    f"t={t:.2f}s  🛑 SHIELD KILL-SWITCH DRN-{self.swarm.drones[kid].id:03d} "
                    f"(trust={self.shield_state.trust[kid]:.2f})"
                )
            self.shield_kill_switched = killed_now

            # Apply new assignments + log a decoy-skip when SHIELD downgrades a target.
            hostile_by_id = {h.id: h for h in self.hostile_fleet.hostiles}
            for fid, hid in shield_new.items():
                h = hostile_by_id.get(hid)
                if h is None:
                    continue
                post = self.shield_state.posteriors.get(hid)
                cls = threat_class(post) if post is not None else "?"
                self.intercept_assignment[fid] = hid
                h.assigned_to = fid
                self.current_task[fid] = f"INTERCEPT {h.callsign} [{cls}]"
                result.event_log_lines.append(
                    f"t={t:.2f}s  🎯 SHIELD DRN-{self.swarm.drones[fid].id:03d} → "
                    f"{h.callsign} class={cls} "
                    f"E[D]={expected_damage(post, self.shield_params):.2f} "
                    f"trust={self.shield_state.trust[fid]:.2f}"
                )

            # Count decoys SHIELD chose NOT to engage despite being TERMINAL.
            for h in self.hostile_fleet.hostiles:
                if not h.alive or h.intent_label != "TERMINAL" or h.assigned_to is not None:
                    continue
                post = self.shield_state.posteriors.get(h.id)
                if post is not None and threat_class(post) == "decoy":
                    self.shield_decoy_skipped += 1

            # ── MAYA strategic posture refresh (every ~30 s) ──
            posteriors_alive = [
                self.shield_state.posteriors[h.id]
                for h in self.hostile_fleet.hostiles
                if h.alive and h.id in self.shield_state.posteriors
            ]
            did_solve, sol = maya_tick(
                sim_time=float(t),
                state=self.maya_state,
                hostile_posteriors=posteriors_alive,
                params=self.maya_params,
            )
            if did_solve and sol is not None:
                pd = posture_dict(sol.posture)
                top = max(pd, key=pd.get)
                result.event_log_lines.append(
                    f"t={t:.2f}s  🧠 MAYA posture: {top.upper()} "
                    f"({pd[top]*100:.0f}%) — value={sol.value:.2f} "
                    f"μ̂_H={dict(zip(['dec','kin','nui'], np.round(self.maya_state.hostile_posterior, 2)))}"
                )
                # Modulate downstream SHIELD/VAJRA params per posture.
                # INTERCEPT-heavy → sharper tropical attention + lower kill-switch
                # DEFEND-heavy   → wider Voronoi margin (commit to your cell)
                # RETREAT-heavy  → looser tropical to spread thin
                self.vajra_params.tropical_beta = 4.0 + 12.0 * pd["intercept"]
                self.vajra_params.voronoi_margin = 1.0 + 2.0 * pd["defend"]
                self.shield_params.trust_kill_threshold = max(
                    0.15, 0.25 - 0.10 * pd["intercept"]
                )

            # ── SHESHNAG offensive psyops tick ──
            # Init panic vector when fleet first becomes non-empty.
            n_h = len(self.hostile_fleet.hostiles)
            if self.panic_vector.shape[0] != n_h:
                self.panic_vector = np.array(
                    [h.panic_level for h in self.hostile_fleet.hostiles]
                )
            # If user armed SHESHNAG and we haven't taken the BFT vote yet, do it.
            if self.sheshnag_armed and not self.sheshnag_bft_voted:
                self.sheshnag_bft_voted = True
                passed, votes = self.raft.propose("authorize_psyops:sheshnag_broadcast")
                for v in votes:
                    m = WireMessage.make(
                        src=v.voter_id,
                        protocol=Protocol.BFT,
                        topic="bft/swarm-raft/vote",
                        payload=v,
                        t=t,
                    )
                    result.new_messages.append(m)
                self.bft_flash = {v.voter_id for v in votes}
                self.bft_flash_ttl = 14
                self.bft_count += 1
                self.bft_history.append({
                    "t": float(t),
                    "proposal": "authorize_psyops:sheshnag_broadcast",
                    "passed": bool(passed),
                    "yes": sum(1 for v in votes if v.decision == "yes"),
                    "no": sum(1 for v in votes if v.decision == "no"),
                    "voters": [int(v.voter_id) for v in votes],
                    "byzantine": [],
                })
                if passed:
                    self.sheshnag_state.authorized = True
                    result.event_log_lines.append(
                        f"t={t:.2f}s  🐍 SHESHNAG AUTHORIZED — psyops broadcasts ENABLED"
                    )
                else:
                    result.event_log_lines.append(
                        f"t={t:.2f}s  🛑 SHESHNAG denied — BFT vote failed"
                    )
            elif not self.sheshnag_armed and self.sheshnag_state.authorized:
                # Disarm
                self.sheshnag_state.authorized = False
                self.sheshnag_bft_voted = False

            alive_hostiles = [h for h in self.hostile_fleet.hostiles if h.alive]
            if alive_hostiles:
                pos = np.array([h.pos for h in alive_hostiles])
                vel = np.array([h.vel for h in alive_hostiles])
                panic_in = np.array([h.panic_level for h in alive_hostiles])
                panic_out = sheshnag_tick(
                    hostile_positions=pos,
                    hostile_velocities=vel,
                    panic=panic_in,
                    dt=self.sim.cfg.dt,
                    state=self.sheshnag_state,
                    params=self.sheshnag_params,
                )
                for h, p in zip(alive_hostiles, panic_out):
                    h.panic_level = float(p)
                # Surface phase transition when SHESHNAG tips the enemy.
                phase = self.sheshnag_state.last_phase["phase"]
                if phase == "MILLING" and self.sheshnag_state.fraction_panicked > 0.4:
                    if not getattr(self, "_milling_alarmed", False):
                        self._milling_alarmed = True
                        result.event_log_lines.append(
                            f"t={t:.2f}s  🌀 SHESHNAG ENEMY MILLING — "
                            f"P={self.sheshnag_state.last_phase['P']:.2f} "
                            f"R={self.sheshnag_state.last_phase['R']:.2f} "
                            f"panic_mean={self.sheshnag_state.mean_panic:.2f}"
                        )
                elif phase != "MILLING":
                    self._milling_alarmed = False

            # ── Record kill events ──
            for kill in tick["kills_this_step"]:
                killer_idx = kill["killer_idx"]
                killer = self.swarm.drones[killer_idx]
                self.kill_events.append(
                    {
                        "t": float(t),
                        "killer_id": int(killer.id),
                        "callsign": kill["callsign"],
                        "pos": kill["pos"],
                    }
                )
                # release intercept
                self.intercept_assignment.pop(killer.id, None)
                self.current_task[killer.id] = _default_task_for_role(killer.role)
            # Kills → BFT engage authorization (first contact only)
            if tick["new_contacts"] > 0 and not self._hostile_first_contact_fired:
                self._hostile_first_contact_fired = True
                passed, votes = self.raft.propose("authorize_engage:hostile_contact")
                for v in votes:
                    m = WireMessage.make(
                        src=v.voter_id,
                        protocol=Protocol.BFT,
                        topic="bft/swarm-raft/vote",
                        payload=v,
                        t=t,
                    )
                    result.new_messages.append(m)
                self.bft_flash = {v.voter_id for v in votes}
                self.bft_flash_ttl = 12
                self.bft_count += 1
                self.bft_history.append(
                    {
                        "t": float(t),
                        "proposal": "authorize_engage:hostile_contact",
                        "passed": bool(passed),
                        "yes": sum(1 for v in votes if v.decision == "yes"),
                        "no": sum(1 for v in votes if v.decision == "no"),
                        "voters": [int(v.voter_id) for v in votes],
                        "byzantine": [],
                    }
                )
                result.event_log_lines.append(
                    f"t={t:.2f}s  ⚔  first hostile contact — engage authorized"
                )
            for kill in tick["kills_this_step"]:
                result.event_log_lines.append(
                    f"t={t:.2f}s  💥 DRN-{self.swarm.drones[kill['killer_idx']].id:03d} → {kill['callsign']} NEUTRALIZED"
                )

        # ── LLM intent refresh ──
        if self.step_i % self.intent_refresh_every == 0:
            for d in self.swarm.drones:
                user = f"id={d.id} pos={d.pos.tolist()} battery={d.battery:.2f}"
                raw = self.backend.complete("terse", user, max_tokens=64)
                try:
                    import json
                    self.intents[d.id] = json.loads(raw).get("intent", "hold_formation")
                except Exception:
                    self.intents[d.id] = "hold_formation"

                m = WireMessage.make(
                    src=d.id,
                    protocol=Protocol.GRPC,
                    topic=f"grpc/cognition/{d.id}/intent",
                    payload=IntentPayload(intent=self.intents[d.id], rationale="mock"),
                    t=t,
                )
                result.new_messages.append(m)

        # ── Peer dialogue (A2A + Zenoh + MAVLink) ──
        wire = self.peer.emit_round(t, self.intents)
        result.new_messages.extend(wire)

        # ── BFT vote at step 30 and 100 ──
        if self.step_i == 30:
            passed, votes = self.raft.propose("advance_phase:engage")
            for v in votes:
                m = WireMessage.make(
                    src=v.voter_id,
                    protocol=Protocol.BFT,
                    topic="bft/swarm-raft/vote",
                    payload=v,
                    t=t,
                )
                result.new_messages.append(m)
            self.bft_flash = {v.voter_id for v in votes}
            self.bft_flash_ttl = 12
            self.bft_count += 1
            tag = f"BFT PASSED · advance_phase:engage  ({sum(1 for v in votes if v.decision=='yes')}/7)"
            result.floating_events.append(FloatingEvent(text=tag, x=0.0, y=24.0, color=(252, 211, 77)))
            result.event_log_lines.append(f"t={t:.2f}s  ⚖  BFT vote PASSED — advance_phase:engage")
            self.bft_history.append(
                {
                    "t": float(t),
                    "proposal": "advance_phase:engage",
                    "passed": bool(passed),
                    "yes": sum(1 for v in votes if v.decision == "yes"),
                    "no": sum(1 for v in votes if v.decision == "no"),
                    "voters": [int(v.voter_id) for v in votes],
                    "byzantine": [],
                }
            )
        if self.step_i == 100 and self.scenario != "hover":
            byz = set(self._rng.sample(range(self.n_drones), k=min(2, self.n_drones)))
            passed, votes = self.raft.propose("re-plan:wind-gust", byzantine_ids=byz)
            for v in votes:
                m = WireMessage.make(
                    src=v.voter_id,
                    protocol=Protocol.BFT,
                    topic="bft/swarm-raft/vote",
                    payload=v,
                    t=t,
                )
                result.new_messages.append(m)
            self.bft_flash = {v.voter_id for v in votes}
            self.bft_flash_ttl = 14
            self.bft_count += 1
            outcome = "PASSED" if passed else "FAILED"
            tag = f"BFT {outcome} · re-plan w/ {len(byz)} spoofed drones"
            color = (252, 211, 77) if passed else (248, 113, 113)
            result.floating_events.append(FloatingEvent(text=tag, x=0.0, y=-24.0, color=color))
            result.event_log_lines.append(
                f"t={t:.2f}s  ⚖  BFT vote {outcome} — re-plan (byzantine={sorted(byz)})"
            )
            self.bft_history.append(
                {
                    "t": float(t),
                    "proposal": "re-plan:wind-gust",
                    "passed": bool(passed),
                    "yes": sum(1 for v in votes if v.decision == "yes"),
                    "no": sum(1 for v in votes if v.decision == "no"),
                    "voters": [int(v.voter_id) for v in votes],
                    "byzantine": [int(b) for b in byz],
                }
            )

        # ── ED-CBBA bidding (coverage scenario, every 25 steps) ──
        if self.cbba_tasks and self.step_i % 25 == 0:
            self.cbba.drones_pos = self.swarm.positions
            bids = self.cbba.round()
            for bid in bids:
                m = WireMessage.make(
                    src=bid.bidder_id,
                    protocol=Protocol.ZENOH,
                    topic="cbba/bid",
                    payload=bid,
                    t=t,
                )
                result.new_messages.append(m)
            if bids:
                self.cbba_count += 1
                for bid in bids:
                    self.cbba_history.append(
                        {
                            "t": float(t),
                            "task_id": str(bid.task_id),
                            "bidder_id": int(bid.bidder_id),
                            "bid_score": float(bid.bid_score),
                        }
                    )
                self.cbba_flash = {}
                for bid in bids[:6]:
                    task_pos = next((t.pos for t in self.cbba_tasks if t.id == bid.task_id), None)
                    if task_pos:
                        self.cbba_flash[bid.task_id] = (task_pos[0], task_pos[1])
                self.cbba_flash_ttl = 10
                result.cbba_flash_cells = dict(self.cbba_flash)
                result.event_log_lines.append(
                    f"t={t:.2f}s  📋 CBBA · {len(bids)} bids · drone {bids[0].bidder_id} claims {bids[0].task_id}"
                )

        # Decay BFT / CBBA flashes
        if self.bft_flash_ttl > 0:
            result.bft_flash_voters = set(self.bft_flash)
            self.bft_flash_ttl -= 1
            if self.bft_flash_ttl == 0:
                self.bft_flash = set()
        if self.cbba_flash_ttl > 0:
            result.cbba_flash_cells = dict(self.cbba_flash)
            self.cbba_flash_ttl -= 1
            if self.cbba_flash_ttl == 0:
                self.cbba_flash = {}

        # ── Bookkeeping ──
        for m in result.new_messages:
            self.message_log.append(m)
            self.bandwidth.record(t, m.protocol.value, m.bytes_size)

        # Record RecentMessage for renderer
        for m in result.new_messages:
            src_xy = (float(self.swarm.drones[m.src].pos[0]), float(self.swarm.drones[m.src].pos[1]))
            if m.dst is not None and 0 <= m.dst < self.n_drones:
                dst_xy = (float(self.swarm.drones[m.dst].pos[0]), float(self.swarm.drones[m.dst].pos[1]))
            else:
                dst_xy = None
            self.recent_msgs.append(
                RecentMessage(
                    src_id=m.src,
                    dst_id=m.dst,
                    protocol=m.protocol,
                    age=0,
                    src_xy=src_xy,
                    dst_xy=dst_xy,
                )
            )
            if m.protocol == Protocol.A2A and m.dst is not None:
                # Surface negotiate.yield as a floating event
                payload = m.payload
                if isinstance(payload, dict) and payload.get("method") == "negotiate.yield":
                    self.event_log.append(
                        f"t={t:.2f}s  ↻ drone {m.src} → drone {m.dst}: negotiate.yield"
                    )

        # Age recent messages
        for rm in list(self.recent_msgs):
            rm.age += 1
        # Age floating events
        new_floats = []
        for fe in list(self.floating):
            fe.age += 1
            if fe.age < 8:
                new_floats.append(fe)
        self.floating = deque(new_floats, maxlen=40)
        for fe in result.floating_events:
            self.floating.append(fe)

        # Append event log lines
        for line in result.event_log_lines:
            self.event_log.append(line)

        self.step_i += 1
        return result

    # ── Snapshot accessors for renderer ────────────────────────────────

    def comm_adjacency(self) -> np.ndarray:
        return self.comm.adjacency(self.swarm.positions)

    def render_stats(self) -> dict:
        now = self.swarm.t
        rates = self.bandwidth.rates_by_protocol(now)
        total_msgs = len(self.message_log)
        msgs_per_s = sum(r["msgs_per_s"] for r in rates.values())
        by_proto = {p: int(r["total_msgs"]) for p, r in rates.items()}
        # SHIELD aggregate telemetry
        loyalty = self.shield_state.loyalty
        trust = self.shield_state.trust
        shield = {
            "loyalty_min": float(loyalty.min()) if loyalty.size else 1.0,
            "loyalty_mean": float(loyalty.mean()) if loyalty.size else 1.0,
            "trust_min": float(trust.min()) if trust.size else 1.0,
            "trust_mean": float(trust.mean()) if trust.size else 1.0,
            "kill_switched": sorted(self.shield_kill_switched),
            "spoofed_ids": sorted(self.spoofed_ids),
            "decoys_skipped": int(self.shield_decoy_skipped),
        }
        # VAJRA aggregate telemetry
        vajra = {
            "lambda2": float(self.vajra_state.lambda2),
            "n_components": int(self.vajra_state.n_components),
            "fragmented": bool(self.fragmentation_alarmed),
            "jamming_factor": float(self.vajra_params.jamming_factor),
            "voronoi_owners": dict(self.vajra_state.voronoi.owner),
            "handover_events": list(self.vajra_state.handover_events[-10:]),
        }
        # MAYA strategic posture
        maya = {
            "posture": posture_dict(self.maya_state.posture),
            "hostile_estimate": self.maya_state.hostile_posterior.tolist(),
            "hostile_worst_case": self.maya_state.worst_case_hostile.tolist(),
            "classifier_entropy": float(self.maya_state.classifier_entropy),
            "value": float(self.maya_state.last_value),
            "n_solves": int(self.maya_state.n_solves),
        }
        # CHANAKYA SEAD-ingress telemetry
        chanakya = {
            "enabled": self.chanakya_state is not None,
            "kills": int(self.chanakya_kills),
            "arrivals": int(self.chanakya_arrivals),
        }
        if self.chanakya_state is not None:
            chanakya.update(chanakya_summary(self.chanakya_state))
            chanakya["n_active_assets"] = len(self.defense_field.active) if self.defense_field else 0
        # SHESHNAG psyops telemetry
        sheshnag = {
            "authorized": bool(self.sheshnag_state.authorized),
            "phase": self.sheshnag_state.last_phase.get("phase", "SWARM"),
            "polarization": float(self.sheshnag_state.last_phase.get("P", 0.0)),
            "rotation": float(self.sheshnag_state.last_phase.get("R", 0.0)),
            "mean_panic": float(self.sheshnag_state.mean_panic),
            "fraction_panicked": float(self.sheshnag_state.fraction_panicked),
            "broadcasts_emitted": int(self.sheshnag_state.broadcasts_emitted),
            "composite_value": float(self.sheshnag_state.composite_value),
        }
        return {
            "t": now,
            "total_msgs": total_msgs,
            "msgs_per_s": msgs_per_s,
            "bft_count": self.bft_count,
            "cbba_count": self.cbba_count,
            "by_proto": by_proto,
            "shield": shield,
            "vajra": vajra,
            "maya": maya,
            "sheshnag": sheshnag,
            "chanakya": chanakya,
        }

    def floating_for_render(self) -> list[FloatingEvent]:
        return list(self.floating)

    def recent_for_render(self) -> list[RecentMessage]:
        return list(self.recent_msgs)
