# ARCHITECTURE_2046 — 20-Year Architectural Hooks

**Purpose:** document the architectural decisions that bridge SARGVISION's 2026
deployment target with its 2046 defensibility target.

Every architectural decision in `sargvision_swarm.cfm.*` must pass two tests:

1. *Does this make us sellable in 2026-2028?* (current procurement window)
2. *Does this make us non-obsolete in 2046?* (20-year horizon)

If a decision only passes one test, defer it. If neither, kill it.

---

## 1. Eight Architectural Hooks

### Hook 1 — Hardware-agnostic coordination
Coordination layer operates on abstract drone entities, not platform-specific
firmware. Currently sits above PX4 + Ardupilot. By 2046 will sit above
hypersonic UCAV, bio-inspired MAV, space-mesh-relayed CCA.

**Realised:** `sim.agent.Agent` is platform-agnostic. Goal positions + velocity
commands are the only required output interface.
**Owed:** MAVLink / uXRCE-DDS adapter for real-hardware integration.

### Hook 2 — Sensor abstraction layer
TWSL consumes `SensorReport` instances. Any modality (radar, RF, EO, acoustic,
hyperspectral, future-quantum) can produce reports. Adding a new modality is
adding an enum value, not rewriting fusion.

**Realised:** `cfm.sensors.report.Modality` enum + `SensorReport` dataclass.
**Owed:** quantum sensor mock when 2032+ deployment looms.

### Hook 3 — PNT-source-agnostic
Position estimates flow through a generic interface — GPS, NavIC, Galileo,
visual-inertial odometry, magnetic anomaly, terrain matching, and (by ~2040)
quantum-INS all expose the same `PNTSolution` shape.

**Realised:** `cfm.sovereign.navic.PNTSolution` dataclass. NavIC implementation
serves as reference interface — alternates plug into the same contract.
**Owed:** quantum-INS placeholder (`cfm.pnt.quantum_ins`) — defer until 2030.

### Hook 4 — Sovereign-first, allies-interoperable
Indian sovereign integration is built first (`cfm.sovereign.*`). Coalition
interop (Quad / INDUS-X / AUKUS-Pillar-2-equivalent) will plug into the same
publish/subscribe contracts.

**Realised:** `cfm.sovereign.iaccs.IACCSMessageBus` shows the pub/sub schema.
A future `cfm.coalition.indus_x` would adopt the same pattern.
**Owed:** coalition adapter scaffolding when first international interest
materialises (target: 2029).

### Hook 5 — AI-augmented, human-supervised
Classical core (TWSL, ED-CBBA) runs without ML. A wrapper layer integrates
MARL policies, foundation-model outputs, and learned heuristics — but ALWAYS
gated by a human-on-the-loop authorisation interface for kinetic actions.

**Realised:** classical core in `core.twsl`, `core.ccg`. Wrappers TBD.
**Owed:** `cfm.autonomy.fm_wrapper` placeholder — VLA-on-edge integration when
Jetson Thor (2027+) brings on-device 8B+ models into reach.

### Hook 6 — Adversarial-AI-resilient Byzantine fusion
TWSL trust operates correctly when adversary is a human-operated drone today
(2026). Architecture must extend cleanly to when adversary is an AI-piloted
swarm (2030+). Attestation flags + cross-modal consistency + hardware-rooted
identity are the bridging primitives.

**Realised:** `cfm.sensors.attestation.AttestationFlags` (TPM + IMU + RF
tamper + secure boot). Cross-modal consistency in `cfm.sensors.fusion`.
**Owed:** PUF-rooted hardware identity (`cfm.sensors.puf`) when first hardware
kit ships.

### Hook 7 — DEW-aware coordination
Iron Beam (2025+), HELIOS (2024+), Anduril Pulsar HPM (2025+) make swarm
engagement against DEW-equipped targets expensive. SARGVISION coordination
must route around laser zones, employ persistent-low-detect approach, and
mass-saturate DEW magazine + cooldown.

**Realised:** `cfm.strategies.c_uas.recommend_strategy` already adapts strategy
to threat composition.
**Owed:** `cfm.engagement.dew_aware_planner` — flag laser-coverage zones in
the threat field, route around. Defer until first DEW-equipped Indian threat
materialises (2028-2030).

### Hook 8 — Open-spec input/output contracts
Schemas published as Pydantic / msgpack / protobuf contracts. When defence
autonomy commoditises (2030-2038), SARGVISION becomes the schema-of-record
for Indian-sovereign coordination, not the implementation-of-record.

**Realised:** `cfm.sovereign.*` dataclasses are the integration contract. They
will be promoted to a standalone `sargvision-spec` repo once stable.
**Owed:** OpenAPI spec for the HTTP layer; protobuf for the binary wire
protocol — defer until Phase 2.

---

## 2. Modules Yet to Build (with Defer-Until-When Markers)

| Module path | Purpose | Defer until |
|---|---|---|
| `cfm.pnt.quantum_ins` | Cold-atom INS adapter | 2030 (first lab-grade hardware) |
| `cfm.autonomy.fm_wrapper` | Vision-Language-Action model integration | 2027 (Jetson Thor + Llama-3-70B-class on edge) |
| `cfm.coalition.indus_x` | US INDUS-X interop adapter | 2029 (first US interest) |
| `cfm.coalition.quad` | Quad coalition mesh adapter | 2030 |
| `cfm.engagement.dew_aware_planner` | Route-around-laser-zone planner | 2028 (first DEW in Indian theatre) |
| `cfm.sensors.puf` | Physically-Unclonable-Function hardware identity | 2026 (first hardware kit) |
| `cfm.sensors.quantum_magnetometer` | Quantum magnetic anomaly nav | 2033 |
| `cfm.cognitive_ew.adaptive_jammer_detector` | Cognitive EW sensor input | 2028 |
| `cfm.manufacturing.dynamic_fleet` | Real-time fleet growth (3D-printed expendables) | 2030 |

Each placeholder must have a documented input/output contract before the
implementation lands. The contracts themselves are the architectural
investment; the implementations follow once the technology arrives.

---

## 3. Decisions Already Made (Locked In)

These cannot be changed without architectural review:

- **Trust scores live in (0, 1] real-valued.** Boolean trust (Byzantine /
  non-Byzantine) was rejected; continuous trust is necessary for sensor-noise
  modelling and survives the move to AI-vs-AI adversary scenarios.

- **Coordination layer is software-only.** SARGVISION will NEVER manufacture
  radar, drones, or DEW hardware. Every customer relationship runs through
  integration with their existing hardware.

- **Math operators are sparse/PSD by construction.** TWSL Laplacian is a
  weighted graph Laplacian — sparse, PSD, structure-preserving. Departures
  from this require an architectural-review note.

- **No hardcoded coordinates.** All scenarios use generic ENU local-tangent
  frames. Real-world geographic coordinates are loaded at runtime via
  Bhuvan / NavIC adapters, never embedded in code.

- **No real operation names in committed code.** Operational doctrine,
  scenario tuning parameters, and adversary-specific knowledge live in
  air-gapped storage (per file 47 patent verdict + file 70 steelman).

---

## 4. Forces We Are Betting Against

Architectural choices we are explicitly making BECAUSE we think the consensus
view is wrong:

- **Consensus says: foundation models will replace classical control.**
  We bet: classical control + trust layer + FM wrapper survives, because
  Byzantine resilience matters more, not less, when individual agents are
  smarter.

- **Consensus says: defence autonomy stays closed-source.** We bet:
  defence autonomy commoditises 2030-2038 like Linux did. SARGVISION's IP
  moat shifts from algorithm to integration + relationships + trade-secret
  operational tuning.

- **Consensus says: India remains hardware buyer not exporter.** We bet:
  Atmanirbhar + Quad open a 2030-2035 export window. CFM is positioned for
  Quad-interop from day 1, not bolted on later.

- **Consensus says: DEW obsoletes swarms.** We bet: DEW makes COORDINATION
  more valuable (route around lasers, persistent low-detect, decoy-mass).
  Coordination layer is the survivor, not the casualty.

---

## 5. Review Cadence

Architectural review every 12 months — re-evaluate each hook against:

- What has the past 12 months of operational data + research shipped?
- Are the defer-until markers still calibrated correctly?
- Are the bets-against still supportable?
- Is the open-spec promotion plan on schedule?

Next review: 2027-Q2.
