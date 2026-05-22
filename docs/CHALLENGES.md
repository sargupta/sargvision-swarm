# Real-world drone-swarm comm challenges

Drawn from field deployments + academic literature 2024-2026. Each challenge cites a source and the design choice this repo makes to mitigate it.

## 1. Latency > 120 ms in complex environments

UAV swarm field tests in cluttered terrain consistently see end-to-end latency exceeding 120 ms. That alone is enough to break inner-loop collision avoidance if those loops route through the cloud.

**Mitigation here:**
- Collision avoidance (BVC + GCBF+) runs **on the drone**, never on the ground.
- LLM intent runs at slow cadence (1-5 Hz); reflex runs at 20-200 Hz.
- Brooks-subsumption discipline.

## 2. Bandwidth saturation

50 drones streaming 1080p video ≈ **25 Gbps aggregate** — infeasible without dedicated mesh radio + 5G slice.

**Mitigation:**
- Drones stream **2-10 Mbps of semantic + key-frame** uplink, never raw HD.
- Pose broadcasts at 10 Hz × ~32 B/msg → ~320 B/s per drone → ~32 kB/s for 100 drones (trivial).
- ED-CBBA cuts task-bidding chatter ~**52%** vs vanilla CBBA.
- Bandwidth panel in the Wire-log tab shows live bytes/sec per protocol.

## 3. Multicast storms with DDS on Wi-Fi

ROS 2 default Fast/Cyclone DDS uses multicast discovery. Collapses at ~20 nodes on shared Wi-Fi.

**Mitigation:** **Zenoh + rmw_zenoh** uses gossip discovery — scales to 100+ on lossy mesh. MDPI 2025 50-UAV paper validated. PX4 v1.17 ships Zenoh in-tree.

## 4. Authentication + key handover

A swarm without per-drone identity is one captured drone away from total compromise.

**Planned for Phase 2:**
- Per-drone X.509 cert from CoE PKI.
- MAVLink v2 signed messages (already protocol-supported).
- A2A Agent Cards carry cryptographic identity.

Reference: [arXiv 2201.05657](https://arxiv.org/pdf/2201.05657) — authentication and handover for drone swarms.

## 5. GPS-denied — 2-5 cm/min positioning drift

In urban canyon / indoor / jamming scenarios, accumulated visual-inertial odometry drift compounds. Coordination becomes unreliable after ~5 minutes.

**Mitigation:**
- **Swarm-SLAM** (MIT, MIT-license) for collaborative loop closure across the squad.
- **UWB** (NoopLoop / Decawave) for relative drone-to-drone 6DOF.
- Vásárhelyi 2018 tuning works on **relative** spacing, not absolute coords.

## 6. Byzantine fault / GNSS spoofing

A spoofed drone votes wrong. Without BFT, single bad actor corrupts the swarm.

**Mitigation:** **SwarmRaft K=7** committee, **⅔ quorum** on irreversible decisions (engage / RTL / abort). Tolerates 3 Byzantine drones in a 7-member committee. The demo runs a spoof test at step 100 (Coverage scenario).

## 7. India regulatory traps

- **DGFT 2022 ban:** drones in CBU/CKD/SKD form **prohibited** to import. Components free. **Must build in India.**
- **NPNT mandatory** for drones sold post Jan 2024.
- **WPC bands:** 2.4 GHz / 5.8 GHz delicensed + **865-867 MHz LoRa only**.
- **NOT 868 MHz** (licensed). **NOT 915 MHz** (licensed).
- **WPC ETA per radio SKU** mandatory before deployment.
- **eGCA** owns registration / type-cert; **DigitalSky** owns NPNT.
- **BVLOS corridors:** Ladakh, Telangana, Andhra Pradesh.

## 8. Sim2real cliff at scale

End-to-end MARL policies trained in Isaac Sim fail outdoors at scale. **No published 100-drone outdoor end-to-end neural deployment exists in 2026.** AttentionSwarm (Mar 2025) is SOTA indoor (~50 Crazyflies w/ Vicon).

**Mitigation:** Hybrid stack:
- Classical reflex layer (Boids / Olfati-Saber / BVC / ED-CBBA / SwarmRaft) — deterministic.
- RL **sub-task** policies — narrow, well-defined.
- LLM cognition — high-level intent.
- Outdoor frontier in 2026: ~10 drones with learned coordination overlaid on classical PX4.

## 9. World records vs autonomy

The 22,580-drone EHang light show (Feb 2026) is **pre-programmed waypoint playback**, not autonomy. True autonomous frontier is 200-250 drones (PLA Atlas, DARPA OFFSET target). Distinguish ambition carefully when target-setting.

## 10. Cost discipline

| Tier | Item | Indicative cost |
|---|---|---|
| Indoor lab | 50 Crazyflie 2.1+ | ~₹21 L |
| Outdoor pilot | 10 Pixhawk 6X + Jetson Orin + Doodle Labs | ~₹40-50 L |
| 100-drone outdoor | full BoM | ~₹3-4 crore |
| Light show fleet | 100 ArduCopter + RTK + Skybrush Pro | ~₹60-75 L |

## Sources

- [Authentication and handover for drone swarms — arXiv 2201.05657](https://arxiv.org/pdf/2201.05657)
- [Drone swarm coordination survey 2024-2025 — Meegle](https://www.meegle.com/en_us/topics/autonomous-drones/drone-swarm-coordination)
- [PRC concepts for UAV swarms in future warfare — CNA Oct 2025](https://www.cna.org/reports/2025/07/PRC-Concepts-for-UAV-Swarms-in-Future-Warfare.pdf)
- [Mission-critical UAV swarm coordination — integrated ROS+LoRa, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0140366424002494)
- [Towards optimal guidance of autonomous swarm drones in dynamic constrained environments — Alqudsi 2025](https://onlinelibrary.wiley.com/doi/10.1111/exsy.70067)
- Master synthesis: `~/Documents/AI_Workspace/drone_swarm_research/00_MASTER.md`.
