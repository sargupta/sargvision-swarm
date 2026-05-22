# Protocols carried by the swarm bus

The sandbox simulates **five protocols** layered across a swarm — each with its own envelope, byte size, and traffic pattern.

| Layer | Protocol | Carries | Wire format | Where it lives |
|---|---|---|---|---|
| Cognition (LLM ↔ LLM) | **gRPC** | Per-drone intent → ground orchestrator | protobuf over HTTP/2 | `Protocol.GRPC` |
| Agent ↔ agent | **A2A** (Agent2Agent) | Capability discovery, intent share, yield negotiation | **JSON-RPC 2.0 over HTTP + SSE** | `Protocol.A2A` |
| Agent ↔ tool | **MCP** (Model Context Protocol) | Per-agent tool access (camera, IMU, GPS) | JSON-RPC over stdio / HTTP | `Protocol.MCP` |
| Robotics middleware | **Zenoh** / DDS | Pose, intent gossip, ED-CBBA bids | binary, gossip-based | `Protocol.ZENOH`, `Protocol.DDS` |
| Autopilot bus | **MAVLink v2** (signed) | Heartbeat, position telem, command | binary, signed | `Protocol.MAVLINK` |
| Mission state | **BFT** (SwarmRaft) | K=7 committee votes | binary | `Protocol.BFT` |

## A2A spec recap (the one you asked about)

- **Origin:** Google, announced **April 2025**, open-sourced under **Apache-2.0**, governed by **Linux Foundation**.
- **Transport:** HTTPS + SSE (Server-Sent Events for streaming).
- **Envelope:** **JSON-RPC 2.0**.
- **Discovery:** "Agent Cards" — JSON documents advertising agent capabilities + transports + auth requirements.
- **Patterns:** synchronous request/response · streaming over SSE · async push notifications.
- **Complements MCP:** A2A is agent-to-**agent**; MCP is agent-to-**tool**.

### Methods we implement (in `agents/peer_dialogue.py`)

| Method | When | Params |
|---|---|---|
| `share.intent` | Every cycle to 1-2 neighbors | `{from, intent, yaw}` |
| `negotiate.yield` | Distance < 2 m | `{requester, reason}` |
| `claim.task` | ED-CBBA bid win | `{task_id, bid_score}` |
| `share.health` | Slow gossip | `{battery, healthy, role}` |

### A2A Agent Card example

```json
{
  "agent_id": 7,
  "name": "sargvision-drone-007",
  "capabilities": ["share.intent", "negotiate.yield", "claim.task", "share.health"],
  "transports": ["HTTP+SSE", "Zenoh"]
}
```

## Why position-only broadcast?

We publish positions on `swarm/<id>/pose` at 10 Hz. We deliberately do **not** broadcast velocity:

1. Velocity leaks intent — adversaries learn maneuvers.
2. Doubles the byte count.
3. **BVC** (Buffered Voronoi Cells) needs only positions to compute collision-safe cells.

Standard discipline in swarm robotics.

## Brooks-subsumption discipline

```
LLM (slow loop ~1-5 Hz)     ──► intent (text)
                                   │
                                   ▼
Reflex (fast loop ~20-200 Hz) ──► velocity command
                                   │
                                   ▼
Sim / autopilot                ──► actuator
```

The LLM never blocks an actuator. RTT to cloud LLM is 50-500 ms; collision avoidance must be <200 ms. The reflex layer is what closes the fast loop — that's the whole point of the layered architecture.

## Sources

- [A2A specification](https://a2a-protocol.org/latest/specification/)
- [A2A GitHub](https://github.com/a2aproject/A2A)
- [Google A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [Anthropic MCP](https://modelcontextprotocol.io/)
- [Eclipse Zenoh](https://zenoh.io/)
- [PX4 v1.17 in-tree Zenoh](https://docs.px4.io/main/en/middleware/zenoh.html)
- [ED-CBBA arXiv 2509.06481](https://arxiv.org/abs/2509.06481)
- See also `~/Documents/AI_Workspace/drone_swarm_research/04_comms.md`.
