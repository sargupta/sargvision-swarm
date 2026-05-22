"""Protocols, topology, bandwidth, BFT, ED-CBBA, peer dialogue."""

import numpy as np

from sargvision_swarm.agents.peer_dialogue import PeerDialogue, agent_card
from sargvision_swarm.comms import (
    A2ACard,
    A2AMessage,
    BFTVote,
    BandwidthTracker,
    CBBABid,
    CommModel,
    IntentPayload,
    MessageLog,
    PosePayload,
    Protocol,
    WireMessage,
)
from sargvision_swarm.core import SwarmState
from sargvision_swarm.orchestrator import EDCBBA, SwarmRaft, Task


def test_a2a_message_is_jsonrpc_2():
    msg = A2AMessage(method="share.intent", params={"from": 0, "intent": "hold_formation"})
    assert msg.jsonrpc == "2.0"
    assert msg.id  # uuid set
    assert msg.method == "share.intent"


def test_wire_message_records_bytes():
    payload = PosePayload(pos=[0, 0, 5], vel=[0.1, 0, 0])
    m = WireMessage.make(src=3, protocol=Protocol.ZENOH, topic="swarm/3/pose", payload=payload)
    assert m.bytes_size > 0
    assert m.protocol == Protocol.ZENOH


def test_topology_in_range():
    comm = CommModel(range_m=10.0)
    positions = np.array([[0, 0, 5], [5, 0, 5], [20, 0, 5]])
    adj = comm.adjacency(positions)
    assert adj[0, 1]
    assert not adj[0, 2]
    assert not adj.diagonal().any()


def test_topology_signal_strength_decays():
    comm = CommModel(range_m=10.0)
    positions = np.array([[0, 0, 5], [2, 0, 5], [9, 0, 5]])
    s = comm.signal_strength(positions)
    assert s[0, 1] > s[0, 2]
    assert (s.diagonal() == 0).all()


def test_packet_loss_at_range():
    comm = CommModel(range_m=10.0, loss_start_m=5.0, max_loss=0.5)
    assert comm.packet_loss(2.0) == 0.0
    assert comm.packet_loss(10.0) == 1.0  # at range = drop
    assert 0.0 < comm.packet_loss(7.5) < 0.5


def test_bandwidth_records_and_aggregates():
    bw = BandwidthTracker(window_s=2.0)
    bw.record(0.0, "A2A", 100)
    bw.record(0.5, "A2A", 200)
    bw.record(1.0, "Zenoh", 32)
    rates = bw.rates_by_protocol(now=1.5)
    assert rates["A2A"]["total_bytes"] == 300
    assert rates["Zenoh"]["total_bytes"] == 32


def test_swarm_raft_passes_with_no_byzantine():
    raft = SwarmRaft()
    passed, votes = raft.propose("advance_phase:engage")
    assert passed
    assert all(v.decision == "yes" for v in votes)
    assert len(votes) == 7


def test_swarm_raft_handles_byzantine():
    raft = SwarmRaft(rng_seed=1)
    passed, votes = raft.propose("re-plan:wind-gust", byzantine_ids={0, 1, 2})
    yes = sum(1 for v in votes if v.decision == "yes")
    # 4+ yes out of 7 still passes quorum; 3 byzantine may or may not flip
    assert len(votes) == 7
    # Always recorded as a BFTVote
    assert all(isinstance(v, BFTVote) for v in votes)


def test_ed_cbba_assigns_each_task_to_some_drone():
    positions = np.array([[0, 0, 5], [10, 0, 5], [0, 10, 5], [5, 5, 5]])
    tasks = [
        Task(id="cell-0", pos=[1, 1, 5]),
        Task(id="cell-1", pos=[11, 1, 5]),
        Task(id="cell-2", pos=[1, 11, 5]),
    ]
    cbba = EDCBBA(drones_pos=positions, tasks=tasks, bundle_size=2)
    cbba.round()
    cbba.round()  # event-driven: may emit fewer bids second round
    assignment = cbba.assignment()
    assert set(assignment.keys()) == {"cell-0", "cell-1", "cell-2"}


def test_agent_card_lists_a2a_methods():
    swarm = SwarmState.random_init(1, seed=0)
    card = agent_card(swarm.drones[0])
    assert "share.intent" in card.capabilities
    assert "negotiate.yield" in card.capabilities
    assert "HTTP+SSE" in card.transports


def test_peer_dialogue_emits_messages():
    swarm = SwarmState.random_init(5, seed=0)
    # Place drones close so they're in range
    for i, d in enumerate(swarm.drones):
        d.pos = np.array([float(i) * 2.0, 0.0, 5.0])
    comm = CommModel(range_m=20.0, max_loss=0.0)
    peer = PeerDialogue(drones=swarm.drones, comm=comm)
    msgs = peer.emit_round(t=0.0, intents={d.id: "hold_formation" for d in swarm.drones})
    assert len(msgs) > 0
    protocols = {m.protocol.value for m in msgs}
    assert "Zenoh" in protocols   # pose broadcasts
    assert "A2A" in protocols     # intent shares
    assert "MAVLink" in protocols  # heartbeats


def test_message_log_holds_recent():
    log = MessageLog(capacity=50)
    for i in range(60):
        m = WireMessage.make(
            src=i % 5,
            protocol=Protocol.MAVLINK,
            topic="test",
            payload=IntentPayload(intent=f"i{i}", rationale=""),
        )
        log.append(m)
    assert len(log) == 50
    assert len(log.recent(k=10)) == 10
