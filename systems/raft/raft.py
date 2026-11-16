"""raft: leader election and log replication (Raft, Ongaro & Ousterhout 2014, Figure 2) on a deterministic
simulated network, with Raft's safety properties checked after every event.

Everything runs in one process on virtual time: a seeded RNG decides message delays, drops, partitions and
crashes, so any failing seed replays exactly. Persistent state (currentTerm, votedFor, log) survives a crash;
volatile state does not.

Scope: elections, replication, commit, crash/restart. Not covered: snapshots, membership changes, client
sessions/deduplication, read-only queries.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Callable

FOLLOWER, CANDIDATE, LEADER = "follower", "candidate", "leader"


@dataclass
class Entry:
    term: int
    cmd: Any


@dataclass
class RequestVote:
    term: int
    candidate: int
    last_log_index: int
    last_log_term: int


@dataclass
class VoteReply:
    term: int
    granted: bool


@dataclass
class AppendEntries:
    term: int
    leader: int
    prev_index: int
    prev_term: int
    entries: list[Entry]
    leader_commit: int


@dataclass
class AppendReply:
    term: int
    success: bool
    match_index: int  # on success: index of the last entry now known to match the leader
    conflict_index: int = 0  # on failure: where the leader should retry from (fast backup)


class SafetyViolation(AssertionError):
    pass


class Sim:
    """Discrete-event simulator: a heap of (time, seq, callback)."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.now = 0.0
        self._q: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = 0

    def at(self, delay: float, fn: Callable[[], None]) -> None:
        self._seq += 1
        heapq.heappush(self._q, (self.now + delay, self._seq, fn))

    def step(self) -> bool:
        if not self._q:
            return False
        self.now, _, fn = heapq.heappop(self._q)
        fn()
        return True


class Network:
    def __init__(self, sim: Sim, drop: float = 0.0, min_delay: float = 1.0, max_delay: float = 10.0):
        self.sim, self.drop, self.min_delay, self.max_delay = sim, drop, min_delay, max_delay
        self.nodes: dict[int, "Node"] = {}
        self.group: dict[int, int] = {}  # node -> partition id; nodes talk only within a partition

    def connected(self, a: int, b: int) -> bool:
        return self.group.get(a, 0) == self.group.get(b, 0)

    def send(self, src: int, dst: int, msg: Any) -> None:
        if self.sim.rng.random() < self.drop or not self.connected(src, dst):
            return
        delay = self.sim.rng.uniform(self.min_delay, self.max_delay)  # reordering happens naturally

        def deliver():
            node = self.nodes[dst]
            if node.alive and self.connected(src, dst):
                node.receive(src, msg)

        self.sim.at(delay, deliver)


class Node:
    ELECTION_MIN, ELECTION_MAX, HEARTBEAT = 150.0, 300.0, 50.0

    def __init__(self, nid: int, peers: list[int], sim: Sim, net: Network, cluster: "Cluster"):
        self.id, self.peers, self.sim, self.net, self.cluster = nid, peers, sim, net, cluster
        # persistent
        self.term, self.voted_for, self.log = 0, None, [Entry(0, None)]  # log[0] is a sentinel
        self.alive = True
        self._restart_volatile()

    def _restart_volatile(self) -> None:
        self.role, self.commit, self.applied = FOLLOWER, 0, 0
        self.votes: set[int] = set()
        self.next_index: dict[int, int] = {}
        self.match_index: dict[int, int] = {}
        self.state_machine: list[Any] = []  # rebuilt from the log on restart (no snapshots)
        self._epoch = getattr(self, "_epoch", 0) + 1  # invalidates timers from before a crash
        self._reset_election_timer()

    # ---- timers ------------------------------------------------------------------------------
    def _timer(self, delay: float, fn: Callable[[], None]) -> None:
        epoch = self._epoch

        def fire():
            if self.alive and self._epoch == epoch:
                fn()
        self.sim.at(delay, fire)

    def _reset_election_timer(self) -> None:
        self._election_deadline = self.sim.now + self.sim.rng.uniform(self.ELECTION_MIN, self.ELECTION_MAX)
        deadline = self._election_deadline
        self._timer(deadline - self.sim.now, lambda: self._election_timeout(deadline))

    def _election_timeout(self, deadline: float) -> None:
        if self.role != LEADER and deadline == self._election_deadline:
            self._start_election()

    # ---- helpers -----------------------------------------------------------------------------
    def last_index(self) -> int:
        return len(self.log) - 1

    def _step_down(self, term: int) -> None:
        if term > self.term:
            self.term, self.voted_for = term, None
        if self.role == LEADER:  # leaders have no election timer running; a deposed one needs it back
            self._reset_election_timer()
        self.role = FOLLOWER

    def _majority(self) -> int:
        return (len(self.peers) + 1) // 2 + 1

    # ---- election ----------------------------------------------------------------------------
    def _start_election(self) -> None:
        self.role, self.term, self.voted_for = CANDIDATE, self.term + 1, self.id
        self.votes = {self.id}
        self._reset_election_timer()
        msg = RequestVote(self.term, self.id, self.last_index(), self.log[-1].term)
        for p in self.peers:
            self.net.send(self.id, p, msg)

    def _become_leader(self) -> None:
        self.role = LEADER
        self.next_index = {p: self.last_index() + 1 for p in self.peers}
        self.match_index = {p: 0 for p in self.peers}
        self.cluster.on_leader(self)
        self._heartbeat()

    def _heartbeat(self) -> None:
        if self.role != LEADER:
            return
        for p in self.peers:
            self._send_append(p)
        self._timer(self.HEARTBEAT, self._heartbeat)

    def _send_append(self, p: int) -> None:
        prev = self.next_index[p] - 1
        entries = [Entry(e.term, e.cmd) for e in self.log[prev + 1: prev + 1 + 20]]
        self.net.send(self.id, p, AppendEntries(self.term, self.id, prev, self.log[prev].term, entries, self.commit))

    # ---- client ------------------------------------------------------------------------------
    def submit(self, cmd: Any) -> bool:
        if not self.alive or self.role != LEADER:
            return False
        self.log.append(Entry(self.term, cmd))
        for p in self.peers:
            self._send_append(p)
        return True

    # ---- message handling --------------------------------------------------------------------
    def receive(self, src: int, msg: Any) -> None:
        if msg.term > self.term:
            self._step_down(msg.term)
        if isinstance(msg, RequestVote):
            self._on_request_vote(msg)
        elif isinstance(msg, VoteReply):
            self._on_vote_reply(src, msg)
        elif isinstance(msg, AppendEntries):
            self._on_append(msg)
        elif isinstance(msg, AppendReply):
            self._on_append_reply(src, msg)
        self.cluster.check()

    def _up_to_date(self, last_index: int, last_term: int) -> bool:
        """§5.4.1: the candidate's log is at least as up-to-date as ours."""
        mine = self.log[-1].term
        return last_term > mine or (last_term == mine and last_index >= self.last_index())

    def _on_request_vote(self, m: RequestVote) -> None:
        grant = (m.term == self.term and self.voted_for in (None, m.candidate)
                 and self._up_to_date(m.last_log_index, m.last_log_term))
        if grant:
            self.voted_for = m.candidate
            self._reset_election_timer()
        self.net.send(self.id, m.candidate, VoteReply(self.term, grant))

    def _on_vote_reply(self, src: int, m: VoteReply) -> None:
        if self.role == CANDIDATE and m.term == self.term and m.granted:
            self.votes.add(src)
            if len(self.votes) >= self._majority():
                self._become_leader()

    def _on_append(self, m: AppendEntries) -> None:
        if m.term < self.term:
            self.net.send(self.id, m.leader, AppendReply(self.term, False, 0, 0))
            return
        self.role = FOLLOWER  # a current leader exists (also turns a candidate back into a follower)
        self._reset_election_timer()
        if m.prev_index > self.last_index():
            self.net.send(self.id, m.leader, AppendReply(self.term, False, 0, self.last_index() + 1))
            return
        if self.log[m.prev_index].term != m.prev_term:
            bad = self.log[m.prev_index].term  # skip back over the whole conflicting term
            i = m.prev_index
            while i > 1 and self.log[i - 1].term == bad:
                i -= 1
            self.net.send(self.id, m.leader, AppendReply(self.term, False, 0, i))
            return
        # Append, truncating only on a real conflict: a stale, reordered message must not cut entries
        # that a newer message already added (Figure 2, AppendEntries step 3).
        for k, e in enumerate(m.entries):
            i = m.prev_index + 1 + k
            if i <= self.last_index():
                if self.log[i].term == e.term:
                    continue
                del self.log[i:]
            self.log.append(Entry(e.term, e.cmd))
        last_new = m.prev_index + len(m.entries)
        if m.leader_commit > self.commit:
            self.commit = min(m.leader_commit, last_new)
            self._apply()
        self.net.send(self.id, m.leader, AppendReply(self.term, True, last_new))

    def _on_append_reply(self, src: int, m: AppendReply) -> None:
        if self.role != LEADER or m.term != self.term:
            return
        if m.success:
            self.match_index[src] = max(self.match_index[src], m.match_index)
            self.next_index[src] = self.match_index[src] + 1
            self._advance_commit()
        else:
            self.next_index[src] = max(1, min(self.next_index[src] - 1, m.conflict_index or self.next_index[src] - 1))
            self._send_append(src)

    def _advance_commit(self) -> None:
        for n in range(self.last_index(), self.commit, -1):
            # §5.4.2: only entries from the current term are committed by counting replicas
            if self.log[n].term != self.term:
                break
            if 1 + sum(1 for p in self.peers if self.match_index[p] >= n) >= self._majority():
                self.commit = n
                self._apply()
                break

    def _apply(self) -> None:
        while self.applied < self.commit:
            self.applied += 1
            self.state_machine.append(self.log[self.applied].cmd)
            self.cluster.on_apply(self, self.applied)

    # ---- crash / restart ---------------------------------------------------------------------
    def crash(self) -> None:
        self.alive = False

    def restart(self) -> None:
        self.alive = True
        self._restart_volatile()


class Cluster:
    """N nodes plus the safety checker. Global history is kept outside the nodes to check Raft's properties."""

    def __init__(self, n: int, seed: int, drop: float = 0.0, node_cls: type[Node] = Node,
                 delay: tuple[float, float] = (1.0, 10.0)):
        self.sim = Sim(seed)
        self.net = Network(self.sim, drop=drop, min_delay=delay[0], max_delay=delay[1])
        ids = list(range(n))
        self.nodes = [node_cls(i, [p for p in ids if p != i], self.sim, self.net, self) for i in ids]
        for node in self.nodes:
            self.net.nodes[node.id] = node
        self.leaders: dict[int, int] = {}  # term -> leader id
        self.committed: dict[int, Entry] = {}  # index -> entry, once any node applied it

    # ---- safety properties (Raft paper, Figure 3) -------------------------------------------
    def on_leader(self, node: Node) -> None:
        prev = self.leaders.setdefault(node.term, node.id)
        if prev != node.id:
            raise SafetyViolation(f"election safety: term {node.term} has leaders {prev} and {node.id}")
        for i, e in self.committed.items():  # leader completeness
            if i > node.last_index() or node.log[i].term != e.term or node.log[i].cmd != e.cmd:
                raise SafetyViolation(f"leader completeness: new leader {node.id} (term {node.term}) lacks "
                                      f"committed entry {i}")

    def on_apply(self, node: Node, index: int) -> None:
        e = node.log[index]
        prev = self.committed.setdefault(index, Entry(e.term, e.cmd))
        if (prev.term, prev.cmd) != (e.term, e.cmd):  # state machine safety
            raise SafetyViolation(f"state machine safety: index {index} applied as {prev} and {e} (node {node.id})")

    def check(self) -> None:
        """Log matching: same index and term => identical logs up to that index."""
        nodes = self.nodes
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                la, lb = nodes[a].log, nodes[b].log
                top = min(len(la), len(lb)) - 1
                for i in range(top, 0, -1):
                    if la[i].term == lb[i].term:
                        if any((la[j].term, la[j].cmd) != (lb[j].term, lb[j].cmd) for j in range(1, i + 1)):
                            raise SafetyViolation(f"log matching: nodes {a},{b} agree at {i} but differ before")
                        break

    # ---- driving -----------------------------------------------------------------------------
    def leader(self) -> Node | None:
        live = [n for n in self.nodes if n.alive and n.role == LEADER]
        return max(live, key=lambda n: n.term) if live else None

    def run_for(self, ms: float) -> None:
        end = self.sim.now + ms
        while self.sim._q and self.sim._q[0][0] <= end:
            self.sim.step()
        self.sim.now = end
