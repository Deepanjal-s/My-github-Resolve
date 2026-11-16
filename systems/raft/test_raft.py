import pytest

from raft import LEADER, Cluster, Node, SafetyViolation


def chaos(cluster: Cluster, ms: float, crash_rate: float = 0.3, partition_rate: float = 0.3):
    """Every 100 ms: maybe crash or restart a node, maybe re-partition the network, submit a command."""
    rng = cluster.sim.rng
    n = len(cluster.nodes)
    t, cmd = 0.0, 0
    while t < ms:
        if rng.random() < crash_rate:
            node = rng.choice(cluster.nodes)
            if node.alive and sum(x.alive for x in cluster.nodes) > 1:
                node.crash()
            elif not node.alive:
                node.restart()
        if rng.random() < partition_rate:
            cluster.net.group = {i: rng.randrange(2) for i in range(n)}
        elif rng.random() < 0.3:
            cluster.net.group = {}
        leader = cluster.leader()
        if leader:
            leader.submit(f"c{cluster.sim.now:.0f}-{cmd}")
            cmd += 1
        cluster.run_for(100)
        t += 100


def heal(cluster: Cluster):
    cluster.net.group = {}
    cluster.net.drop = 0.0
    for node in cluster.nodes:
        if not node.alive:
            node.restart()


def test_elects_one_leader_and_replicates_in_order():
    c = Cluster(5, seed=1)
    c.run_for(1000)
    leader = c.leader()
    assert leader is not None
    for i in range(50):
        assert leader.submit(i)
        c.run_for(5)
    c.run_for(500)
    assert all(n.state_machine == list(range(50)) for n in c.nodes)


def test_leader_crash_triggers_new_election_and_no_committed_entry_is_lost():
    c = Cluster(5, seed=2)
    c.run_for(1000)
    old = c.leader()
    for i in range(10):
        old.submit(i)
    c.run_for(300)
    old.crash()
    c.run_for(1000)
    new = c.leader()
    assert new is not None and new.id != old.id and new.term > old.term
    new.submit("after")
    c.run_for(300)
    old.restart()
    c.run_for(1000)
    assert all(n.state_machine == list(range(10)) + ["after"] for n in c.nodes)


def test_minority_partition_cannot_commit():
    c = Cluster(5, seed=3)
    c.run_for(1000)
    leader = c.leader()
    c.net.group = {leader.id: 1, (leader.id + 1) % 5: 1}  # leader + 1 follower cut off
    leader.submit("lost")
    c.run_for(2000)
    assert all("lost" not in n.state_machine for n in c.nodes)
    majority_leader = c.leader()
    assert majority_leader.id != leader.id
    majority_leader.submit("kept")
    c.run_for(500)
    c.net.group = {}
    c.run_for(2000)
    assert all(n.state_machine == ["kept"] for n in c.nodes)  # the uncommitted entry was overwritten


@pytest.mark.parametrize("seed", range(150))
def test_safety_under_chaos_and_liveness_after_heal(seed):
    """Drops, reordering, partitions, crashes and restarts for 8 s of virtual time; every event is checked
    for election safety, log matching, leader completeness and state machine safety. Then the network heals
    and the cluster must elect a leader and commit a new command on every node."""
    c = Cluster(5, seed=seed, drop=0.1)
    chaos(c, 8000)
    heal(c)
    c.run_for(3000)
    leader = c.leader()
    assert leader is not None, "no leader 3 s after healing"
    assert leader.submit("final")
    c.run_for(2000)
    sms = [n.state_machine for n in c.nodes]
    assert all(sm == sms[0] for sm in sms) and sms[0][-1] == "final"


# ---- the checker must catch broken Raft variants --------------------------------------------------

class VotesForStaleLogs(Node):
    """Bug: grants votes without the §5.4.1 up-to-date check."""

    def _up_to_date(self, last_index, last_term):
        return True


class ForgetsVoteOnRestart(Node):
    """Bug: votedFor is not persisted, so a restarted node can vote twice in one term."""

    def restart(self):
        self.voted_for = None
        super().restart()


class CommitsOldTermEntries(Node):
    """Bug: counts replicas for entries from earlier terms (the Figure 8 problem)."""

    def _advance_commit(self):
        for n in range(self.last_index(), self.commit, -1):
            if 1 + sum(1 for p in self.peers if self.match_index[p] >= n) >= self._majority():
                self.commit = n
                self._apply()
                break


def _first_violation(node_cls, seeds, **chaos_kw):
    for seed in seeds:
        c = Cluster(5, seed=seed, drop=0.1, node_cls=node_cls)
        try:
            chaos(c, 8000, **chaos_kw)
        except SafetyViolation as e:
            return seed, str(e)
    return None


def test_checker_catches_votes_for_stale_logs_under_chaos():
    found = _first_violation(VotesForStaleLogs, range(300), crash_rate=0.5)
    assert found is not None and "leader completeness" in found[1], found


# Directed scenarios: these bugs need a precise interleaving that random chaos rarely produces.

def _directed(node_cls, n: int = 5) -> Cluster:
    # 1 ms delays, so no election timer (>= 150 ms) fires mid-scenario. node_cls goes into the constructor:
    # swapping nodes in afterwards left the original nodes' timers running as ghosts in the simulator.
    return Cluster(n, seed=0, node_cls=node_cls, delay=(1.0, 1.0))


def _connect(c: Cluster, together):
    c.net.group = {i: (0 if i in together else 10 + i) for i in range(len(c.nodes))}


def _elect(c: Cluster, cand: int, voters, tries: int = 3) -> bool:
    """Like an election timeout firing up to `tries` times (each try is a new term)."""
    _connect(c, [cand, *voters])
    for _ in range(tries):
        c.nodes[cand]._start_election()
        c.run_for(5)
        if c.nodes[cand].role == LEADER:
            return True
    return False


def _double_vote(c: Cluster):
    """S0 wins term 1 with S1, S2. S2 restarts. S3 then runs in the same term 1 with S2 and S4."""
    assert _elect(c, 0, [1, 2])
    c.nodes[2].crash()
    c.nodes[2].restart()
    c.nodes[3].term = 0  # S3 never heard of term 1, so its election is also for term 1
    _elect(c, 3, [2, 4], tries=1)


def _figure8(c: Cluster):
    """Raft paper, Figure 8, with S1..S5 as nodes 0..4."""
    s1, s2, s3, s4, s5 = c.nodes
    assert _elect(c, 0, [1, 2, 3, 4])   # (a) S1 leads term 1 and replicates x to S2 only
    _connect(c, [0, 1])
    s1.submit("x")
    c.run_for(5)
    s1.crash()                          # (b) S5 leads term 2 with S3, S4; appends y locally only
    assert _elect(c, 4, [2, 3])
    _connect(c, [4])
    s5.submit("y")
    c.run_for(5)
    s5.crash()                          # (c) S1 leads term 3 with S2, S3 and copies x to S3 (a majority)
    s1.restart()
    assert _elect(c, 0, [1, 2])
    c.run_for(5)
    s1.crash()                          # (d) S5 leads term 4 with S2, S3, S4 and overwrites x with y
    s5.restart()
    _elect(c, 4, [1, 2, 3])
    c.run_for(5)


def test_double_vote_scenario_is_safe_with_persisted_vote():
    c = _directed(Node)
    _double_vote(c)
    assert c.nodes[3].role != LEADER and list(c.leaders.values()) == [0]


def test_checker_catches_vote_forgotten_on_restart():
    with pytest.raises(SafetyViolation, match="election safety"):
        _double_vote(_directed(ForgetsVoteOnRestart))


def test_figure8_scenario_is_safe_with_current_term_commit_rule():
    c = _directed(Node)
    _figure8(c)
    assert c.nodes[4].role == LEADER  # S5 legitimately overwrote x, which was never committed
    assert 1 not in c.committed or c.committed[1].cmd == "y"


def test_checker_catches_figure8_commit_bug():
    with pytest.raises(SafetyViolation, match="leader completeness|state machine"):
        _figure8(_directed(CommitsOldTermEntries))


def test_deposed_leader_stands_again_when_only_it_can_win():
    """Regression: leaders run no election timer, and stepping down didn't restart it. If the only node
    that can win (the only up-to-date log) is a deposed leader, the cluster then stayed leaderless."""
    c = _directed(Node, n=3)
    a, b, cand = c.nodes
    assert _elect(c, 0, [1, 2])
    _connect(c, [0, 1])
    a.submit("x")                  # committed on A and B; C missed it
    c.run_for(400)                 # A leads past the deadline of the timer left over from its own election
    b.crash()
    _connect(c, [0, 2])
    cand._start_election()         # C has a higher term but a stale log: A steps down and refuses its vote
    c.run_for(5)
    assert a.role != LEADER
    c.run_for(3000)
    leader = c.leader()
    assert leader is a, "no leader: the deposed leader never started an election"
