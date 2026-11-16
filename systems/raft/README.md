# raft: elections and log replication, checked under chaos

Roadmap Track D, topic 3 (distributed systems). Raft's core (Figure 2 of the
[paper](https://raft.github.io/raft.pdf)) in ~350 lines of Python on a **deterministic simulated network**. A seeded
RNG decides delays, reordering, drops, partitions and crashes, so any failure replays exactly from its seed.

Implemented: leader election with randomized timeouts, the §5.4.1 voting restriction, log replication with
conflict truncation and fast backup, the §5.4.2 commit rule (only current-term entries are committed by counting),
and crash/restart with persistent `currentTerm`, `votedFor` and log. Not implemented: snapshots, membership changes,
client sessions, read-only queries.

## How it is checked

After **every** delivered message, the cluster checks Raft's safety properties (paper, Figure 3):

| Property | Check |
|---|---|
| Election safety | At most one leader per term, recorded across the whole run |
| Log matching | Two logs with the same term at an index are identical up to it |
| Leader completeness | A newly elected leader holds every entry already applied anywhere |
| State machine safety | No index is ever applied with two different entries |

## Tests (159, ~11 s)

- **150 chaos seeds:** 5 nodes, 10% message loss, random two-way partitions, crashes and restarts every 100 ms,
  and a client submitting throughout, for 8 s of virtual time. Then the network heals, and **liveness** is
  required: within 3 s a leader exists, and a new command reaches every node's state machine.
- Directed cases: replication in order, a leader crash with no committed entry lost, and a minority partition
  that can't commit (its uncommitted entry is later overwritten).

## Do the checks catch real Raft bugs?

A checker that never fires proves nothing, so three classic bugs are implemented as subclasses and must be caught:

| Broken variant | How it is caught |
|---|---|
| Votes without the up-to-date check (§5.4.1) | Random chaos, first seed: *leader completeness* violated |
| `votedFor` not persisted across a restart | Directed scenario: one node votes twice in one term → *election safety* violated |
| Counts replicas for old-term entries (the **Figure 8** problem) | Figure 8 replayed step by step → *leader completeness* violated |

The last two need an exact interleaving, which random chaos did not produce in 300–600 seeds. That is a useful
lesson about randomized testing: rare-interleaving bugs need directed scenarios too. Each directed scenario also
runs against the correct implementation, where it must stay safe. In Figure 8, S5 legitimately overwrites the
uncommitted entry.

## Two bugs found while building it

1. **A deposed leader never stood for election again.** Leaders run no election timer, and stepping down didn't
   restart it. Usually this was hidden: a timer left over from the leader's own election, or a heartbeat from the
   next leader, re-armed it. It only bit when the leader had served past that leftover deadline *and* was the only
   node that could win (the only up-to-date log). Then the cluster stayed leaderless. None of the 150 chaos seeds
   hit it, so a directed regression test does: it fails without the fix and passes with it.
2. **A test-harness bug: ghost nodes.** The first version of the broken-variant tests swapped new node objects
   into an existing cluster. The nodes the cluster had built first still had live timers in the simulator, so
   they ran elections, even for a "crashed" node, and sent messages under the real nodes' ids. A trace of terms
   that went *backwards* exposed it. The node class is now a constructor parameter. In a simulation, everything
   that owns a timer must be accounted for.
