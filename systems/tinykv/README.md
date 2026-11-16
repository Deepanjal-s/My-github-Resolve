# tinykv: a write-ahead log, and the tests that prove it recovers

Roadmap Track D, topic 1 (databases). About 200 lines of Python: a key-value store with a write-ahead log (WAL),
atomic batches, snapshots (checkpoints) and crash recovery. The point is the **recovery rules** and the
**tests that check them**, not speed.

```python
from tinykv import TinyKV, PUT, DELETE
with TinyKV("data") as db:          # recovers from whatever a crash left behind
    db.put(b"user:1", b"Asha")
    db.write_batch([(PUT, b"a", b"1"), (DELETE, b"b", b"")])   # all-or-nothing
    db.checkpoint()                  # snapshot + fresh WAL
```

## Design

| Piece | Rule | Why |
|---|---|---|
| Record | `[length][crc32(length)][crc32(payload)][payload]` | The header CRC makes the length trustworthy. Without it, one flipped bit in a length field mid-log looks like a torn tail, and recovery silently drops every write after it. |
| Write | Append, `fsync`, *then* update memory and return | A returned write is durable. Memory never holds something the disk doesn't. |
| Batch | One record per batch | One CRC covers the whole batch, so it applies completely or not at all. |
| Torn tail | Truncated away on open | A crash mid-append leaves a partial record that was never acknowledged. |
| Corruption | `CorruptionError`, never silent | A bad record *followed by more data* can't come from a crash. Skipping it would lose acknowledged writes. |
| Checkpoint | Write `snapshot.tmp`, fsync → create new WAL → rename snapshot, fsync dir → delete old files | At every step either the old (snapshot, WAL) pair or the new one is complete. Recovery also deletes leftovers. |

The one case that cannot be told apart from a torn write is a flipped bit inside the **last** record's payload
(or its payload CRC). Recovery drops that record, which was the newest write. Every other single-bit flip raises.

## Tests (19, `pytest`)

- **Every truncation point:** for 5 random workloads (puts, deletes, overwrites, 3-op batches), the WAL is cut at
  *every byte offset*. Recovery must give exactly the state after the last complete batch, and must truncate the
  file there.
- **Bit flips:** 3 bits of every byte of the log. Each flip must either raise `CorruptionError` or, only inside
  the last payload, drop just the newest batch. A wrong state never passes.
- **Crash at each checkpoint step:** an exception is injected after each of the 4 steps. Reopen, keep writing,
  reopen again: no data lost and no leftover files.
- **Killed process:** a real writer process with `fsync` on is killed (`TerminateProcess`/SIGKILL) at 3 points,
  including during checkpoints. Every write it acknowledged (printed after `put()` returned) is present, plus at
  most one durable but unacknowledged write.
- Plus zero-filled tails (the file grew but the data never landed), damaged snapshots and batch atomicity.

**Mutation check:** deleting the old WAL before the new snapshot is renamed fails 3 checkpoint tests. Removing the
header-CRC check fails the bit-flip test.

## Limits (deliberately out of scope)

- The whole state lives in memory, and a checkpoint rewrites all of it (no SSTables or compaction levels).
- The process-kill test covers process crashes. Power loss also needs the disk to honour `fsync`. On Windows,
  directory `fsync` isn't available, so durability of renames there relies on NTFS metadata journaling.
- Single writer, no concurrency control.

Next in Track D: a rate-limited async job queue with retries and idempotency (Q2 2027).
