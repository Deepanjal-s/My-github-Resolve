# lsm: sorted runs, tombstones, and crash recovery

Roadmap Track D (databases). A small Python LSM-tree store for byte keys and
values. Run the example from the repository root.

```python
from systems.lsm.lsm import LSM

with LSM("data") as db:
    db.put(b"user:1", b"Asha")
    db.flush()
    db.delete(b"user:1")
    db.put(b"user:2", b"Mira")
    print(list(db.scan(b"user:", b"user;")))  # [(b'user:2', b'Mira')]
```

## Design

| Piece | Rule | Why |
|---|---|---|
| WAL | Append a tinykv record and fsync before applying it to the memtable. | An acknowledged mutation survives a process crash. The record checksums protect both the length and the payload. |
| Memtable | Keep a dictionary and a sorted key list. Flush when the encoded size of the latest entries exceeds the configured byte limit. | Point reads use the dictionary and flushes already have the keys in order. Replacing an entry adjusts the size estimate. |
| SSTable | Write immutable sorted data blocks, a sparse index of first keys and offsets, a Bloom filter, and a footer. Check the footer, index and filter on open and each block when read. | The Bloom filter can rule out a table, and the index locates the one candidate block. CRC failures raise `CorruptionError`. |
| Reads | Check the memtable first, then the SSTables newest first. A tombstone ends the search. Scans merge the same sources over a half-open range. | Each key exposes its latest version, including deletes. |
| Flush | Fsync the temporary table, rename it, and fsync the directory. Create and sync a fresh WAL, then publish the table and WAL together through the manifest. | The prior manifest can still replay the old WAL. The new manifest can recover from the new table and fresh WAL. |
| Compaction | At the table-count threshold (four by default), merge all tables into one. Keep the newest version of each key and discard tombstones. | Every older on-disk value participates in the merge. The active WAL and memtable continue to hold newer writes. |
| MANIFEST | Fsync a checksummed temporary manifest, atomically replace the current one, then fsync the directory before unlinking obsolete files. | The manifest selects one complete file set. |
| Recovery | Open the manifest's tables, replay its WAL, truncate a torn tail, and remove unreferenced store files. | Temporary and obsolete files from interrupted maintenance are safe to remove once the observed manifest is durable. |

The record framing, torn-tail rules, and directory sync helper come from
[tinykv](../tinykv/README.md).

## Tests

- Four seeded workloads compare lookups and scans against a dictionary across
  repeated automatic flushes, compactions, and reopens.
- Exception hooks interrupt flush and compaction after each table, WAL,
  manifest, and cleanup boundary that applies. Each case checks the recovered
  model, leftover files, and another round of writes and recovery.
- Every incomplete prefix of one final WAL record is truncated. The test
  verifies recovery retains the complete WAL prefix and accepts later writes.
- Bit flips in a newer SSTable's block raise during lookup, scanning, and
  compaction. Separate cases exercise the footer, sparse index, and Bloom CRCs.
- Range tests cover half-open bounds, empty values, and an all-deleted table set.
- With 10,000 inserted keys at ten bits per key and seven probes, the Bloom
  test found zero false negatives and 865 (0.865%) false positives among 100,000
  distinct absent keys.

Run from the repository root with `python -m pytest -q -s systems/lsm/test_lsm.py`.

## Limits

- Single process and single writer. Consume scans before mutating or closing the
  store. Flush and compaction run synchronously.
- Count-triggered full compaction is a simplified size-tiered policy. It
  rewrites the entire on-disk state and temporarily retains the old files.
  The writer keeps output keys in memory to build the Bloom filter, and WAL
  recovery reads the whole log. Sustained overwrites of a small set of keys can
  grow the WAL without crossing the memtable limit.
- A checksum failure in the last WAL payload is treated as a torn write, as in
  tinykv. Reopen after a maintenance error. A write that raised after WAL sync
  may already be durable.
- Crash tests use exception injection. Power-loss durability depends on the
  filesystem and hardware honoring sync and atomic replace. The shared
  directory-sync helper skips directory fsync on Windows.
