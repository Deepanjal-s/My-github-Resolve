# btree: disk pages, leaf links, and redo recovery

Roadmap Track D (databases). A small Python B+tree for byte keys and values.
Run the example from the repository root.

```python
from systems.btree.btree import BTree

with BTree("data.btree") as db:
    db.put(b"user:1", b"Asha")
    db.put(b"user:2", b"Mira")
    print(db.get(b"user:2"))  # b'Mira'
    db.delete(b"user:1")
    print(list(db.scan(b"user:", b"user;")))  # [(b'user:2', b'Mira')]
```

## Design

| Piece | Rule | Why |
|---|---|---|
| Pages | Store the tree in one file with 4 KiB pages. Each page has a magic, kind, page ID, entry count, link, and a CRC32 that covers the header, payload, and padding. Page zero holds the root, allocation limit, and free-list head. | Page IDs map directly to file offsets. Damaged pages raise `CorruptionError` when read. |
| Branches and leaves | Each branch entry stores a child's minimum key and page ID. Leaves hold key/value pairs and link to the next leaf. | Descend once for a lookup or the start of a half-open range. Follow the leaf links for the rest of a scan. |
| Splits and deletes | Non-root pages hold four to eight entries. Split an overflowing page in two. On underflow, merge with a sibling if both fit or redistribute their entries. Collapse a root with just one child. | Entry counts keep the tree balanced and leaf depths equal. Merges return pages to a reusable free list. |
| Buffer pool | Keep a bounded LRU of clean pages, with 16 slots by default. Hold dirty pages separately until the mutation commits. | Cache eviction cannot write uncommitted pages. The dirty set grows with the height of the tree. |
| Redo WAL | Write the dirty page images and superblock image to a persistent WAL and append a commit record with a batch CRC. Sync the WAL before overwriting any data pages. Sync the data file before truncating and syncing the WAL. | Redo keeps page IDs stable, so updating a leaf does not require copying predecessor leaves to repair sibling links. |
| Recovery | Validate the commit record and batch CRC before replaying the WAL. Replay before reading the superblock. Discard incomplete batches. | Each page image can be applied again after an interrupted replay. |

The directory sync helper and `CorruptionError` come from
[tinykv](../tinykv/README.md).

## Complexity

For `n` keys, lookup and mutation use O(log n) page visits. A mutation
writes O(log n) page images in the worst case and syncs each commit. A scan
returning `k` rows takes O(log n + k) time, with one tree descent followed by
sequential traversal of the relevant leaf links. Leaf page IDs may be scattered
on disk. The file grows to its high-water mark and reuses freed pages.

## Tests

- A fixed-seed 10,000-operation workload compares the tree with a dict across
  ten opens. The invariant checker verifies ordering, child minima, equal leaf
  depths, slot occupancy, sibling links, and free-list ownership.
- Targeted cases cover redistribution, multi-level splits and merges, page
  reuse, half-open ranges, empty byte strings, and reopen persistence.
- Exception and partial-write injection cover bootstrap, a root split, and a
  root collapse. The tests interrupt recovery too, then verify the recovered
  tree and perform another durable write.
- A power-loss model restores each file to its last synced bytes at a fixed
  set of root-split crash points, including after WAL cleanup.
- Bit flips test CRC detection in leaf pages and the superblock.

Run from the repository root with `python -m pytest systems/btree -q` or
`python systems/btree/test_btree.py`.

## Limits

- One open handle and one writer. Consume scans before mutating or closing.
  After a write error, reopen to recover. A mutation that raised after WAL
  sync may already be durable.
- Keys and values can total up to 498 bytes per row. Fill factor is measured
  by entry count. Short rows leave more unused space within each page.
- Power-loss durability depends on the filesystem honoring file sync and durable
  filename creation. Initial filenames are established with a write-through move on
  Windows or a directory sync on POSIX. Subsequent commits update the existing
  WAL in place.

## What I'd do next

- Pack variable-length entries into slotted pages and rebalance by byte usage.
- Add overflow pages for larger values and batch mutations to amortize syncs.
- Add a full-file verifier that checks every page and a compaction command
  that returns free pages at the end of the file to the filesystem.
