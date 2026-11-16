"""tinykv: a small key-value store with a write-ahead log (WAL) and crash recovery.

Files in the data directory:
    wal-<gen>.log    append-only log of committed batches since snapshot <gen>
    snapshot-<gen>   full state as of the start of wal-<gen>.log (absent for gen 0)

Record framing (WAL and snapshot):  [length u32][crc32(length) u32][crc32(payload) u32][payload]
The header checksum makes the length trustworthy: without it, one flipped bit in a length field mid-log
would look like a torn tail and recovery would silently drop every acknowledged write after it.
Payload: a batch of operations, applied all-or-nothing:  [count u32] then per op
    [op u8][key len u32][key][value len u32][value]     op: 1 = put, 2 = delete (value empty)

Durability: with sync=True (default) a write returns only after fsync of the WAL, so an acknowledged
write survives a process crash and (if the disk honours fsync) a power loss.

Recovery rules:
- Torn write from a crash (never acknowledged, so dropped and the file truncated to the last good record):
  fewer than 12 header bytes left; an all-zero tail (file grew but data never landed); a valid header whose
  payload runs past end-of-file; or the last record failing its payload CRC.
- Anything else (bad header with data after it, bad payload with more records after it) is corruption: raise
  CorruptionError rather than silently drop acknowledged writes. A flip inside the last record's payload is
  indistinguishable from a torn write and is dropped; that record was the newest write.
"""
from __future__ import annotations

import os
import re
import struct
import zlib
from typing import Callable, Iterable, Iterator

PUT, DELETE = 1, 2
_HEADER = struct.Struct("<III")
_U32 = struct.Struct("<I")


class CorruptionError(Exception):
    """The log is damaged somewhere other than a torn tail."""


def encode_batch(ops: Iterable[tuple[int, bytes, bytes]]) -> bytes:
    ops = list(ops)
    parts = [_U32.pack(len(ops))]
    for op, key, value in ops:
        if op not in (PUT, DELETE):
            raise ValueError(f"bad op {op}")
        parts += [bytes([op]), _U32.pack(len(key)), key, _U32.pack(len(value)), value]
    payload = b"".join(parts)
    n = len(payload)
    return _HEADER.pack(n, zlib.crc32(_U32.pack(n)), zlib.crc32(payload)) + payload


def decode_batch(payload: bytes) -> list[tuple[int, bytes, bytes]]:
    (count,), pos, ops = _U32.unpack_from(payload, 0), 4, []
    for _ in range(count):
        op = payload[pos]
        (klen,) = _U32.unpack_from(payload, pos + 1)
        key = payload[pos + 5: pos + 5 + klen]
        pos += 5 + klen
        (vlen,) = _U32.unpack_from(payload, pos)
        value = payload[pos + 4: pos + 4 + vlen]
        pos += 4 + vlen
        ops.append((op, key, value))
    if pos != len(payload):
        raise CorruptionError("batch payload has trailing bytes")
    return ops


def read_records(data: bytes) -> tuple[list[list[tuple[int, bytes, bytes]]], int]:
    """Parses records; returns (batches, length of the valid prefix). Raises on mid-file corruption."""
    batches, pos = [], 0
    while pos < len(data):
        rest = data[pos:]
        if len(rest) < _HEADER.size or not any(rest):
            break  # torn header, or a tail that was allocated but never written
        length, hcrc, crc = _HEADER.unpack_from(data, pos)
        if zlib.crc32(_U32.pack(length)) != hcrc:
            raise CorruptionError(f"bad record header at offset {pos}")
        end = pos + _HEADER.size + length
        if end > len(data):
            break  # torn payload
        payload = data[pos + _HEADER.size: end]
        if zlib.crc32(payload) != crc:
            if end == len(data):
                break  # last record torn
            raise CorruptionError(f"payload CRC mismatch at offset {pos}, {len(data) - end} bytes after it")
        batches.append(decode_batch(payload))
        pos = end
    return batches, pos


def _fsync_dir(path: str) -> None:
    """Makes a rename/create durable. Not supported on Windows, where NTFS journals metadata."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _apply(state: dict[bytes, bytes], ops: list[tuple[int, bytes, bytes]]) -> None:
    for op, key, value in ops:
        if op == PUT:
            state[key] = value
        else:
            state.pop(key, None)


class TinyKV:
    def __init__(self, path: str, sync: bool = True,
                 crash_hook: Callable[[str], None] | None = None):
        """``crash_hook(step)`` is called between checkpoint steps; tests raise from it to simulate a crash."""
        self.path, self.sync = path, sync
        self._hook = crash_hook or (lambda step: None)
        os.makedirs(path, exist_ok=True)
        self._state: dict[bytes, bytes] = {}
        self.gen = self._recover()
        self._wal = open(self._wal_path(self.gen), "ab")

    # ---- paths -------------------------------------------------------------------------------
    def _wal_path(self, gen: int) -> str:
        return os.path.join(self.path, f"wal-{gen}.log")

    def _snap_path(self, gen: int) -> str:
        return os.path.join(self.path, f"snapshot-{gen}")

    def _gens(self, prefix: str) -> list[int]:
        pat = re.compile(rf"^{prefix}-(\d+)(\.log)?$")
        return sorted(int(m.group(1)) for f in os.listdir(self.path) if (m := pat.match(f)))

    # ---- recovery ----------------------------------------------------------------------------
    def _recover(self) -> int:
        snaps = self._gens("snapshot")
        gen = snaps[-1] if snaps else 0
        if snaps:
            with open(self._snap_path(gen), "rb") as f:
                data = f.read()
            batches, good = read_records(data)
            if good != len(data):  # a snapshot is complete before its rename, so any damage is real
                raise CorruptionError(f"snapshot-{gen} is damaged")
            for ops in batches:
                _apply(self._state, ops)
        wal = self._wal_path(gen)
        if os.path.exists(wal):
            with open(wal, "rb") as f:
                data = f.read()
            batches, good = read_records(data)
            for ops in batches:
                _apply(self._state, ops)
            if good != len(data):
                with open(wal, "r+b") as f:
                    f.truncate(good)
                    f.flush()
                    os.fsync(f.fileno())
        # Leftovers of an interrupted checkpoint: older generations and an unfinished snapshot.
        for g in self._gens("wal"):
            if g < gen:
                os.remove(self._wal_path(g))
            elif g > gen:  # new WAL created, but its snapshot never became durable: it holds no writes yet
                os.remove(self._wal_path(g))
        for g in snaps[:-1]:
            os.remove(self._snap_path(g))
        for f in os.listdir(self.path):
            if f.endswith(".tmp"):
                os.remove(os.path.join(self.path, f))
        return gen

    # ---- writes ------------------------------------------------------------------------------
    def write_batch(self, ops: Iterable[tuple[int, bytes, bytes]]) -> None:
        ops = list(ops)
        record = encode_batch(ops)
        self._wal.write(record)
        self._wal.flush()
        if self.sync:
            os.fsync(self._wal.fileno())
        _apply(self._state, ops)  # only after the record is durable

    def put(self, key: bytes, value: bytes) -> None:
        self.write_batch([(PUT, key, value)])

    def delete(self, key: bytes) -> None:
        self.write_batch([(DELETE, key, b"")])

    # ---- reads -------------------------------------------------------------------------------
    def get(self, key: bytes, default: bytes | None = None) -> bytes | None:
        return self._state.get(key, default)

    def items(self) -> Iterator[tuple[bytes, bytes]]:
        return iter(sorted(self._state.items()))

    def __len__(self) -> int:
        return len(self._state)

    # ---- checkpoint --------------------------------------------------------------------------
    def checkpoint(self) -> None:
        """Writes the full state to snapshot-<gen+1>, then starts wal-<gen+1> and deletes the old files.

        Order matters: the new snapshot is fsynced and renamed into place *before* anything old is deleted,
        so a crash at any step leaves either the old (snapshot, WAL) pair or the new one intact.
        """
        new = self.gen + 1
        tmp = self._snap_path(new) + ".tmp"
        with open(tmp, "wb") as f:
            f.write(encode_batch((PUT, k, v) for k, v in sorted(self._state.items())))
            f.flush()
            os.fsync(f.fileno())
        self._hook("snapshot-written")
        open(self._wal_path(new), "wb").close()
        self._hook("new-wal-created")
        os.replace(tmp, self._snap_path(new))
        _fsync_dir(self.path)
        self._hook("snapshot-renamed")
        self._wal.close()
        self._wal = open(self._wal_path(new), "ab")
        old = self.gen
        self.gen = new
        os.remove(self._wal_path(old))
        if os.path.exists(self._snap_path(old)):
            os.remove(self._snap_path(old))
        self._hook("old-files-removed")

    def close(self) -> None:
        self._wal.close()

    def __enter__(self) -> "TinyKV":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
