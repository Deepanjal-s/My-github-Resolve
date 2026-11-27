"""A single-writer LSM tree for byte keys and values.

The MANIFEST names the active WAL and the SSTables in newest-first order.
Publishing it is the commit point for both flush and compaction.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from hashlib import blake2b
from heapq import merge
from itertools import groupby
from operator import itemgetter
from pathlib import Path
import json
import os
import re
import struct
import zlib

# Reuse tinykv's record framing, batch codec, torn-tail recovery and directory sync.
from systems.tinykv.tinykv import (
    DELETE, PUT, CorruptionError, _fsync_dir, encode_batch, read_records,
)

# Blocks are tinykv records. Index entries store the first key and block offset.
# The footer locates the index and Bloom filter and protects both and itself.
_INDEX = struct.Struct("<IQ")  # key length, block offset
_FOOTER = struct.Struct("<8sQQIII")  # magic, index offset, Bloom offset, three CRCs
_MAGIC = b"PYLSM001"
_CRC = struct.Struct("<I")
_KEY = itemgetter(0)
_OWNED = re.compile(r"MANIFEST\.tmp|wal-\d+\.log|sst-\d+\.sst(?:\.tmp)?")


def _sync(file):
    file.flush()
    os.fsync(file.fileno())


def _operations(entries):
    for key, value in entries:
        yield (DELETE, key, b"") if value is None else (PUT, key, value)


def _entries(operations):
    for op, key, value in operations:
        if op == PUT:
            yield key, value
        elif op == DELETE and not value:
            yield key, None
        else:
            raise CorruptionError("invalid put/delete operation")


class Bloom:
    """Ten bits per key (rounded up to bytes), seven probes."""

    def __init__(self, bits: bytes | bytearray):
        self.bits = bytearray(bits)

    @classmethod
    def build(cls, keys):
        bloom = cls(bytearray(max(1, (len(keys) * 10 + 7) // 8)))
        for key in keys:
            for bit in bloom._positions(key):
                bloom.bits[bit // 8] |= 1 << (bit % 8)
        return bloom

    def _positions(self, key):
        start, step = struct.unpack("<QQ", blake2b(key, digest_size=16).digest())
        for i in range(7):
            yield (start + i * (step | 1)) % (8 * len(self.bits))

    def __contains__(self, key):
        return all(self.bits[bit // 8] & (1 << (bit % 8)) for bit in self._positions(key))


def _write_table(file, entries, block_size):
    keys, index, block = [], [], []
    size = 0

    def write_block():
        first = block[0][0]
        index.extend((_INDEX.pack(len(first), file.tell()), first))
        file.write(encode_batch(_operations(block)))
        block.clear()

    for key, value in entries:
        entry_size = 9 + len(key) + (0 if value is None else len(value))
        if block and size + entry_size > block_size:
            write_block()
            size = 0
        keys.append(key)
        block.append((key, value))
        size += entry_size
    if block:
        write_block()
    index_offset = file.tell()
    index_bytes = b"".join(index)
    file.write(index_bytes)
    bloom_offset = file.tell()
    bits = Bloom.build(keys).bits
    file.write(bits)
    footer = _FOOTER.pack(_MAGIC, index_offset, bloom_offset,
                          zlib.crc32(index_bytes), zlib.crc32(bits), 0)
    file.write(footer[:-4] + _CRC.pack(zlib.crc32(footer[:-4])))


class _Table:
    def __init__(self, path):
        self.path = path
        self.keys, self.offsets = [], []
        try:
            with path.open("rb") as file:
                end = file.seek(0, os.SEEK_END) - _FOOTER.size
                if end < 0:
                    raise ValueError("truncated footer")
                file.seek(end)
                footer = file.read(_FOOTER.size)
                magic, index, bloom, icrc, bcrc, fcrc = _FOOTER.unpack(footer)
                if (magic != _MAGIC or zlib.crc32(footer[:-4]) != fcrc
                        or not 0 <= index <= bloom < end):
                    raise ValueError("invalid footer")
                file.seek(index)
                index_bytes = file.read(bloom - index)
                bits = file.read(end - bloom)
                if zlib.crc32(index_bytes) != icrc or zlib.crc32(bits) != bcrc:
                    raise ValueError("index or Bloom filter CRC mismatch")
            pos = 0
            while pos < len(index_bytes):
                length, offset = _INDEX.unpack_from(index_bytes, pos)
                pos += _INDEX.size
                key = index_bytes[pos:pos + length]
                pos += length
                if pos > len(index_bytes) or offset >= index:
                    raise ValueError("invalid index entry")
                if self.keys and (key <= self.keys[-1] or offset <= self.offsets[-1]):
                    raise ValueError("unordered index")
                self.keys.append(key)
                self.offsets.append(offset)
            if (self.offsets or [index])[0] != 0:
                raise ValueError("index does not cover data")
            self.offsets.append(index)
            self.bloom = Bloom(bits)
        except (ValueError, struct.error, FileNotFoundError) as exc:
            raise CorruptionError(f"{path.name}: {exc}") from exc

    def _block(self, file, number):
        start, end = self.offsets[number:number + 2]
        file.seek(start)
        data = file.read(end - start)
        try:
            batches, good = read_records(data)
            if len(data) != end - start or good != len(data) or len(batches) != 1:
                raise CorruptionError("incomplete block or CRC mismatch")
            entries = list(_entries(batches[0]))
            keys = [key for key, _ in entries]
            if (not keys or keys[0] != self.keys[number]
                    or any(a >= b for a, b in zip(keys, keys[1:]))
                    or (number + 1 < len(self.keys) and keys[-1] >= self.keys[number + 1])):
                raise CorruptionError("block keys disagree with index")
            return entries
        except (CorruptionError, struct.error, IndexError) as exc:
            raise CorruptionError(f"{self.path.name}: data block at {start}: {exc}") from exc

    def find(self, key):
        if key in self.bloom and (number := bisect_right(self.keys, key) - 1) >= 0:
            with self.path.open("rb") as file:
                for candidate, value in self._block(file, number):
                    if candidate == key:
                        return True, value
        return False, None

    def rows(self, start=None, stop=None):
        first = 0 if start is None else max(0, bisect_right(self.keys, start) - 1)
        with self.path.open("rb") as file:
            for number in range(first, len(self.keys)):
                if stop is not None and self.keys[number] >= stop:
                    break
                for key, value in self._block(file, number):
                    if stop is not None and key >= stop:
                        return
                    if start is None or key >= start:
                        yield key, value


def _newest(sources):
    # heapq.merge is stable: for equal keys the newest source comes first.
    for _, versions in groupby(merge(*sources, key=_KEY), key=_KEY):
        yield next(versions)


class LSM:
    """Single-writer store. Reopen after an exception from a write or maintenance I/O."""

    def __init__(self, path, *, memtable_limit=64 * 1024, block_size=4096,
                 compact_at=4, crash_hook=None):
        if memtable_limit < 1 or block_size < 1 or compact_at < 2:
            raise ValueError("positive sizes and compact_at >= 2 required")
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        _fsync_dir(self.path.parent)
        self.memtable_limit, self.block_size, self.compact_at = memtable_limit, block_size, compact_at
        self._hook = crash_hook or (lambda step: None)
        self._keys, self._mem, self._size = [], {}, 0
        self._wal = None
        manifest = self.path / "MANIFEST"
        if not manifest.exists():
            # Only bootstrap when there are no published SSTables. A bootstrap
            # interrupted before publishing can leave an empty WAL and a temp manifest.
            wal = self._wal_path(0)
            if (any(self.path.glob("sst-*.sst"))
                    or any(p != wal or p.stat().st_size for p in self.path.glob("wal-*.log"))):
                raise CorruptionError("MANIFEST missing from an existing store")
            self._create_wal(0)
            self._publish(0, 0, [], "open")
        self._epoch, self._wal_id, table_ids = self._read_manifest(manifest)
        self._tables = [_Table(self._table_path(number)) for number in table_ids]
        wal = self._wal_path(self._wal_id)
        try:
            with wal.open("r+b") as file:
                data = file.read()
                batches, good = read_records(data)
                for batch in batches:
                    for key, value in _entries(batch):
                        self._remember(key, value)
                if good != len(data):
                    file.truncate(good)
                    _sync(file)
        except (FileNotFoundError, struct.error, IndexError) as exc:
            raise CorruptionError(f"{wal.name}: {exc}") from exc
        # Persist the observed manifest before deleting its predecessor's files.
        _fsync_dir(self.path)
        self._cleanup()
        self._wal = wal.open("ab")

    @staticmethod
    def _read_manifest(path):
        try:
            data = path.read_bytes()
            (crc,) = _CRC.unpack_from(data)
            if zlib.crc32(data[4:]) != crc:
                raise ValueError("CRC mismatch")
            epoch, wal, tables = json.loads(data[4:])
            if (type(epoch) is not int or type(wal) is not int or not 0 <= wal <= epoch
                    or not isinstance(tables, list)
                    or any(type(number) is not int or not 0 < number <= epoch for number in tables)
                    or tables != sorted(set(tables), reverse=True)):
                raise ValueError("invalid generations")
            return epoch, wal, tables
        except (ValueError, TypeError, struct.error) as exc:
            raise CorruptionError(f"MANIFEST: {exc}") from exc

    def _wal_path(self, number):
        return self.path / f"wal-{number}.log"

    def _table_path(self, number):
        return self.path / f"sst-{number}.sst"

    def _cleanup(self, phase=None):
        live = {self._wal_path(self._wal_id), *(table.path for table in self._tables)}
        for path in sorted(self.path.iterdir()):
            if _OWNED.fullmatch(path.name) and path not in live:
                path.unlink()
                if phase is not None:
                    self._hook(f"{phase}:obsolete-unlinked")
        _fsync_dir(self.path)
        if phase is not None:
            self._hook(f"{phase}:cleanup-synced")

    def _publish(self, epoch, wal, tables, phase):
        payload = json.dumps([epoch, wal, tables]).encode("ascii")
        temp = self.path / "MANIFEST.tmp"
        with temp.open("wb") as file:
            file.write(_CRC.pack(zlib.crc32(payload)) + payload)
            _sync(file)
        self._hook(f"{phase}:manifest-written")
        temp.replace(self.path / "MANIFEST")
        self._hook(f"{phase}:manifest-renamed")
        _fsync_dir(self.path)
        self._hook(f"{phase}:manifest-synced")

    def _create_wal(self, number, phase=None):
        with self._wal_path(number).open("wb") as file:
            _sync(file)
        if phase is not None:
            self._hook(f"{phase}:wal-created")
        _fsync_dir(self.path)
        if phase is not None:
            self._hook(f"{phase}:wal-synced")

    def _install_table(self, number, entries, phase):
        path = self._table_path(number)
        temp = path.with_suffix(".sst.tmp")
        with temp.open("wb") as file:
            _write_table(file, entries, self.block_size)
            _sync(file)
        self._hook(f"{phase}:table-written")
        temp.replace(path)
        self._hook(f"{phase}:table-renamed")
        _fsync_dir(self.path)
        self._hook(f"{phase}:table-synced")
        return _Table(path)

    def _remember(self, key, value):
        if key in self._mem:
            self._size -= len(self._mem[key] or b"")
        else:
            self._keys.insert(bisect_left(self._keys, key), key)
            self._size += 9 + len(key)
        self._mem[key] = value
        self._size += len(value or b"")

    def _check_open(self):
        if self._wal is None:
            raise ValueError("store closed; reopen to continue")

    def _write(self, key, value):
        self._check_open()
        record = encode_batch(_operations([(key, value)]))
        try:
            self._wal.write(record)
            _sync(self._wal)
            self._remember(key, value)
            if self._size > self.memtable_limit:
                self.flush()
        except BaseException:
            self.close()
            raise

    def put(self, key: bytes, value: bytes):
        if not isinstance(key, bytes) or not isinstance(value, bytes):
            raise TypeError("keys and values must be bytes")
        self._write(key, value)

    def delete(self, key: bytes):
        if not isinstance(key, bytes):
            raise TypeError("keys must be bytes")
        self._write(key, None)

    def get(self, key: bytes, default=None):
        self._check_open()
        if key in self._mem:
            value = self._mem[key]
            return default if value is None else value
        for table in self._tables:
            found, value = table.find(key)
            if found:
                return default if value is None else value
        return default

    def scan(self, start: bytes | None = None, stop: bytes | None = None):
        """Yield live rows in [start, stop). Consume before mutating or closing."""
        self._check_open()
        left = 0 if start is None else bisect_left(self._keys, start)
        right = len(self._keys) if stop is None else bisect_left(self._keys, stop)
        memory = ((key, self._mem[key]) for key in self._keys[left:right])
        sources = [memory, *(table.rows(start, stop) for table in self._tables)]
        return ((key, value) for key, value in _newest(sources) if value is not None)

    def flush(self):
        self._check_open()
        if not self._keys:
            return
        try:
            new = self._epoch + 1
            table = self._install_table(new, ((k, self._mem[k]) for k in self._keys), "flush")
            self._create_wal(new, "flush")
            old_ids = [int(t.path.stem.split("-")[1]) for t in self._tables]
            self._publish(new, new, [new, *old_ids], "flush")
            self.close()
            self._wal = self._wal_path(new).open("ab")
            self._epoch = self._wal_id = new
            self._tables.insert(0, table)
            self._keys.clear()
            self._mem.clear()
            self._size = 0
            self._cleanup("flush")
            if len(self._tables) >= self.compact_at:
                self.compact()
        except BaseException:
            self.close()
            raise

    def compact(self):
        self._check_open()
        if len(self._tables) < 2:
            return
        try:
            new = self._epoch + 1
            rows = _newest([table.rows() for table in self._tables])
            # Every older table participates. No older value can outlive its tombstone.
            table = self._install_table(new, ((k, v) for k, v in rows if v is not None), "compact")
            self._publish(new, self._wal_id, [new], "compact")
            self._epoch, self._tables = new, [table]
            self._cleanup("compact")
        except BaseException:
            self.close()
            raise

    def close(self):
        wal, self._wal = self._wal, None
        if wal is not None:
            wal.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
