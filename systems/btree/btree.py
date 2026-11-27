"""A disk-backed B+tree for byte keys and values."""
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import zlib

from systems.tinykv.tinykv import CorruptionError, _fsync_dir

PAGE_SIZE = 4096
_HEADER = struct.Struct("<4sBQHQ")  # magic, kind, page ID, item count, link
_CRC = struct.Struct("<I")
_COMMIT = struct.Struct("<4sI")
_ID = struct.Struct("<Q")
_LENGTH = struct.Struct("<H")
_META = struct.Struct("<QQQ")  # root, next unallocated page, free-list head
_LEAF, _BRANCH, _FREE, _SUPER = range(4)
_MAX_ITEM = 498  # eight branch records of at most 508 bytes fit in a page
_MAX_SLOTS = 8


@dataclass(frozen=True)
class _Node:
    kind: int
    rows: tuple = ()
    link: int = 0


def _page(number, kind, count, link, payload):
    body = (_HEADER.pack(b"BPT1", kind, number, count, link) + payload)
    body = body.ljust(PAGE_SIZE - _CRC.size, b"\0")
    return body + _CRC.pack(zlib.crc32(body))


def _encode(number, node):
    parts = []
    for key, value in node.rows:
        parts.extend((_LENGTH.pack(len(key)), key))
        if node.kind == _LEAF:
            parts.extend((_LENGTH.pack(len(value)), value))
        else:
            parts.append(_ID.pack(value))
    return _page(number, node.kind, len(node.rows), node.link, b"".join(parts))


def _decode(number, data):
    try:
        if (len(data) != PAGE_SIZE
                or zlib.crc32(data[:-4]) != _CRC.unpack_from(data, PAGE_SIZE - 4)[0]):
            raise ValueError("short page or CRC mismatch")
        magic, kind, stored, count, link = _HEADER.unpack_from(data)
        if magic != b"BPT1" or stored != number:
            raise ValueError("invalid page header")
        if number == 0:
            if kind != _SUPER or count or link:
                raise ValueError("invalid superblock")
            root, end, free = _META.unpack_from(data, _HEADER.size)
            if not 0 < root < end or free >= end or free == root:
                raise ValueError("invalid superblock pointers")
            return root, end, free
        if (kind not in (_LEAF, _BRANCH, _FREE) or count > _MAX_SLOTS
                or (kind == _FREE and count) or (kind == _BRANCH and (not count or link))):
            raise ValueError("invalid node header")
        pos = _HEADER.size
        rows = []
        for _ in range(count):
            size = _LENGTH.unpack_from(data, pos)[0]
            pos += _LENGTH.size
            key = data[pos:pos + size]
            pos += size
            if kind == _LEAF:
                length = _LENGTH.unpack_from(data, pos)[0]
                pos += _LENGTH.size
                value = data[pos:pos + length]
                pos += length
            else:
                length = 0
                value = _ID.unpack_from(data, pos)[0]
                pos += _ID.size
            if (size + length > _MAX_ITEM or pos > PAGE_SIZE - _CRC.size
                    or (rows and rows[-1][0] >= key)):
                raise ValueError("invalid node entry")
            rows.append((key, value))
        return _Node(kind, tuple(rows), link)
    except (ValueError, struct.error) as exc:
        raise CorruptionError(f"page {number}: {exc}") from exc


def _sync(file):
    os.fsync(file.fileno())


def _write(file, data):
    if file.write(data) != len(data):
        raise OSError("short write")


def _create_empty(path, temp):
    # Establish the filename before relying on file syncs for subsequent writes.
    with temp.open("wb", buffering=0) as file:
        _sync(file)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        MOVEFILE_WRITE_THROUGH = 8
        if not move(str(temp), str(path), MOVEFILE_WRITE_THROUGH):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        temp.rename(path)
        _fsync_dir(path.parent)


def _batch_crc(payload):
    # Including each page's own CRC would give a fixed residue per page.
    crc = 0
    for pos in range(0, len(payload), PAGE_SIZE):
        crc = zlib.crc32(payload[pos:pos + PAGE_SIZE - _CRC.size], crc)
    return crc


class BTree:
    """Single-writer B+tree. Each successful mutation is durable."""

    def __init__(self, path, *, cache_size=16, crash_hook=None):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.path = Path(path)
        self._wal = self.path.with_name(self.path.name + ".wal")
        self._temp = self.path.with_name(self.path.name + ".wal.tmp")
        self._hook = crash_hook or (lambda step: None)
        self._limit = cache_size
        self._cache = OrderedDict()
        self._dirty = {}
        self._file = None
        try:
            if not self._wal.exists():
                _create_empty(self._wal, self._temp)
            pages = self._read_wal()
            if pages:
                self._install(pages)
            elif self._wal.stat().st_size:
                self._clear_wal()
            if not self.path.exists():
                self._root, self._end, self._free = 1, 2, 0
                self._dirty[1] = _Node(_LEAF)
                self._commit()
            if self._file is None:
                self._file = self.path.open("r+b", buffering=0)
            self._file.seek(0)
            self._root, self._end, self._free = _decode(0, self._file.read(PAGE_SIZE))
            if self._file.seek(0, os.SEEK_END) != self._end * PAGE_SIZE:
                raise CorruptionError("file size disagrees with superblock")
            self._node(self._root)
            self._temp.unlink(missing_ok=True)
        except BaseException:
            self.close()
            raise

    def _read_wal(self):
        data = self._wal.read_bytes()
        if (len(data) >= _CRC.size + 2 * PAGE_SIZE
                and (len(data) - _CRC.size) % PAGE_SIZE == 0
                and data[_CRC.size:_CRC.size + 4] == b"BPT1"):
            payload = data[_CRC.size:]  # WAL from the rename-based format
            if zlib.crc32(payload) != _CRC.unpack_from(data)[0]:
                raise CorruptionError("invalid legacy redo WAL")
        else:
            payload = data[:-_COMMIT.size]
            if (len(payload) < 2 * PAGE_SIZE or len(payload) % PAGE_SIZE
                    or _COMMIT.unpack_from(data, len(payload)) != (b"CMIT", _batch_crc(payload))):
                return {}
        pages = {}
        for pos in range(0, len(payload), PAGE_SIZE):
            page = payload[pos:pos + PAGE_SIZE]
            number = _HEADER.unpack_from(page)[2]
            if number in pages:
                raise CorruptionError("duplicate page in redo WAL")
            _decode(number, page)
            pages[number] = page
        if 0 not in pages or max(pages) >= _decode(0, pages[0])[1]:
            raise CorruptionError("invalid page IDs in redo WAL")
        return pages

    def _install(self, pages):
        if self._file is None:
            if not self.path.exists():
                _create_empty(self.path, self._temp)
            self._file = self.path.open("r+b", buffering=0)
        for number, data in pages.items():
            self._file.seek(number * PAGE_SIZE)
            _write(self._file, data)
            self._hook(f"page-written:{number}")
        _sync(self._file)
        self._hook("data-synced")
        self._clear_wal()

    def _clear_wal(self):
        with self._wal.open("r+b", buffering=0) as file:
            file.truncate(0)
            self._hook("wal-truncated")
            _sync(file)
        self._hook("cleanup-synced")

    def _commit(self):
        pages = {number: _encode(number, node) for number, node in self._dirty.items()}
        meta = _META.pack(self._root, self._end, self._free)
        pages[0] = _page(0, _SUPER, 0, 0, meta)
        payload = b"".join(pages.values())
        with self._wal.open("r+b", buffering=0) as file:
            _write(file, payload)
            self._hook("wal-written")
            _write(file, _COMMIT.pack(b"CMIT", _batch_crc(payload)))
            _sync(file)
            self._hook("wal-synced")
        self._install(pages)
        for number, node in self._dirty.items():
            self._remember(number, node)
        self._dirty.clear()

    def _remember(self, number, node):
        self._cache[number] = node
        self._cache.move_to_end(number)
        if len(self._cache) > self._limit:
            self._cache.popitem(last=False)

    def _load(self, number):
        if self._file is None:
            raise ValueError("store closed; reopen to continue")
        if not 0 < number < self._end:
            raise CorruptionError(f"invalid page pointer {number}")
        if number in self._dirty:
            return self._dirty[number]
        if number in self._cache:
            self._cache.move_to_end(number)
            return self._cache[number]
        self._file.seek(number * PAGE_SIZE)
        node = _decode(number, self._file.read(PAGE_SIZE))
        if (node.link >= self._end or node.link == number
                or (node.kind == _BRANCH
                    and any(not 0 < child < self._end or child == number
                            for _, child in node.rows))):
            raise CorruptionError(f"invalid pointer in page {number}")
        self._remember(number, node)
        return node

    def _node(self, number):
        node = self._load(number)
        if node.kind == _FREE:
            raise CorruptionError(f"tree points to free page {number}")
        return node

    def _allocate(self):
        if self._free:
            number = self._free
            node = self._load(number)
            if node.kind != _FREE:
                raise CorruptionError(f"free list points to live page {number}")
            self._free = node.link
        else:
            number = self._end
            self._end += 1
        return number

    def _release(self, number):
        self._dirty[number] = _Node(_FREE, link=self._free)
        self._free = number

    def _find(self, key):
        number = self._root
        path = []
        seen = set()
        while (node := self._node(number)).kind == _BRANCH:
            if number in seen:
                raise CorruptionError("cycle in tree")
            seen.add(number)
            keys = [item[0] for item in node.rows]
            index = 0 if key is None else max(0, bisect_right(keys, key) - 1)
            path.append((number, node, index))
            number = node.rows[index][1]
        return number, node, path

    @staticmethod
    def _index(rows, key):
        return bisect_left(rows, key, key=lambda row: row[0])

    @staticmethod
    def _check_key(key):
        if not isinstance(key, bytes):
            raise TypeError("keys must be bytes")

    def get(self, key, default=None):
        self._check_key(key)
        _, node, _ = self._find(key)
        index = self._index(node.rows, key)
        if index < len(node.rows) and node.rows[index][0] == key:
            return node.rows[index][1]
        return default

    def scan(self, start=None, stop=None):
        """Yield rows in [start, stop). Consume before mutating or closing."""
        for bound in (start, stop):
            if bound is not None:
                self._check_key(bound)
        _, node, _ = self._find(start)
        index = 0 if start is None else self._index(node.rows, start)
        while True:
            for key, value in node.rows[index:]:
                if stop is not None and key >= stop:
                    return
                yield key, value
            if not node.link:
                return
            sibling = self._node(node.link)
            if (sibling.kind != _LEAF or not sibling.rows or not node.rows
                    or node.rows[-1][0] >= sibling.rows[0][0]):
                raise CorruptionError("invalid leaf sibling")
            node, index = sibling, 0

    def put(self, key, value):
        self._check_key(key)
        if not isinstance(value, bytes):
            raise TypeError("values must be bytes")
        if len(key) + len(value) > _MAX_ITEM:
            raise ValueError(f"key and value must total at most {_MAX_ITEM} bytes")
        self._change(key, value)

    def delete(self, key):
        self._check_key(key)
        self._change(key, None)

    def _change(self, key, value):
        number, node, path = self._find(key)
        rows = list(node.rows)
        index = self._index(rows, key)
        exists = index < len(rows) and rows[index][0] == key
        if value is None:
            if not exists:
                return
            del rows[index]
        elif exists:
            rows[index] = (key, value)
        else:
            rows.insert(index, (key, value))
        try:
            self._propagate(number, _Node(_LEAF, tuple(rows), node.link), path)
            self._commit()
        except BaseException:
            self.close()
            raise

    def _split(self, number, node):
        right = self._allocate()
        mid = len(node.rows) // 2
        self._dirty[number] = _Node(node.kind, node.rows[:mid],
                                    right if node.kind == _LEAF else 0)
        self._dirty[right] = _Node(node.kind, node.rows[mid:], node.link)
        return [(node.rows[0][0], number), (node.rows[mid][0], right)]

    def _rebalance(self, number, node, parent, index):
        left_index = index - 1 if index else 0
        if index:
            left_id, left = parent[left_index][1], self._node(parent[left_index][1])
            right_id, right = number, node
        else:
            left_id, left = number, node
            right_id, right = parent[1][1], self._node(parent[1][1])
        if left.kind != right.kind or (left.kind == _LEAF and left.link != right_id):
            raise CorruptionError("invalid siblings during rebalance")
        rows = left.rows + right.rows
        if len(rows) <= _MAX_SLOTS:
            self._dirty[left_id] = _Node(node.kind, rows, right.link)
            self._release(right_id)
            parent[left_index:left_index + 2] = [(rows[0][0], left_id)]
        else:
            mid = len(rows) // 2
            self._dirty[left_id] = _Node(node.kind, rows[:mid], left.link)
            self._dirty[right_id] = _Node(node.kind, rows[mid:], right.link)
            parent[left_index:left_index + 2] = [(rows[0][0], left_id),
                                                 (rows[mid][0], right_id)]

    def _propagate(self, number, node, path):
        for parent_id, parent, index in reversed(path):
            rows = list(parent.rows)
            if len(node.rows) > _MAX_SLOTS:
                rows[index:index + 1] = self._split(number, node)
            elif len(node.rows) < _MAX_SLOTS // 2:
                if len(rows) < 2:
                    raise CorruptionError("parent has no sibling to rebalance")
                self._rebalance(number, node, rows, index)
            else:
                self._dirty[number] = node
                if rows[index][0] == node.rows[0][0]:
                    return
                rows[index] = (node.rows[0][0], number)
            number, node = parent_id, _Node(_BRANCH, tuple(rows))
        if len(node.rows) > _MAX_SLOTS:
            children = self._split(number, node)
            self._root = self._allocate()
            self._dirty[self._root] = _Node(_BRANCH, tuple(children))
        elif node.kind == _BRANCH and len(node.rows) == 1:
            self._root = node.rows[0][1]
            self._release(number)
        else:
            self._dirty[number] = node

    def close(self):
        file, self._file = self._file, None
        if file is not None:
            file.close()
        self._cache.clear()
        self._dirty.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
