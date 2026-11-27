"""Run with pytest or directly from the repository root."""
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from systems.btree import btree
from systems.btree.btree import BTree, CorruptionError, PAGE_SIZE


def key(number):
    return number.to_bytes(2, "big")


def check_tree(db, expected):
    seen, leaves, depths = set(), [], set()

    def visit(number, depth=0):
        assert number not in seen
        seen.add(number)
        node = db._node(number)
        keys = [k for k, _ in node.rows]
        assert keys == sorted(set(keys))
        minimum = 4 if depth else (2 if node.kind == btree._BRANCH else 0)
        assert minimum <= len(keys) <= 8
        if node.kind == btree._LEAF:
            leaves.append((number, node))
            depths.add(depth)
            return list(node.rows)
        rows = []
        for lower, child in node.rows:
            subtree = visit(child, depth + 1)
            assert subtree[0][0] == lower
            assert not rows or rows[-1][0] < lower
            rows.extend(subtree)
        return rows

    assert visit(db._root) == sorted(expected.items())
    assert len(depths) == 1
    links = [number for number, _ in leaves[1:]] + [0]
    assert [node.link for _, node in leaves] == links
    number = db._free
    while number:
        assert number not in seen
        seen.add(number)
        page = db._load(number)
        assert page.kind == btree._FREE
        number = page.link
    assert seen == set(range(1, db._end))
    assert len(db._cache) <= db._limit
    assert not db._dirty
    return depths.pop()


def expect(error, action):
    try:
        action()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def test_randomized_model():
    rng = random.Random(1729)
    expected = {}
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "tree"
        for _ in range(10):
            with BTree(path, cache_size=3) as db:
                check_tree(db, expected)
                for i in range(1000):
                    k = key(rng.randrange(600))
                    op = rng.randrange(10)
                    if op < 4:
                        value = rng.randbytes(rng.randrange(497))
                        db.put(k, value)
                        expected[k] = value
                    elif op < 7:
                        db.delete(k)
                        expected.pop(k, None)
                    elif op < 9:
                        assert db.get(k, False) == expected.get(k, False)
                    else:
                        end = key(rng.randrange(600))
                        assert list(db.scan(k, end)) == [
                            (k2, expected[k2]) for k2 in sorted(expected) if k <= k2 < end
                        ]
                    if i % 100 == 0:
                        check_tree(db, expected)
                check_tree(db, expected)


def test_splits_merges_and_page_reuse():
    with TemporaryDirectory() as tmp, BTree(Path(tmp) / "tree", cache_size=1) as db:
        expected = {}
        # Force redistribution of a 3-entry page with a 6-entry neighbor.
        for n in range(10):
            db.put(key(n), b"")
            expected[key(n)] = b""
        db.delete(key(0))
        del expected[key(0)]
        assert not db._free
        assert [len(db._node(pid).rows) for _, pid in db._node(db._root).rows] == [4, 5]
        for n in range(10, 210):
            db.put(key(n), b"")
            expected[key(n)] = b""
        assert check_tree(db, expected) >= 3
        allocated = db._end
        for n in range(1, 210):
            db.delete(key(n))
            del expected[key(n)]
            check_tree(db, expected)
        assert check_tree(db, {}) == 0
        for n in range(210):
            db.put(key(n), b"")
            expected[key(n)] = b""
        assert db._end <= allocated + 1
        check_tree(db, expected)


def test_range_and_reopen():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "tree"
        expected = {key(n): b"" if n % 2 else bytes([n]) for n in range(40)}
        expected[b""] = b"empty key"
        expected[b"\xff" * 498] = b""
        with BTree(path) as db:
            for k, value in expected.items():
                db.put(k, value)
            db.put(key(2), b"replacement")
            expected[key(2)] = b"replacement"
            db.delete(key(4))
            del expected[key(4)]
            expect(ValueError, lambda: db.put(b"x" * 499, b""))
        with BTree(path) as db:
            check_tree(db, expected)
            for start, stop in [(None, None), (None, b""), (key(5), key(25)),
                                (key(25), None), (key(3), key(3)), (key(5), key(3))]:
                assert list(db.scan(start, stop)) == [
                    (k, v) for k, v in sorted(expected.items())
                    if (start is None or start <= k) and (stop is None or k < stop)
                ]
            assert db.get(b"") == b"empty key"
            assert db.get(key(4)) is None
        # Fill an internal page with maximum-length separators.
        with BTree(Path(tmp) / "wide") as db:
            wide = {key(n) + bytes(496): b"" for n in range(36)}
            for k, value in wide.items():
                db.put(k, value)
        with BTree(Path(tmp) / "wide") as db:
            check_tree(db, wide)


class Crash(Exception):
    pass


def exercise_crashes(root, before, after, mutate):
    original, path = root / "original", root / "case"
    if before is not None:
        with BTree(original) as db:
            for k, value in before.items():
                db.put(k, value)

    def reset():
        for suffix in ("", ".wal", ".wal.tmp"):
            (root / ("case" + suffix)).unlink(missing_ok=True)
        if before is not None:
            shutil.copyfile(original, path)

    def run(hook):
        with BTree(path, crash_hook=hook) as db:
            mutate(db)

    def verify(model):
        with BTree(path) as db:
            check_tree(db, model or {})
            assert list(db.scan()) == sorted((model or {}).items())
            db.put(b"new", b"write after recovery")
        with BTree(path) as db:
            assert db.get(b"new") == b"write after recovery"

    reset()
    steps = []
    run(steps.append)
    published = steps.index("wal-synced")
    for i, target in enumerate(steps):
        reset()

        def interrupt(step):
            if step == target:
                raise Crash(step)

        expect(Crash, lambda: run(interrupt))
        verify(after if i >= published else before)

    real_write = btree._write
    writes = []
    reset()

    def record(file, data):
        writes.append(len(data))
        real_write(file, data)

    try:
        btree._write = record
        run(None)
        for i, size in enumerate(writes):
            for prefix in (1, size // 2, size - 1):
                reset()
                remaining = iter(range(len(writes)))

                def torn(file, data):
                    if next(remaining) == i:
                        real_write(file, data[:prefix])
                        raise Crash("torn write")
                    real_write(file, data)

                btree._write = torn
                expect(Crash, lambda: run(None))
                btree._write = real_write
                # Interrupt recovery too. The published WAL remains replayable.
                if i >= 2:
                    expect(Crash, lambda: run(
                        lambda step: interrupt_replay(step)))
                verify(after if i >= 2 else before)
    finally:
        btree._write = real_write


def interrupt_replay(step):
    if step.startswith("page-written:"):
        raise Crash(step)


def test_crash_and_torn_writes():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases = [
            (None, {}, lambda db: None),
            ({key(n): b"x" for n in range(8)}, {key(n): b"x" for n in range(9)},
             lambda db: db.put(key(8), b"x")),
            ({key(n): b"x" for n in range(9)}, {key(n): b"x" for n in range(1, 9)},
             lambda db: db.delete(key(0))),
        ]
        for i, (before, after, mutate) in enumerate(cases):
            directory = root / str(i)
            directory.mkdir()
            exercise_crashes(directory, before, after, mutate)


def test_power_loss_discards_unsynced_writes():
    before = {key(n): b"x" for n in range(8)}
    after = {**before, key(8): b"x"}
    cases = (
        ("wal-written", before),
        ("wal-synced", after),
        ("page-written:1", after),
        ("wal-truncated", after),
        ("cleanup-synced", after),
    )
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "tree"
        wal = path.with_suffix(".wal")
        for target, expected in cases:
            with BTree(path) as db:
                for k, value in before.items():
                    db.put(k, value)

            # Start with a persisted baseline, then take independent snapshots
            # only when each file is actually synced, including WAL truncation.
            persisted = {file: file.read_bytes() for file in (path, wal)}
            real_sync = btree._sync

            def sync(file):
                real_sync(file)
                filename = Path(file.name)
                persisted[filename] = filename.read_bytes()

            def interrupt(step):
                if step == target:
                    raise Crash(step)

            try:
                btree._sync = sync
                with BTree(path, crash_hook=interrupt) as db:
                    expect(Crash, lambda: db.put(key(8), b"x"))
            finally:
                btree._sync = real_sync

            # Reconstruct the disk image after dropping all unsynced changes.
            for filename, contents in persisted.items():
                filename.write_bytes(contents)
            with BTree(path) as db:
                assert list(db.scan()) == sorted(expected.items()), target
                check_tree(db, expected)
            path.unlink()


def test_crc_corruption():
    def flip(path, offset):
        with path.open("r+b") as file:
            file.seek(offset)
            byte = file.read(1)
            file.seek(offset)
            file.write(bytes([byte[0] ^ 1]))

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "tree"
        with BTree(path) as db:
            for n in range(20):
                db.put(key(n), b"data")
            leaf, _, _ = db._find(key(19))
        flip(path, leaf * PAGE_SIZE + 100)
        with BTree(path) as db:
            expect(CorruptionError, lambda: db.get(key(19)))
            expect(CorruptionError, lambda: list(db.scan()))
        flip(path, 100)
        expect(CorruptionError, lambda: BTree(path))


if __name__ == "__main__":
    tests = [test for name, test in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
