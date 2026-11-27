import random

import pytest

from systems.lsm.lsm import Bloom, CorruptionError, LSM
from systems.tinykv.tinykv import PUT, encode_batch


def assert_model(db, model, keys):
    assert list(db.scan()) == sorted(model.items())
    for key in keys:
        assert db.get(key) == model.get(key)


@pytest.mark.parametrize("seed", range(4))
def test_random_workload(tmp_path, seed):
    rng = random.Random(seed)
    options = dict(memtable_limit=132, block_size=70, compact_at=4)
    keys = [b"", *(bytes([i]) for i in range(32))]
    model, steps = {}, []
    db = LSM(tmp_path, crash_hook=steps.append, **options)
    try:
        for i in range(450):
            key = rng.choice(keys)
            if rng.random() < 0.35:
                db.delete(key)
                model.pop(key, None)
            else:
                value = rng.randbytes(rng.randrange(35))
                db.put(key, value)
                model[key] = value
            if i % 13 == 0:
                assert_model(db, model, keys)
                start, stop = rng.choices(keys, k=2)
                expected = sorted((k, v) for k, v in model.items() if start <= k < stop)
                assert list(db.scan(start, stop)) == expected
            if i % 71 == 0:
                db.close()
                db = LSM(tmp_path, crash_hook=steps.append, **options)
                assert_model(db, model, keys)
        assert_model(db, model, keys)
        assert steps.count("flush:manifest-synced") >= 20
        assert steps.count("compact:manifest-synced") >= 5
    finally:
        db.close()


class Crash(Exception):
    pass


TABLE = ["table-written", "table-renamed", "table-synced"]
MANIFEST = ["manifest-written", "manifest-renamed", "manifest-synced"]
CLEANUP = ["obsolete-unlinked", "cleanup-synced"]


@pytest.mark.parametrize(("phase", "step", "occurrence"), [
    *(("flush", step, 1) for step in TABLE + ["wal-created", "wal-synced"] + MANIFEST + CLEANUP),
    *(("compact", step, 1) for step in TABLE + MANIFEST + CLEANUP),
    ("compact", "obsolete-unlinked", 2),
])
def test_crash_at_each_step(tmp_path, phase, step, occurrence):
    observed = 0

    def hook(event):
        nonlocal observed
        if event == f"{phase}:{step}":
            observed += 1
            if observed == occurrence:
                raise Crash

    model = {b"same": b"new", b"keep": b"", b"pending": b"memtable"}
    keys = [*model, b"gone", b"absent"]
    with LSM(tmp_path, compact_at=20) as db:
        db.put(b"same", b"old")
        db.put(b"gone", b"old")
        db.flush()
        db.put(b"same", b"new")
        db.put(b"keep", b"")
        db.delete(b"gone")
        db.flush()
        db.put(b"pending", b"memtable")

    with LSM(tmp_path, crash_hook=hook, compact_at=20) as db:
        with pytest.raises(Crash):
            db.flush() if phase == "flush" else db.compact()

    with LSM(tmp_path) as db:
        assert_model(db, model, keys)
        assert not list(tmp_path.glob("*.tmp"))
        assert len(list(tmp_path.glob("wal-*.log"))) == 1
        assert sorted(tmp_path.glob("sst-*.sst")) == sorted(t.path for t in db._tables)
        db.delete(b"same")
        db.flush()
        db.compact()
    model.pop(b"same")
    with LSM(tmp_path) as db:
        assert_model(db, model, keys)


def test_torn_wal_tail(tmp_path):
    with LSM(tmp_path) as db:
        db.put(b"on disk", b"value")
        db.flush()
        db.put(b"in wal", b"value")
    wal, = tmp_path.glob("wal-*.log")
    prefix = wal.read_bytes()
    record = encode_batch([(PUT, b"in wal", b"torn overwrite")])
    for end in range(len(record)):
        wal.write_bytes(prefix + record[:end])
        with LSM(tmp_path) as db:
            assert_model(db, {b"on disk": b"value", b"in wal": b"value"}, [b"in wal"])
        assert wal.read_bytes() == prefix
    with LSM(tmp_path) as db:
        db.put(b"after recovery", b"value")
    with LSM(tmp_path) as db:
        assert db.get(b"after recovery") == b"value"


@pytest.mark.parametrize("offset", [0, 8, 12])
def test_corrupt_block_raises_instead_of_falling_back(tmp_path, offset):
    with LSM(tmp_path, block_size=1) as db:
        db.put(b"a", b"old")
        db.flush()
        db.put(b"a", b"new")
        db.put(b"b", b"intact")
        db.flush()
    table = tmp_path / "sst-2.sst"
    data = bytearray(table.read_bytes())
    data[offset] ^= 1
    table.write_bytes(data)
    with LSM(tmp_path) as db:
        assert db.get(b"b") == b"intact"
        with pytest.raises(CorruptionError, match=r"sst-2.sst: data block at 0"):
            db.get(b"a")
        with pytest.raises(CorruptionError, match="data block"):
            list(db.scan())
        with pytest.raises(CorruptionError, match="data block"):
            db.compact()


@pytest.mark.parametrize("region", ["footer", "index", "bloom"])
def test_corrupt_table_metadata_raises_on_open(tmp_path, region):
    with LSM(tmp_path) as db:
        db.put(b"k", b"v")
        db.flush()
    table, = tmp_path.glob("sst-*.sst")
    data = bytearray(table.read_bytes())
    offset = {"footer": -1, "bloom": -37, "index": len(encode_batch([(PUT, b"k", b"v")]))}
    data[offset[region]] ^= 1
    table.write_bytes(data)
    with pytest.raises(CorruptionError, match="sst-1.sst"):
        LSM(tmp_path)


def test_all_deleted_and_range_boundaries(tmp_path):
    with LSM(tmp_path, block_size=1) as db:
        db.put(b"a", b"")
        db.put(b"b", b"2")
        db.put(b"c", b"3")
        db.flush()
        db.delete(b"b")
        assert list(db.scan(stop=b"c")) == [(b"a", b"")]
        assert list(db.scan(start=b"b")) == [(b"c", b"3")]
        assert not list(db.scan(b"c", b"c"))
        db.delete(b"a")
        db.delete(b"c")
        db.flush()
        db.compact()
    with LSM(tmp_path) as db:
        assert not list(db.scan())
        assert all(db.get(key) is None for key in (b"a", b"b", b"c"))


def test_bloom_false_positive_rate():
    keys = [f"present:{i}".encode() for i in range(10_000)]
    bloom = Bloom.build(keys)
    assert all(key in bloom for key in keys)
    trials = 100_000
    positives = sum(f"absent:{i}".encode() in bloom for i in range(trials))
    print(f"Bloom false positives: {positives}/{trials} ({positives / trials:.3%})")
    assert positives / trials < 0.02
