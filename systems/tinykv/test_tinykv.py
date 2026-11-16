import os
import random
import subprocess
import sys
import time

import pytest

from tinykv import DELETE, PUT, CorruptionError, TinyKV, encode_batch, read_records


def _ops(seed: int, n: int):
    """Random puts/deletes/batches over a small key space, so overwrites and deletes are common."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        batch = []
        for _ in range(rng.choice([1, 1, 1, 3])):
            key = f"k{rng.randrange(12)}".encode()
            if rng.random() < 0.25:
                batch.append((DELETE, key, b""))
            else:
                batch.append((PUT, key, os.urandom(rng.randrange(0, 40))))
        out.append(batch)
    return out


def _model(batches):
    state = {}
    for ops in batches:
        for op, k, v in ops:
            if op == PUT:
                state[k] = v
            else:
                state.pop(k, None)
    return state


def _write(path, batches, sync=False):
    """Writes batches; returns the WAL size after each one (sync off: these tests don't need fsync)."""
    ends = []
    with TinyKV(path, sync=sync) as db:
        for ops in batches:
            db.write_batch(ops)
            ends.append(os.path.getsize(os.path.join(path, "wal-0.log")))
    return ends


def test_put_get_delete_and_reopen(tmp_path):
    with TinyKV(str(tmp_path)) as db:
        db.put(b"a", b"1")
        db.put(b"b", b"2")
        db.put(b"a", b"3")
        db.delete(b"b")
        db.delete(b"missing")
        assert db.get(b"a") == b"3" and db.get(b"b") is None
    with TinyKV(str(tmp_path)) as db:
        assert dict(db.items()) == {b"a": b"3"}


def test_batch_is_all_or_nothing_across_a_torn_write(tmp_path):
    batches = [[(PUT, b"x", b"0")], [(PUT, b"x", b"1"), (PUT, b"y", b"1"), (DELETE, b"z", b"")]]
    ends = _write(str(tmp_path), batches)
    wal = tmp_path / "wal-0.log"
    data = wal.read_bytes()
    for cut in range(ends[0], ends[1]):
        wal.write_bytes(data[:cut])
        with TinyKV(str(tmp_path)) as db:
            assert dict(db.items()) == {b"x": b"0"}
        assert wal.stat().st_size == ends[0]  # torn tail truncated away


@pytest.mark.parametrize("seed", range(5))
def test_every_truncation_recovers_exactly_the_committed_prefix(tmp_path, seed):
    """Cut the WAL at every byte offset: recovery must give the state after the last complete batch."""
    batches = _ops(seed, 40)
    ends = _write(str(tmp_path), batches)
    wal = tmp_path / "wal-0.log"
    data = wal.read_bytes()
    for cut in range(len(data) + 1):
        wal.write_bytes(data[:cut])
        committed = sum(1 for e in ends if e <= cut)
        with TinyKV(str(tmp_path)) as db:
            assert dict(db.items()) == _model(batches[:committed]), f"cut={cut}"
        # recovery truncated the tail, so new writes append after the last good record
        assert wal.stat().st_size == (ends[committed - 1] if committed else 0)


def test_zero_filled_tail_is_treated_as_torn(tmp_path):
    batches = _ops(1, 10)
    _write(str(tmp_path), batches)
    wal = tmp_path / "wal-0.log"
    wal.write_bytes(wal.read_bytes() + b"\0" * 300)
    with TinyKV(str(tmp_path)) as db:
        assert dict(db.items()) == _model(batches)


def test_bit_flips_never_silently_lose_acknowledged_writes(tmp_path):
    """Flip bits across the whole log. Every flip must raise CorruptionError, except a flip inside the last
    record's payload, which looks exactly like a torn write: that drops only the newest batch. Never a wrong state."""
    batches = _ops(2, 12)
    ends = _write(str(tmp_path), batches)
    wal = tmp_path / "wal-0.log"
    data = wal.read_bytes()
    without_last = _model(batches[:-1])
    # Header: length (4) + length CRC (4) + payload CRC (4). A flip in the payload CRC or the payload of the
    # last record shows up as a payload mismatch on the newest record, which is handled like a torn write.
    last_payload = ends[-2] + 8
    for i in range(len(data)):
        for bit in (0, 3, 7):
            flipped = bytearray(data)
            flipped[i] ^= 1 << bit
            wal.write_bytes(bytes(flipped))
            try:
                with TinyKV(str(tmp_path)) as db:
                    got = dict(db.items())
            except CorruptionError:
                # A crash cannot produce a complete header with a wrong checksum, so a flip in the last
                # record's header is corruption too; only a flip in the last payload may pass as torn.
                assert i < last_payload, f"flip at {i} in the last payload should be treated as a torn write"
                continue
            assert i >= last_payload, f"flip at {i} (outside the last payload) was not detected"
            assert got == without_last


@pytest.mark.parametrize("step", ["snapshot-written", "new-wal-created", "snapshot-renamed", "old-files-removed"])
def test_crash_at_each_checkpoint_step_loses_nothing(tmp_path, step):
    class Crash(Exception):
        pass

    def hook(s):
        if s == step:
            raise Crash

    batches = _ops(3, 30)
    db = TinyKV(str(tmp_path), crash_hook=hook)
    for ops in batches[:20]:
        db.write_batch(ops)
    with pytest.raises(Crash):
        db.checkpoint()
    db._wal.close()  # the "process" dies; nothing else is flushed
    with TinyKV(str(tmp_path)) as db2:
        assert dict(db2.items()) == _model(batches[:20])
        for ops in batches[20:]:
            db2.write_batch(ops)
    with TinyKV(str(tmp_path)) as db3:
        assert dict(db3.items()) == _model(batches)
        files = sorted(os.listdir(tmp_path))
        assert not any(f.endswith(".tmp") for f in files), files
        assert len([f for f in files if f.startswith("wal-")]) == 1, files


def test_checkpoint_then_more_writes_then_reopen(tmp_path):
    batches = _ops(4, 60)
    with TinyKV(str(tmp_path)) as db:
        for i, ops in enumerate(batches):
            db.write_batch(ops)
            if i % 15 == 14:
                db.checkpoint()
    with TinyKV(str(tmp_path)) as db:
        assert dict(db.items()) == _model(batches)
        assert db.gen == 4
    assert sorted(os.listdir(tmp_path)) == ["snapshot-4", "wal-4.log"]


def test_damaged_snapshot_raises(tmp_path):
    with TinyKV(str(tmp_path)) as db:
        db.put(b"a", b"1")
        db.checkpoint()
    snap = tmp_path / "snapshot-1"
    snap.write_bytes(snap.read_bytes()[:-1])
    with pytest.raises(CorruptionError):
        TinyKV(str(tmp_path))


def test_record_roundtrip():
    ops = [(PUT, b"", b""), (PUT, b"k" * 1000, os.urandom(5000)), (DELETE, b"k", b"")]
    batches, good = read_records(encode_batch(ops) * 2)
    assert batches == [ops, ops] and good == 2 * len(encode_batch(ops))


WRITER = r"""
import sys
sys.path.insert(0, sys.argv[2])
from tinykv import TinyKV
db = TinyKV(sys.argv[1], sync=True)
i = 0
while True:
    db.put(b"key%06d" % i, b"value%06d" % i)
    if i % 50 == 49:
        db.checkpoint()
    print(i, flush=True)   # acknowledged only after put() (and its fsync) returned
    i += 1
"""


@pytest.mark.parametrize("delay", [0.3, 0.7, 1.2])
def test_killed_writer_keeps_every_acknowledged_write(tmp_path, delay):
    """A real process is killed (SIGKILL / TerminateProcess) mid-stream, also during checkpoints.
    This tests process crashes; power loss also depends on the disk honouring fsync."""
    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    data = tmp_path / "db"
    proc = subprocess.Popen([sys.executable, str(script), str(data), os.path.dirname(os.path.abspath(__file__))],
                            stdout=subprocess.PIPE, text=True)
    time.sleep(delay)
    proc.kill()
    out, _ = proc.communicate()
    acked = [int(x) for x in out.split()]
    assert acked, "writer acknowledged nothing before it was killed; raise the delay"
    with TinyKV(str(data)) as db:
        state = dict(db.items())
    n = acked[-1] + 1
    for i in range(n):
        assert state[b"key%06d" % i] == b"value%06d" % i
    assert len(state) in (n, n + 1)  # at most one write that was durable but not yet acknowledged
