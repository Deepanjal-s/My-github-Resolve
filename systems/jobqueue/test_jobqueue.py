import asyncio
import os
import random
import sqlite3
import subprocess
import sys
import time

import pytest

from jobqueue import JobQueue, Runner


class FakeClock:
    """Virtual time. sleep(d) wakes at now + d. When every task is blocked, time jumps to the earliest
    wake-up, so concurrent sleeps overlap as they would in real time, and tests run instantly."""

    def __init__(self):
        self.t = 1000.0
        self._sleepers = []  # heap of (wake time, seq, future)
        self._seq = 0
        self._driver = None

    def __call__(self):
        return self.t

    async def _drive(self):
        import heapq
        while True:
            for _ in range(50):  # let every runnable task reach its next await
                await asyncio.sleep(0)
            if not self._sleepers:
                if all(t.done() for t in asyncio.all_tasks() if t is not asyncio.current_task()):
                    return
                continue
            wake, _, fut = heapq.heappop(self._sleepers)
            self.t = max(self.t, wake)
            if not fut.done():
                fut.set_result(None)

    async def sleep(self, d):
        import heapq
        if self._driver is None or self._driver.done():
            self._driver = asyncio.get_running_loop().create_task(self._drive())
        fut = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._sleepers, (self.t + max(d, 0), self._seq, fut))
        await fut


def make(tmp_path, clock, **kw):
    return JobQueue(str(tmp_path / "q.db"), clock=clock, rng=random.Random(0), **kw)


def run(q, handler, clock, **kw):
    r = Runner(q, handler, sleep=clock.sleep, **kw)
    asyncio.run(r.run_until_idle())
    return r


def test_same_idempotency_key_is_one_job_and_one_effect(tmp_path):
    clock = FakeClock()
    q = make(tmp_path, clock)
    ids = {q.enqueue("welcome:42", {"user": 42}) for _ in range(5)}
    assert len(ids) == 1

    async def handler(job):
        return lambda txn: txn.put_effect(job.key, "sent")

    run(q, handler, clock)
    assert q.counts() == {"done": 1}
    q.enqueue("welcome:42", {"user": 42})  # enqueue after completion: still the same, finished job
    run(q, handler, clock)
    assert q.conn.execute("SELECT count(*) FROM effects").fetchone()[0] == 1


def test_retries_with_capped_exponential_backoff_then_success(tmp_path):
    clock = FakeClock()
    q = make(tmp_path, clock, base_backoff=1.0, max_backoff=4.0, max_attempts=10)
    q.enqueue("flaky", {})
    attempts_at = []

    async def handler(job):
        attempts_at.append(clock())
        if job.attempts < 6:
            raise RuntimeError(f"attempt {job.attempts} failed")

    run(q, handler, clock, poll=0.01)
    assert q.counts() == {"done": 1}
    gaps = [b - a for a, b in zip(attempts_at, attempts_at[1:])]
    caps = [min(4.0, 1.0 * 2 ** (n - 1)) for n in range(1, 6)]
    for gap, cap in zip(gaps, caps):
        assert gap <= cap + 1.0 + 1e-9  # full jitter in [0, cap], plus at most one poll/sleep step


def test_dead_letter_after_max_attempts_keeps_last_error(tmp_path):
    clock = FakeClock()
    q = make(tmp_path, clock, max_attempts=3, base_backoff=0.1)
    q.enqueue("always-fails", {})

    async def handler(job):
        raise ValueError(f"boom {job.attempts}")

    run(q, handler, clock)
    row = q.conn.execute("SELECT state, attempts, last_error FROM jobs").fetchone()
    assert row == ("dead", 3, "ValueError: boom 3")


def test_concurrency_limit_is_respected(tmp_path):
    clock = FakeClock()
    q = make(tmp_path, clock)
    for i in range(40):
        q.enqueue(f"j{i}", i)

    async def handler(job):
        await clock.sleep(0.5)

    r = run(q, handler, clock, concurrency=3, rate_per_sec=1000)
    assert r.peak_active == 3 and q.counts() == {"done": 40}


def test_rate_limit_holds_in_every_window(tmp_path):
    """Token bucket invariant: starts in any window of length w <= burst + rate * w."""
    clock = FakeClock()
    q = make(tmp_path, clock)
    for i in range(100):
        q.enqueue(f"j{i}", i)

    async def handler(job):
        return None

    r = run(q, handler, clock, concurrency=8, rate_per_sec=5, burst=3)
    s = sorted(r.starts)
    assert len(s) == 100
    for i in range(len(s)):
        for j in range(i, len(s)):
            assert j - i + 1 <= 3 + 5 * (s[j] - s[i]) + 1e-6
    assert s[-1] - s[0] >= (100 - 3) / 5 - 1e-6  # and it really was throttled


def test_random_failures_no_job_lost_or_done_twice(tmp_path):
    clock = FakeClock()
    q = make(tmp_path, clock, max_attempts=4, base_backoff=0.2)
    rng = random.Random(7)
    for i in range(200):
        q.enqueue(f"j{i}", i)
    runs = {}

    async def handler(job):
        runs[job.key] = runs.get(job.key, 0) + 1
        await clock.sleep(rng.random() * 0.1)
        if rng.random() < 0.4:
            raise RuntimeError("transient")
        return lambda txn: txn.put_effect(job.key, str(job.payload))

    run(q, handler, clock, concurrency=6, rate_per_sec=50)
    c = q.counts()
    assert c.get("done", 0) + c.get("dead", 0) == 200 and set(c) <= {"done", "dead"}
    effects = q.conn.execute("SELECT count(*) FROM effects").fetchone()[0]
    assert effects == c["done"]  # one effect per done job, none for dead ones
    dead_attempts = q.conn.execute("SELECT DISTINCT attempts FROM jobs WHERE state = 'dead'").fetchall()
    assert dead_attempts in ([], [(4,)])


def test_stale_worker_cannot_complete_after_lease_takeover(tmp_path):
    clock = FakeClock()
    q = make(tmp_path, clock, lease_seconds=5)
    q.enqueue("slow", {})
    first = q.claim()
    clock.t += 6  # first worker stalls past its lease
    second = q.claim()
    assert second is not None and second.id == first.id and second.attempts == 2
    q.complete(second, lambda txn: txn.put_effect("slow", "by-second"))
    from jobqueue import StaleLease
    with pytest.raises(StaleLease):
        q.complete(first, lambda txn: txn.put_effect("slow", "by-first"))
    assert q.fail(first, "late failure") == "stale"
    assert q.conn.execute("SELECT value FROM effects").fetchall() == [("by-second",)]
    assert q.counts() == {"done": 1}


WORKER = r"""
import asyncio, sys, time
sys.path.insert(0, sys.argv[2])
from jobqueue import JobQueue, Runner
q = JobQueue(sys.argv[1], lease_seconds=1.0)
async def handler(job):
    print("start", job.key, flush=True)
    await asyncio.sleep(10)          # killed during this
asyncio.run(Runner(q, handler, concurrency=2).run_until_idle())
"""


def test_killed_worker_job_is_redelivered_after_lease_and_effect_happens_once(tmp_path):
    db = str(tmp_path / "q.db")
    q = JobQueue(db, lease_seconds=1.0)
    for i in range(4):
        q.enqueue(f"j{i}", i)
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    proc = subprocess.Popen([sys.executable, str(script), db, os.path.dirname(os.path.abspath(__file__))],
                            stdout=subprocess.PIPE, text=True)
    started = [proc.stdout.readline().split()[1] for _ in range(2)]  # it holds two leases
    proc.kill()
    proc.communicate()
    assert q.counts() == {"running": 2, "queued": 2}

    async def handler(job):
        return lambda txn: txn.put_effect(job.key, "ok")

    t = time.time()
    asyncio.run(Runner(q, handler, concurrency=2).run_until_idle())
    assert time.time() - t >= 0.5  # had to wait for the dead worker's leases to expire
    assert q.counts() == {"done": 4}
    rows = dict(q.conn.execute("SELECT key, attempts FROM jobs").fetchall())
    assert all(rows[k] == 2 for k in started)  # redelivered once
    assert q.conn.execute("SELECT count(*) FROM effects").fetchone()[0] == 4


def test_effect_and_done_are_atomic(tmp_path):
    """If committing the effect fails, the job is not marked done and the effect is not half-written."""
    clock = FakeClock()
    q = make(tmp_path, clock)
    q.enqueue("dup-effect", {})
    q.conn.execute("INSERT INTO effects VALUES ('dup-effect', 'pre-existing')")
    job = q.claim()
    with pytest.raises(sqlite3.IntegrityError):
        q.complete(job, lambda txn: (txn.execute("INSERT INTO effects VALUES ('other', 'x')"),
                                     txn.put_effect(job.key, "again")))
    assert q.counts() == {"running": 1}
    assert q.conn.execute("SELECT key FROM effects ORDER BY key").fetchall() == [("dup-effect",)]


def test_throttled_jobs_do_not_lose_their_lease_while_waiting(tmp_path):
    """Regression: workers used to claim a job and then wait for a rate token, so under heavy throttling
    the lease expired and other workers re-claimed (and re-ran) the same job."""
    clock = FakeClock()
    q = make(tmp_path, clock, lease_seconds=2)
    for i in range(30):
        q.enqueue(f"j{i}", i)
    calls = []

    class TooManyRuns(BaseException):  # BaseException: escapes the runner's per-attempt error handling
        pass

    async def handler(job):
        calls.append(job.key)
        if len(calls) > 60:
            raise TooManyRuns(f"{len(calls)} runs for 30 jobs: leases expire while workers wait for tokens")

    run(q, handler, clock, concurrency=8, rate_per_sec=1, burst=1)
    assert len(calls) == 30
    assert q.counts() == {"done": 30}
    assert q.conn.execute("SELECT max(attempts) FROM jobs").fetchone()[0] == 1
