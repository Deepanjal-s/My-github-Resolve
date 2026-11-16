"""jobqueue: a durable, rate-limited async job queue with retries, leases and idempotency keys (SQLite).

Delivery is at-least-once: a job is leased to a worker for `lease_seconds`; if the worker dies, the lease
expires and another worker runs the job again. Exactly-once *effects* come from the handler: it receives a
`Txn` and writes its effect in the same SQLite transaction that marks the job done, so a re-run either sees
the job already done or finds no effect from the crashed attempt (that transaction never committed).

    q = JobQueue("jobs.db")
    q.enqueue("send-welcome:42", {"user": 42})            # same key again -> same job, not a new one
    await Runner(q, handler, concurrency=4, rate_per_sec=10).run_until_idle()

Retries: a failing attempt is retried after exponential backoff with full jitter,
    delay = uniform(0, min(max_backoff, base_backoff * 2 ** (attempt - 1)))
and after `max_attempts` the job moves to state 'dead' with its last error kept.
"""
from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,          -- idempotency key
    payload     TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'queued',-- queued | running | done | dead
    attempts    INTEGER NOT NULL DEFAULT 0,
    run_at      REAL NOT NULL,
    lease_until REAL,
    lease_token TEXT,
    last_error  TEXT
);
CREATE INDEX IF NOT EXISTS jobs_ready ON jobs (state, run_at);
CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class Job:
    id: int
    key: str
    payload: Any
    attempts: int
    lease_token: str


class Txn:
    """Handed to a handler: write effects here and they commit atomically with 'done'."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def put_effect(self, key: str, value: str) -> None:
        self._conn.execute("INSERT INTO effects (key, value) VALUES (?, ?)", (key, value))

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)


class StaleLease(Exception):
    """The job's lease expired and another worker took it over; this attempt's result is discarded."""


class JobQueue:
    def __init__(self, path: str, lease_seconds: float = 30.0, max_attempts: int = 5,
                 base_backoff: float = 0.5, max_backoff: float = 60.0,
                 clock: Callable[[], float] = time.time, rng: random.Random | None = None):
        self.conn = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(SCHEMA)
        self.lease_seconds, self.max_attempts = lease_seconds, max_attempts
        self.base_backoff, self.max_backoff = base_backoff, max_backoff
        self.clock, self.rng = clock, rng or random.Random()

    def enqueue(self, key: str, payload: Any, delay: float = 0.0) -> int:
        """Adds a job unless one with this key exists; returns the job id either way."""
        self.conn.execute("INSERT OR IGNORE INTO jobs (key, payload, run_at) VALUES (?, ?, ?)",
                          (key, json.dumps(payload), self.clock() + delay))
        return self.conn.execute("SELECT id FROM jobs WHERE key = ?", (key,)).fetchone()[0]

    def claim(self) -> Job | None:
        """Atomically leases the next ready job: queued and due, or running with an expired lease."""
        now = self.clock()
        token = f"{now:.6f}-{self.rng.getrandbits(64):016x}"
        row = self.conn.execute(
            """UPDATE jobs SET state = 'running', attempts = attempts + 1, lease_until = ?, lease_token = ?
               WHERE id = (SELECT id FROM jobs
                           WHERE (state = 'queued' AND run_at <= ?) OR (state = 'running' AND lease_until <= ?)
                           ORDER BY run_at, id LIMIT 1)
               RETURNING id, key, payload, attempts""",
            (now + self.lease_seconds, token, now, now)).fetchone()
        if row is None:
            return None
        return Job(row[0], row[1], json.loads(row[2]), row[3], token)

    def backoff(self, attempt: int) -> float:
        return self.rng.uniform(0, min(self.max_backoff, self.base_backoff * 2 ** (attempt - 1)))

    def _owned(self, job: Job) -> bool:
        row = self.conn.execute("SELECT state, lease_token FROM jobs WHERE id = ?", (job.id,)).fetchone()
        return row == ("running", job.lease_token)

    def complete(self, job: Job, effect: Callable[[Txn], None] | None = None) -> None:
        """Marks done and applies `effect` in one transaction, only if this worker still holds the lease."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if not self._owned(job):
                raise StaleLease(job.key)
            if effect:
                effect(Txn(self.conn))
            self.conn.execute("UPDATE jobs SET state = 'done', lease_until = NULL, last_error = NULL WHERE id = ?",
                              (job.id,))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def fail(self, job: Job, error: str) -> str:
        """Schedules a retry with backoff, or moves the job to 'dead'. Returns the new state."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if not self._owned(job):
                self.conn.execute("ROLLBACK")
                return "stale"
            if job.attempts >= self.max_attempts:
                state, run_at = "dead", self.clock()
            else:
                state, run_at = "queued", self.clock() + self.backoff(job.attempts)
            self.conn.execute("UPDATE jobs SET state = ?, run_at = ?, lease_until = NULL, last_error = ? WHERE id = ?",
                              (state, run_at, error[:2000], job.id))
            self.conn.execute("COMMIT")
            return state
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def counts(self) -> dict[str, int]:
        return dict(self.conn.execute("SELECT state, count(*) FROM jobs GROUP BY state").fetchall())

    def next_due(self) -> float | None:
        row = self.conn.execute(
            "SELECT min(CASE WHEN state = 'queued' THEN run_at ELSE lease_until END) FROM jobs "
            "WHERE state IN ('queued', 'running')").fetchone()
        return row[0]


class TokenBucket:
    """`rate` tokens per second, at most `burst` stored. acquire() waits until a token is available."""

    def __init__(self, rate: float, burst: int, clock: Callable[[], float], sleep: Callable[[float], Awaitable]):
        self.rate, self.burst, self.clock, self.sleep = rate, burst, clock, sleep
        self.tokens, self.last = float(burst), clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:  # FIFO: waiters are served in arrival order
            while True:
                now = self.clock()
                self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
                self.last = now
                # Tolerance: refilling to exactly 1.0 can leave 0.9999999999999 after float rounding, and the
                # matching sleep is then below the clock's resolution, so time never advances (a busy loop).
                if self.tokens >= 1 - 1e-9:
                    self.tokens = max(0.0, self.tokens - 1)
                    return
                await self.sleep((1 - self.tokens) / self.rate)

    def refund(self) -> None:
        """Returns an unused token (the worker found nothing to run)."""
        self.tokens = min(self.burst, self.tokens + 1)


Handler = Callable[[Job], Awaitable[Callable[[Txn], None] | None]]


class Runner:
    """Runs jobs with at most `concurrency` handlers at once and at most `rate_per_sec` starts per second.

    A handler does its work and returns an optional `effect(txn)` that is committed with 'done'.
    Raising marks the attempt failed (retry or dead-letter)."""

    def __init__(self, queue: JobQueue, handler: Handler, concurrency: int = 4, rate_per_sec: float = 10.0,
                 burst: int | None = None, sleep: Callable[[float], Awaitable] = asyncio.sleep,
                 poll: float = 0.05):
        self.q, self.handler, self.concurrency, self.sleep, self.poll = queue, handler, concurrency, sleep, poll
        self.bucket = TokenBucket(rate_per_sec, burst or concurrency, queue.clock, sleep)
        self.active = self.peak_active = 0
        self.starts: list[float] = []

    async def _worker(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            # Rate token first, claim second: a job is never leased while its worker waits for the limiter.
            # (Claiming first let leases expire under heavy throttling, so jobs were re-claimed over and over.)
            await self.bucket.acquire()
            job = self.q.claim()
            if job is None:
                self.bucket.refund()
                due = self.q.next_due()
                if due is None:
                    return  # nothing queued or running anywhere: idle
                await self.sleep(max(self.poll, min(1.0, due - self.q.clock())))
                continue
            self.starts.append(self.q.clock())
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                effect = await self.handler(job)
            except Exception as e:  # noqa: BLE001 - any handler error is a failed attempt
                self.q.fail(job, f"{type(e).__name__}: {e}")
            else:
                try:
                    self.q.complete(job, effect)
                except StaleLease:
                    pass  # another worker owns it now; its result counts
            finally:
                self.active -= 1

    async def run_until_idle(self) -> None:
        stop = asyncio.Event()
        await asyncio.gather(*(self._worker(stop) for _ in range(self.concurrency)))
