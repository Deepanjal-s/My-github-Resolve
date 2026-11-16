# jobqueue: retries, rate limits and idempotency, with the failure cases tested

Roadmap Track D, topic 2 (concurrency). About 250 lines of Python on SQLite: a durable async job queue with
idempotency keys, leases, retries with backoff, a dead-letter state, a concurrency cap and a token-bucket rate
limit.

```python
q = JobQueue("jobs.db", lease_seconds=30, max_attempts=5)
q.enqueue("send-welcome:42", {"user": 42})          # same key again -> the same job, not a second one

async def handler(job):
    await send_email(job.payload["user"])
    return lambda txn: txn.put_effect(job.key, "sent")   # committed together with 'done'

await Runner(q, handler, concurrency=4, rate_per_sec=10).run_until_idle()
```

## Guarantees and how each one is enforced

| Guarantee | Mechanism |
|---|---|
| One job per idempotency key | `UNIQUE(key)` + `INSERT OR IGNORE`; enqueue returns the existing id |
| A job is leased by one worker at a time | `UPDATE … RETURNING` claims and leases in one statement |
| A crashed worker's job runs again (at-least-once) | Leases expire; expired `running` jobs are claimable |
| A stale worker can't overwrite the new owner's result | Each claim gets a random lease token; complete/fail check it inside `BEGIN IMMEDIATE` |
| Effects happen once per job | The handler's effect and `state='done'` commit in one transaction |
| Retries don't stampede | Exponential backoff with full jitter, `uniform(0, min(cap, base·2^(n-1)))` |
| Poison jobs stop | After `max_attempts`, state `dead` with the last error |
| Rate and concurrency limits | Token bucket (rate, burst) and N workers |

## Two bugs the tests found

1. **Leases expired while workers waited for the rate limiter.** The first version claimed a job and *then* waited
   for a rate token. Under heavy throttling the lease ran out while the job sat in that wait, so another worker
   re-claimed it, and that happened again and again. Fix: take the token first, then claim, and refund the token
   if nothing is ready. Regression test: 30 jobs at 1 job/s with a 2 s lease must run exactly 30 times (the old
   order makes 61+ runs).
2. **Floating-point starvation in the token bucket.** After a refill, tokens were 0.9999999999999 instead of 1.0.
   The bucket then asked for a sleep shorter than the clock's resolution, so time never advanced and it looped
   forever. Fix: accept `tokens ≥ 1 − 1e-9`. The rate-limit test caught this as a hang.

## Tests (10, `pytest`, ~5 s)

A virtual clock is used: sleeps overlap as in real time, and time jumps to the next wake-up once every task is
blocked. Backoff and rate tests are therefore exact and instant.

- Idempotent enqueue, including after completion. Backoff gaps within their jittered caps. Dead-letter with the
  last error.
- The concurrency cap is reached and never exceeded. **The rate limit is checked on every window** of start times:
  `starts ≤ burst + rate·w`.
- 200 jobs with 40% random failures: every job ends `done` or `dead`, with exactly one effect per done job.
- A stale worker can't complete or fail a job another worker took over. Effect and done are atomic (a failing
  effect leaves no partial write).
- **Killed process:** a real worker holding two leases is killed. A new runner waits for the leases to expire, runs
  both jobs again once each, and every job ends with exactly one effect.

Mutation check: removing the lease-token check fails the stale-worker test. Restoring claim-before-token fails
the throttling regression test.

## Limits

- One SQLite file, so throughput is bounded by its single writer. Fine for one machine, not a distributed queue.
- Exactly-once covers effects written through `txn`. An external side effect (an email, an HTTP call) is still
  at-least-once. Pass `job.key` to the external API as its idempotency key.
