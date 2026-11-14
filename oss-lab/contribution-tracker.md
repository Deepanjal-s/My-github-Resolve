# Contribution tracker

Status vocabulary: `discovered` → `investigating` → `implemented` (local, verified) →
`PR prepared` → `submitted` (PR open) → `merged`. A local commit is not a
submission, and an open PR is not a merge.

## Own repositories

| Repository | Issue | Difficulty | Root cause | Solution | Verification | Status | Learning |
|---|---|---|---|---|---|---|---|
| [street-vendor-react](https://github.com/Deepanjal-s/street-vendor-react) | Backend security audit: authorization, pricing, OTP, password hashing | Intermediate | Routes checked authentication but not ownership; client-supplied prices trusted; OTP limits not enforced; Mongoose hook continued after `next()` | Ownership checks, server-side pricing, order state machine, OTP lockout + secure RNG, hook fix | 26 API tests on in-memory MongoDB (14 failed before the fix, all pass after); frontend lint and build pass | implemented (local, not pushed) | IDOR, state machines, atomic updates, Mongoose middleware |
| street-vendor-react | No rate limiting on login/register/OTP | Beginner | Missing limiter | `express-rate-limit` (configurable), trust proxy on Render | 2 new tests; 28/28 pass | implemented (local) | Rate limiting behind proxies |
| street-vendor-react | 1,981 `node_modules` files committed | Beginner | Missing `.gitignore` entry | Untracked them, fixed `.gitignore` | `git ls-files` shows 0 under node_modules | implemented (local) | What belongs in version control |
| [vehicle-tracking-backend](https://github.com/Deepanjal-s/vehicle-tracking-backend) | Auth configuration, startup resilience and GPS ingestion hardening | Intermediate | Signing secret not loaded from config; synchronous broker connect at startup; DB session not closed on failure; unvalidated MQTT payloads | Env-loaded secret with fail-fast, async MQTT connect with backoff, try/finally session handling, `parse_gps()` validation, paho v2 + FastAPI lifespan | 7 pytest tests (SQLite, no broker); all reproduced the bugs first | implemented (local, not pushed) | Secrets management, resilience, resource cleanup, input validation |
| [wdc-learning-portal](https://github.com/Deepanjal-s/wdc-learning-portal) | Malformed ids returned 500 | Beginner | Mongoose `CastError` not mapped in error handler | Map to 400 | 2 unit tests (1 failed before); 17/17 pass; client builds | implemented (local, not pushed) | Client vs server errors, error middleware |
| [Portfolio](https://github.com/Deepanjal-s/Portfolio) | Empty contact form opened a blank email; vulnerable build deps; missing project | Beginner | `noValidate` with no JS validation | Native form validation, `npm audit fix`, big-oh-no card | `npm audit` 0 vulnerabilities, lint and build pass | implemented (local, not pushed) | HTML constraint validation |
| [Deepanjal-s](https://github.com/Deepanjal-s/Deepanjal-s) (profile) | Profile omitted big-oh-no and the engineering notebook | n/a | n/a | Added both (no unverified claims) | n/a | implemented (local) | n/a |
| [raktkoshika](https://github.com/Deepanjal-s/raktkoshika), [chargenest](https://github.com/Deepanjal-s/chargenest) | Health check | n/a | esbuild advisory GHSA-g7r4-m6w7-qqqr affects only the dev server, which raktkoshika never starts (`build()` only) | No change (bump would cross a 0.x minor) | tests 2/2, builds, lint clean; chargenest 0 prod vulns | audited, healthy | Reading whether an advisory is actually reachable |
| [ai-chat-memory](https://github.com/Deepanjal-s/ai-chat-memory) | Keyword search kept the answer in a 4k context only 56% of the time (measured with memory-recall-bench) | Intermediate | `ts_rank_cd` has no IDF and favours long messages; flat all-terms bonus | Postgres selects candidates, BM25 re-ranks with document frequencies under the same filters | 479 LongMemEval_s questions: 56%→85% (any), 28%→61% (all-evidence); 163 fixed / 8 broken; 53/53 tests | implemented (local, not pushed) | IDF, BM25, measuring search quality |
| [big-oh-no](https://github.com/Deepanjal-s/big-oh-no) | Health check | n/a | n/a | No changes needed | 49/49 and 47/47 tests pass, typecheck clean, CI green | audited, healthy | n/a |

## Flagship project

| Project | What | Verification | Status |
|---|---|---|---|
| [memory-recall-bench](https://github.com/Deepanjal-s/memory-recall-bench) | Eval harness: does a chat-memory layer keep the facts it will need? 3-method adapter, baselines, BM25, bootstrap CIs, CLI | 18 tests (FIFO exact-output test mutation-checked); matches the independent Kaggle implementation exactly on LongMemEval | **public**, CI green. Embedding + hybrid retrievers, external-rankings interface (any language), ai-chat-memory case study. v0.2: real llama_index `Memory` adapter + exact port (68/68 identical contexts); v0.3: full LongMemEval_s (479 Q) results with per-type and all-evidence metrics |

## Upstream open source

| Repository | Issue | Difficulty | Root cause | Solution | Verification | Status | Learning |
|---|---|---|---|---|---|---|---|
| [arktypeio/arktype](https://github.com/arktypeio/arktype) | [#1726](https://github.com/arktypeio/arktype/issues/1726) `string.date.iso` accepts non-existent days; `.parse` rolls into next month | Intermediate | ISO regex bounded day to 01–31 only; `new Date()` overflows | Per-month day ranges + leap-year lookahead in the regex (keeps JSON Schema `pattern`) | 3 new tests (2 failed before); 1,816 passing; 4.4M-string exhaustive fuzz with 0 differences; Python `re` compatible | **submitted**: [PR #1727](https://github.com/arktypeio/arktype/pull/1727), all 14 CI checks green incl. `prChecks`; awaiting review (not merged) | Regex groups/backrefs, lookahead, property testing ([case study](case-studies/arktype-1726-iso-date-validation.md)) |
| [TanStack/router](https://github.com/TanStack/router) | [#8928](https://github.com/TanStack/router/issues/8928) masked navigation drops a hash character | Beginner | `__tempLocation.hash` stored without `#` | n/a | n/a | skipped: existing PR #8821 already fixes it | Always search PRs by keyword, not just by issue link |
| [unjs/ofetch](https://github.com/unjs/ofetch) | [#645](https://github.com/unjs/ofetch/issues/645) falsy JSON bodies sent as text/plain | Beginner | `if (body && ...)` skips 0/false | n/a | n/a | skipped: reporter opened PR #646 the same day | |
| [expressjs/multer](https://github.com/expressjs/multer) | [#1504](https://github.com/expressjs/multer/issues/1504) uploaded file left on disk after fsync failure | Intermediate | failed file not in cleanup list | n/a | n/a | skipped: PR #1505 opened the same day | |
| [agno-agi/agno](https://github.com/agno-agi/agno) | #11653, #11657, #11659, #11661 (Chroma/DocxReader bugs) | Beginner–Intermediate | see scout notes | n/a | n/a | closed upstream before work started | Repos with a `ready`-label gate need a triage comment first |

## Kaggle

| Notebook / dataset | Question | Technique | Result | Status |
|---|---|---|---|---|
| [dev-pay-2025](../kaggle/dev-pay-2025) | Do daily AI users earn less? | Within-country percentiles, OLS ladder with country FE + robust SE, LightGBM CV ablation, bootstrap CIs | Raw −22% reverses to +6.8% after controls (Simpson's paradox); AI adds little predictive power | Uploaded and ran successfully on Kaggle ([link](https://www.kaggle.com/code/deepanjalshukla95/the-ai-pay-penalty-that-isnt)); **published publicly** (version 2, ran successfully on Kaggle) |
| [memory-strategies](../kaggle/memory-strategies) | Which chat-memory strategy keeps the facts needed later? | Simulation on LongMemEval, self-implemented BM25, closed-form sanity check, random-retrieval control, bootstrap CIs | BM25 hybrid 86–100% evidence recall vs FIFO ≤58%; default flush leaves context 46% full (llama_index #24117) | **published publicly** ([link](https://www.kaggle.com/code/deepanjalshukla95/what-does-your-chatbot-forget)) |
| [longmemeval-full](../kaggle/longmemeval-full) | BM25 vs embeddings vs hybrid on all 479 LongMemEval_s questions | bge-small on a T4 GPU (183k messages), RRF fusion, paired per-question comparison, bootstrap CIs | Hybrid 90% all-evidence at 16k vs BM25 82% (37 fixed, 1 broken); embeddings alone trade wins and losses at 4k | ran successfully on Kaggle ([link](https://www.kaggle.com/code/deepanjalshukla95/chat-memory-retrieval-on-longmemeval)) |

## Observation: upstream timing

Every good recent bug I found in an established repo had a PR opened within hours, usually by the reporter
(TanStack/router, ofetch, multer). A fix only has value if it reaches the PR stage quickly, so the PR step
needs to be unblocked before more upstream work is worth preparing.
