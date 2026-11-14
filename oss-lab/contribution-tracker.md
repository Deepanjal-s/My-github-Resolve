# Contribution tracker

Status vocabulary: `discovered` → `investigating` → `implemented` (local, verified) →
`PR prepared` → `submitted` (PR open) → `merged`. A local commit is not a
submission, and an open PR is not a merge.

## Own repositories

| Repository | Issue | Difficulty | Root cause | Solution | Verification | Status | Learning |
|---|---|---|---|---|---|---|---|
| [street-vendor-react](https://github.com/Deepanjal-s/street-vendor-react) | Backend security audit: authorization, pricing, OTP, password hashing | Intermediate | Routes checked authentication but not ownership; client-supplied prices trusted; OTP limits not enforced; Mongoose hook continued after `next()` | Ownership checks, server-side pricing, order state machine, OTP lockout + secure RNG, hook fix | 26 API tests on in-memory MongoDB (14 failed before the fix, all pass after); frontend lint and build pass | implemented (local, not pushed) | IDOR, state machines, atomic updates, Mongoose middleware |
| street-vendor-react | 1,981 `node_modules` files committed | Beginner | Missing `.gitignore` entry | Untracked them, fixed `.gitignore` | `git ls-files` shows 0 under node_modules | implemented (local) | What belongs in version control |
| [vehicle-tracking-backend](https://github.com/Deepanjal-s/vehicle-tracking-backend) | Auth configuration, startup resilience and GPS ingestion hardening | Intermediate | Signing secret not loaded from config; synchronous broker connect at startup; DB session not closed on failure; unvalidated MQTT payloads | Env-loaded secret with fail-fast, async MQTT connect with backoff, try/finally session handling, `parse_gps()` validation, paho v2 + FastAPI lifespan | 7 pytest tests (SQLite, no broker); all reproduced the bugs first | implemented (local, not pushed) | Secrets management, resilience, resource cleanup, input validation |
| [wdc-learning-portal](https://github.com/Deepanjal-s/wdc-learning-portal) | Malformed ids returned 500 | Beginner | Mongoose `CastError` not mapped in error handler | Map to 400 | 2 unit tests (1 failed before); 17/17 pass; client builds | implemented (local, not pushed) | Client vs server errors, error middleware |
| [Portfolio](https://github.com/Deepanjal-s/Portfolio) | Empty contact form opened a blank email; vulnerable build deps; missing project | Beginner | `noValidate` with no JS validation | Native form validation, `npm audit fix`, big-oh-no card | `npm audit` 0 vulnerabilities, lint and build pass | implemented (local, not pushed) | HTML constraint validation |
| [ai-chat-memory](https://github.com/Deepanjal-s/ai-chat-memory), [big-oh-no](https://github.com/Deepanjal-s/big-oh-no) | Health check | n/a | n/a | No changes needed | 49/49 and 47/47 tests pass, typecheck clean, CI green | audited, healthy | n/a |

## Upstream open source

| Repository | Issue | Difficulty | Root cause | Solution | Verification | Status | Learning |
|---|---|---|---|---|---|---|---|
| [arktypeio/arktype](https://github.com/arktypeio/arktype) | [#1726](https://github.com/arktypeio/arktype/issues/1726) `string.date.iso` accepts non-existent days; `.parse` rolls into next month | Intermediate | ISO regex bounded day to 01–31 only; `new Date()` overflows | Per-month day ranges + leap-year lookahead in the regex (keeps JSON Schema `pattern`) | 3 new tests (2 failed before); 1,816 passing; 4.4M-string exhaustive fuzz with 0 differences; Python `re` compatible | PR prepared (local branch, awaiting submission) | Regex groups/backrefs, lookahead, property testing ([case study](case-studies/arktype-1726-iso-date-validation.md)) |
| [TanStack/router](https://github.com/TanStack/router) | [#8928](https://github.com/TanStack/router/issues/8928) masked navigation drops a hash character | Beginner | `__tempLocation.hash` stored without `#` | n/a | n/a | skipped: existing PR #8821 already fixes it | Always search PRs by keyword, not just by issue link |

## Kaggle

| Notebook / dataset | Question | Technique | Result | Status |
|---|---|---|---|---|
| [dev-pay-2025](../kaggle/dev-pay-2025) | Do daily AI users earn less? | Within-country percentiles, OLS ladder with country FE + robust SE, LightGBM CV ablation, bootstrap CIs | Raw −22% reverses to +6.8% after controls (Simpson's paradox); AI adds little predictive power | Uploaded and ran successfully on Kaggle ([link](https://www.kaggle.com/code/deepanjalshukla95/the-ai-pay-penalty-that-isnt)); still **private** until the account is phone-verified |
