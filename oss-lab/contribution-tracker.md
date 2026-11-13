# Contribution tracker

Status vocabulary: `discovered` → `investigating` → `implemented` (local, verified) →
`PR prepared` → `submitted` (PR open) → `merged`. A local commit is not a
submission, and an open PR is not a merge.

## Own repositories

| Repository | Issue | Difficulty | Root cause | Solution | Verification | Status | Learning |
|---|---|---|---|---|---|---|---|
| [street-vendor-react](https://github.com/Deepanjal-s/street-vendor-react) | Backend security audit: authorization, pricing, OTP, password hashing | Intermediate | Routes checked authentication but not ownership; client-supplied prices trusted; OTP limits not enforced; Mongoose hook continued after `next()` | Ownership checks, server-side pricing, order state machine, OTP lockout + secure RNG, hook fix | 26 API tests on in-memory MongoDB (14 failed before the fix, all pass after); frontend lint and build pass | implemented (local, not pushed) | IDOR, state machines, atomic updates, Mongoose middleware |
| street-vendor-react | 1,981 `node_modules` files committed | Beginner | Missing `.gitignore` entry | Untracked them, fixed `.gitignore` | `git ls-files` shows 0 under node_modules | implemented (local) | What belongs in version control |

## Upstream open source

| Repository | Issue | Difficulty | Root cause | Solution | Verification | Status | Learning |
|---|---|---|---|---|---|---|---|
| _scouting in progress_ | | | | | | | |

## Kaggle

| Notebook / dataset | Question | Technique | Result | Status |
|---|---|---|---|---|
| _planned_ | | | | |
