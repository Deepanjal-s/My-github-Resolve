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
| [arktypeio/arktype](https://github.com/arktypeio/arktype) | [#1726](https://github.com/arktypeio/arktype/issues/1726) `string.date.iso` accepts non-existent days; `.parse` rolls into next month | Intermediate | ISO regex bounded day to 01–31 only; `new Date()` overflows | Per-month day ranges + leap-year lookahead in the regex (keeps JSON Schema `pattern`) | 3 new tests (2 failed before); 1,816 passing; 4.4M-string exhaustive fuzz with 0 differences; Python `re` compatible | PR prepared (local branch, awaiting submission) | Regex groups/backrefs, lookahead, property testing ([case study](case-studies/arktype-1726-iso-date-validation.md)) |
| [TanStack/router](https://github.com/TanStack/router) | [#8928](https://github.com/TanStack/router/issues/8928) masked navigation drops a hash character | Beginner | `__tempLocation.hash` stored without `#` | n/a | n/a | skipped: existing PR #8821 already fixes it | Always search PRs by keyword, not just by issue link |

## Kaggle

| Notebook / dataset | Question | Technique | Result | Status |
|---|---|---|---|---|
| _planned_ | | | | |
