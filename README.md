# My GitHub Resolve: an engineering lab notebook

A running record of the bugs I investigate, the fixes I ship, and the ML
experiments I run, written so that each entry explains **why**, not just **what**.

Built by [Deepanjal Shukla](https://github.com/Deepanjal-s) (B.Tech CSE, NIT Sikkim, 2028),
with AI pair-programming assistance (Claude). Every claim in here is backed by a test, a
command you can rerun, or a link to the upstream PR.

## Layout

| Path | What's in it |
|---|---|
| [`oss-lab/contribution-tracker.md`](oss-lab/contribution-tracker.md) | Every contribution and its honest status (local → submitted → merged) |
| [`oss-lab/case-studies/`](oss-lab/case-studies) | Write-ups: what broke, how it was reproduced, why, what fixed it |
| `oss-lab/investigations/` | Notes taken while tracing a bug |
| `oss-lab/reproduction-cases/` | Minimal scripts that trigger a bug |
| `oss-lab/regression-tests/` | Tests written to pin a fix (copies/links of upstream tests) |
| `oss-lab/experiments/` | Comparisons of alternative fixes or approaches |
| `oss-lab/learning-notes/` | Concepts each fix made me learn |
| [`kaggle/`](kaggle) | Notebooks published to Kaggle, with source kept here |

Third-party code is **linked, not copied**. Upstream repositories are cloned
locally for work and referenced here by URL and PR number.

## Ground rules

- No artificial activity: no empty commits, no drive-by typo PRs to farm counts.
- A bug isn't "fixed" until a test that failed before the fix passes after it.
- Respect each project's contribution and AI-assistance policies; disclose AI help where asked.
- Security write-ups are published only after the fix is deployed.
