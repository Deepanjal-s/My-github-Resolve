fix: reject non-existent days in string.date.iso (#1726)

Fixes #1726.

`string.date.iso` only bounded the day to 01–31, so `2024-06-31` passed, and `string.date.iso.parse("2023-02-29")` silently returned March 1st.

On the question in the issue (predicate vs. a check in `.parse`), this PR takes a third option: **the per-month day limits and the leap-year rule go into the regex itself.**

- `string.date.iso` and `string.date.iso.parse` both reject days that don't exist.
- `toJsonSchema()` keeps emitting a plain `pattern` and doesn't start throwing, which a predicate would cause for anyone generating OpenAPI from `string.date.iso`.
- `string.date` with `format: "iso"` gets the fix too, because it uses the same matcher.

### How the regex changed
- A leading lookahead rejects `YYYY-02-29` / `YYYY0229` unless YYYY is a Gregorian leap year (divisible by 4 and not by 100, or divisible by 400).
- The month/day part is split into 31-day months, 30-day months and February (01–29).
- The new groups are non-capturing where needed, so the regex still has **24 capture groups** and the existing `\3` (date separator) and `\17` (time separator) backreferences still point at the same groups.
- Week dates (`2023-W09-3`), ordinal dates (`2023-060`), month-only (`2023-02`), times and offsets behave as before.

### Verification
- New tests in `ark/type/__tests__/keywords/date.test.ts` covering:
  - 30-day months, Feb 29 in common years and in century years (1900, 2100), basic and extended formats, and with a time part;
  - the leap days that must still pass (2024, 2000);
  - the `.parse` rollover;
  - `toJsonSchema()` still producing a `pattern`.

  The first two of these tests failed before the change.
- Outside the repo, I also compared the new and old regex on **4.4 million strings**: every day 00–32 of every month of every year 0000–9999, plus basic, signed and timed variants and a list of week, ordinal and time edge cases. The new regex matched a `Date`-based reference on every calendar date and the old regex on everything else, with 0 differences. The script is linked below if that's useful.
- The pattern also compiles and gives the same answers in Python's `re`, for JSON Schema consumers.
- `pnpm test`: 1816 passing. `pnpm typecheckRepo`: clean. Prettier and ESLint are clean on the changed files.

  `pnpm lint` fails on `tsconfig.json` on an untouched `main` checkout too (Windows), so that failure is unrelated to this change. I haven't run the bench and TS-version parts of `prChecks`.
- Performance: in a quick micro-benchmark (200k ISO timestamps), the new regex was faster than the old one (~41 ms vs ~70 ms on my machine), probably because the month split cuts backtracking.

### Trade-off
The regex is longer. I've added a comment above it describing each part. If you'd rather keep the matcher as it was and accept the JSON Schema change, the predicate version from the issue is a small change on top of the existing code.

AI disclosure: I used Claude (an AI assistant) while working on this. I reviewed and understand the change, and the verification above was run locally.
