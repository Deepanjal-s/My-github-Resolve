# Case study: arktype accepted dates that don't exist

- Upstream: [arktypeio/arktype#1726](https://github.com/arktypeio/arktype/issues/1726)
- Status: **PR prepared locally** (branch `fix/iso-date-nonexistent-days`), not yet submitted. See the [tracker](../contribution-tracker.md).
- Files: [`experiments/arktype-1726/`](../experiments/arktype-1726) (prototype, exhaustive fuzz, patch, PR text)

## What was broken

`type("string.date.iso")("2024-06-31")` passed, and
`type("string.date.iso.parse")("2023-02-29")` returned **1 March 2023** with no error.
The ISO 8601 regex only checked that the day was between 01 and 31. `.parse` then calls `new Date(s)`,
which quietly rolls day 31 of a 30-day month into the next month.

## The real question: where does the check go?

The reporter had already found the obvious fix (add a `predicate` next to the
`pattern`, the way `creditCard` does) and the catch:

| Option | Rejects `2024-06-31` in `string.date.iso`? | `toJsonSchema()` | Notes |
|---|---|---|---|
| A. Predicate on the root | ✅ | ❌ throws unless the caller passes a `fallback` | Breaks OpenAPI generation for existing users |
| B. Check only in `.parse` | ❌ (still passes) | ✅ | Fixes only half the bug |
| **C. Encode it in the regex** | ✅ | ✅ still a `pattern` | Longer regex; needs very thorough testing |

I first checked `ark/schema/shared/toJsonSchema.ts` to confirm option A's cost:
the default handler for a predicate is `ToJsonSchema.throw("predicate", ctx)`.

## Building option C

Three pieces:

1. **Month-specific day ranges**: `(0[13578]|1[02])` → 01–31, `(0[469]|11)` → 01–30, `02` → 01–29.
2. **Leap years without lookbehind.** Feb 29 depends on the year, which the regex has
   already consumed by the time it reaches the day. Lookbehind would fix that, but
   Python's `re` only supports fixed-width lookbehind, and JSON Schema patterns get used
   from many languages. Instead, a **lookahead at the start** rejects
   "a non-leap year, then 02, then 29":
   `^(?![+-]?(?!LEAP)\d{4}-?02-?29)`, where
   `LEAP = \d\d(0[48]|[2468][048]|[13579][26]) | ([02468][048]|[13579][26])00`.
3. **Keep the capture-group count at 24.** The original regex uses backreferences
   `\3` (so `2024-0101` with mixed separators is rejected) and `\17`. Adding a
   capturing group anywhere earlier silently renumbers them. My first prototype had 25
   groups because the lookahead used `(-?)`. I wrote a check for the group count before trusting anything else.

## Testing a regex properly

I wrote unit tests in arktype's style, and also an **exhaustive comparison**:

- Every year 0000–9999 × every month × days 00–32, compared to a reference
  implementation (`isLeap`, `daysIn`), plus basic (`YYYYMMDD`), signed and with-time
  variants for a sample of years, including every century.
- For inputs that aren't calendar dates (week dates, ordinal dates, times,
  offsets, malformed input), the new regex must give **exactly the original's answer**.

The first fuzz run found **49 failures**, all of the form `20000229`. My leap-year
lookahead had a `(?!\d)` guard that is wrong for the basic format, where the year is
always followed by a digit. The handwritten tests didn't cover this case, so the fuzz
was what caught it. After the fix: 4.4 million strings, 0 failures. I then re-ran the fuzz against the
regex **copied from the committed source file** (not my prototype), so what
was tested is exactly what ships.

## Verification

- 2 new tests failed before the fix and pass after; a 3rd guards `toJsonSchema()`.
- `pnpm test`: 1,816 passing. `pnpm typecheckRepo` clean. Prettier and ESLint clean on changed files.
- Python `re` compiles the pattern and agrees on the key cases.
- Micro-benchmark: the new regex is ~40% faster than the old one on ISO timestamps.

## Side finding

arktype's build scripts fail if the repo path contains a space
(`Cannot find module 'C:\Projects\My'`): an unquoted path passed to `node`. I worked
around it by cloning to a path without spaces. It could be a separate small issue.

## Concepts learned

Regex capture-group numbering and backreferences, lookahead as a way to get conditional logic without
lookbehind, regex portability across engines (ECMA vs. Python), Gregorian
leap-year rules, property/exhaustive testing vs. example tests, and reading
a library's design constraints (JSON Schema compatibility) before choosing a fix.
