# Session summaries (LLM) for chat-memory retrieval

Next research step after session-diverse packing (negative result, see
`projects/memory-recall-bench/results/README.md`): rank whole sessions by an LLM summary
instead of single messages.

`summarize_sessions.py` writes one summary per unique LongMemEval_s session with
`gemini-3.5-flash-lite`. Summaries are question-independent, so each session is summarized once.

## Size (dry-run, from `../longmemeval-full/convs.jsonl`)

| Scope | Unique sessions | Input tokens (est.) | Output tokens (upper bound) |
|---|---|---|---|
| 60-question pilot (`subset60_ids.json`) | 2,747 | 7.5 M | 0.8 M |
| All 479 questions | 18,527 | 49.8 M | 5.6 M |

Run the pilot first and check the real usage numbers printed at the end before the full run.

## Run

```powershell
cd "C:\Projects\My git Resolve\kaggle\session-summaries"
python summarize_sessions.py --convs ..\longmemeval-full\convs.jsonl --dry-run --questions subset60_ids.json
$env:GEMINI_API_KEY = "<your key>"   # set in your own shell; never commit it
python summarize_sessions.py --convs ..\longmemeval-full\convs.jsonl --questions subset60_ids.json --out summaries_subset60.jsonl --max-calls 3000
```

The run is resumable: rerunning skips sessions already in the output file.

## Status

- Script written and compiled; dry-run counts above. **Not yet run against the API**
  (no key in this session).
