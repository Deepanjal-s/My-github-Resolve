"""One LLM summary per LongMemEval session, cached to JSONL (resumable).

Why: a session-level summary lets a retriever rank whole sessions instead of single
messages, which is the next candidate after per-message ranking stopped helping
(see memory-recall-bench results/README, session-diverse packing).

Summaries depend only on the session, not on the question, so each unique session id
is summarized once even when several questions share it.

Usage
  python summarize_sessions.py --convs convs.jsonl --questions subset60_ids.json --dry-run
  set GEMINI_API_KEY=...               (PowerShell: $env:GEMINI_API_KEY = "...")
  python summarize_sessions.py --convs convs.jsonl --questions subset60_ids.json \
      --out summaries_subset60.jsonl --max-calls 3000

The API key is read from the environment only (GEMINI_API_KEY). It is never written to disk or printed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

MODEL = "gemini-3.5-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
MAX_OUT_TOKENS = 1024  # headroom: the model may spend tokens on hidden reasoning before the summary
MAX_SESSION_CHARS = 60_000  # longer sessions are cut and the cut is recorded in the output

PROMPT = (
    "Below is one chat session between a user and an assistant. Write a faithful summary "
    "in at most 120 words. Keep every concrete fact a later question could ask about: names, "
    "numbers, dates, places, preferences, decisions, plans, and what the user said about "
    "themselves. Do not add anything that is not in the session.\n\n<session>\n{body}\n</session>"
)


def session_text(session: dict) -> tuple[str, bool]:
    lines = [f"{m.get('role', '?')}: {m.get('content', '')}" for m in session["messages"]]
    body = "\n".join(lines)
    if len(body) > MAX_SESSION_CHARS:
        return body[:MAX_SESSION_CHARS], True
    return body, False


def load_sessions(convs_path: str, question_ids: set[str] | None) -> dict[str, dict]:
    """Unique sessions, keyed by session id, restricted to the chosen questions."""
    sessions: dict[str, dict] = {}
    with open(convs_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if question_ids is not None and q["id"] not in question_ids:
                continue
            for s in q["sessions"]:
                sessions.setdefault(s["id"], s)
    return sessions


def load_done(out_path: str) -> set[str]:
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["session_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn last line from an interrupted run is retried
    return done


def call_api(api_key: str, prompt: str) -> dict:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": MAX_OUT_TOKENS},
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "x-goog-api-key": api_key,
        "content-type": "application/json",
    })
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            # covers dropped connections (RemoteDisconnected) and timeouts
            if attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError("network error after retries") from None
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convs", required=True, help="convs.jsonl (one LongMemEval question per line)")
    ap.add_argument("--questions", help="JSON list of question ids to include (default: all)")
    ap.add_argument("--out", default="summaries.jsonl")
    ap.add_argument("--max-calls", type=int, default=3000, help="hard cap on API calls this run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="count sessions and estimate tokens; no network")
    args = ap.parse_args()

    qids = None
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            qids = set(json.load(f))
    sessions = load_sessions(args.convs, qids)
    done = load_done(args.out)
    todo = {sid: s for sid, s in sessions.items() if sid not in done}

    chars = sum(len(session_text(s)[0]) for s in todo.values())
    est_in = chars // 4 + len(todo) * (len(PROMPT) // 4)
    print(f"questions: {len(qids) if qids is not None else 'all'}")
    print(f"unique sessions: {len(sessions)}  already done: {len(done)}  to do: {len(todo)}")
    print(f"estimated input tokens: {est_in:,}  (~4 chars per token)")
    print(f"estimated output tokens: {len(todo) * MAX_OUT_TOKENS:,} (upper bound)")
    if args.dry_run:
        return 0
    if len(todo) > args.max_calls:
        print(f"refusing: {len(todo)} calls exceeds --max-calls {args.max_calls}", file=sys.stderr)
        return 2

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY is not set in this shell", file=sys.stderr)
        return 2

    usage = {"input": 0, "output": 0, "calls": 0, "errors": 0}

    def work(item: tuple[str, dict]) -> dict | None:
        sid, s = item
        body, cut = session_text(s)
        try:
            r = call_api(api_key, PROMPT.format(body=body))
        except Exception as e:  # one bad session must not abort the whole run
            print(f"error on {sid}: {e}", file=sys.stderr)
            return None
        cands = r.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        u = r.get("usageMetadata", {})
        return {"session_id": sid, "summary": text.strip(), "truncated": cut,
                "finish_reason": cands[0].get("finishReason"),
                "input_tokens": u.get("promptTokenCount"), "output_tokens": u.get("candidatesTokenCount"),
                "model": MODEL}

    with open(args.out, "a", encoding="utf-8") as out, cf.ThreadPoolExecutor(args.workers) as pool:
        for rec in pool.map(work, todo.items()):
            usage["calls"] += 1
            if rec is None:
                usage["errors"] += 1
                continue
            usage["input"] += rec["input_tokens"] or 0
            usage["output"] += rec["output_tokens"] or 0
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if usage["calls"] % 100 == 0:
                print(f"{usage['calls']}/{len(todo)} calls, in {usage['input']:,} out {usage['output']:,}",
                      flush=True)

    print(f"done: calls {usage['calls']}, errors {usage['errors']}, "
          f"input tokens {usage['input']:,}, output tokens {usage['output']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
