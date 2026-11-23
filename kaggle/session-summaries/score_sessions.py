"""Score SessionBoost (session-summary term) against plain BM25 on the 60-question pilot.

    python score_sessions.py summaries_qwen_0.jsonl summaries_qwen_1.jsonl [--weights 0.5,1,2]

Only questions whose sessions are all summarized are scored (coverage is printed).
Writes session_boost_summary.json next to this script.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.adapters import RecentPlusBM25  # noqa: E402
from memory_recall_bench.datasets import load_longmemeval  # noqa: E402
from memory_recall_bench.runner import run, summarize  # noqa: E402
from memory_recall_bench.summaries import SessionBoost, load_summaries, session_of  # noqa: E402

HERE = Path(__file__).parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
weights = [1.0]
for a in sys.argv[1:]:
    if a.startswith("--weights="):
        weights = [float(w) for w in a.split("=", 1)[1].split(",")]

summ = load_summaries(*args)
ids = set(json.loads((HERE / "subset60_ids.json").read_text()))
convs = [c for c in load_longmemeval(HERE.parent / "longmemeval-full" / "data" / "longmemeval_s_cleaned.json")
         if c.id in ids]
full = [c for c in convs if all(session_of(m.id) in summ for m in c.messages)]
print(f"{len(summ)} summaries; {len(full)}/{len(convs)} questions fully covered")

def boost(w):
    a = SessionBoost(RecentPlusBM25(), summ, weight=w)
    a.name += f" w{w:g}"
    return a


adapters = [RecentPlusBM25] + [(lambda w=w: boost(w)) for w in weights]
res = run(adapters, full, [4000, 16000])
out = {}
for metric in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=metric) + summarize(res, metric=metric, by_kind=True):
        name = s["adapter"]
        key = f'{metric}|{name}|{s["budget"]}|{s.get("kind") or "all"}'
        out[key] = [round(s["recall"] * 100, 1), round(s["ci_low"] * 100, 1), round(s["ci_high"] * 100, 1)]
for k, v in out.items():
    if k.endswith("|all"):
        print(k, v)
(HERE / "session_boost_summary.json").write_text(json.dumps(out, indent=1))
