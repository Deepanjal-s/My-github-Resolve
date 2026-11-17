"""Experiment: session-diverse packing for multi-session questions.

Re-orders any ranking so the best message of each session comes first (sessions in order of their best
message), then the remaining messages in their original order. Goal: cover *several* answer sessions in a small
budget. Applied to BM25 and to mem0's rankings, all 479 questions, paired against the originals.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.adapters import RecentPlusBM25  # noqa: E402
from memory_recall_bench.datasets import load_longmemeval  # noqa: E402
from memory_recall_bench.external import RankedFromFile  # noqa: E402
from memory_recall_bench.runner import run, summarize  # noqa: E402

FULL = r"C:\Projects\My git Resolve\kaggle\longmemeval-full"


def session(mid: str) -> str:
    return mid.rsplit(":", 1)[0]


K = int(os.environ.get("DIVERSE_K", "1000000"))


def diversify(ranked: list) -> list:
    """Best message of each of the top-K sessions first (K bounds how many weak sessions get a slot)."""
    seen, first, rest = set(), [], []
    for m in ranked:
        sid = session(m if isinstance(m, str) else m.id)
        (first if sid not in seen and len(seen) < K else rest).append(m)
        seen.add(sid)
    return first + rest


class DiverseBM25(RecentPlusBM25):
    name = "recent+bm25 diverse"

    def rank(self, query, archive):
        return diversify(super().rank(query, archive))


class DiverseFromFile(RankedFromFile):
    def rank(self, query, archive):
        return diversify(super().rank(query, archive))


convs = load_longmemeval(os.path.join(FULL, "data", "longmemeval_s_cleaned.json"))
res = run([RecentPlusBM25, DiverseBM25,
           lambda: RankedFromFile("mem0_rankings_full.json", convs, name="mem0"),
           lambda: DiverseFromFile("mem0_rankings_full.json", convs, name="mem0 diverse")],
          convs, [4000, 16000])
out = {}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m) + summarize(res, metric=m, by_kind=True):
        k = f'{m}|{s["adapter"]}|{s["budget"]}|{s.get("kind") or "all"}'
        out[k] = [round(s["recall"] * 100, 1), round(s["ci_low"] * 100, 1), round(s["ci_high"] * 100, 1)]
by = defaultdict(dict)
for r in res:
    by[(r.budget, r.conversation, r.kind)][r.adapter] = r
for base, div in [("recent+bm25", "recent+bm25 diverse"), ("mem0", "mem0 diverse")]:
    for b in [4000, 16000]:
        for m in ["recalled", "recalled_all"]:
            for kind in [None, "multi-session"]:
                f = br = 0
                for (bb, _, k), d in by.items():
                    if bb != b or (kind and k != kind):
                        continue
                    x, y = getattr(d[base], m), getattr(d[div], m)
                    f += y and not x
                    br += x and not y
                out[f"paired|{div}|{b}|{m}|{kind or 'all'}"] = [f, br]
for k, v in out.items():
    if k.startswith("paired") or k.endswith("|all") or "multi-session" in k:
        print(k, v)
json.dump(out, open(f"session_diverse_summary_k{os.environ.get('DIVERSE_K', 'all')}.json", "w"), indent=1)
