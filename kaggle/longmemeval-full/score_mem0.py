"""Scores mem0 rankings against the other retrievers on the same questions (run after rank_mem0.py)."""
import json
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.adapters import RecentPlusBM25  # noqa: E402
from memory_recall_bench.datasets import load_longmemeval  # noqa: E402
from memory_recall_bench.external import RankedFromFile  # noqa: E402
from memory_recall_bench.runner import run, summarize  # noqa: E402

done = set(json.load(open("mrb_rankings_mem0.json")))
convs = [c for c in load_longmemeval("data/longmemeval_s_cleaned.json") if c.id in done]
print(len(convs), "conversations")
names = {
    "bm25": "recent+bm25",
    "aim": "ai-chat-memory keyword",
    "aimh": "ai-chat-memory hybrid (1500)",
    "mem0": "mem0 (verbatim, bge-small)",
}
ads = [
    RecentPlusBM25,
    lambda: RankedFromFile("aim_rankings_bm25.json", convs, name=names["aim"]),
    lambda: RankedFromFile("aim_rankings_hybrid_1500.json", convs, name=names["aimh"]),
    lambda: RankedFromFile("mrb_rankings_mem0.json", convs, name=names["mem0"]),
]
res = run(ads, convs, [4000, 16000])
out = {}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, by_kind=False, metric=m) + summarize(res, by_kind=True, metric=m):
        print(m, s["adapter"], s["budget"], s.get("kind", "ALL"), s["n"], round(s["recall"] * 100),
              f'[{round(s["ci_low"] * 100)}-{round(s["ci_high"] * 100)}]')
        out.setdefault(m, []).append(s)
by = defaultdict(dict)
for r in res:
    by[(r.budget, r.conversation)][r.adapter] = r
for base in ["bm25", "aimh"]:
    for b in [4000, 16000]:
        for m in ["recalled", "recalled_all"]:
            f = br = 0
            for (bb, _), d in by.items():
                if bb != b:
                    continue
                x, y = getattr(d[names[base]], m), getattr(d[names["mem0"]], m)
                f += y and not x
                br += x and not y
            print(f"paired mem0 vs {base}", b, m, "fixed", f, "broken", br)
            out[f"paired_vs_{base}_{b}_{m}"] = {"fixed": f, "broken": br}
json.dump(out, open("mem0_summary.json", "w"), indent=1, default=str)
