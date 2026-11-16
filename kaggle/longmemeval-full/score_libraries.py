"""One table, same 60 questions: real memory libraries vs baselines (run after rank_mem0.py)."""
import json
import sys

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.adapters import FIFO, RecentPlusBM25, RecentPlusRandom  # noqa: E402
from memory_recall_bench.datasets import load_longmemeval  # noqa: E402
from memory_recall_bench.external import RankedFromFile  # noqa: E402
from memory_recall_bench.integrations.llamaindex import LlamaIndexMemory  # noqa: E402
from memory_recall_bench.runner import run, summarize  # noqa: E402

done = set(json.load(open("mrb_rankings_mem0.json")))
convs = [c for c in load_longmemeval("data/longmemeval_s_cleaned.json") if c.id in done]
ads = [
    FIFO, LlamaIndexMemory, RecentPlusRandom, RecentPlusBM25,
    lambda: RankedFromFile("aim_rankings_bm25.json", convs, name="ai-chat-memory keyword"),
    lambda: RankedFromFile("mrb_rankings_mem0.json", convs, name="mem0 (verbatim)"),
    lambda: RankedFromFile("aim_rankings_hybrid_1500.json", convs, name="ai-chat-memory hybrid"),
]
res = run(ads, convs, [4000, 16000])
rows = {}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m):
        rows.setdefault(s["adapter"], {})[f'{m}_{s["budget"]}'] = (round(s["recall"] * 100), round(s["ci_low"] * 100), round(s["ci_high"] * 100))
for a, r in rows.items():
    print(a.ljust(26), " ".join(f'{k}={v[0]}[{v[1]}-{v[2]}]' for k, v in sorted(r.items())))
json.dump({"n": len(convs), "rows": rows}, open("libraries_summary.json", "w"), indent=1)
