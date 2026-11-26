"""ai-chat-memory keyword search vs keyword + on-device cross-encoder (MiniLM-L-6), all 479, paired.

    python score_rerank.py aim_kw.json aim_kw_rr.json
"""
import json, math, sys
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile
from memory_recall_bench.runner import run, summarize

kw, rr = sys.argv[1], sys.argv[2]
done = set(json.load(open(kw))) & set(json.load(open(rr)))
convs = [c for c in load_longmemeval("data/longmemeval_s_cleaned.json") if c.id in done]
print(len(convs), "conversations")
ads = [RankedFromFile(kw, convs, name="keyword (BM25)"), RankedFromFile(rr, convs, name="keyword + rerank")]
budgets = [2000, 4000, 8000, 16000]
res = run(ads, convs, budgets)
sign_p = lambda a, b: 1.0 if a + b == 0 else min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
out = {}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m):
        print(m, s["adapter"], s["budget"], round(s["recall"] * 100, 1), f'[{s["ci_low"]*100:.1f}-{s["ci_high"]*100:.1f}]')
        out[f'{m}|{s["adapter"]}|{s["budget"]}'] = round(s["recall"] * 100, 1)
by = {}
for r in res:
    by.setdefault((r.budget, r.conversation), {})[r.adapter] = r
for b in budgets:
    for m in ["recalled", "recalled_all"]:
        f = br = 0
        for (bb, c), d in by.items():
            if bb == b:
                x, y = getattr(d["keyword (BM25)"], m), getattr(d["keyword + rerank"], m)
                f += y and not x; br += x and not y
        print(f"paired {b} {m}: fixed {f} broken {br} p={sign_p(f, br):.3g}")
        out[f"paired|{b}|{m}"] = {"fixed": f, "broken": br, "p": round(sign_p(f, br), 4)}
json.dump(out, open("aim_rerank_summary.json", "w"), indent=1)
