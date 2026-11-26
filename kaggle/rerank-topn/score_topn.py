"""Recall of ai-chat-memory keyword search vs re-ranking its top N (30/50/100/200), all 479, paired vs N=30.

    python score_topn.py aim_kw.json aim_kw_rr.json out/rerank_topn.json
"""
import json, math, sys, tempfile
from pathlib import Path
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile
from memory_recall_bench.runner import run, summarize

kw, app, topn = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
convs = load_longmemeval(r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json")
tmp = Path(tempfile.mkdtemp())
files = {"keyword": kw, "app rerank N=30": app}
for k in ("N=30", "N=50", "N=100", "N=200"):
    p = tmp / f"{k}.json"; p.write_text(json.dumps(topn[k])); files[f"rerank {k}"] = str(p)
ads = [RankedFromFile(f, convs, name=n) for n, f in files.items()]
budgets = [2000, 4000, 8000, 16000]
res = run(ads, convs, budgets)
sign_p = lambda a, b: 1.0 if a + b == 0 else min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
out = {"timing": topn["timing"]}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m):
        print(m, s["adapter"], s["budget"], round(s["recall"] * 100, 1))
        out[f'{m}|{s["adapter"]}|{s["budget"]}'] = round(s["recall"] * 100, 1)
by = {}
for r in res:
    by.setdefault((r.budget, r.conversation), {})[r.adapter] = r
for n in ("N=50", "N=100", "N=200"):
    for b in budgets:
        f = br = 0
        for (bb, c), d in by.items():
            if bb == b:
                x, y = d["rerank N=30"].recalled_all, d[f"rerank {n}"].recalled_all
                f += y and not x; br += x and not y
        print(f"paired vs N=30, {n} {b} all: fixed {f} broken {br} p={sign_p(f, br):.3g}")
        out[f"paired_vs_30|{n}|{b}"] = {"fixed": f, "broken": br, "p": round(sign_p(f, br), 4)}
json.dump(out, open(Path(__file__).parent / "topn_summary.json", "w"), indent=1)
