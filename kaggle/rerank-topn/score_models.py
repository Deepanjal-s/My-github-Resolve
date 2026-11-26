"""Recall of the app's BM25 top 30 re-scored by each re-ranker model, all 479, paired vs MiniLM-L-6 (the app default).

    python score_models.py mout/rerank_models.json
"""
import json, math, sys, tempfile
from pathlib import Path
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile
from memory_recall_bench.runner import run, summarize

d = json.load(open(sys.argv[1]))
models = [k for k in d if k != "timing"]
convs = load_longmemeval(r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json")
tmp = Path(tempfile.mkdtemp()); ads = []
for i, k in enumerate(models):
    p = tmp / f"{i}.json"; p.write_text(json.dumps(d[k])); ads.append(RankedFromFile(str(p), convs, name=k))
budgets = [2000, 4000, 8000, 16000]
res = run(ads, convs, budgets)
out = {"timing": d["timing"]}
for s in summarize(res, metric="recalled_all"):
    out[f'{s["adapter"]}|{s["budget"]}'] = round(s["recall"] * 100, 1)
by = {}
for r in res: by.setdefault((r.budget, r.conversation), {})[r.adapter] = r
sp = lambda a, b: 1.0 if a + b == 0 else min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
base = models[0]
for k in models:
    row = [out[f"{k}|{b}"] for b in budgets]
    f = sum(1 for (bb, c), x in by.items() if bb == 4000 and x[k].recalled_all and not x[base].recalled_all)
    br = sum(1 for (bb, c), x in by.items() if bb == 4000 and x[base].recalled_all and not x[k].recalled_all)
    out[f"paired_vs_base|{k}|4000"] = {"fixed": f, "broken": br, "p": round(sp(f, br), 4)}
    print(f"{k:42s}", row, f"4k vs L-6: {f}/{br} p={sp(f, br):.3g}", d["timing"][k])
json.dump(out, open(Path(__file__).parent / "models_summary.json", "w"), indent=1)
