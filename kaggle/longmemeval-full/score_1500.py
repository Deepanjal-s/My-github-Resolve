import json, sys
from collections import defaultdict
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile
from memory_recall_bench.runner import run, summarize
ids = {json.loads(l)["id"] for l in open("convs_embed_subset.jsonl", encoding="utf-8")}
convs = [c for c in load_longmemeval("data/longmemeval_s_cleaned.json") if c.id in ids]
done = set(json.load(open("aim_rankings_hybrid_1500.json")))
convs = [c for c in convs if c.id in done]
print(len(convs), "conversations")
names = {"bm25": "app keyword (BM25)", "h4k": "hybrid, 4000-char chunks", "h15": "hybrid, 1500-char chunks"}
ads = [RankedFromFile("aim_rankings_bm25.json", convs, name=names["bm25"]),
       RankedFromFile("aim_rankings_hybrid_subset.json", convs, name=names["h4k"]),
       RankedFromFile("aim_rankings_hybrid_1500.json", convs, name=names["h15"])]
res = run(ads, convs, [4000, 16000])
out = {}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, by_kind=True, metric=m):
        print(m, s["adapter"], s["budget"], s.get("kind"), s["n"], round(s["recall"]*100), f'[{round(s["ci_low"]*100)}-{round(s["ci_high"]*100)}]')
        out.setdefault(m, []).append(s)
by = defaultdict(dict)
for r in res: by[(r.budget, r.conversation)][r.adapter] = r
for base in ["bm25", "h4k"]:
  for b in [4000, 16000]:
    for m in ["recalled", "recalled_all"]:
      f = br = 0
      for (bb, c), d in by.items():
          if bb != b: continue
          x, y = getattr(d[names[base]], m), getattr(d[names["h15"]], m)
          f += (y and not x); br += (x and not y)
      print(f"paired 1500 vs {base}", b, m, "fixed", f, "broken", br); out[f"paired_vs_{base}_{b}_{m}"] = {"fixed": f, "broken": br}
json.dump(out, open("aim_hybrid_1500_summary.json", "w"), indent=1, default=str)
