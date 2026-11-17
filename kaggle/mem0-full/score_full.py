"""Merges the 3 Kaggle shard outputs and scores mem0 on all 479 questions, paired per question against the
GPU run's recent+bm25 and recent+hybrid (kaggle/longmemeval-full/kaggle_out/results.csv).

    python score_full.py out/shard1 out/shard2 out/shard3
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval  # noqa: E402
from memory_recall_bench.external import RankedFromFile  # noqa: E402
from memory_recall_bench.runner import run, summarize  # noqa: E402

FULL = r"C:\Projects\My git Resolve\kaggle\longmemeval-full"
rankings = {}
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "mem0_rankings_shard*.json")):
        part = json.load(open(f))
        assert not set(part) & set(rankings), "shards overlap"
        rankings.update(part)
json.dump(rankings, open("mem0_rankings_full.json", "w"))
convs = load_longmemeval(os.path.join(FULL, "data", "longmemeval_s_cleaned.json"))
missing = [c.id for c in convs if c.id not in rankings]
print(len(rankings), "ranked;", len(missing), "missing", missing[:5])
convs = [c for c in convs if c.id in rankings]

res = run([lambda: RankedFromFile("mem0_rankings_full.json", convs, name="mem0")], convs, [4000, 16000, 64000])
out = {"n": len(convs), "summary": {}, "paired": {}}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m) + summarize(res, metric=m, by_kind=True):
        key = f'{m}|{s["budget"]}|{s.get("kind", "ALL")}'
        out["summary"][key] = [round(s["recall"] * 100, 1), round(s["ci_low"] * 100, 1), round(s["ci_high"] * 100, 1), s["n"]]
        print(key, out["summary"][key])

base = defaultdict(dict)
with open(os.path.join(FULL, "kaggle_out", "results.csv")) as f:
    for r in csv.DictReader(f):
        base[(r["adapter"], int(r["budget"]), r["conversation"])] = r
ids = {c.id for c in convs}
for other in ["recent+bm25", "recent+hybrid", "recent+embed"]:
    for b in [4000, 16000, 64000]:
        for m in ["recalled", "recalled_all"]:
            fixed = broken = 0
            for r in res:
                if r.budget != b or r.conversation not in ids:
                    continue
                o = base.get((other, b, r.conversation))
                if o is None:
                    continue
                x, y = o[m] == "True", getattr(r, m)
                fixed += y and not x
                broken += x and not y
            out["paired"][f"{other}|{b}|{m}"] = [fixed, broken]
            print(f"mem0 vs {other} {b} {m}: fixed {fixed}, broken {broken}")
json.dump(out, open("mem0_full_summary.json", "w"), indent=1)
