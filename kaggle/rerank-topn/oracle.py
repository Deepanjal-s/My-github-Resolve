"""Ceiling: a perfect re-ranker over the app's BM25 top N (evidence first, rest in BM25 order).

    python oracle.py aim_kw.json
"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile
from memory_recall_bench.runner import run, summarize

kw = json.load(open(sys.argv[1]))
convs = load_longmemeval(r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json")
ev = {c.id: c.evidence_ids for c in convs}
tmp = Path(tempfile.mkdtemp()); ads = []
for n in (30, 100, 200):
    r = {}
    for cid, ids in kw.items():
        top = ids[:n]
        r[cid] = [i for i in top if i in ev[cid]] + [i for i in top if i not in ev[cid]] + ids[n:]
    p = tmp / f"{n}.json"; p.write_text(json.dumps(r)); ads.append(RankedFromFile(str(p), convs, name=f"oracle N={n}"))
out = {}
for s in summarize(run(ads, convs, [2000, 4000, 8000, 16000]), metric="recalled_all"):
    out[f'{s["adapter"]}|{s["budget"]}'] = round(s["recall"] * 100, 1)
for k, v in sorted(out.items()): print(k, v)
json.dump(out, open(Path(__file__).parent / "oracle_summary.json", "w"), indent=1)
