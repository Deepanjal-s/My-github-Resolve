"""Same recency-window question for the bench's plain recent+bm25 (all 479). Writes packing_bm25_summary.json."""
import sys, json
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.adapters import RecentPlusBM25
from memory_recall_bench.runner import run, summarize
convs = load_longmemeval(r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json")
out = {}
for share in (0.5, 0.25, 0.0):
    class A(RecentPlusBM25):
        name = f"recent+bm25 share={share}"
        def __init__(self): super().__init__(recent_share=share)
    for s in summarize(run([A()], convs, [2000, 4000, 8000, 16000]), metric="recalled_all"):
        out[f'{s["adapter"]}|{s["budget"]}'] = round(100 * s["recall"], 1); print(s["adapter"], s["budget"], out[f'{s["adapter"]}|{s["budget"]}'], flush=True)
json.dump(out, open(r"C:\Projects\My git Resolve\kaggle\rerank-topn\packing_bm25_summary.json", "w"), indent=1)
