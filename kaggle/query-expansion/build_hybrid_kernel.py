"""Private Kaggle GPU kernel: query expansion on top of the embedding hybrid (BM25 + bge-small, RRF), all 479
questions, 4k/16k. Writes /kaggle/working/hybrid_expansion_results.csv (one row per adapter x budget x question).

    python build_hybrid_kernel.py   then   kaggle kernels push -p hybrid_kernel
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
EXP = {}
for line in (HERE / "expansions.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    EXP[r["question"]] = r["queries"]

CODE = r'''
import json, time, urllib.request
from collections import defaultdict
from pathlib import Path
import pandas as pd, torch
from memory_recall_bench import load_longmemeval, run
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusHybrid, sentence_transformers_encoder

EXP = json.loads(%(exp)r)
DATA = Path("longmemeval_s_cleaned.json")
if not DATA.exists():
    urllib.request.urlretrieve("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json", DATA)
convs = load_longmemeval(DATA)
cache = EmbeddingCache(sentence_transformers_encoder("BAAI/bge-small-en-v1.5", device="cuda"), max_chars=512, batch_size=512)
t = time.time(); cache.get([m.content for c in convs for m in c.messages]); print(f"embedded in {time.time()-t:.0f}s", flush=True)
QP = "Represent this sentence for searching relevant passages: "

class ExpandedHybrid(RecentPlusHybrid):
    def __init__(self, w0, **kw):
        super().__init__(cache, query_prefix=QP, **kw)
        self.w0, self.name = w0, f"hybrid+expansion w0={w0:g}"
    def rank(self, query, archive):
        score = defaultdict(float)
        for q, w in [(query, self.w0)] + [(s, 1.0) for s in EXP.get(query, [])]:
            for r, m in enumerate(super().rank(q, archive)):
                score[m.id] += w / (self.k + r + 1)
        by = {m.id: m for m in archive}
        return [by[i] for i in sorted(score, key=lambda i: -score[i])]

ads = [lambda: RecentPlusHybrid(cache, query_prefix=QP), lambda: ExpandedHybrid(1.0), lambda: ExpandedHybrid(2.0)]
t = time.time(); res = run(ads, convs, [4000, 16000]); print(f"ran in {time.time()-t:.0f}s")
pd.DataFrame([r.__dict__ for r in res]).to_csv("/kaggle/working/hybrid_expansion_results.csv", index=False)
DATA.unlink()
print(pd.DataFrame([r.__dict__ for r in res]).groupby(["adapter", "budget"])[["recalled", "recalled_all"]].mean().round(3))
''' % {"exp": json.dumps(EXP)}

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell("# Query expansion on the hybrid retriever (LongMemEval_s)\n\nWork kernel for memory-recall-bench."),
            nbf.v4.new_code_cell('!pip install -q "memory-recall-bench[embeddings] @ git+https://github.com/Deepanjal-s/memory-recall-bench.git"'),
            nbf.v4.new_code_cell(CODE.strip())]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / "hybrid_kernel"
d.mkdir(exist_ok=True)
nbf.write(nb, d / "hybrid-expansion.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-hybrid-expansion", "title": "lme hybrid expansion",
    "code_file": "hybrid-expansion.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
