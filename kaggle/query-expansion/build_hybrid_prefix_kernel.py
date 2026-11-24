"""Private Kaggle GPU kernel: confirm the budget-aware "prefix" expansion on the hybrid (BM25 + bge-small).
f=0.5 was picked on BM25 (score_budget_aware.py); here it is fixed in advance, so this run is the check.
The original query's hybrid ranking fills the first half of the archive budget; the fused (original +
sub-queries, RRF) ranking fills the rest. Writes /kaggle/working/hybrid_prefix_results.csv.

    python build_hybrid_prefix_kernel.py   then   kaggle kernels push -p hybrid_prefix_kernel
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
import pandas as pd
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

class PrefixHybrid(RecentPlusHybrid):
    def __init__(self, f):
        super().__init__(cache, query_prefix=QP)
        self.f, self.name = f, f"hybrid+exp prefix f={f:g}"
    def rank(self, query, archive):
        orig = super().rank(query, archive)
        subs = EXP.get(query, [])
        if not subs:
            return orig
        score = defaultdict(float)
        for q, ranked in [(query, orig)] + [(s, super().rank(s, archive)) for s in subs]:
            for r, m in enumerate(ranked):
                score[m.id] += 1.0 / (self.k + r + 1)
        by = {m.id: m for m in archive}
        cap, used, head = self.budget * (1 - self.recent_share) * self.f, 0, []
        for m in orig:
            if used + m.tokens > cap:
                break
            head.append(m); used += m.tokens
        seen = {m.id for m in head}
        return head + [by[i] for i in sorted(score, key=lambda i: -score[i]) if i not in seen]

ads = [lambda: RecentPlusHybrid(cache, query_prefix=QP), lambda: PrefixHybrid(0.5)]
t = time.time(); res = run(ads, convs, [4000, 16000]); print(f"ran in {time.time()-t:.0f}s")
pd.DataFrame([r.__dict__ for r in res]).to_csv("/kaggle/working/hybrid_prefix_results.csv", index=False)
DATA.unlink()
print(pd.DataFrame([r.__dict__ for r in res]).groupby(["adapter", "budget"])[["recalled", "recalled_all"]].mean().round(3))
''' % {"exp": json.dumps(EXP)}

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell("# Budget-aware query expansion on the hybrid (LongMemEval_s)\n\nWork kernel for memory-recall-bench."),
            nbf.v4.new_code_cell('!pip install -q "memory-recall-bench[embeddings] @ git+https://github.com/Deepanjal-s/memory-recall-bench.git"'),
            nbf.v4.new_code_cell(CODE.strip())]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / "hybrid_prefix_kernel"
d.mkdir(exist_ok=True)
nbf.write(nb, d / "hybrid-prefix.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-hybrid-prefix-expansion", "title": "lme hybrid prefix expansion",
    "code_file": "hybrid-prefix.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
