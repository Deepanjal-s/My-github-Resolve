"""Builds longmemeval-retrieval.ipynb: the full LongMemEval_s run with embeddings on a Kaggle GPU."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Keyword vs. embedding vs. hybrid retrieval for chat memory (LongMemEval_s, 479 questions)

The previous notebook ([What does your chatbot forget?](https://www.kaggle.com/code/deepanjalshukla95/what-does-your-chatbot-forget))
found that recency-only memory loses most facts and that BM25 retrieval recovers them, but BM25 was weakest on
**preference** questions and on **multi-session** questions that need several facts.

This notebook runs the full benchmark with [memory-recall-bench](https://github.com/Deepanjal-s/memory-recall-bench)
to test whether **embedding** retrieval, or a **hybrid** of the two, fixes those weaknesses. It embeds ~240k messages
on a GPU, then replays every conversation at three context budgets.

**Metric:** *evidence recall*, whether the answer-bearing message(s) are in the context the model would see.
"Any" needs one answer message; "all" needs one from every answer session.
""")

code(r"""
!pip install -q "memory-recall-bench[embeddings] @ git+https://github.com/Deepanjal-s/memory-recall-bench.git"
""")

code(r"""
import json, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from memory_recall_bench import load_longmemeval, run, summarize, FIFO, FlushToArchive, RecentPlusBM25, RecentPlusRandom
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusEmbedding, RecentPlusHybrid, sentence_transformers_encoder
import memory_recall_bench as mrb
print("memory-recall-bench", mrb.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

DATA = Path("longmemeval_s_cleaned.json")
if not DATA.exists():
    url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    urllib.request.urlretrieve(url, DATA)
convs = load_longmemeval(DATA)
print(len(convs), "conversations with labelled evidence;", sum(len(c.messages) for c in convs), "messages")
""")

md("## Embed every message once (GPU)\n`BAAI/bge-small-en-v1.5`, the first 512 characters of each message (same setting as the CPU pilot).")

code(r"""
device = "cuda" if torch.cuda.is_available() else "cpu"
cache = EmbeddingCache(sentence_transformers_encoder("BAAI/bge-small-en-v1.5", device=device), max_chars=512, batch_size=512)
t = time.time()
cache.get([m.content for c in convs for m in c.messages])
print(f"{len(cache):,} distinct messages embedded in {time.time()-t:.0f}s")
""")

md("## Replay all conversations")

code(r"""
QP = "Represent this sentence for searching relevant passages: "
ADAPTERS = [FIFO, FlushToArchive, RecentPlusRandom, RecentPlusBM25,
            lambda: RecentPlusEmbedding(cache, query_prefix=QP), lambda: RecentPlusHybrid(cache, query_prefix=QP)]
BUDGETS = [4000, 16000, 64000]
t = time.time()
results = run(ADAPTERS, convs, BUDGETS)
print(f"{len(results):,} runs in {time.time()-t:.0f}s")
res = pd.DataFrame([r.__dict__ for r in results])
res.to_csv("results.csv", index=False)
""")

code(r"""
NAMES = {"fifo": "FIFO", "flush": "llama_index Memory (default)", "recent+random": "Recent + random (control)",
         "recent+bm25": "Recent + BM25", "recent+embed": "Recent + embedding", "recent+hybrid": "Recent + hybrid"}
def table(metric):
    rows = pd.DataFrame(summarize(results, metric=metric))
    t = rows.pivot(index="adapter", columns="budget", values="recall").rename(index=NAMES)
    t.columns = [f"{b//1000}k" for b in t.columns]
    return t.loc[[NAMES[k] for k in NAMES]]
print("Any answer message kept"); display((table("recalled") * 100).round(0).astype(int).astype(str) + "%")
print("Evidence from every answer session kept"); display((table("recalled_all") * 100).round(0).astype(int).astype(str) + "%")
""")

md("## Where each retriever wins: by question type (16k budget, all-evidence)")

code(r"""
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#d6d5d0",
                     "axes.grid": True, "grid.color": "#ebeae6", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10, "figure.dpi": 110})
rows = pd.DataFrame(summarize(results, by_kind=True, metric="recalled_all"))
sub = rows[(rows.budget == 16000) & rows.adapter.isin(["recent+bm25", "recent+embed", "recent+hybrid"])]
piv = sub.pivot(index="kind", columns="adapter", values="recall").sort_values("recent+bm25")
n = sub.groupby("kind").n.first()
fig, ax = plt.subplots(figsize=(8, 4.4))
y = np.arange(len(piv)); h = 0.26
for i, (col, c) in enumerate([("recent+bm25", GRAY), ("recent+embed", ORANGE), ("recent+hybrid", BLUE)]):
    ax.barh(y + (i - 1) * h, piv[col] * 100, height=h - 0.04, color=c, label=NAMES[col])
ax.set_yticks(y, [f"{k} (n={n[k]})" for k in piv.index])
ax.set_xlabel("Evidence from every answer session kept (%)"); ax.set_xlim(0, 105)
ax.set_title("Which retriever keeps the facts, by question type (16k budget)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()
display((piv * 100).round(0))
""")

md(r"""
## Reading the results

*(Interpretation is written after the run, from the numbers above. See the repository's `results/` folder for the
version committed alongside the package.)*

## Limitations

- **Recall, not answer accuracy**: the answer being in context is necessary, not sufficient.
- One small embedding model (bge-small) and the first 512 characters of each message; a larger model or chunking
  could do better.
- Token counts are approximate (characters / 4).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "longmemeval-retrieval.ipynb")
print("wrote longmemeval-retrieval.ipynb")
