"""Builds the public CPU notebook over the re-ranking results dataset.

    python build_notebook.py            # writes notebook/reranking.ipynb + kernel-metadata.json
    python build_notebook.py --run DIR  # also runs the code cells locally against DIR (the dataset files)
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# More context didn't help my chatbot. A re-ranker did.

Chat assistants with "memory" retrieve old messages and paste the best matches into the prompt. Earlier I
found that giving a small reader 4x more retrieved context (4k → 16k tokens) raised **recall** from 63% to
90% but left **answer accuracy** flat at ~59%: the extra near-misses distract the model.

This notebook tests the obvious next idea: **spend the effort on precision instead of volume.** A
cross-encoder (`BAAI/bge-reranker-v2-m3`) re-scores the hybrid retriever's top 100 candidates, and the
context is packed from the new order.

- Data: the 479 evidence questions of LongMemEval_s (MIT).
- Harness: [memory-recall-bench](https://github.com/Deepanjal-s/memory-recall-bench) (open source).
- Reader and judge: Qwen3-4B-Instruct-2507, LongMemEval judge prompts, greedy decoding, on a free Kaggle T4.
- This notebook only re-scores the saved per-question outputs (CPU, ~10 s). The GPU run took ~4.7 h.

Arms: **recent+hybrid** (recent messages + BM25/embedding hybrid with RRF), **hybrid+rerank N=100**
(cross-encoder order replaces the hybrid's), **hybrid+rerank rrf** (cross-encoder order fused with the hybrid's)."""),
    code("""import glob, json, math, os
from collections import defaultdict
import matplotlib.pyplot as plt

DATA = os.environ.get("RESULTS_DIR") or os.path.dirname(
    (glob.glob("/kaggle/input/**/rerank_recall.jsonl", recursive=True) or ["./rerank_recall.jsonl"])[0])
read = lambda name: [json.loads(l) for l in open(os.path.join(DATA, name), encoding="utf-8") if l.strip()]

def sign_p(a, b):
    \"\"\"Two-sided sign test on paired wins/losses.\"\"\"
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)

recall = defaultdict(dict)
for r in read("rerank_recall.jsonl"):
    recall[(r["adapter"], r["budget"])][r["qid"]] = r

def judged(ctx_file, ans_file):
    ans = {(r["qid"], tuple(sorted(r["ids"]))): r["correct"] for r in read(ans_file)}
    out = defaultdict(dict)
    for c in read(ctx_file):
        c["correct"] = ans[(c["qid"], tuple(sorted(c["ids"])))]
        out[(c["adapter"], c["budget"])][c["qid"]] = c
    return out

acc = judged("rerank_contexts.jsonl", "rerank_answers.jsonl")
base = judged("base_contexts.jsonl", "base_answers.jsonl")
acc[("recent+hybrid", 16000)] = base[("recent+hybrid", 16000)]
ARMS = ["recent+hybrid", "hybrid+rerank N=100", "hybrid+rerank rrf"]
print(len(recall[("recent+hybrid", 4000)]), "questions;", sorted({b for _, b in recall}), "token budgets")"""),
    md("""## 1. Recall: does all the evidence reach the context?

"All evidence" means every session that holds part of the answer contributed a message. Paired against
recent+hybrid on the same questions: *fixed* = re-ranking newly gets all evidence in, *broken* = it loses it."""),
    code("""budgets = [2000, 4000, 8000, 16000]
pct = lambda q, m: 100 * sum(r[m] for r in q.values()) / len(q)
print(f"{'arm':22s}" + "".join(f"{b//1000:>5}k" for b in budgets))
for a in ARMS:
    print(f"{a:22s}" + "".join(f"{pct(recall[(a, b)], 'recalled_all'):6.1f}" for b in budgets))
print()
for a in ARMS[1:]:
    for b in budgets:
        q, bq = recall[(a, b)], recall[("recent+hybrid", b)]
        fx = sum(q[c]["recalled_all"] and not bq[c]["recalled_all"] for c in bq)
        br = sum(bq[c]["recalled_all"] and not q[c]["recalled_all"] for c in bq)
        print(f"{a:22s} {b//1000:>2}k  fixed {fx:3d} / broken {br:3d}  p={sign_p(fx, br):.3g}")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
for a, style in zip(ARMS, ["o-", "s-", "^--"]):
    ax.plot([b // 1000 for b in budgets], [pct(recall[(a, b)], "recalled_all") for b in budgets], style, label=a)
ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(["2k", "4k", "8k", "16k"])
ax.set_xlabel("context budget (tokens)"); ax.set_ylabel("all evidence in context (%)")
ax.grid(alpha=.3); ax.legend(frameon=False); ax.set_title("Re-ranking lifts recall most at small budgets")
plt.tight_layout(); plt.show()"""),
    md("""## 2. Answer accuracy: does the recall turn into correct answers?

Answers were generated at 4k and 8k for every arm (16k for the baseline comes from the earlier run on the
same contexts)."""),
    code("""rows = [("recent+hybrid", 4000), ("recent+hybrid", 8000), ("recent+hybrid", 16000),
        ("hybrid+rerank N=100", 4000), ("hybrid+rerank N=100", 8000),
        ("hybrid+rerank rrf", 4000), ("hybrid+rerank rrf", 8000)]
for a, b in rows:
    q = acc[(a, b)]
    print(f"{a:22s} {b//1000:>2}k  accuracy {pct(q, 'correct'):5.1f}   all-evidence recall {pct(q, 'recalled_all'):5.1f}")

def paired(x, y):
    X, Y = acc[x], acc[y]
    fx = sum(Y[q]["correct"] and not X[q]["correct"] for q in X)
    br = sum(X[q]["correct"] and not Y[q]["correct"] for q in X)
    print(f"{x[0]} {x[1]//1000}k -> {y[0]} {y[1]//1000}k: fixed {fx} / broken {br}, p={sign_p(fx, br):.3g}")

print()
best = ("hybrid+rerank N=100", 8000)
paired(("recent+hybrid", 4000), best)
paired(("recent+hybrid", 16000), best)
paired(("recent+hybrid", 8000), best)
paired(("recent+hybrid", 4000), ("hybrid+rerank N=100", 4000))

fig, ax = plt.subplots(figsize=(6.5, 3.6))
labels = [f"{a.replace('hybrid+rerank', 'rerank')}\\n{b//1000}k" for a, b in rows]
vals = [pct(acc[r], "correct") for r in rows]
bars = ax.bar(range(len(rows)), vals, color=["#9aa5b1"] * 3 + ["#3b6fd8"] * 2 + ["#7fa3e8"] * 2)
ax.bar_label(bars, fmt="%.1f", fontsize=8); ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(50, 70); ax.set_ylabel("judged accuracy (%)"); ax.set_title("Best answers: re-ranked, mid-sized context")
plt.tight_layout(); plt.show()"""),
    md("""## 3. The distractor penalty is still there

Accuracy *on questions where the context holds all the evidence*. If extra context were free, this would not
drop as the budget grows."""),
    code("""for a, b in rows:
    q = [r for r in acc[(a, b)].values() if r["recalled_all"]]
    print(f"{a:22s} {b//1000:>2}k  accuracy given all evidence {100*sum(r['correct'] for r in q)/len(q):5.1f}  (n={len(q)})")"""),
    md("""## What it shows

- **Re-ranking is the first change in this benchmark that moves recall a lot at small budgets**: +7.8 points
  at 2k and +8.4 at 4k, every budget significant. Replacing the hybrid order beats fusing with it.
- **Some of it becomes accuracy**: re-ranked 8k (64.9%) is the best configuration measured, clearly better
  than the hybrid at 4k or 16k. Against the hybrid at the *same* budget the gain is ~3 points (borderline).
- **Distractors still cost the reader**: accuracy given all evidence still falls from 4k to 8k. The re-ranker
  wins by fitting the evidence into fewer tokens.

**Caveats.** One reader and judge (Qwen3-4B). I hand-graded 50 random judge verdicts and agreed on ~95%; the
judge can be fooled by junk strings, so accuracy carries roughly ±2-3 points of judge noise (paired
comparisons less, since both arms share it). A larger reader in 16-bit is the open test.

Code, kernels and all earlier results (including two negative results):
[memory-recall-bench results](https://github.com/Deepanjal-s/memory-recall-bench/tree/main/results).
Written with AI assistance; every number above is computed in this notebook from the saved outputs."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "notebook"
out.mkdir(exist_ok=True)
nbf.write(nb, out / "reranking.ipynb")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/more-context-didn-t-help-a-re-ranker-did",
    "title": "More context didn't help. A re-ranker did.",
    "code_file": "reranking.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": False, "enable_gpu": False, "enable_internet": False,
    "dataset_sources": ["deepanjalshukla95/longmemeval-reranking-results"],
    "competition_sources": [], "kernel_sources": [], "keywords": ["nlp", "retrieval"],
}, indent=1))
print("wrote", out / "reranking.ipynb")

if "--run" in sys.argv:
    import os
    import matplotlib
    matplotlib.use("Agg")
    os.environ["RESULTS_DIR"] = sys.argv[sys.argv.index("--run") + 1]
    ns = {}
    for c in cells:
        if c.cell_type == "code":
            exec(c.source, ns)
