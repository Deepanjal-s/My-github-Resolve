"""Builds memory-strategies.ipynb. memsim.py is embedded as a cell so the notebook runs offline on Kaggle."""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# What does your chatbot forget?
### Comparing chat-memory strategies on LongMemEval, measured without calling an LLM

Every chat agent has a context budget. When the conversation outgrows it, the memory layer decides what stays.
This notebook replays real long conversations from **LongMemEval** through four memory strategies and asks
one measurable question:

> When the user finally asks *"What degree did I graduate with?"*, is the message that contains the answer
> still in the context the model will see?

That's **evidence recall**. It is an upper bound on answer accuracy: a model can't answer from a message it never sees.
It needs no API calls, so the whole study is free, fast and fully reproducible.

| Strategy | What it does |
|---|---|
| **FIFO** | Keep the most recent messages that fit |
| **Flush-to-archive** | The `llama_index` `Memory` policy: when history exceeds 70% of the limit, move the oldest messages out in chunks of `token_flush_size` (default 3000) |
| **Recent + BM25** | Half the budget for recent messages, half for archived messages that match the question (BM25 keyword retrieval) |
| **Recent + random** | Same split, but the archive half is filled at random. This **control** separates "retrieval helps" from "having more budget helps" |

*Data: LongMemEval (Wu et al., 2024, MIT license), dev split via the Kaggle dataset `bonepyaesone/longmemeval-dev-json`.*
""")

code(r"""
import json, sys, random, statistics as st
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

candidates = list(Path("/kaggle/input").glob("**/longmemeval_dev.json")) + [Path("data/longmemeval_dev.json")]
DATA = next(p for p in candidates if p.exists())
questions = json.load(open(DATA, encoding="utf8"))
print(f"{len(questions)} questions; types: {pd.Series([q['question_type'] for q in questions]).value_counts().to_dict()}")
""")

md("## 1. The simulator\nEverything is in one small, self-contained module: token counting, the four strategies and a BM25 implementation (Kaggle notebooks run offline, so there are no extra packages).")

code(Path("memsim.py").read_text(encoding="utf8"))

code(r"""
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#d6d5d0",
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10, "figure.dpi": 110,
})
convs = [(q, flatten(q)) for q in questions]
ages = np.array([evidence_age_tokens(t) for _, t in convs])
sizes = np.array([sum(x.tokens for x in t) for _, t in convs])
print(f"Conversation length: median {np.median(sizes)/1000:.0f}k tokens, {np.median([len(t) for _, t in convs]):.0f} messages")
print(f"Answer age (tokens between the answer and the question): median {np.median(ages)/1000:.0f}k, range {ages.min()/1000:.0f}k-{ages.max()/1000:.0f}k")
""")

md(r"""
## 2. Sanity check before trusting any number

FIFO has an exact closed form: the answer survives **if and only if** the answer message and everything after it fit in
the budget. If the simulator disagrees with that rule even once, nothing else in this notebook can be trusted.
""")

code(r"""
mismatches = 0
for B in [2000, 4000, 8000, 16000, 32000, 64000]:
    for q, t in convs:
        fits = any(sum(y.tokens for y in t[e.idx:]) <= B for e in t if e.evidence)
        mismatches += fits != evidence_recalled(t, fifo(t, B))
print("FIFO vs closed form, mismatches:", mismatches)
assert mismatches == 0
""")

md("## 3. Evidence recall vs. context budget")

code(r"""
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
STRATS = {
    "FIFO": lambda t, B, q: fifo(t, B),
    "Flush-to-archive (default 3000)": lambda t, B, q: flush_to_archive(t, B),
    # Stable per-question seed (Python's hash() is randomised per process, which would make the control irreproducible).
    "Recent + random (control)": lambda t, B, q: recent_plus_random(t, B, seed=sum(map(ord, q))),
    "Recent + BM25": lambda t, B, q: recent_plus_bm25(t, B, question=q),
}
rows = []
for B in BUDGETS:
    for name, fn in STRATS.items():
        for q, t in convs:
            rows.append({"budget": B, "strategy": name, "qid": q["question_id"],
                         "hit": evidence_recalled(t, fn(t, B, q["question"]))})
res = pd.DataFrame(rows)

rng = np.random.default_rng(0)
def boot(x, n=2000):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(n, len(x))).mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])

summary = (res.groupby(["strategy", "budget"]).hit.apply(lambda s: pd.Series(boot(s), index=["recall", "lo", "hi"]))
              .unstack().reset_index())
display(summary.pivot(index="budget", columns="strategy", values="recall").map("{:.0%}".format))
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4.6))
colors = {"FIFO": GRAY, "Flush-to-archive (default 3000)": ORANGE, "Recent + random (control)": YELLOW, "Recent + BM25": BLUE}
for name, c in colors.items():
    s = summary[summary.strategy == name].sort_values("budget")
    x = np.arange(len(s))
    ax.fill_between(x, s.lo * 100, s.hi * 100, color=c, alpha=0.12, linewidth=0)
    ax.plot(x, s.recall * 100, color=c, lw=2, marker="o", ms=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
ax.set_xticks(np.arange(len(BUDGETS)), [f"{b//1000}k" for b in BUDGETS])
ax.set_xlabel("Context budget for chat history (tokens)"); ax.set_ylabel("Answer still in context (%)")
ax.set_ylim(0, 105); ax.set_xlim(-0.2, len(BUDGETS) - 0.4)
ax.set_title("Retrieval keeps the answer; recency-based strategies lose it")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=9)
plt.tight_layout(); plt.show()
""")

md(r"""
**Reading it:** the median answer is ~55k tokens old, so any recency-based policy needs a budget at least that big.
FIFO only reaches ~58% at 64k. The **random control sits on top of FIFO**, so the jump for BM25 isn't about having
budget to spare: it comes from *choosing* the right old messages. Flush-to-archive does worse than FIFO at the same budget,
because it keeps the history well below the limit, as the next section shows.

## 4. Why flush-to-archive underperforms: the sawtooth (llama_index issue #24117)

`Memory.from_defaults()` flushes 3000 tokens at a time. With `token_limit=3644` (what chat engines use for a 3,900-token
model), the history budget is 0.7 × 3644 ≈ 2,551 tokens, *smaller than one flush*, so every flush empties almost everything.
Below, one conversation is replayed with the default and with a flush size scaled to 10% of the budget.
""")

code(r"""
q0, t0 = convs[0]
LIMIT = 3644
traces = {}
for label, fs in [("default flush (3000)", None), ("scaled flush (10% of budget)", int(LIMIT * 0.7 * 0.1))]:
    tr = []; flush_to_archive(t0, LIMIT, flush_size=fs, trace=tr); traces[label] = tr

fig, ax = plt.subplots(figsize=(8, 3.6))
for (label, tr), c in zip(traces.items(), [ORANGE, BLUE]):
    ax.plot(tr[:160], color=c, lw=1.6, label=f"{label}: this conversation averages {np.mean(tr[50:])/(LIMIT*0.7):.0%} full")
ax.axhline(LIMIT * 0.7, color=INK2, lw=1, ls="--"); ax.text(1, LIMIT * 0.7 + 60, "history budget", color=INK2, fontsize=9)
ax.set_xlabel("Message number"); ax.set_ylabel("Tokens in active history")
ax.set_title("Default flush size empties the history again and again")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
plt.tight_layout(); plt.show()

fill = {}
for lab, fs in [("default", None), ("scaled 10%", int(LIMIT * 0.7 * 0.1))]:
    per_conv = []
    for _, t in convs:
        tr = []
        flush_to_archive(t, LIMIT, flush_size=fs, trace=tr)
        per_conv.append(np.mean(tr[50:]) / (LIMIT * 0.7))  # skip the warm-up before the first flush
    fill[lab] = np.mean(per_conv)
print("Average fill of the history budget across all 50 conversations:", {k: f"{v:.0%}" for k, v in fill.items()})
""")

md(r"""
With the default, the model sees **about half** the history it has room for, on average. A flush size scaled to the
budget keeps it close to full. The fix proposed upstream (scale the default flush size with the budget) is the right shape.

## 5. Which answers survive? Recall by answer age
""")

code(r"""
B = 16000
AGE_LABELS = ["< 8k", "8k-32k", "32k-64k", "> 64k"]
age_of = dict(zip([q["question_id"] for q, _ in convs],
                  pd.cut(ages, [-1, 8000, 32000, 64000, 1e9], labels=AGE_LABELS).astype(str)))
age_bins = pd.Categorical([age_of[q["question_id"]] for q, _ in convs], categories=AGE_LABELS, ordered=True)
sub = res[res.budget == B].copy()
sub["age"] = pd.Categorical(sub.qid.map(age_of), categories=AGE_LABELS, ordered=True)
by_age = (sub
            .groupby(["age", "strategy"], observed=True).hit.mean().unstack() * 100)
n_per_bin = pd.Series(age_bins).value_counts().reindex(by_age.index)

fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.2
for i, (name, c) in enumerate(colors.items()):
    ax.bar(np.arange(len(by_age)) + (i - 1.5) * w, by_age[name], width=w - 0.03, color=c, label=name)
ax.set_xticks(np.arange(len(by_age)), [f"{a}\n(n={n})" for a, n in zip(by_age.index, n_per_bin)])
ax.set_xlabel("How long ago the answer was said (tokens)"); ax.set_ylabel("Recall (%)")
ax.set_title(f"At a {B//1000}k budget, only retrieval reaches answers older than the window")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25)); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
""")

md("## 6. Ablation: how much of the budget should go to recent messages?")

code(r"""
shares = [0.0, 0.25, 0.5, 0.75, 0.9]
abl = pd.DataFrame([{"recent_share": s, "budget": B,
                     "recall": np.mean([evidence_recalled(t, recent_plus_bm25(t, B, question=q["question"], recent_share=s)) for q, t in convs])}
                    for s in shares for B in [2000, 4000, 8000]])
display(abl.pivot(index="recent_share", columns="budget", values="recall").map("{:.0%}".format))
""")

md(r"""
Recall stays high until most of the budget goes to recent messages. In a real agent the recent window matters for
coherence (pronouns, follow-ups), which this metric doesn't measure. So the split is a product decision,
and this table shows what each choice costs in recall.

## 7. Limitations: please read before quoting the numbers

- **Easy questions for keyword search.** This dev split is all `single-session-user` questions ("What degree did I graduate with?"),
  which share words with the answer message. Paraphrased or multi-hop questions (other LongMemEval types) would favour embedding
  retrieval and lower BM25's numbers. Treat the BM25 curve as **optimistic**.
- **Recall, not accuracy.** Having the answer in context is necessary, not sufficient; models still miss facts buried in long contexts.
- **The question is known at retrieval time.** That's realistic for answering, but not for proactive memory (e.g. summarising
  before anyone asks).
- **Approximate tokens** (characters / 4) and **n = 50 questions**; confidence bands are shown where it matters.

## Takeaways

1. **Recency is not memory.** FIFO and flush policies forget anything older than the window, and that is most answers.
2. **Retrieval over an archive is what makes old facts reachable.** A random-retrieval control with the same budget
   shows the gain comes from retrieval, not from budget.
3. **Small configuration defaults matter:** a flush size larger than the budget leaves the model using about half its context.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "memory-strategies.ipynb")
print("wrote memory-strategies.ipynb with", len(cells), "cells")
