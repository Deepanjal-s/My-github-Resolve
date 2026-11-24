"""Score query expansion: BM25 on the original question fused (RRF) with BM25 on each LLM sub-query.

    python score_expansion.py expansions.jsonl

All questions with evidence (479). Prints recall with CIs and paired fixed/broken vs plain BM25, overall and for
multi-session questions. Writes expansion_summary.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.adapters import RecentPlusBM25  # noqa: E402
from memory_recall_bench.bm25 import BM25, tokenize  # noqa: E402
from memory_recall_bench.datasets import load_longmemeval  # noqa: E402
from memory_recall_bench.runner import run, summarize  # noqa: E402

HERE = Path(__file__).parent
exp = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    exp[r["question"]] = r["queries"]


class ExpandedBM25(RecentPlusBM25):
    """RRF over the original query (weight `w0`) and each sub-query (weight 1)."""

    def __init__(self, w0=1.0, k=60, **kw):
        super().__init__(**kw)
        self.w0, self.k = w0, k
        self.name = f"bm25+expansion w0={w0:g}"

    def rank(self, query, archive):
        index = BM25([tokenize(m.content) for m in archive])
        score = defaultdict(float)
        for q, w in [(query, self.w0)] + [(s, 1.0) for s in exp.get(query, [])]:
            sc = index.scores(tokenize(q))
            for r, i in enumerate(i for i in sorted(range(len(archive)), key=lambda i: -sc[i]) if sc[i] > 0):
                score[i] += w / (self.k + r + 1)
        return [archive[i] for i in sorted(score, key=lambda i: -score[i])]


convs = load_longmemeval(HERE.parent / "longmemeval-full" / "data" / "longmemeval_s_cleaned.json")
missing = sum(c.question not in exp for c in convs)
print(f"{len(convs)} questions, {missing} without an expansion")
ads = [RecentPlusBM25, lambda: ExpandedBM25(1.0), lambda: ExpandedBM25(2.0)]
res = run(ads, convs, [4000, 16000])
out = {}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m) + summarize(res, metric=m, by_kind=True):
        key = f'{m}|{s["adapter"]}|{s["budget"]}|{s.get("kind") or "all"}'
        out[key] = [round(s["recall"] * 100, 1), round(s["ci_low"] * 100, 1), round(s["ci_high"] * 100, 1)]
        if not s.get("kind") or s["kind"] == "multi-session":
            print(key, out[key])
by = defaultdict(dict)
for r in res:
    by[(r.adapter, r.budget)][r.conversation] = r
kinds = {c.id: c.kind for c in convs}
for name in {r.adapter for r in res} - {"recent+bm25"}:
    for b in (4000, 16000):
        for metric in ("recalled", "recalled_all"):
            base, new = by[("recent+bm25", b)], by[(name, b)]
            fixed = sum(getattr(new[c], metric) and not getattr(base[c], metric) for c in base)
            broken = sum(getattr(base[c], metric) and not getattr(new[c], metric) for c in base)
            fm = sum(getattr(new[c], metric) and not getattr(base[c], metric) for c in base if kinds[c] == "multi-session")
            bm = sum(getattr(base[c], metric) and not getattr(new[c], metric) for c in base if kinds[c] == "multi-session")
            out[f"paired|{name}|{b}|{metric}"] = {"fixed": fixed, "broken": broken, "multi_fixed": fm, "multi_broken": bm}
            print(f"{name} {b} {metric}: fixed {fixed} / broken {broken} (multi-session {fm}/{bm})")
(HERE / "expansion_summary.json").write_text(json.dumps(out, indent=1))
