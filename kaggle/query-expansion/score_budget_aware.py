"""Budget-aware query expansion (BM25). Expansion helped at 16k and hurt at 4k: sub-query hits push the
original query's best messages out of a small budget. Variants that protect the original ranking:

  gate          plain BM25 below 16k, expansion at >= 16k (a policy, not a ranker)
  prefix f      the original query's ranking fills the first fraction f of the archive budget (tokens),
                then the fused (original + sub-queries) ranking fills the rest

    python score_budget_aware.py expansions.jsonl

All 479 questions with evidence, 4k/16k. Paired fixed/broken vs plain BM25 with a two-sided sign test.
Writes budget_aware_summary.json.
"""
import json
import math
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


def bm25_order(index, q, n):
    sc = index.scores(tokenize(q))
    return [i for i in sorted(range(n), key=lambda i: -sc[i]) if sc[i] > 0]


class BudgetAware(RecentPlusBM25):
    def __init__(self, mode, f=0.0, k=60, **kw):
        super().__init__(**kw)
        self.mode, self.f, self.k = mode, f, k
        self.name = "bm25+exp gate16k" if mode == "gate" else f"bm25+exp prefix f={f:g}"

    def rank(self, query, archive):
        index = BM25([tokenize(m.content) for m in archive])
        orig = bm25_order(index, query, len(archive))
        subs = exp.get(query, [])
        if not subs or (self.mode == "gate" and self.budget < 16000):
            return [archive[i] for i in orig]
        score = defaultdict(float)
        for r, i in enumerate(orig):
            score[i] += 1.0 / (self.k + r + 1)
        for s in subs:
            for r, i in enumerate(bm25_order(index, s, len(archive))):
                score[i] += 1.0 / (self.k + r + 1)
        fused = sorted(score, key=lambda i: -score[i])
        if self.mode == "gate":
            return [archive[i] for i in fused]
        cap, used, head = self.budget * (1 - self.recent_share) * self.f, 0, []
        for i in orig:
            if used + archive[i].tokens > cap:
                break
            head.append(i)
            used += archive[i].tokens
        seen = set(head)
        return [archive[i] for i in head + [i for i in fused if i not in seen]]


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


convs = load_longmemeval(HERE.parent / "longmemeval-full" / "data" / "longmemeval_s_cleaned.json")
ads = [RecentPlusBM25, lambda: BudgetAware("gate")] + [lambda f=f: BudgetAware("prefix", f) for f in (0.25, 0.5, 0.75)]
res = run(ads, convs, [4000, 16000])
out = {}
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m):
        key = f'{m}|{s["adapter"]}|{s["budget"]}'
        out[key] = [round(s["recall"] * 100, 1), round(s["ci_low"] * 100, 1), round(s["ci_high"] * 100, 1)]
        print(key, out[key])
by = defaultdict(dict)
for r in res:
    by[(r.adapter, r.budget)][r.conversation] = r
kinds = {c.id: c.kind for c in convs}
for name in sorted({r.adapter for r in res} - {"recent+bm25"}):
    for b in (4000, 16000):
        for metric in ("recalled", "recalled_all"):
            base, new = by[("recent+bm25", b)], by[(name, b)]
            fx = [c for c in base if getattr(new[c], metric) and not getattr(base[c], metric)]
            br = [c for c in base if getattr(base[c], metric) and not getattr(new[c], metric)]
            fm = sum(kinds[c] == "multi-session" for c in fx)
            bm = sum(kinds[c] == "multi-session" for c in br)
            out[f"paired|{name}|{b}|{metric}"] = {"fixed": len(fx), "broken": len(br), "p": round(sign_p(len(fx), len(br)), 3),
                                                  "multi_fixed": fm, "multi_broken": bm}
            print(f"{name} {b} {metric}: fixed {len(fx)} / broken {len(br)} p={sign_p(len(fx), len(br)):.3f} (multi {fm}/{bm})")
(HERE / "budget_aware_summary.json").write_text(json.dumps(out, indent=1))
