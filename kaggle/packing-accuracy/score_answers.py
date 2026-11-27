"""Score the packing-accuracy kernel output (copy of answer-accuracy/score_answers.py, other baseline).

    python score_answers.py out

Joins out/contexts.jsonl with out/answers_*.jsonl on (question, context). Prints judged accuracy per
retriever and budget, paired fixed/broken (sign test), accuracy split by whether all evidence was in
context, and per question kind. Writes packing_accuracy_summary.json.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
ctx = [json.loads(l) for l in (d / "contexts.jsonl").read_text().splitlines()]
ans = {}
for p in d.glob("answers_[0-9]*.jsonl"):  # answers_raw_* are unjudged
    for l in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        ans[(r["qid"], tuple(sorted(r["ids"])))] = r
for r in ctx:
    r["correct"] = ans[(r["qid"], tuple(sorted(r["ids"])))]["correct"]
by = defaultdict(dict)
for r in ctx:
    by[(r["adapter"], r["budget"])][r["qid"]] = r


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
out = {}
for (ad, b), q in sorted(by.items()):
    rs = list(q.values())
    s = {"acc": pct([r["correct"] for r in rs]), "n": len(rs),
         "acc_if_all_evidence": pct([r["correct"] for r in rs if r["recalled_all"]]),
         "acc_if_missing_evidence": pct([r["correct"] for r in rs if not r["recalled_all"]]),
         "recall_all": pct([r["recalled_all"] for r in rs])}
    kinds = defaultdict(list)
    for r in rs:
        kinds[r["kind"]].append(r["correct"])
    s["by_kind"] = {k: [pct(v), len(v)] for k, v in sorted(kinds.items())}
    out[f"{ad}|{b}"] = s
    print(f"{ad:28s} {b:6d} acc {s['acc']:5.1f}  recall_all {s['recall_all']:5.1f}  "
          f"acc|all-ev {s['acc_if_all_evidence']}  acc|missing {s['acc_if_missing_evidence']}")
    print("    ", s["by_kind"])
base = "rerank+prior recent=0.5"
for (ad, b), q in sorted(by.items()):
    if ad == base:
        continue
    bq = by[(base, b)]
    fx = [c for c in bq if q[c]["correct"] and not bq[c]["correct"]]
    br = [c for c in bq if bq[c]["correct"] and not q[c]["correct"]]
    same_ctx = sum(sorted(q[c]["ids"]) == sorted(bq[c]["ids"]) for c in bq)
    multi = lambda cs: sum(bq[c]["kind"] == "multi-session" for c in cs)
    out[f"paired|{ad}|{b}"] = {"fixed": len(fx), "broken": len(br), "p": round(sign_p(len(fx), len(br)), 3),
                              "multi_fixed": multi(fx), "multi_broken": multi(br), "same_context": same_ctx}
    print(f"{ad} {b}: fixed {len(fx)} / broken {len(br)} p={sign_p(len(fx), len(br)):.3f} "
          f"(multi {multi(fx)}/{multi(br)}; identical context on {same_ctx})")
(Path(__file__).parent / "packing_accuracy_summary.json").write_text(json.dumps(out, indent=1))
