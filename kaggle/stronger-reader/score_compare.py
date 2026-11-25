"""Compare the Qwen3-8B reader (this kernel) with the Qwen3-4B reader (../answer-accuracy) on identical contexts.

    python score_compare.py out ../answer-accuracy/out
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load(d):
    d = Path(d)
    ctx = [json.loads(l) for l in (d / "contexts.jsonl").read_text().splitlines()]
    ans = {}
    for p in d.glob("answers_[0-9]*.jsonl"):
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            ans[(r["qid"], tuple(sorted(r["ids"])))] = r["correct"]
    out = {}
    for r in ctx:
        if r["adapter"] == "recent+hybrid":
            key = (r["qid"], tuple(sorted(r["ids"])))
            out[(r["qid"], r["budget"])] = dict(r, correct=ans[key], key=key)
    return out


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


big, small = load(sys.argv[1]), load(sys.argv[2])
same = sum(big[k]["key"] == small[k]["key"] for k in big if k in small)
print(f"identical contexts: {same} / {len(big)}")
pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
res = {}
for name, d in (("8B", big), ("4B", small)):
    for b in (4000, 16000):
        rs = [r for (q, bb), r in d.items() if bb == b]
        res[f"{name}|{b}"] = {"acc": pct([r["correct"] for r in rs]),
                              "acc_if_all_evidence": pct([r["correct"] for r in rs if r["recalled_all"]]),
                              "by_kind": {k: pct([r["correct"] for r in rs if r["kind"] == k])
                                          for k in sorted({r["kind"] for r in rs})}}
        print(name, b, res[f"{name}|{b}"])
for name, d in (("8B", big), ("4B", small)):
    qs = [q for (q, b) in d if b == 4000 and d[(q, 4000)]["recalled_all"] and d[(q, 16000)]["recalled_all"]]
    a4, a16 = [d[(q, 4000)]["correct"] for q in qs], [d[(q, 16000)]["correct"] for q in qs]
    fx = sum(y and not x for x, y in zip(a4, a16)); br = sum(x and not y for x, y in zip(a4, a16))
    res[f"{name}|both-all-ev"] = {"n": len(qs), "acc4k": pct(a4), "acc16k": pct(a16), "fixed": fx, "broken": br,
                                  "p": round(sign_p(fx, br), 3)}
    print(name, "both all-evidence:", res[f"{name}|both-all-ev"])
for b in (4000, 16000):
    ks = [k for k in big if k[1] == b and k in small]
    fx = sum(big[k]["correct"] and not small[k]["correct"] for k in ks)
    br = sum(small[k]["correct"] and not big[k]["correct"] for k in ks)
    res[f"8B vs 4B|{b}"] = {"fixed": fx, "broken": br, "p": round(sign_p(fx, br), 3)}
    print("8B vs 4B", b, res[f"8B vs 4B|{b}"])
(Path(__file__).parent / "stronger_reader_summary.json").write_text(json.dumps(res, indent=1))
