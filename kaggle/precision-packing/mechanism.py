"""Why does top-5 beat the full 8k context? Split the paired fixed/broken questions by evidence state.

    python mechanism.py [--a "top5 recent=0" --a-dir out] [--b "full recent=0" --b-dir out-matched]

For each question where exactly one arm is correct, records whether each arm's context held all evidence.
"Distractor removal" = A right, B wrong, both contexts hold all evidence.
"Recall loss" = A wrong, B right, A lacks evidence that B has.
Writes mechanism_<a>_vs_<b>.json.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load(d, adapter):
    d = Path(d)
    ans = {}
    for p in sorted(d.glob("answers_[0-9]*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ans[(r["qid"], tuple(sorted(r["ids"])))] = r["correct"]
    out = {}
    for line in (d / "contexts.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["adapter"] == adapter:
            r["correct"] = ans[(r["qid"], tuple(sorted(r["ids"])))]
            out[r["qid"]] = r
    return out


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--a", default="top5 recent=0")
ap.add_argument("--a-dir", default="out")
ap.add_argument("--b", default="full recent=0")
ap.add_argument("--b-dir", default="out-matched")
args = ap.parse_args()
A, B = load(args.a_dir, args.a), load(args.b_dir, args.b)
assert A.keys() == B.keys() and len(A) == 479, (len(A), len(B))

cells = Counter()
kinds = Counter()
for q in A:
    a, b = A[q], B[q]
    if a["correct"] == b["correct"]:
        continue
    who = "A_wins" if a["correct"] else "B_wins"
    ev = f"A_all={a['recalled_all']},B_all={b['recalled_all']}"
    cells[(who, ev)] += 1
    kinds[(who, a["kind"])] += 1

both_all = [q for q in A if A[q]["recalled_all"] and B[q]["recalled_all"]]
fx = sum(A[q]["correct"] and not B[q]["correct"] for q in both_all)
br = sum(B[q]["correct"] and not A[q]["correct"] for q in both_all)
acc = lambda S, qs: round(100 * sum(S[q]["correct"] for q in qs) / len(qs), 1)
summary = {
    "a": args.a, "b": args.b,
    "cells": {f"{w} | {e}": n for (w, e), n in sorted(cells.items())},
    "both_contexts_hold_all_evidence": {"n": len(both_all), "acc_a": acc(A, both_all), "acc_b": acc(B, both_all),
                                        "a_fixed": fx, "a_broken": br, "p": round(sign_p(fx, br), 4)},
    "by_kind": {f"{w} | {k}": n for (w, k), n in sorted(kinds.items())},
    "mean_msgs": {"a": round(sum(len(r["ids"]) for r in A.values()) / 479, 1),
                  "b": round(sum(len(r["ids"]) for r in B.values()) / 479, 1)},
}
print(json.dumps(summary, indent=1))
name = f"mechanism_{args.a.split()[0]}_vs_{args.b.split()[0]}.json"
Path(name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
