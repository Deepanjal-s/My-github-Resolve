"""Paired comparison from a kernel results CSV (one row per adapter x budget x conversation).

    python score_csv.py hybrid_prefix_out/hybrid_prefix_results.csv "recent+hybrid"

Prints recall per adapter/budget and fixed/broken vs the baseline adapter with a two-sided sign test,
overall and for multi-session questions.
"""
import csv
import math
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
base_name = sys.argv[2]
by = defaultdict(dict)
for r in rows:
    by[(r["adapter"], int(r["budget"]))][r["conversation"]] = r
truthy = lambda v: v in ("True", "true", "1")


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


for (ad, b), d in sorted(by.items()):
    print(f"{ad:32s} {b:6d} any {100 * sum(truthy(r['recalled']) for r in d.values()) / len(d):5.1f}"
          f"  all {100 * sum(truthy(r['recalled_all']) for r in d.values()) / len(d):5.1f}  n={len(d)}")
for (ad, b), d in sorted(by.items()):
    if ad == base_name:
        continue
    base = by[(base_name, b)]
    for m in ("recalled", "recalled_all"):
        fx = [c for c in base if truthy(d[c][m]) and not truthy(base[c][m])]
        br = [c for c in base if truthy(base[c][m]) and not truthy(d[c][m])]
        multi = lambda cs: sum(base[c]["kind"] == "multi-session" for c in cs)
        print(f"{ad} {b} {m}: fixed {len(fx)} / broken {len(br)} p={sign_p(len(fx), len(br)):.3f}"
              f" (multi {multi(fx)}/{multi(br)})")
