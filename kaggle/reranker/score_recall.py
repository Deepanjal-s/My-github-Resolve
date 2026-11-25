"""Recall of the re-ranker arms vs recent+hybrid, paired, from the kernel's recall.jsonl.

    python score_recall.py out
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in (Path(sys.argv[1]) / "recall.jsonl").read_text().splitlines()]
by = defaultdict(dict)
for r in rows:
    by[(r["adapter"], r["budget"])][r["qid"]] = r


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


base, out = "recent+hybrid", {}
for (ad, b), q in sorted(by.items(), key=lambda x: (x[0][1], x[0][0] != base, x[0][0])):
    s = {m: round(100 * sum(r[m] for r in q.values()) / len(q), 1) for m in ("recalled", "recalled_all")}
    s["multi_all"] = round(100 * sum(r["recalled_all"] for r in q.values() if r["kind"] == "multi-session")
                           / sum(r["kind"] == "multi-session" for r in q.values()), 1)
    line = f"{ad:22s} {b:6d}  any {s['recalled']:5.1f}  all-ev {s['recalled_all']:5.1f}  multi all-ev {s['multi_all']:5.1f}"
    if ad != base:
        bq = by[(base, b)]
        fx = sum(q[c]["recalled_all"] and not bq[c]["recalled_all"] for c in bq)
        br = sum(bq[c]["recalled_all"] and not q[c]["recalled_all"] for c in bq)
        s.update(fixed=fx, broken=br, p=round(sign_p(fx, br), 3))
        line += f"  all-ev {fx} / {br} p={s['p']}"
    out[f"{ad}|{b}"] = s
    print(line)
(Path(__file__).parent / "recall_summary.json").write_text(json.dumps(out, indent=1))
