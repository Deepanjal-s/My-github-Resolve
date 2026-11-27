"""Score the packing-accuracy kernel output (copy of answer-accuracy/score_answers.py, other baseline).

    python score_answers.py out [--workers N]

Joins out/contexts.jsonl with out/answers_*.jsonl on (question, context). Prints judged accuracy per
retriever and budget, paired fixed/broken (sign test), accuracy split by whether all evidence was in
context, and per question kind. Writes packing_accuracy_summary.json.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--workers", type=int, help="worker count for legacy outputs (default: 2)")
args = parser.parse_args()
d = args.output_dir
contexts_file = d / "contexts.jsonl"
worker_count_file = d / "worker_count.txt"
workers = int(worker_count_file.read_text()) if worker_count_file.is_file() else args.workers
workers = 2 if workers is None else workers
if workers < 1:
    parser.error("worker count must be positive")

answer_files = [d / f"answers_{i}.jsonl" for i in range(workers)]
missing_files = [str(p) for p in [contexts_file, *answer_files] if not p.is_file()]
if missing_files:
    raise SystemExit("Missing required output files: " + ", ".join(missing_files))

ctx = [json.loads(line) for line in contexts_file.read_text(encoding="utf-8").splitlines()]
if not ctx:
    raise SystemExit(f"No question IDs in {contexts_file}")

ans = {}
for p in answer_files:
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ans[(r["qid"], tuple(sorted(r["ids"])))] = r

missing_answers = set()
by = defaultdict(dict)
for r in ctx:
    key = (r["qid"], tuple(sorted(r["ids"])))
    if key in ans:
        r["correct"] = ans[key]["correct"]
    else:
        missing_answers.add(r["qid"])
    by[(r["adapter"], r["budget"])][r["qid"]] = r

base = "rerank+prior recent=0.5"
qids = {r["qid"] for r in ctx}
errors = []
if missing_answers:
    errors.append("Missing judged answers for question IDs: " + ", ".join(sorted(missing_answers)))
for adapter, budget in sorted(by.keys() | {(base, r["budget"]) for r in ctx}):
    missing_qids = qids - by.get((adapter, budget), {}).keys()
    if missing_qids:
        errors.append(f"Missing question IDs in contexts.jsonl for {adapter} at budget {budget}: "
                      + ", ".join(sorted(missing_qids)))
if errors:
    raise SystemExit("\n".join(errors))


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
for (ad, b), q in sorted(by.items()):
    if ad == base:
        continue
    bq = by[(base, b)]
    fx = [c for c in bq if q[c]["correct"] and not bq[c]["correct"]]
    br = [c for c in bq if bq[c]["correct"] and not q[c]["correct"]]
    same_ctx = sum(sorted(q[c]["ids"]) == sorted(bq[c]["ids"]) for c in bq)
    multi = lambda cs: sum(bq[c]["kind"] == "multi-session" for c in cs)
    p = sign_p(len(fx), len(br))
    p_display = "p<0.001" if p < 0.001 else f"p={p:.3g}"
    out[f"paired|{ad}|{b}"] = {"fixed": len(fx), "broken": len(br), "p": p,
                              "multi_fixed": multi(fx), "multi_broken": multi(br), "same_context": same_ctx}
    print(f"{ad} {b}: fixed {len(fx)} / broken {len(br)} {p_display} "
          f"(multi {multi(fx)}/{multi(br)}; identical context on {same_ctx})")
(Path(__file__).parent / "packing_accuracy_summary.json").write_text(json.dumps(out, indent=1))
