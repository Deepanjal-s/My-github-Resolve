"""After re-ranking the top 30: cap each session at K messages before any session gets more (the overflow keeps its
order and goes after the capped list). Targets the remaining gap, which is mostly questions that need 2+ sessions
(temporal 27, multi-session 22 of the 64 questions between re-rank and the perfect re-ranker at 4k). K is picked on
half the questions and reported on the other half. Runs on top of plain re-ranking and of the user-turn prior.

    python session_cap.py aim_kw.json out/rerank_topn.json
"""
import json, math, re, sys, tempfile
from pathlib import Path
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile
from memory_recall_bench.runner import run

N = 30
kw = json.load(open(sys.argv[1]))
scores = json.load(open(sys.argv[2]))["scores"]
DATA = r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json"
raw = json.load(open(DATA, encoding="utf-8"))
role = {f"{sid}:{i}": m["role"] for q in raw for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        for i, m in enumerate(sess)}
qtext = {q["question_id"]: q["question"] for q in raw}
ASKS_ASSISTANT = re.compile(r"\byou\b.*\b(said|say|told|tell|mention|recommend|suggest|gave|give|list|provide|explain)", re.I)
convs = load_longmemeval(DATA)
order_ids = [c.id for c in convs]
half_a, half_b = set(order_ids[0::2]), set(order_ids[1::2])
kind = {c.id: c.kind for c in convs}


def ranking(k, prior):
    out = {}
    for cid, ids in kw.items():
        top = ids[:N]
        b = 0.0 if (not prior or ASKS_ASSISTANT.search(qtext[cid])) else 3.0
        s = [scores[cid][j] + (b if role[i] == "user" else 0.0) for j, i in enumerate(top)]
        rr = [top[j] for j in sorted(range(len(top)), key=lambda j: (-s[j], j))]
        if k:
            seen, first, later = {}, [], []
            for i in rr:
                sess = i.rsplit(":", 1)[0]
                seen[sess] = seen.get(sess, 0) + 1
                (first if seen[sess] <= k else later).append(i)
            rr = first + later
        out[cid] = rr + ids[N:]
    return out


def results(r, subset, budgets):
    p = Path(tempfile.mkdtemp()) / "r.json"; p.write_text(json.dumps(r))
    cs = [c for c in convs if c.id in subset]
    return run([RankedFromFile(str(p), cs, name="x")], cs, list(budgets))


sign_p = lambda a, b: 1.0 if a + b == 0 else min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
summary = {}
for prior in (False, True):
    tune = {}
    for k in (0, 1, 2, 3, 4):
        rs = results(ranking(k, prior), half_a, [4000])
        tune[k] = sum(x.recalled_all for x in rs) / len(rs)
    best = max(tune, key=lambda k: (tune[k], -k if k else -99))
    tag = "prior" if prior else "plain"
    print(tag, "tune", {k: round(100 * v, 1) for k, v in tune.items()}, "chosen K", best)
    summary[f"{tag}|tune_half_4k"] = {str(k): round(100 * v, 1) for k, v in tune.items()}
    summary[f"{tag}|K"] = best
    for label, subset in (("held-out half", half_b), ("all 479", half_a | half_b)):
        for b in (2000, 4000, 8000, 16000):
            x = {r.conversation: r.recalled_all for r in results(ranking(0, prior), subset, [b])}
            y = {r.conversation: r.recalled_all for r in results(ranking(best, prior), subset, [b])}
            f = sum(y[c] and not x[c] for c in x); br = sum(x[c] and not y[c] for c in x)
            multi = [c for c in x if kind[c] in ("multi-session", "temporal-reasoning")]
            row = {"base": round(100 * sum(x.values()) / len(x), 1), "capped": round(100 * sum(y.values()) / len(y), 1),
                   "fixed": f, "broken": br, "p": round(sign_p(f, br), 5),
                   "multi+temporal base/capped": [sum(x[c] for c in multi), sum(y[c] for c in multi), len(multi)]}
            summary[f"{tag}|{label}|{b}"] = row
            print(tag, label, b, row)
json.dump(summary, open(Path(__file__).parent / "session_cap_summary.json", "w"), indent=1)
