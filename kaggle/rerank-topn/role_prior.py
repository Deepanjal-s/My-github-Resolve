"""Role prior on the re-ranker: 94% of LongMemEval evidence is in user turns, but 63% of the BM25 top 30 are
assistant turns. Score = MiniLM-L-6 logit + B if the message is the user's own, unless the question asks about what
the assistant said ("you said / told / recommended / suggested ..."), in which case no boost.

B is picked on half the questions (even position in the file) and reported on the other half, then on all 479.

    python role_prior.py aim_kw.json out/rerank_topn.json
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
role = {}
for q in raw:
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for i, m in enumerate(sess):
            role[f"{sid}:{i}"] = m["role"]
qtext = {q["question_id"]: q["question"] for q in raw}
ASKS_ASSISTANT = re.compile(r"\byou\b.*\b(said|say|told|tell|mention|recommend|suggest|gave|give|list|provide|explain)",
                            re.I)
convs = load_longmemeval(DATA)
ids_in_order = [c.id for c in convs]
half_a = set(ids_in_order[0::2]); half_b = set(ids_in_order[1::2])


def ranking(b, rule=True):
    out = {}
    for cid, ids in kw.items():
        top = ids[:N]
        boost = 0.0 if (rule and ASKS_ASSISTANT.search(qtext[cid])) else b
        s = [scores[cid][j] + (boost if role[i] == "user" else 0.0) for j, i in enumerate(top)]
        order = sorted(range(len(top)), key=lambda j: (-s[j], j))
        out[cid] = [top[j] for j in order] + ids[N:]
    return out


def recall(r, subset, budgets=(4000,)):
    p = Path(tempfile.mkdtemp()) / "r.json"; p.write_text(json.dumps(r))
    cs = [c for c in convs if c.id in subset]
    res = run([RankedFromFile(str(p), cs, name="x")], cs, list(budgets))
    return {b: [x for x in res if x.budget == b] for b in budgets}


print("assistant-question rule matches", sum(bool(ASKS_ASSISTANT.search(q)) for q in qtext.values()), "questions")
tune = {}
for b in (0, 1, 2, 3, 4, 6, 8):
    rs = recall(ranking(b), half_a)[4000]
    tune[b] = sum(x.recalled_all for x in rs) / len(rs)
    print(f"tune half, B={b}: {tune[b]:.3f}")
best = max(tune, key=lambda b: (tune[b], -b))
print("chosen B", best)
sign_p = lambda a, b: 1.0 if a + b == 0 else min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
summary = {"chosen_B": best, "tune_half_4k": {str(k): round(100 * v, 1) for k, v in tune.items()}}
budgets = (2000, 4000, 8000, 16000)
for label, subset in (("held-out half", half_b), ("all 479", half_a | half_b)):
    base, boosted, norule = recall(ranking(0), subset, budgets), recall(ranking(best), subset, budgets), \
        recall(ranking(best, rule=False), subset, budgets)
    for b in budgets:
        x = {r.conversation: r.recalled_all for r in base[b]}; y = {r.conversation: r.recalled_all for r in boosted[b]}
        z = sum(r.recalled_all for r in norule[b]) / len(norule[b])
        f = sum(y[c] and not x[c] for c in x); br = sum(x[c] and not y[c] for c in x)
        row = {"rerank": round(100 * sum(x.values()) / len(x), 1), "rerank+user prior": round(100 * sum(y.values()) / len(y), 1),
               "prior without assistant rule": round(100 * z, 1), "fixed": f, "broken": br, "p": round(sign_p(f, br), 5)}
        summary[f"{label}|{b}"] = row
        print(label, b, row)
json.dump(summary, open(Path(__file__).parent / "role_prior_summary.json", "w"), indent=1)
