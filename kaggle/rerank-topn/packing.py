"""Packing, not ranking: of the 46 temporal questions still missing evidence at 4k after re-ranking + user-turn
prior, many have every evidence message in the top 5. The bench's standard context spends half the budget on the
most recent messages (recent_share=0.5) and brings each hit's reply partner (pair=True; LongMemEval assistant
replies are long). This measures all-evidence recall for the same ranking under other packings.

recent_share/pair change what the reader sees, so recall alone can't pick a winner: a hit without its reply is
less readable. Reported to size the cost, not as a free gain.

    python packing.py aim_kw.json out/rerank_topn.json
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
AA = re.compile(r"\byou\b.*\b(said|say|told|tell|mention|recommend|suggest|gave|give|list|provide|explain)", re.I)
convs = load_longmemeval(DATA)
r = {}
for cid, ids in kw.items():
    top = ids[:N]; b = 0.0 if AA.search(qtext[cid]) else 3.0
    s = [scores[cid][j] + (b if role[i] == "user" else 0.0) for j, i in enumerate(top)]
    r[cid] = [top[j] for j in sorted(range(len(top)), key=lambda j: (-s[j], j))] + ids[N:]
p = Path(tempfile.mkdtemp()) / "r.json"; p.write_text(json.dumps(r))
sign_p = lambda a, b: 1.0 if a + b == 0 else min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
budgets = [2000, 4000, 8000, 16000]
res = {}
for share in (0.5, 0.25, 0.0):
    for pair in (True, False):
        out = run([RankedFromFile(str(p), convs, name="x", recent_share=share, pair=pair)], convs, budgets)
        res[(share, pair)] = {(x.budget, x.conversation): (x.recalled_all, x.kind) for x in out}
base = res[(0.5, True)]
summary = {}
for (share, pair), d in res.items():
    for b in budgets:
        keys = [k for k in d if k[0] == b]
        f = sum(d[k][0] and not base[k][0] for k in keys); br = sum(base[k][0] and not d[k][0] for k in keys)
        temp = [k for k in keys if d[k][1] == "temporal-reasoning"]
        row = {"all_ev": round(100 * sum(d[k][0] for k in keys) / len(keys), 1),
               "temporal": round(100 * sum(d[k][0] for k in temp) / len(temp), 1),
               "fixed": f, "broken": br, "p": round(sign_p(f, br), 5)}
        summary[f"recent_share={share} pair={pair}|{b}"] = row
        print(f"recent_share={share} pair={pair} {b}", row)
json.dump(summary, open(Path(__file__).parent / "packing_summary.json", "w"), indent=1)
