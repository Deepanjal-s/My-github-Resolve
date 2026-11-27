"""Step 2: recall with a date window, on the questions date_window.py can parse (26 of 479).

Ranking for those questions: messages from sessions inside the target window first (in re-ranked order where the
app's keyword search found them, then the window's remaining user turns in session order), then everything else in
the usual re-ranked order. Other questions are unchanged. Packing: no recent window, pairs (packing.py).

    python date_window_recall.py aim_kw.json out/rerank_topn.json
"""
import json, math, re, sys, tempfile
from pathlib import Path
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile
from memory_recall_bench.runner import run
sys.path.insert(0, str(Path(__file__).parent))
from date_window import parse_date, target

N = 30
kw = json.load(open(sys.argv[1]))
scores = json.load(open(sys.argv[2]))["scores"]
DATA = r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json"
raw = {q["question_id"]: q for q in json.load(open(DATA, encoding="utf-8"))}
role = {f"{sid}:{i}": m["role"] for q in raw.values() for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        for i, m in enumerate(sess)}
AA = re.compile(r"\byou\b.*\b(said|say|told|tell|mention|recommend|suggest|gave|give|list|provide|explain)", re.I)
convs = load_longmemeval(DATA)

base, dated = {}, {}
for cid, ids in kw.items():
    q = raw[cid]
    top = ids[:N]; b = 0.0 if AA.search(q["question"]) else 3.0
    s = [scores[cid][j] + (b if role[i] == "user" else 0.0) for j, i in enumerate(top)]
    r = [top[j] for j in sorted(range(len(top)), key=lambda j: (-s[j], j))] + ids[N:]
    base[cid] = dated[cid] = r
    t = target(q["question"], parse_date(q["question_date"]))
    if not t:
        continue
    tgt, tol = t
    inwin = {sid for sid, d in zip(q["haystack_session_ids"], q["haystack_dates"]) if abs((parse_date(d) - tgt).days) <= tol}
    first = [i for i in r if i.rsplit(":", 1)[0] in inwin]
    seen = set(first)
    extra = [f"{sid}:{k}" for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]) if sid in inwin
             for k, m in enumerate(sess) if m["role"] == "user" and f"{sid}:{k}" not in seen]
    dated[cid] = first + extra + [i for i in r if i not in seen]

parsed = [c for c in convs if dated[c.id] is not base[c.id]]
sign_p = lambda a, b: 1.0 if a + b == 0 else min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
out = {}
for name, rk in (("re-rank + prior", base), ("+ date window", dated)):
    p = Path(tempfile.mkdtemp()) / "r.json"; p.write_text(json.dumps(rk))
    res = run([RankedFromFile(str(p), parsed, name=name, recent_share=0.0)], parsed, [2000, 4000, 8000, 16000])
    out[name] = {(x.budget, x.conversation): x.recalled_all for x in res}
summary = {"questions": len(parsed)}
for bgt in (2000, 4000, 8000, 16000):
    a = {k: v for k, v in out["re-rank + prior"].items() if k[0] == bgt}
    d = {k: v for k, v in out["+ date window"].items() if k[0] == bgt}
    f = sum(d[k] and not a[k] for k in a); br = sum(a[k] and not d[k] for k in a)
    row = {"base": sum(a.values()), "date window": sum(d.values()), "fixed": f, "broken": br, "p": round(sign_p(f, br), 4)}
    summary[str(bgt)] = row
    print(bgt, f"of {len(parsed)}:", row)
json.dump(summary, open(Path(__file__).parent / "date_window_summary.json", "w"), indent=1)
