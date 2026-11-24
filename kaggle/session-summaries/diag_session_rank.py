"""Diagnostic: rank of the first answer session when sessions are ranked by (a) BM25 over the summary,
(b) the best BM25 message score in the session, (c) BM25 over the summary with boilerplate removed."""
import json, re, sys, statistics
from pathlib import Path
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.bm25 import BM25, tokenize
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.summaries import load_summaries, session_of

HERE = Path(__file__).parent
summ = load_summaries("summaries_qwen_0.jsonl", "summaries_qwen_1.jsonl")
BOILER = re.compile(r"[^.]*\b(no|not|none)\b[^.]*\b(names?|numbers?|dates?|places?|preferences?|decisions?|plans?)\b[^.]*\.", re.I)
ids = set(json.loads((HERE / "subset60_ids.json").read_text()))
convs = [c for c in load_longmemeval(HERE.parent / "longmemeval-full/data/longmemeval_s_cleaned.json") if c.id in ids]

def rank_of(scores, sids, gold):
    order = [s for _, s in sorted(zip(scores, sids), key=lambda x: -x[0])]
    return min(order.index(g) for g in gold) + 1

res = {"summary": [], "summary_clean": [], "best_message": []}
for c in convs:
    sids = sorted({session_of(m.id) for m in c.messages})
    if not all(s in summ for s in sids):
        continue
    gold = {session_of(i) for i in c.evidence_ids}
    q = tokenize(c.question)
    res["summary"].append(rank_of(BM25([tokenize(summ[s]) for s in sids]).scores(q), sids, gold))
    res["summary_clean"].append(rank_of(BM25([tokenize(BOILER.sub("", summ[s])) for s in sids]).scores(q), sids, gold))
    ms = BM25([tokenize(m.content) for m in c.messages]).scores(q)
    best = {}
    for m, sc in zip(c.messages, ms):
        best[session_of(m.id)] = max(best.get(session_of(m.id), 0), sc)
    res["best_message"].append(rank_of([best[s] for s in sids], sids, gold))
n = len(res["summary"])
print(f"{n} questions, ~{statistics.mean(len({session_of(m.id) for m in c.messages}) for c in convs):.0f} sessions each")
for k, v in res.items():
    print(f"{k:14s} median rank {statistics.median(v):5.1f}  top1 {sum(r <= 1 for r in v)/n:.0%}  top5 {sum(r <= 5 for r in v)/n:.0%}  top10 {sum(r <= 10 for r in v)/n:.0%}")
