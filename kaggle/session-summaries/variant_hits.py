"""Variant test: the session term only re-orders messages that already have a BM25 hit."""
import json, sys
from pathlib import Path
sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.adapters import RecentPlusBM25
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.runner import run, summarize
from memory_recall_bench.summaries import SessionBoost, load_summaries, session_of

HERE = Path(__file__).parent
summ = load_summaries("summaries_qwen_0.jsonl", "summaries_qwen_1.jsonl")
ids = set(json.loads((HERE / "subset60_ids.json").read_text()))
convs = [c for c in load_longmemeval(HERE.parent / "longmemeval-full/data/longmemeval_s_cleaned.json") if c.id in ids]
convs = [c for c in convs if all(session_of(m.id) in summ for m in c.messages)]

class HitsOnly(SessionBoost):
    def rank(self, query, archive):
        base = self.base.rank(query, archive)
        hit = {m.id for m in base}
        full = super().rank(query, archive)
        return [m for m in full if m.id in hit] + [m for m in full if m.id not in hit]

def mk(w, k):
    a = HitsOnly(RecentPlusBM25(), summ, weight=w, k=k); a.name = f"hits w{w} k{k}"; return a

ads = [RecentPlusBM25] + [(lambda w=w, k=k: mk(w, k)) for w in (0.5, 1, 2) for k in (10, 60)]
res = run(ads, convs, [4000, 16000])
for m in ["recalled", "recalled_all"]:
    for s in summarize(res, metric=m):
        print(m, s["adapter"], s["budget"], round(s["recall"] * 100, 1))
