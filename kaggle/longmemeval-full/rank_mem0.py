"""mem0 (verbatim messages, on-device bge-small + Qdrant + its BM25/entity boosts) rankings for
the 60-question embedding subset. Resumable: finished conversations are kept in the output file.

    python rank_mem0.py            (run from kaggle/longmemeval-full)
"""
import json
import os
import sys
import time

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval  # noqa: E402
from memory_recall_bench.integrations.mem0 import Mem0Retriever  # noqa: E402

OUT = "mrb_rankings_mem0.json"
ids = [json.loads(line)["id"] for line in open("convs_embed_subset.jsonl", encoding="utf-8")]
convs = {c.id: c for c in load_longmemeval("data/longmemeval_s_cleaned.json") if c.id in set(ids)}
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
retriever = Mem0Retriever()
start = time.time()
for cid in ids:
    if cid in done:
        continue
    c = convs[cid]
    done[cid] = retriever.rank_all(c.messages, [c.question])[c.question]
    json.dump(done, open(OUT + ".tmp", "w"))
    os.replace(OUT + ".tmp", OUT)
    print(f"{len(done)} conversations ({time.time() - start:.0f}s)", flush=True)
print(f"wrote {len(done)} rankings to {OUT}")
