"""Private Kaggle GPU kernel: how many candidates should ai-chat-memory's on-device re-ranker re-score?

Input: ai-chat-memory's own keyword search rankings for all 479 LongMemEval_s questions (aim_kw.json, top 200
message ids, best first), embedded in the notebook. The app re-ranks the top N with
Xenova/ms-marco-MiniLM-L-6-v2; this uses the original cross-encoder/ms-marco-MiniLM-L-6-v2 weights on GPU
(same model, fp32, not q8), text = message content[:2000] as in the app. N in {30, 50, 100, 200}.
N=30 is checked against the app's own run (aim_kw_rr.json): recall should be close to 70.1% at 4k.

Writes rerank_topn.json: {"N=<n>": {conversation id: [ids]}} and per-pair GPU timing.

    python build_kernel.py   then   kaggle kernels push -p kernel
"""
import base64
import gzip
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SCRATCH = Path(r"C:\Users\deepa\AppData\Local\Temp\claude\9961769b500bcff6\54ae22e9-19eb-420a-8fe5-e1d097cc00fc\scratchpad")
blob = base64.b64encode(gzip.compress((SCRATCH / "aim_kw.json").read_bytes())).decode()

SETUP = (
    "import os, urllib.request\n"
    "if not os.path.exists('/kaggle/working/lme.json'):\n"
    "    urllib.request.urlretrieve('https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json', '/kaggle/working/lme.json')\n"
)

RUN = r'''
import base64, gzip, json, time
import torch
from sentence_transformers import CrossEncoder

kw = json.loads(gzip.decompress(base64.b64decode(BLOB)))
data = json.load(open("/kaggle/working/lme.json"))
text = {}
for q in data:
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for i, m in enumerate(sess):
            text[f"{sid}:{i}"] = m["content"][:2000]
missing = sum(1 for v in kw.values() for i in v if i not in text)
print(len(kw), "conversations;", missing, "ids without text", flush=True)

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cuda")
qtext = {q["question_id"]: q["question"] for q in data}
t = time.time(); pairs = 0
scores = {}
for cid, ids in kw.items():
    s = ce.predict([(qtext[cid], text.get(i, "")) for i in ids], batch_size=128, show_progress_bar=False)
    scores[cid] = [float(x) for x in s]; pairs += len(ids)
gpu_s = time.time() - t
print(f"scored {pairs} pairs in {gpu_s:.0f}s", flush=True)

out = {"timing": {"pairs": pairs, "gpu_seconds": round(gpu_s, 1)}}
for n in (30, 50, 100, 200):
    ranked = {}
    for cid, ids in kw.items():
        top = list(zip(ids[:n], scores[cid][:n], range(n)))
        top.sort(key=lambda x: (-x[1], x[2]))  # ties keep the original order, as in the app
        ranked[cid] = [i for i, _, _ in top] + ids[n:]
    out[f"N={n}"] = ranked
    # RRF of the first-stage order and the re-ranker order over the top n (k=60), rest unchanged
    fused = {}
    for cid, ids in kw.items():
        top = ids[:n]
        by_ce = [i for i, _ in sorted(zip(top, scores[cid][:n]), key=lambda x: -x[1])]
        sc = {}
        for ranked_list in (top, by_ce):
            for r, i in enumerate(ranked_list):
                sc[i] = sc.get(i, 0.0) + 1.0 / (60 + r + 1)
        fused[cid] = sorted(top, key=lambda i: -sc[i]) + ids[n:]
    out[f"RRF N={n}"] = fused
    out[f"scores"] = scores
json.dump(out, open("/kaggle/working/rerank_topn.json", "w"))
print("done")
'''

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nb.cells = [nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(f"BLOB = {blob!r}\n"), nbf.v4.new_code_cell(RUN)]
out = HERE / "kernel"
out.mkdir(exist_ok=True)
nbf.write(nb, out / "rerank-topn.ipynb")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/aim-rerank-topn", "title": "aim rerank topn", "code_file": "rerank-topn.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("wrote", out)
