"""Private Kaggle GPU kernel: can the re-ranker close the gap to the perfect-reranker ceiling (83.9 at 4k) with
more of each message, or with its neighbouring turn? App BM25 top 30 (all 479), MiniLM-L-6 unless noted:

  trunc         content[:2000] (what the app does; 33% of top-30 candidates are longer)
  maxp          max score over 1,200-char windows (stride 1,000) of the whole message
  prev          previous turn[:600] + this message[:1400]
  maxp+prev     max over windows, each prefixed by previous turn[:300]
  m3-long       bge-reranker-v2-m3, max_length 1024, content[:6000] (long-context model)

    python build_context_kernel.py   then   kaggle kernels push -p context_kernel
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
import base64, gc, gzip, json, time
import torch
from sentence_transformers import CrossEncoder

N = 30
kw = json.loads(gzip.decompress(base64.b64decode(BLOB)))
data = json.load(open("/kaggle/working/lme.json"))
text, prev = {}, {}
for q in data:
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for i, m in enumerate(sess):
            text[f"{sid}:{i}"] = m["content"]
            prev[f"{sid}:{i}"] = sess[i - 1]["content"] if i else ""
qtext = {q["question_id"]: q["question"] for q in data}

def windows(t, size=1200, stride=1000):
    return [t[s:s + size] for s in range(0, max(1, len(t) - size + stride), stride)] or [""]

VARIANTS = {
    "trunc": lambda i: [text[i][:2000]],
    "maxp": lambda i: windows(text[i]),
    "prev": lambda i: [(prev[i][:600] + " \n " + text[i][:1400]).strip()],
    "maxp+prev": lambda i: [(prev[i][:300] + " \n " + w).strip() for w in windows(text[i])],
}

def rank_with(ce, make_docs, name):
    t = time.time(); ranked = {}; pairs = 0
    for cid, ids in kw.items():
        top = ids[:N]
        docs, owner = [], []
        for j, i in enumerate(top):
            for d in make_docs(i):
                docs.append((qtext[cid], d)); owner.append(j)
        s = ce.predict(docs, batch_size=64, show_progress_bar=False); pairs += len(docs)
        best = [-1e9] * len(top)
        for j, v in zip(owner, s):
            best[j] = max(best[j], float(v))
        order = sorted(range(len(top)), key=lambda j: (-best[j], j))
        ranked[cid] = [top[j] for j in order] + ids[N:]
    print(name, pairs, "pairs", round(time.time() - t), "s", flush=True)
    return ranked, pairs

out = {"pairs": {}}
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cuda")
for name, f in VARIANTS.items():
    out[name], out["pairs"][name] = rank_with(ce, f, name)
del ce; gc.collect(); torch.cuda.empty_cache()
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device="cuda", model_kwargs={"torch_dtype": torch.float16})
out["m3-long"], out["pairs"]["m3-long"] = rank_with(ce, lambda i: [text[i][:6000]], "m3-long")
json.dump(out, open("/kaggle/working/rerank_context.json", "w"))
print("done")
'''

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nb.cells = [nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(f"BLOB = {blob!r}\n"), nbf.v4.new_code_cell(RUN)]
out = HERE / "context_kernel"
out.mkdir(exist_ok=True)
nbf.write(nb, out / "rerank-context.ipynb")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/aim-rerank-context", "title": "aim rerank context", "code_file": "rerank-context.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("wrote", out)
