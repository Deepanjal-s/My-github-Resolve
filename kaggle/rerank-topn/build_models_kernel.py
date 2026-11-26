"""Private Kaggle GPU kernel: which re-ranker model should ai-chat-memory use? Same input as build_kernel.py
(the app's BM25 rankings, all 479), top 30 re-scored by each model. All models have ONNX ports that
transformers.js can load (checked on the Hugging Face API), so any winner is usable in the app.

    python build_models_kernel.py   then   kaggle kernels push -p models_kernel
"""
import base64
import gzip
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SCRATCH = Path(r"C:\Users\deepa\AppData\Local\Temp\claude\9961769b500bcff6\54ae22e9-19eb-420a-8fe5-e1d097cc00fc\scratchpad")
blob = base64.b64encode(gzip.compress((SCRATCH / "aim_kw.json").read_bytes())).decode()
MODELS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2",
          "mixedbread-ai/mxbai-rerank-xsmall-v1", "mixedbread-ai/mxbai-rerank-base-v1",
          "BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"]

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
text = {}
for q in data:
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for i, m in enumerate(sess):
            text[f"{sid}:{i}"] = m["content"][:2000]
qtext = {q["question_id"]: q["question"] for q in data}
out = {"timing": {}}
for name in MODELS:
    kwargs = {"torch_dtype": torch.float16} if "v2-m3" in name else {}
    ce = CrossEncoder(name, max_length=512, device="cuda", model_kwargs=kwargs)
    t = time.time(); pairs = 0; ranked = {}
    for cid, ids in kw.items():
        top = ids[:N]
        s = ce.predict([(qtext[cid], text[i]) for i in top], batch_size=64, show_progress_bar=False)
        order = sorted(range(len(top)), key=lambda j: (-float(s[j]), j))
        ranked[cid] = [top[j] for j in order] + ids[N:]; pairs += len(top)
    out["timing"][name] = {"pairs": pairs, "gpu_seconds": round(time.time() - t, 1),
                           "params_m": round(sum(p.numel() for p in ce.model.parameters()) / 1e6)}
    out[name] = ranked
    print(name, out["timing"][name], flush=True)
    del ce; gc.collect(); torch.cuda.empty_cache()
json.dump(out, open("/kaggle/working/rerank_models.json", "w"))
print("done")
'''

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nb.cells = [nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(f"BLOB = {blob!r}\nMODELS = {MODELS!r}\n"),
            nbf.v4.new_code_cell(RUN)]
out = HERE / "models_kernel"
out.mkdir(exist_ok=True)
nbf.write(nb, out / "rerank-models.ipynb")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/aim-rerank-models", "title": "aim rerank models", "code_file": "rerank-models.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("wrote", out)
