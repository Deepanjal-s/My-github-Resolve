"""Builds a private Kaggle GPU kernel that writes one summary per LongMemEval_s session with a local
open model (no API key, no cost). Same prompt and session text as ../summarize_sessions.py.

    python build_kernel.py            then   kaggle kernels push -p kernel

Output in the kernel: /kaggle/working/summaries_qwen_{0,1}.jsonl (one worker per T4), checkpointed per batch.
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
IDS = json.loads((HERE.parent / "subset60_ids.json").read_text())
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

WORKER = r'''
import json, os, sys, time, urllib.request
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKER, WORKERS = int(sys.argv[1]), int(sys.argv[2])
MODEL = "%(model)s"
IDS = set(%(ids)s)
MAX_SESSION_CHARS = 60_000
TOKEN_BUDGET = 24_000          # prompt tokens per batch
MAX_NEW = 220
PROMPT = (
    "Below is one chat session between a user and an assistant. Write a faithful summary "
    "in at most 120 words. Keep every concrete fact a later question could ask about: names, "
    "numbers, dates, places, preferences, decisions, plans, and what the user said about "
    "themselves. Do not add anything that is not in the session.\n\n<session>\n{body}\n</session>"
)

data = json.loads(Path("/kaggle/working/lme.json").read_text())
sessions = {}
for q in data:
    if q["question_id"] not in IDS:
        continue
    for sid, msgs in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        sessions.setdefault(sid, msgs)
sids = sorted(sessions)[WORKER::WORKERS]
out_path = Path(f"/kaggle/working/summaries_qwen_{WORKER}.jsonl")
done = set()
if out_path.exists():
    for line in out_path.read_text().splitlines():
        try: done.add(json.loads(line)["session_id"])
        except Exception: pass
print(f"worker {WORKER}: {len(sids)} sessions, {len(done)} already done", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda:0").eval()

items = []
for sid in sids:
    if sid in done:
        continue
    body = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in sessions[sid])
    cut = len(body) > MAX_SESSION_CHARS
    body = body[:MAX_SESSION_CHARS]
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(body=body)}],
                                   tokenize=False, add_generation_prompt=True)
    items.append((sid, text, cut, len(tok(text).input_ids)))
items.sort(key=lambda x: x[3])

t0, n = time.time(), 0
with out_path.open("a", encoding="utf-8") as out:
    i = 0
    while i < len(items):
        j = i + 1
        while j < len(items) and (j - i + 1) * items[j][3] <= TOKEN_BUDGET and j - i < 16:
            j += 1
        batch = items[i:j]
        enc = tok([b[1] for b in batch], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        for b, g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
            out.write(json.dumps({"session_id": b[0], "summary": tok.decode(g, skip_special_tokens=True).strip(),
                                  "truncated": b[2], "input_tokens": b[3], "model": MODEL}) + "\n")
        out.flush()
        n += len(batch); i = j
        print(f"worker {WORKER}: {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
print(f"worker {WORKER}: finished", flush=True)
''' % {"model": MODEL, "ids": json.dumps(IDS)}

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Session summaries for LongMemEval_s (pilot, 60 questions)\n\n"
        f"Work kernel for memory-recall-bench: one summary per session with `{MODEL}` on 2x T4. "
        "Output is scored offline."),
    nbf.v4.new_code_cell(
        "import urllib.request, os\n"
        "if not os.path.exists('/kaggle/working/lme.json'):\n"
        "    urllib.request.urlretrieve('https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json', '/kaggle/working/lme.json')\n"
        "!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
        "import transformers; print(transformers.__version__)"),
    nbf.v4.new_code_cell("%%writefile /kaggle/working/worker.py\n" + WORKER.strip()),
    nbf.v4.new_code_cell(
        "import subprocess, os, torch\n"
        "W = max(1, torch.cuda.device_count())\n"
        "procs = [subprocess.Popen(['python', '/kaggle/working/worker.py', str(i), str(W)],\n"
        "          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i)}) for i in range(W)]\n"
        "print([p.wait() for p in procs])\n"
        "os.remove('/kaggle/working/lme.json')\n"
        "os.remove('/kaggle/working/worker.py')"),
]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / "kernel"
d.mkdir(exist_ok=True)
nbf.write(nb, d / "session-summaries.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-session-summaries-qwen",
    "title": "lme session summaries qwen",
    "code_file": "session-summaries.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d, len(IDS), "questions")
