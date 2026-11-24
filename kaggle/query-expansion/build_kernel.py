"""Private Kaggle GPU kernel: rewrite every LongMemEval_s question into short search queries (one per event or
fact it needs) with a local open model. Output /kaggle/working/expansions.jsonl:
{"question_id", "question", "queries": [...], "raw"}.

    python build_kernel.py   then   kaggle kernels push -p kernel
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

CODE = r'''
import json, os, re, time, urllib.request
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "%(model)s"
P = Path("/kaggle/working/lme.json")
if not P.exists():
    urllib.request.urlretrieve("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json", P)
qs = [(q["question_id"], q["question"], q.get("question_date", "")) for q in json.loads(P.read_text())]
PROMPT = (
    "A user asks an assistant a question about their own past chats with it. Write 2 to 4 short search "
    "queries that would find the past messages needed to answer it. If the question combines several "
    "events, items or time points, write one query per event or item. Use the words the user would likely "
    "have typed back then (concrete nouns, names, activities), not the question's wording. "
    "Output only the queries, one per line, no numbering.\n\nQuestion: {q}"
)
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda:0").eval()
out = Path("/kaggle/working/expansions.jsonl")
t0 = time.time()
with out.open("w", encoding="utf-8") as f:
    for i in range(0, len(qs), 16):
        batch = qs[i:i + 16]
        texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q)}], tokenize=False,
                                         add_generation_prompt=True) for _, q, _ in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120, do_sample=False)
        for (qid, q, _), g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
            raw = tok.decode(g, skip_special_tokens=True).strip()
            queries = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip().strip('"') for l in raw.splitlines()]
            f.write(json.dumps({"question_id": qid, "question": q, "queries": [x for x in queries if x][:4], "raw": raw}) + "\n")
        print(i + len(batch), f"{time.time() - t0:.0f}s", flush=True)
P.unlink()
''' % {"model": MODEL}

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell("# Query expansion for LongMemEval_s\n\nWork kernel for memory-recall-bench."),
            nbf.v4.new_code_cell(CODE.strip())]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / "kernel"
d.mkdir(exist_ok=True)
nbf.write(nb, d / "query-expansion.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-query-expansion-qwen", "title": "lme query expansion qwen",
    "code_file": "query-expansion.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
