"""Private Kaggle GPU kernel: is the 16k accuracy penalty a small-model effect? Same recent+hybrid contexts as
kaggle/answer-accuracy (4k and 16k, all 479 questions), read by Qwen3-8B (4-bit NF4, thinking off) instead of
Qwen3-4B-Instruct. The judge stays Qwen3-4B-Instruct with the same prompts, so only the reader changes.

    python build_kernel.py   then   kaggle kernels push -p kernel --accelerator NvidiaTeslaT4

Compare with ../answer-accuracy/out (4B reader) on identical contexts: score_compare.py.
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # judge
READER = "Qwen/Qwen3-8B"

CONTEXTS = r'''
import json, time
from pathlib import Path
from memory_recall_bench import load_longmemeval
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusHybrid, sentence_transformers_encoder

convs = load_longmemeval("/kaggle/working/lme.json")
cache = EmbeddingCache(sentence_transformers_encoder("BAAI/bge-small-en-v1.5", device="cuda"), max_chars=512, batch_size=512)
t = time.time(); cache.get([m.content for c in convs for m in c.messages]); print(f"embedded in {time.time()-t:.0f}s", flush=True)
QP = "Represent this sentence for searching relevant passages: "

makers = [lambda: RecentPlusHybrid(cache, query_prefix=QP)]
with open("/kaggle/working/contexts.jsonl", "w") as f:
    for make in makers:
        for budget in (4000, 16000):
            for c in convs:
                a = make(); a.reset(budget)
                for m in c.messages:
                    a.add(m)
                ids = list(a.context(c.question))
                kept = set(ids)
                f.write(json.dumps({"qid": c.id, "kind": c.kind, "adapter": a.name, "budget": budget, "ids": ids,
                                    "recalled": bool(c.evidence_ids & kept),
                                    "recalled_all": all(g & kept for g in c.evidence_groups)}) + "\n")
            print(a.name, budget, "done", flush=True)
'''

WORKER = r'''
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

WORKER, WORKERS = int(sys.argv[1]), int(sys.argv[2])
MODEL = "%(model)s"
TOKEN_BUDGET = 12_000
data = {q["question_id"]: q for q in json.loads(Path("/kaggle/working/lme.json").read_text())}
rows = [json.loads(l) for l in Path("/kaggle/working/contexts.jsonl").read_text().splitlines()]
# one item per distinct (question, context) pair
items = {}
for r in rows:
    items.setdefault((r["qid"], tuple(sorted(r["ids"]))), r)
keys = sorted(items)[WORKER::WORKERS]

READ = ("I will give you several history chats between you and a user. Please answer the question based on "
        "the relevant chat history. Answer concisely.\n\nHistory Chats:\n\n{ctx}\n\nCurrent Date: {date}\n"
        "Question: {q}\nAnswer:")
BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
        "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
        "the response only contains a subset of the information required by the answer, answer no.{extra}\n\n"
        "Question: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only.")
EXTRA = {
    "temporal-reasoning": " In addition, do not penalize off-by-one errors for the number of days. If the question "
        "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 "
        "days when the answer is 20), the model's response is still correct.",
    "knowledge-update": " If the response contains some previous information along with an updated answer, the "
        "response should be considered as correct as long as the updated answer is the required answer.",
}
PREF = ("I will give you a question, a rubric for desired personalized response, and a response from a model. Please "
        "answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to "
        "reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's "
        "personal information correctly.\n\nQuestion: {q}\n\nRubric: {a}\n\nModel Response: {r}\n\nIs the model "
        "response correct? Answer yes or no only.")

def context_text(q, ids):
    sdate = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
    order = {sid: i for i, sid in enumerate(q["haystack_session_ids"])}
    turns = {}
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for t, turn in enumerate(sess):
            turns[f"{sid}:{t}"] = turn
    by_s = {}
    for i in ids:
        sid, t = i.rsplit(":", 1)
        by_s.setdefault(sid, []).append(int(t))
    parts = []
    for sid in sorted(by_s, key=lambda s: (sdate.get(s, ""), order.get(s, 0))):
        lines = [f"{turns[f'{sid}:{t}']['role']}: {turns[f'{sid}:{t}']['content']}" for t in sorted(by_s[sid])]
        parts.append(f"### Session Date: {sdate.get(sid, '?')}\n" + "\n".join(lines))
    return "\n\n".join(parts)

from transformers import BitsAndBytesConfig
READER = "%(reader)s"
tok = AutoTokenizer.from_pretrained(READER); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(READER, device_map="cuda:0", quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()

def chat(t):
    return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)

def generate(texts, max_new):
    lens = [len(tok(t).input_ids) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    outs, i, t0 = [None] * len(texts), 0, time.time()
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * lens[order[j]] <= TOKEN_BUDGET and j - i < 16:
            j += 1
        idx = order[i:j]
        enc = tok([texts[k] for k in idx], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        for k, g in zip(idx, gen[:, enc.input_ids.shape[1]:]):
            outs[k] = tok.decode(g, skip_special_tokens=True).strip()
        i = j
        if i %% 50 < len(idx):
            print(f"worker {WORKER}: {i}/{len(order)} ({time.time()-t0:.0f}s)", flush=True)
    return outs

EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
gc_eos = model.generation_config.eos_token_id
EOS |= set(gc_eos if isinstance(gc_eos, list) else [gc_eos])

def long_answer(text, max_new=120, chunk=2048):
    """Greedy decode with chunked prefill: attention memory stays ~chunk x length, so 16k+ prompts fit a T4."""
    ids = tok(text, return_tensors="pt").input_ids.to("cuda:0")
    cache = DynamicCache()
    with torch.no_grad():
        for s in range(0, ids.shape[1], chunk):
            out = model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = []
        for _ in range(max_new):
            t = nxt.item()
            if t in EOS:
                break
            gen.append(t)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    del cache
    return tok.decode(gen, skip_special_tokens=True).strip()

part = Path(f"/kaggle/working/answers_raw_{WORKER}.jsonl")
done = {}
if part.exists():
    for l in part.read_text().splitlines():
        r = json.loads(l); done[(r["qid"], tuple(r["ids"]))] = r["answer"]
print(f"worker {WORKER}: {len(keys)} answers to write, {len(done)} done", flush=True)
t0 = time.time()
with part.open("a") as f:
    for n, (qid, ids) in enumerate(keys):
        if (qid, ids) in done:
            continue
        q = data[qid]
        try:
            a = long_answer(chat(READ.format(ctx=context_text(q, ids), date=q.get("question_date", "?"), q=q["question"])))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); a = "[OOM]"
        done[(qid, ids)] = a
        f.write(json.dumps({"qid": qid, "ids": list(ids), "answer": a}) + "\n"); f.flush()
        if n %% 25 == 0:
            print(f"worker {WORKER}: {n}/{len(keys)} ({time.time()-t0:.0f}s)", flush=True)
answers = [done[k] for k in keys]
import gc
del model; gc.collect(); torch.cuda.empty_cache()
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda:0").eval()
judges = []
for (qid, ids), ans in zip(keys, answers):
    q = data[qid]
    kind = q["question_type"]
    tmpl = PREF if kind == "single-session-preference" else BASE.replace("{extra}", EXTRA.get(kind, ""))
    judges.append(chat(tmpl.format(q=q["question"], a=q["answer"], r=ans)))
verdicts = generate(judges, 4)
with open(f"/kaggle/working/answers_{WORKER}.jsonl", "w") as f:
    for (qid, ids), ans, v in zip(keys, answers, verdicts):
        f.write(json.dumps({"qid": qid, "ids": list(ids), "answer": ans, "verdict": v,
                            "correct": ans != "[OOM]" and v.strip().lower().startswith("yes")}) + "\n")
print(f"worker {WORKER}: finished", flush=True)
''' % {"model": MODEL, "reader": READER}

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Stronger reader: Qwen3-8B on the answer-accuracy contexts (LongMemEval_s)\n\nWork kernel for memory-recall-bench."),
    nbf.v4.new_code_cell(
        '!pip install -q bitsandbytes "memory-recall-bench[embeddings] @ git+https://github.com/Deepanjal-s/memory-recall-bench.git"\n'
        "import urllib.request, os, json\n"
        "if not os.path.exists('/kaggle/working/lme.json'):\n"
        "    urllib.request.urlretrieve('https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json', '/kaggle/working/lme.json')\n"
        ),
    nbf.v4.new_code_cell("%%writefile /kaggle/working/contexts.py\n" + CONTEXTS.strip()),
    nbf.v4.new_code_cell("!python /kaggle/working/contexts.py"),
    nbf.v4.new_code_cell("%%writefile /kaggle/working/worker.py\n" + WORKER.strip()),
    nbf.v4.new_code_cell(
        "import subprocess, os, torch\n"
        "W = max(1, torch.cuda.device_count())\n"
        "procs = [subprocess.Popen(['python', '/kaggle/working/worker.py', str(i), str(W)],\n"
        "          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}) for i in range(W)]\n"
        "print([p.wait() for p in procs])\n"
        "for p in ['lme.json', 'worker.py', 'contexts.py']:\n"
        "    os.remove('/kaggle/working/' + p)"),
]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / "kernel"
d.mkdir(exist_ok=True)
nbf.write(nb, d / "stronger-reader.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-stronger-reader", "title": "lme stronger reader",
    "code_file": "stronger-reader.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
