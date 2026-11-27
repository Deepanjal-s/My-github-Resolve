"""Private Kaggle GPU kernel: does better recall give better answers? For every LongMemEval_s question with
evidence (479), build the retrieved context of two retrievers (recent+hybrid, and hybrid + prefix expansion
f=0.5) at 4k and 16k, let Qwen3-4B-Instruct answer from it, then judge each answer with the same model using
LongMemEval-style yes/no judge prompts. Identical contexts share one answer.

    python build_kernel.py   then   kaggle kernels push -p kernel

Outputs: /kaggle/working/contexts.jsonl (qid, adapter, budget, ids), answers_{0,1}.jsonl (one per T4).
The judge is a 4B model, not GPT-4o: absolute accuracy is rough, but both retrievers share the same reader
and judge, so the paired difference is what this measures.
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
EXP = {}
for line in (HERE.parent / "query-expansion" / "expansions.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    EXP[r["question"]] = r["queries"]
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

CONTEXTS = r'''
import json, time
from collections import defaultdict
from pathlib import Path
from memory_recall_bench import load_longmemeval
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusHybrid, sentence_transformers_encoder

EXP = json.loads(Path("/kaggle/working/exp.json").read_text())
convs = load_longmemeval("/kaggle/working/lme.json")
cache = EmbeddingCache(sentence_transformers_encoder("BAAI/bge-small-en-v1.5", device="cuda"), max_chars=512, batch_size=512)
t = time.time(); cache.get([m.content for c in convs for m in c.messages]); print(f"embedded in {time.time()-t:.0f}s", flush=True)
QP = "Represent this sentence for searching relevant passages: "

class PrefixHybrid(RecentPlusHybrid):
    def __init__(self, f):
        super().__init__(cache, query_prefix=QP)
        self.f, self.name = f, f"hybrid+exp prefix f={f:g}"
    def rank(self, query, archive):
        orig = super().rank(query, archive)
        subs = EXP.get(query, [])
        if not subs:
            return orig
        score = defaultdict(float)
        for q, ranked in [(query, orig)] + [(s, super().rank(s, archive)) for s in subs]:
            for r, m in enumerate(ranked):
                score[m.id] += 1.0 / (self.k + r + 1)
        by = {m.id: m for m in archive}
        cap, used, head = self.budget * (1 - self.recent_share) * self.f, 0, []
        for m in orig:
            if used + m.tokens > cap:
                break
            head.append(m); used += m.tokens
        seen = {m.id for m in head}
        return head + [by[i] for i in sorted(score, key=lambda i: -score[i]) if i not in seen]

makers = [lambda: RecentPlusHybrid(cache, query_prefix=QP), lambda: PrefixHybrid(0.5)]
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
    sdate = {}
    order = {}
    for i, (sid, date) in enumerate(zip(q["haystack_session_ids"], q["haystack_dates"])):
        sdate.setdefault(sid, date)
        order.setdefault(sid, i)
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

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda:0").eval()

def chat(t):
    return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)

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
''' % {"model": MODEL}

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Answer accuracy vs retrieval (LongMemEval_s)\n\nWork kernel for memory-recall-bench."),
    nbf.v4.new_code_cell(
        '!pip install -q "memory-recall-bench[embeddings] @ git+https://github.com/Deepanjal-s/memory-recall-bench.git"\n'
        "import urllib.request, os, json\n"
        "if not os.path.exists('/kaggle/working/lme.json'):\n"
        "    urllib.request.urlretrieve('https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json', '/kaggle/working/lme.json')\n"
        f"open('/kaggle/working/exp.json', 'w').write({json.dumps(EXP)!r})"),
    nbf.v4.new_code_cell("%%writefile /kaggle/working/contexts.py\n" + CONTEXTS.strip()),
    nbf.v4.new_code_cell("!python /kaggle/working/contexts.py"),
    nbf.v4.new_code_cell("%%writefile /kaggle/working/worker.py\n" + WORKER.strip()),
    nbf.v4.new_code_cell(
        "import subprocess, os, torch\n"
        "W = max(1, torch.cuda.device_count())\n"
        "with open('/kaggle/working/worker_count.txt', 'w') as f:\n"
        "    f.write(str(W))\n"
        "procs = [subprocess.Popen(['python', '/kaggle/working/worker.py', str(i), str(W)],\n"
        "          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}) for i in range(W)]\n"
        "print([p.wait() for p in procs])\n"
        "for p in ['lme.json', 'exp.json', 'worker.py', 'contexts.py']:\n"
        "    os.remove('/kaggle/working/' + p)"),
]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / "kernel"
d.mkdir(exist_ok=True)
nbf.write(nb, d / "answer-accuracy.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-answer-accuracy", "title": "lme answer accuracy",
    "code_file": "answer-accuracy.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
