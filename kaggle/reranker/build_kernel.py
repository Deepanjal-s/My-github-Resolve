"""Private Kaggle GPU kernel: does a cross-encoder re-ranker make small budgets more precise, and does that
become answer accuracy? For all 479 LongMemEval_s questions with evidence:

  recent+hybrid          the baseline (bge-small + BM25, RRF)
  hybrid+rerank N=100    bge-reranker-v2-m3 re-scores the hybrid's top 100 archive messages
  hybrid+rerank rrf      RRF of the hybrid ranking and the re-ranker ranking (top 100), rest in hybrid order

Recall at 2k/4k/8k/16k for all three (recall.jsonl). Qwen3-4B-Instruct answers and judges (same prompts as
kaggle/answer-accuracy) at 4k and 8k only. Re-ranker scores are cached per (question, message).

    python build_kernel.py   then   kaggle kernels push -p kernel
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RERANKER = "BAAI/bge-reranker-v2-m3"
TOP_N = 100

CONTEXTS = r'''
import json, time
from pathlib import Path
import torch
from sentence_transformers import CrossEncoder
from memory_recall_bench import load_longmemeval
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusHybrid, sentence_transformers_encoder

convs = load_longmemeval("/kaggle/working/lme.json")
cache = EmbeddingCache(sentence_transformers_encoder("BAAI/bge-small-en-v1.5", device="cuda"), max_chars=512, batch_size=512)
t = time.time(); cache.get([m.content for c in convs for m in c.messages]); print(f"embedded in {time.time()-t:.0f}s", flush=True)
QP = "Represent this sentence for searching relevant passages: "
ce = CrossEncoder("%(reranker)s", max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})
TOP_N = %(top_n)d
CE = {}  # (query, message id) -> score
t_ce = [0.0, 0]

def ce_scores(query, msgs):
    todo = [m for m in msgs if (query, m.id) not in CE]
    if todo:
        t = time.time()
        s = ce.predict([(query, m.content[:2000]) for m in todo], batch_size=64, show_progress_bar=False)
        for m, v in zip(todo, s):
            CE[(query, m.id)] = float(v)
        t_ce[0] += time.time() - t; t_ce[1] += len(todo)
    return [CE[(query, m.id)] for m in msgs]

class Rerank(RecentPlusHybrid):
    def __init__(self, fuse):
        super().__init__(cache, query_prefix=QP)
        self.fuse = fuse
        self.name = "hybrid+rerank rrf" if fuse else f"hybrid+rerank N={TOP_N}"
    def rank(self, query, archive):
        orig = super().rank(query, archive)
        top, rest = orig[:TOP_N], orig[TOP_N:]
        s = ce_scores(query, top)
        by_ce = [m for _, m in sorted(zip(s, top), key=lambda x: -x[0])]
        if not self.fuse:
            return by_ce + rest
        score = {}
        for ranked in (top, by_ce):
            for r, m in enumerate(ranked):
                score[m.id] = score.get(m.id, 0.0) + 1.0 / (self.k + r + 1)
        return sorted(top, key=lambda m: -score[m.id]) + rest

makers = [lambda: RecentPlusHybrid(cache, query_prefix=QP), lambda: Rerank(False), lambda: Rerank(True)]
ANSWER_BUDGETS = (4000, 8000)
with open("/kaggle/working/contexts.jsonl", "w") as f, open("/kaggle/working/recall.jsonl", "w") as g:
    for make in makers:
        for budget in (2000, 4000, 8000, 16000):
            for c in convs:
                a = make(); a.reset(budget)
                for m in c.messages:
                    a.add(m)
                ids = list(a.context(c.question))
                kept = set(ids)
                row = {"qid": c.id, "kind": c.kind, "adapter": a.name, "budget": budget, "ids": ids,
                       "recalled": bool(c.evidence_ids & kept),
                       "recalled_all": all(gr & kept for gr in c.evidence_groups)}
                g.write(json.dumps({k: v for k, v in row.items() if k != "ids"}) + "\n")
                if budget in ANSWER_BUDGETS:
                    f.write(json.dumps(row) + "\n")
            print(a.name, budget, "done", f"ce {t_ce[1]} pairs in {t_ce[0]:.0f}s", flush=True)
del ce; torch.cuda.empty_cache()
''' % {"reranker": RERANKER, "top_n": TOP_N}

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
                            "correct": v.strip().lower().startswith("yes")}) + "\n")
print(f"worker {WORKER}: finished", flush=True)
''' % {"model": MODEL}

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Cross-encoder re-ranker: recall and answer accuracy (LongMemEval_s)\n\nWork kernel for memory-recall-bench."),
    nbf.v4.new_code_cell(
        '!pip install -q "memory-recall-bench[embeddings] @ git+https://github.com/Deepanjal-s/memory-recall-bench.git"\n'
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
nbf.write(nb, d / "reranker.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-reranker", "title": "lme reranker",
    "code_file": "reranker.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
