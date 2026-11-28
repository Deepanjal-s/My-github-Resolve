"""Private Kaggle GPU kernel: does packing FEWER, better hits beat filling the budget? ("precision, not quantity")

Follow-up to kaggle/packing-accuracy (recall up 72.9 -> 83.1, accuracy flat) and the answer-accuracy run
(more context with distractors hurt a small reader). Same ranking as there: ai-chat-memory keyword search, top 30
re-ranked by MiniLM-L-6 + user-turn prior (B=3). Arms:
  control  : bench standard packing, recent_share=0.5, 4k (the best earlier config, 64.3%)
  top-k    : only the k best re-ranked hits (+ reply pairs), no recent window, capped at 8k; k = 5, 10, 20
Same Qwen3-4B reader/judge and worker code as kaggle/answer-accuracy, so numbers line up with earlier runs.

    python build_kernel.py <aim_kw.json>   then   kaggle kernels push -p kernel
"""
import ast, base64, gzip, json, re, sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, r"C:\Projects\My git Resolve\projects\memory-recall-bench\src")
from memory_recall_bench.datasets import load_longmemeval
from memory_recall_bench.external import RankedFromFile

HERE = Path(__file__).parent
DATA = r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json"
N, B = 30, 3.0
kw = json.load(open(sys.argv[1]))
scores = json.load(open(HERE.parent / "rerank-topn" / "out" / "rerank_topn.json"))["scores"]
raw = json.load(open(DATA, encoding="utf-8"))
role = {f"{sid}:{i}": m["role"] for q in raw for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        for i, m in enumerate(sess)}
qtext = {q["question_id"]: q["question"] for q in raw}
AA = re.compile(r"\byou\b.*\b(said|say|told|tell|mention|recommend|suggest|gave|give|list|provide|explain)", re.I)
rank, srt = {}, {}
for cid, ids in kw.items():
    top = ids[:N]; b = 0.0 if AA.search(qtext[cid]) else B
    s = [scores[cid][j] + (b if role[i] == "user" else 0.0) for j, i in enumerate(top)]
    rank[cid] = [top[j] for j in sorted(range(len(top)), key=lambda j: (-s[j], j))] + ids[N:]
    srt[cid] = sorted(s, reverse=True)

convs = load_longmemeval(DATA)
ARMS = [("control recent=0.5", None, 0.5, 4000)] + [(f"top{k} recent=0", k, 0.0, 8000) for k in (5, 10, 20)]
KID, NB = "lme-precision-packing", "precision-packing"
if "--matched" in sys.argv:  # Codex review: matched comparator = full ranking, same 8k cap, no recent window
    ARMS, KID, NB = [("full recent=0", None, 0.0, 8000)], "lme-precision-packing-matched", "precision-packing-matched"
if "--adaptive" in sys.argv:  # variable k: keep hits scoring within d of the best hit (1..20), vs fixed k at a similar mean
    ARMS = [("top3 recent=0", 3, 0.0, 8000), ("top5 recent=0", 5, 0.0, 8000),
            ("cut4 recent=0", ("cut", 4.0), 0.0, 8000), ("cut6 recent=0", ("cut", 6.0), 0.0, 8000)]
    KID, NB = "lme-precision-packing-adaptive", "precision-packing-adaptive"


def keep(c, k):
    if k is None:
        return len(rank[c])
    if isinstance(k, tuple):
        v = srt[c]
        return min(20, max(1, sum(x >= v[0] - k[1] for x in v)))
    return k


rows = []
for name, k, share, budget in ARMS:
    rp = HERE / "ranking.json"
    rp.write_text(json.dumps({c: ids[:keep(c, k)] for c, ids in rank.items()}))
    a = RankedFromFile(str(rp), convs, name=name, recent_share=share, pair=True)
    rp.unlink()
    for c in convs:
        a.reset(budget)
        for m in c.messages:
            a.add(m)
        ids = list(a.context(c.question)); kept = set(ids)
        rows.append({"qid": c.id, "kind": c.kind, "adapter": name, "budget": budget, "ids": ids,
                     "recalled": bool(c.evidence_ids & kept),
                     "recalled_all": all(g & kept for g in c.evidence_groups)})
for name, *_ in ARMS:
    rs = [r for r in rows if r["adapter"] == name]
    print(f"{name:22s} recall_all {100 * sum(r['recalled_all'] for r in rs) / len(rs):5.1f}  "
          f"mean msgs {sum(len(r['ids']) for r in rs) / len(rs):5.1f}")
print("distinct (question, context) pairs:", len({(r["qid"], tuple(sorted(r["ids"]))) for r in rows}))

# reuse the reader/judge worker verbatim
src = (HERE.parent / "answer-accuracy" / "build_kernel.py").read_text(encoding="utf-8")
tree = ast.parse(src)
worker_tmpl = next(n.value.left.value for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", "") == "WORKER")
WORKER = worker_tmpl % {"model": "Qwen/Qwen3-4B-Instruct-2507"}

ctx_jsonl = "\n".join(json.dumps(r) for r in rows) + "\n"
blob = base64.b64encode(gzip.compress(ctx_jsonl.encode(), 9)).decode()  # plain JSON this size fails to push
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Precision-limited packing vs answer accuracy (LongMemEval_s)\n\n"
                             "Work kernel for memory-recall-bench."),
    nbf.v4.new_code_cell(
        "import urllib.request, os, gzip, base64\n"
        "if not os.path.exists('/kaggle/working/lme.json'):\n"
        "    urllib.request.urlretrieve('https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json', '/kaggle/working/lme.json')\n"
        f"open('/kaggle/working/contexts.jsonl', 'wb').write(gzip.decompress(base64.b64decode({blob!r})))"),
    nbf.v4.new_code_cell("%%writefile /kaggle/working/worker.py\n" + WORKER.strip()),
    nbf.v4.new_code_cell(
        "import subprocess, os, torch\n"
        "W = max(1, torch.cuda.device_count())\n"
        "procs = [subprocess.Popen(['python', '/kaggle/working/worker.py', str(i), str(W)],\n"
        "          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}) for i in range(W)]\n"
        "print([p.wait() for p in procs])\n"
        "for p in ['lme.json', 'worker.py']:\n"
        "    os.remove('/kaggle/working/' + p)"),
]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / {"precision-packing": "kernel", "precision-packing-matched": "kernel-matched"}.get(NB, "kernel-adaptive"); d.mkdir(exist_ok=True)
nbf.write(nb, d / f"{NB}.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"deepanjalshukla95/{KID}", "title": KID.replace("-", " "),
    "code_file": f"{NB}.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
