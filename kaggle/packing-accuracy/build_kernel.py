"""Private Kaggle GPU kernel: does the packing recall gain (kaggle/rerank-topn/packing.py) reach answers?

Same ranking for both arms: ai-chat-memory keyword search, top 30 re-ranked by MiniLM-L-6 + user-turn prior (B=3).
Arm A packs the bench's standard way (recent_share=0.5: half the budget for the latest messages). Arm B gives the
whole budget to the ranked hits (recent_share=0.0). Both keep reply pairs. Budgets 4k and 8k. Contexts are built
here (CPU) and shipped in the notebook; the kernel only reads and judges, with the same Qwen3-4B reader/judge and
worker code as kaggle/answer-accuracy, so the numbers line up with earlier runs.

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
rank = {}
for cid, ids in kw.items():
    top = ids[:N]; b = 0.0 if AA.search(qtext[cid]) else B
    s = [scores[cid][j] + (b if role[i] == "user" else 0.0) for j, i in enumerate(top)]
    rank[cid] = [top[j] for j in sorted(range(len(top)), key=lambda j: (-s[j], j))] + ids[N:]
rp = HERE / "ranking.json"; rp.write_text(json.dumps(rank))
convs = load_longmemeval(DATA)
rows = []
for share, name in ((0.5, "rerank+prior recent=0.5"), (0.0, "rerank+prior recent=0")):
    for budget in (4000, 8000):
        for c in convs:
            a = RankedFromFile(str(rp), convs, name=name, recent_share=share, pair=True)
            a.reset(budget)
            for m in c.messages:
                a.add(m)
            ids = list(a.context(c.question)); kept = set(ids)
            rows.append({"qid": c.id, "kind": c.kind, "adapter": name, "budget": budget, "ids": ids,
                         "recalled": bool(c.evidence_ids & kept),
                         "recalled_all": all(g & kept for g in c.evidence_groups)})
rp.unlink()
for name in sorted({r["adapter"] for r in rows}):
    for b in (4000, 8000):
        rs = [r for r in rows if r["adapter"] == name and r["budget"] == b]
        print(name, b, "recall_all", round(100 * sum(r["recalled_all"] for r in rs) / len(rs), 1))
print("distinct (question, context) pairs:", len({(r["qid"], tuple(sorted(r["ids"]))) for r in rows}))

# reuse the reader/judge worker verbatim
src = (HERE.parent / "answer-accuracy" / "build_kernel.py").read_text(encoding="utf-8")
tree = ast.parse(src)
worker_tmpl = next(n.value.left.value for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", "") == "WORKER")
WORKER = worker_tmpl % {"model": "Qwen/Qwen3-4B-Instruct-2507"}

ctx_jsonl = "\n".join(json.dumps(r) for r in rows) + "\n"
blob = base64.b64encode(gzip.compress(ctx_jsonl.encode(), 9)).decode()  # 1.3 MB of plain JSON made the push fail
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Packing vs answer accuracy (LongMemEval_s)\n\nWork kernel for memory-recall-bench."),
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
        "with open('/kaggle/working/worker_count.txt', 'w') as f:\n"
        "    f.write(str(W))\n"
        "procs = [subprocess.Popen(['python', '/kaggle/working/worker.py', str(i), str(W)],\n"
        "          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}) for i in range(W)]\n"
        "print([p.wait() for p in procs])\n"
        "for p in ['lme.json', 'worker.py']:\n"
        "    os.remove('/kaggle/working/' + p)"),
]
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
d = HERE / "kernel"; d.mkdir(exist_ok=True)
nbf.write(nb, d / "packing-accuracy.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "deepanjalshukla95/lme-packing-accuracy", "title": "lme packing accuracy",
    "code_file": "packing-accuracy.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, indent=1))
print("built", d)
