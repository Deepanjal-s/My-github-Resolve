"""Builds 3 private Kaggle kernels that rank all 479 LongMemEval_s questions with mem0 (one third each).

    python build_shards.py        then  kaggle kernels push -p shard0  (and shard1, shard2)

Each kernel writes mem0_rankings_shard{i}.json ({question id: [message ids, best first]}), checkpointed
after every conversation so a timeout still leaves the finished part.
"""
import json
from pathlib import Path

import nbformat as nbf

SHARDS = 3
for i in range(SHARDS):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"# mem0 retrieval on LongMemEval_s, shard {i + 1}/{SHARDS}\n\n"
            "Work kernel for [memory-recall-bench](https://github.com/Deepanjal-s/memory-recall-bench): "
            "mem0 2.2's own `search()` over verbatim messages (`add(infer=False)`, no LLM; fastembed bge-small, "
            "local Qdrant, spaCy; telemetry off). Output: rankings scored offline."),
        nbf.v4.new_code_cell(
            '!pip install -q "memory-recall-bench[mem0] @ git+https://github.com/Deepanjal-s/memory-recall-bench@v0.4.0"\n'
            "!python -m spacy download -q en_core_web_sm"),
        nbf.v4.new_code_cell(f'''
import json, os, time, urllib.request
os.environ["MEM0_TELEMETRY"] = "False"
from pathlib import Path
from memory_recall_bench import load_longmemeval
from memory_recall_bench.integrations.mem0 import Mem0Retriever

SHARD, SHARDS = {i}, {SHARDS}
DATA = Path("longmemeval_s_cleaned.json")
if not DATA.exists():
    urllib.request.urlretrieve("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json", DATA)
convs = sorted(load_longmemeval(DATA), key=lambda c: c.id)[SHARD::SHARDS]
print(len(convs), "conversations in this shard")
OUT = Path(f"/kaggle/working/mem0_rankings_shard{{SHARD}}.json")
done = json.loads(OUT.read_text()) if OUT.exists() else {{}}
r, t = Mem0Retriever(), time.time()
for c in convs:
    if c.id in done:
        continue
    done[c.id] = r.rank_all(c.messages, [c.question])[c.question]
    OUT.write_text(json.dumps(done))
    if len(done) % 10 == 0:
        print(len(done), f"done ({{time.time() - t:.0f}}s)", flush=True)
DATA.unlink()
print("finished", len(done))
'''.strip()),
    ]
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}}
    d = Path(f"shard{i}")
    d.mkdir(exist_ok=True)
    nbf.write(nb, d / "mem0-shard.ipynb")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"deepanjalshukla95/mem0-longmemeval-shard-{i + 1}",
        "title": f"mem0 LongMemEval shard {i + 1}",
        "code_file": "mem0-shard.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=1))
print("built", SHARDS, "shards")
