"""Self-test the scorer CLI with synthetic worker outputs."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).parent
BASE = "recent+hybrid"
SUMMARY = HERE.name.replace("-", "_") + "_summary.json"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main():
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        script = work / "score_answers.py"
        shutil.copyfile(HERE / script.name, script)

        def score(*args):
            return subprocess.run([sys.executable, str(script), str(work), *args],
                                  capture_output=True, text=True)

        contexts = []
        answers = []
        for i in range(16):
            for budget, fixes in ((4000, 16), (8000, 7)):
                for adapter in (BASE, "candidate"):
                    key = {"qid": f"q{i:02d}", "ids": [f"{adapter}:{budget}"]}
                    contexts.append({**key, "adapter": adapter, "budget": budget,
                                     "kind": "multi-session", "recalled_all": True})
                    answers.append({**key, "correct": adapter == "candidate" and i < fixes})

        write_jsonl(work / "contexts.jsonl", contexts)
        write_jsonl(work / "answers_0.jsonl", answers)
        result = score()
        assert result.returncode != 0 and "answers_1.jsonl" in result.stderr, result
        assert not (work / SUMMARY).exists()

        write_jsonl(work / "answers_0.jsonl", answers[2::2])
        write_jsonl(work / "answers_1.jsonl", answers[3::2])
        result = score()
        assert result.returncode != 0 and "Missing judged answers for question IDs: q00" in result.stderr, result
        assert not result.stdout and not (work / SUMMARY).exists()

        write_jsonl(work / "answers_0.jsonl", answers[::2])
        write_jsonl(work / "answers_1.jsonl", answers[1::2])
        write_jsonl(work / "contexts.jsonl", contexts[:1] + contexts[2:])
        result = score()
        assert result.returncode != 0 and "candidate at budget 4000: q00" in result.stderr, result
        assert not result.stdout and not (work / SUMMARY).exists()

        write_jsonl(work / "contexts.jsonl", contexts)
        result = score()
        assert result.returncode == 0, result.stderr
        assert "candidate 4000: fixed 16 / broken 0 p<0.001" in result.stdout, result.stdout
        assert "candidate 8000: fixed 7 / broken 0 p=0.0156" in result.stdout, result.stdout
        summary = json.loads((work / SUMMARY).read_text(encoding="utf-8"))
        assert summary["paired|candidate|4000"]["p"] == 2 / 2**16, summary
        assert summary["paired|candidate|8000"]["p"] == 2 / 2**7, summary

        (work / "worker_count.txt").write_text("3")
        result = score()
        assert result.returncode != 0 and "answers_2.jsonl" in result.stderr, result

    print(f"{HERE.name} scorer self-test passed")


if __name__ == "__main__":
    main()
