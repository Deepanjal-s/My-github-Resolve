# What does your chatbot forget?

**On Kaggle:** https://www.kaggle.com/code/deepanjalshukla95/what-does-your-chatbot-forget

Replays 50 LongMemEval conversations (~122k tokens each) through four chat-memory strategies and measures
**evidence recall**: whether the message holding the answer is still in context when the question is asked.
No LLM calls; fully reproducible.

| Budget | FIFO | Flush-to-archive (llama_index default) | Recent + random (control) | Recent + BM25 |
|---|---|---|---|---|
| 2k | 4% | 2% | 4% | **86%** |
| 8k | 8% | 4% | 16% | **98%** |
| 16k | 16% | 8% | 10% | **100%** |
| 64k | 58% | 44% | 58% | **100%** |

- **Validation:** FIFO matches its closed-form rule on all 300 (budget × question) cases. The random-retrieval control
  matches FIFO, so the BM25 gain comes from retrieval, not from extra budget.
- **llama_index #24117, quantified:** with `token_limit=3644`, the default 3000-token flush leaves the history budget
  only **46% full** on average (sawtooth), vs **88%** with a flush size scaled to the budget.
- **Caveat:** this split contains only `single-session-user` questions, which share words with the answer, so BM25's curve is optimistic.

Files: `memsim.py` (simulator, BM25, strategies), `build_notebook.py` → `memory-strategies.ipynb`.
Data: LongMemEval (Wu et al., 2024, MIT), Kaggle mirror `bonepyaesone/longmemeval-dev-json` (not committed).
