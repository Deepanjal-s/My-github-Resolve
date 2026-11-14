"""Simulate chat-memory strategies on LongMemEval haystacks and measure evidence recall.

A "turn" is one message. Each strategy decides which past turns are in the model's
context when the final question is asked. Recall = the answer-bearing turn is in it.
Token counts use a simple, deterministic approximation (characters / 4) so results
don't depend on a tokenizer download.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("""a an the and or but if then of to in on at for with by from as is are was were be been being i you he she it we they me
my your our their this that these those do does did have has had not no so can could would should will just about what which who
how when where why there here up out into over than too very also any some all more most other such only own same s t don""".split())


def ntok(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def words(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOP]


@dataclass
class Turn:
    idx: int
    role: str
    text: str
    tokens: int
    evidence: bool


def flatten(question: dict) -> list[Turn]:
    out, i = [], 0
    for session in question["haystack_sessions"]:
        for t in session:
            out.append(Turn(i, t["role"], t["content"], ntok(t["content"]), bool(t.get("has_answer"))))
            i += 1
    return out


# ---------------------------------------------------------------- strategies
def fifo(turns: list[Turn], budget: int, **_) -> set[int]:
    """Keep the most recent turns that fit in the budget."""
    kept, used = set(), 0
    for t in reversed(turns):
        if used + t.tokens > budget:
            break
        kept.add(t.idx)
        used += t.tokens
    return kept


def flush_to_archive(turns: list[Turn], budget: int, flush_size: int | None = None, ratio: float = 0.7,
                     trace: list | None = None, **_) -> set[int]:
    """Port of llama_index ``Memory._manage_queue`` (llama-index-core 0.14.x), no memory blocks.

    When the active history exceeds ratio*budget, flush the oldest messages in chunks of at
    least `flush_size` tokens; always keep one message; the remaining history must start
    with a user message; if that would leave nothing, restore the last complete user turn.
    The library archives the N oldest rows, so the active history is queue[N:].
    Checked message-for-message against the real library in memory-recall-bench.
    """
    limit = budget * ratio
    if flush_size is None:
        flush_size = 3000
        if flush_size > budget:            # Memory.from_defaults rule
            flush_size = int(budget * 0.7)
    queue: list[Turn] = []
    for t in turns:
        queue.append(t)
        if sum(x.tokens for x in queue) > limit:
            rev = queue[::-1]
            archived = 0
            to_remove = sum(x.tokens for x in queue) - limit
            while to_remove > 0 and len(rev) > 1:
                flushed, ft = [], 0
                while ft < flush_size and rev and len(rev) > 1:
                    m = rev.pop(); flushed.append(m); ft += m.tokens
                if not rev and flushed:
                    rev.append(flushed.pop())
                while rev and rev[-1].role != "user" and len(rev) > 1:
                    flushed.append(rev.pop())
                if (not rev or (len(rev) == 1 and rev[0].role != "user")) and flushed:
                    if rev and rev[0].role != "user":
                        flushed.append(rev.pop(0))
                    start = next((i for i in range(len(flushed) - 1, -1, -1) if flushed[i].role == "user"), -1)
                    if start >= 0:
                        rev = flushed[start:][::-1] + rev
                        flushed = flushed[:start]
                archived += len(flushed)
                to_remove = sum(x.tokens for x in rev) - limit
                if not flushed:
                    break
            queue = queue[archived:]
        if trace is not None:
            trace.append(sum(x.tokens for x in queue))
    return {t.idx for t in queue}


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs))
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for w in query:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


def recent_plus_bm25(turns: list[Turn], budget: int, question: str = "", recent_share: float = 0.5,
                     pair: bool = True, **_) -> set[int]:
    """Keep a FIFO window of recent turns, then fill the rest of the budget with the
    archived turns that best match the question (BM25). With `pair`, a retrieved
    turn brings its neighbour (user<->assistant) so the exchange stays readable."""
    recent = fifo(turns, int(budget * recent_share))
    used = sum(turns[i].tokens for i in recent)
    archive = [t for t in turns if t.idx not in recent]
    if not archive:
        return recent
    bm = BM25([words(t.text) for t in archive])
    ranked = sorted(zip(bm.scores(words(question)), archive), key=lambda x: -x[0])
    kept = set(recent)
    for score, t in ranked:
        if score <= 0:
            break
        group = [t]
        if pair:
            nb = t.idx + 1 if t.role == "user" else t.idx - 1
            if 0 <= nb < len(turns) and nb not in kept:
                group.append(turns[nb])
        cost = sum(g.tokens for g in group if g.idx not in kept)
        if used + cost > budget:
            continue
        for g in group:
            kept.add(g.idx)
        used += cost
    return kept


def evidence_recalled(turns: list[Turn], kept: set[int]) -> bool:
    return any(t.evidence and t.idx in kept for t in turns)


def evidence_age_tokens(turns: list[Turn]) -> int:
    """Tokens between the (latest) evidence turn and the end of the conversation."""
    last = max(t.idx for t in turns if t.evidence)
    return sum(t.tokens for t in turns[last + 1:])


def recent_plus_random(turns: list[Turn], budget: int, recent_share: float = 0.5, seed: int = 0, **_) -> set[int]:
    """Control: same budget split as recent_plus_bm25, but archive turns are chosen at
    random. Separates the effect of *retrieval* from the effect of *having budget*."""
    import random
    rng = random.Random(seed)
    recent = fifo(turns, int(budget * recent_share))
    used = sum(turns[i].tokens for i in recent)
    archive = [t for t in turns if t.idx not in recent]
    rng.shuffle(archive)
    kept = set(recent)
    for t in archive:
        if used + t.tokens <= budget:
            kept.add(t.idx)
            used += t.tokens
    return kept
