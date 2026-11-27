"""Relative dates ("two weeks ago", "last Saturday", "10 days ago"): can date arithmetic find evidence that keyword
search never ranks? LongMemEval gives each session a date and each question a "current date". For questions with a
parseable relative expression, compute the target date and check how close the evidence sessions are to it.

Step 1 (this script): coverage. How many questions parse, how far the evidence is from the target date, and how
many messages a +-W day window would add. Step 2 (date_window_recall.py) measures recall with the window.

    python date_window.py
"""
import json, re
from collections import Counter
from datetime import datetime, timedelta

DATA = r"C:\Projects\My git Resolve\kaggle\longmemeval-full\data\longmemeval_s_cleaned.json"
raw = json.load(open(DATA, encoding="utf-8"))
NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
       "nine": 9, "ten": 10, "a couple of": 2, "couple of": 2, "a few": 3, "few": 3}
UNIT = {"day": 1, "week": 7, "month": 30}
DOW = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
AGO = re.compile(r"\b(\d+|a couple of|couple of|a few|few|an?|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                 r"(day|week|month)s?\s+ago\b", re.I)
LAST = re.compile(r"\blast\s+(week|month|" + "|".join(DOW) + r")\b", re.I)
YDAY = re.compile(r"\byesterday\b", re.I)


def parse_date(s):
    return datetime.strptime(s.split(" (")[0], "%Y/%m/%d")


def target(question, now):
    """(target date, tolerance in days) or None."""
    m = AGO.search(question)
    if m:
        n = m.group(1).lower()
        n = int(n) if n.isdigit() else NUM[n]
        days = n * UNIT[m.group(2).lower()]
        return now - timedelta(days=days), max(2, days // 7 + 1)
    m = LAST.search(question)
    if m:
        w = m.group(1).lower()
        if w == "week":
            return now - timedelta(days=7), 4
        if w == "month":
            return now - timedelta(days=30), 15
        back = (now.weekday() - DOW.index(w)) % 7 or 7
        return now - timedelta(days=back), 1
    if YDAY.search(question):
        return now - timedelta(days=1), 1
    return None


if __name__ == "__main__":
    hits, dist, sizes = 0, Counter(), []
    for q in raw:
        now = parse_date(q["question_date"])
        t = target(q["question"], now)
        if not t:
            continue
        hits += 1
        tgt, tol = t
        sdate = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        ev_sessions = set(q["answer_session_ids"])
        d = min(abs((parse_date(sdate[s]) - tgt).days) for s in ev_sessions if s in sdate)
        dist["within tol" if d <= tol else f"off by {d}"] += 1
        sizes.append(sum(len(sess) for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
                         if abs((parse_date(sdate[sid]) - tgt).days) <= tol))
    print("questions with a parseable relative date:", hits)
    print("evidence session vs target:", dist.most_common(12))
    sizes.sort()
    print("messages inside the window: median", sizes[len(sizes) // 2], "max", sizes[-1])
