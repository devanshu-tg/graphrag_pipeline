"""LLM-as-a-judge — a deliberately LENIENT second opinion alongside the deterministic score.

WHY A JUDGE AT ALL
------------------
The set scorer is exact and reproducible, but it is literal: it asks whether each gold string
appears in the answer. It cannot tell that "Tessa Morgan" and "tessa_morgan" are one person,
that "#eng-sre" and "eng-sre" are one channel, or that an answer which names 3 of 4 channels
and explains why is substantially right.

WHY IT IS NOT THE HEADLINE METRIC
---------------------------------
On the SEC run an LLM judge flipped a question between two passes over IDENTICAL answers.
A judge is a measuring instrument that changes its mind, so it is reported BESIDE the
deterministic number, never instead of it. If the two disagree sharply, that disagreement is
itself the finding and both get published.

BIAS CONTROL
------------
The judge model is deliberately DIFFERENT from the model that wrote the answers
(gemini-2.5-pro judging gemini-2.5-flash). A model scoring its own output shows measurable
self-preference, and using the same model for both would make every number here arguable.
The judge never sees which pipeline produced an answer — rows are scored blind, so it cannot
favour "the GraphRAG one".

    python build/llm_judge.py results/enterprise/eval_mixed.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, load_env, ropen, wopen   # noqa: E402

init_console()
load_env()

# Must stay a DIFFERENT (and stronger) model than the one that wrote the answers, which is
# now gemini-3.6-flash. A model grading its own output shows measurable self-preference.
JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "gemini-3.1-pro-preview")

RUBRIC = """You are grading answers to questions about a company's internal records.
Grade GENEROUSLY. You are checking whether the answer captures the right information, not
whether it is formatted nicely.

Question: {question}

CORRECT ANSWER: {gold}

ANSWER TO GRADE: {answer}

Give a PERCENTAGE from 0 to 100 for how much of the correct answer this captures.
Use the full range — 37 and 84 are valid scores, not only round numbers.

  100     names everything, nothing wrong (extra wording or ordering is fine)
  80-99   names everything but adds one or two items that do not belong
  50-79   names most of the correct items
  20-49   names a minority of the correct items
  1-19    names one small part, or is mostly wrong
  0       names none of it, is empty, or refuses to answer

Weigh two things: how much of the correct answer is present (coverage), and how much of what
the answer asserts is wrong (noise). A short correct answer beats a long one padded with
wrong items.

Treat these as THE SAME and never penalise them:
  - a username and a full name for one person ("tessa_morgan" = "Tessa Morgan")
  - a channel with or without "#" ("#eng-sre" = "eng-sre")
  - different capitalisation, punctuation, separators or order
  - an answer written as a sentence rather than a list

An empty answer or a refusal scores 0 — it conveys no information, however reasonable it may
have been to decline.

Reply with STRICT JSON only: {{"score": <0-100 number>, "why": "<one short sentence>"}}"""


def judge_one(question: str, gold: str, answer: str) -> tuple[float, str]:
    import requests
    key = os.getenv("GOOGLE_API_KEY", "")
    if not (answer or "").strip():
        return 0.0, "empty answer"          # no need to spend a call on nothing
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}"
           f":generateContent?key={key}")
    body = {"contents": [{"parts": [{"text": RUBRIC.format(
                question=question, gold=gold, answer=answer)}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096,
                                 "responseMimeType": "application/json"}}
    for attempt in range(4):
        try:
            r = requests.post(url, json=body, timeout=120)
            if r.status_code == 200:
                cand = (r.json().get("candidates") or [{}])[0]
                txt = "".join(p.get("text", "")
                              for p in cand.get("content", {}).get("parts", []))
                d = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M))
                pct = max(0.0, min(100.0, float(d.get("score", 0))))
                return pct, str(d.get("why", ""))[:90]
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt + 1)
                continue
            return -1.0, f"http {r.status_code}"
        except Exception as e:
            if attempt == 3:
                return -1.0, str(e)[:60]
            time.sleep(2 ** attempt + 1)
    return -1.0, "retries exhausted"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    import statistics as st
    for path in sys.argv[1:]:
        p = Path(path)
        rows = [json.loads(l) for l in ropen(p) if l.strip()]
        print(f"\n{p.name}: judging {len(rows)} answers with {JUDGE_MODEL} (blind to pipeline)")
        failed = 0
        for i, r in enumerate(rows, 1):
            s, why = judge_one(r["question"], r["gold"], r.get("answer", ""))
            if s < 0:
                failed += 1
                r["judge_score"] = None
                r["judge_why"] = why
            else:
                r["judge_pct"] = round(s, 1)
                r["judge_score"] = round(s / 100.0, 4)   # 0-1 copy, so the dashboard keeps working
                r["judge_why"] = why
            if i % 20 == 0:
                print(f"    {i}/{len(rows)}")
        with wopen(p) as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if failed:
            print(f"    WARNING: {failed} judgements failed and are excluded, not scored 0")

        names = {"p1": "P1 LLM-only", "p2": "P2 Basic RAG", "p2xl": "P2-XL Big context", "p2rr": "P2-RR Reranked",
                 "p2p": "P2+ Iterative RAG", "p3": "P3 GraphRAG"}
        by: dict = {}
        for r in rows:
            by.setdefault(r["arm"], []).append(r)
        print(f"\n  {'pipeline':<20}{'judge':>8}{'setF1':>8}{'gap':>8}")
        print("  " + "-" * 44)
        for a in ("p1", "p2", "p2rr", "p2xl", "p2p", "p3"):
            R = [x for x in by.get(a, []) if x.get("judge_score") is not None]
            if not R:
                continue
            j = st.mean([x.get("judge_pct", x["judge_score"] * 100) for x in R])
            f = st.mean([x["f1"] for x in R]) * 100
            print(f"  {names[a]:<20}{j:>7.1f}%{f:>7.1f}%{j-f:>+8.1f}")
        print("\n  (the judge is a lenient second opinion; the deterministic setF1 is the "
              "number of record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
