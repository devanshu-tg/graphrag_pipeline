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


# ---------------------------------------------------------------------------------------------
# v2. Selected with LLM_JUDGE_RUBRIC=v2. v1 is left exactly as it was so any number can be
# reported under either and the difference between them is itself the disclosure.
#
# v2 exists because v1 penalised answers for omitting DERIVED detail. On the SEC set the gold is
# written like "$57,048M (~$57.0B), down 2% YoY" — one fact plus two annotations of it. An answer
# of "$57,048 million" is the fact, exactly, and scored 0.85 under v1 for missing the annotation.
# That is a rubric problem, not an answer problem.
#
# What v2 does NOT do: relax multi-part questions. "Which company, and who is its auditor"
# answered with only the auditor is half an answer under both rubrics. Marking that correct
# would be moving the bar until the score improves.
RUBRIC_V2 = """You are grading answers to factual questions about companies and their filings.
Grade GENEROUSLY on form and STRICTLY on substance. You are checking whether the answer states
the facts the question asked for — not whether it repeats every word of the reference.

Question: {question}

REFERENCE ANSWER: {gold}

ANSWER TO GRADE: {answer}

First decide what the question actually asked for. The reference answer often contains MORE than
that: a figure restated in other units, an approximation in brackets, a year-on-year change, or
the losing side of a comparison. Those are annotations of the answer, not extra required facts.

  100     states every fact the question asked for, and states them correctly
  80-99   states every required fact, plus one or two assertions that do not belong
  50-79   states most of the required facts, or answers one half of a two-part question
  20-49   states a minority of the required facts
  1-19    states one small part, or is mostly wrong
  0       states none of it, is empty, or refuses to answer

Treat these as THE SAME and never penalise them:
  - the same number in different units or formats:
      "$57,048 million" = "$57,048M" = "~$57.0B" = "57,048"
      "$416,161M" = "~$416.2B"
  - a ticker and a company name ("CL" = "Colgate-Palmolive", "NVDA" = "NVIDIA")
  - a username and a full name ("tessa_morgan" = "Tessa Morgan")
  - a channel with or without "#" ("#eng-sre" = "eng-sre")
  - different capitalisation, punctuation, separators, ordering or wording
  - an answer written as a sentence rather than a list
  - omitting derived commentary that only annotates the answer: a YoY change, a bracketed
    approximation, a unit conversion, or a restatement of the same figure
  - on a "which of A or B" question: naming the correct one with its figure. Repeating the other
    one's figure is not required.
  - citations, source ids or provenance lines appended to the answer. Ignore them entirely.

DO still penalise, and score around 50:
  - a two-part question answered in one part (asked for a company AND its auditor, gave only one)
  - a set answer missing members, in proportion to how many are missing

An empty answer or a refusal scores 0 — it conveys no information, however reasonable it may
have been to decline.

Reply with STRICT JSON only: {{"score": <0-100 number>, "why": "<one short sentence>"}}"""


# v3 — LENIENT. Selected with LLM_JUDGE_RUBRIC=v3.
#
# Grades usefulness rather than string similarity. It scores a correct-but-incomplete set on
# what it got right instead of dragging it down for omissions, and treats a bundled secondary
# lookup ("and who is its auditor") as a smaller part of the question than the main fact.
#
# It deliberately keeps two floors: a FALSE assertion costs more than an omission, and a wrong
# central fact stays low. Without those, "lenient" would just mean "everything passes" and the
# score would stop carrying information.
RUBRIC_V3 = """You are grading answers to factual questions about companies and their filings.

Judge the answer the way the person who ASKED the question would: is this a useful, correct
answer to what I asked? Not: does it match the reference wording.

Question: {question}

REFERENCE ANSWER: {gold}

ANSWER TO GRADE: {answer}

Work out what the question is really asking for, and what is secondary. A reference answer often
carries extra material — a figure restated in other units, a bracketed approximation, a
year-on-year change, the losing side of a comparison, or the bridging entity used to find the
subject. None of that is a required fact.

  100     answers the question. Everything it asserts is correct, and it covers the main thing
          asked. A shorter answer than the reference is still 100 if nothing required is missing
          and nothing stated is wrong.
  85-99   answers the main thing correctly, and omits a secondary detail the question also
          mentioned (for example a bundled "and who is its auditor")
  70-84   a set answer where everything named is CORRECT but some members are missing
  40-69   gets roughly half of what was asked, OR names correct items alongside incorrect ones
  10-39   a small part right, or mostly padded with items that do not belong
  0       nothing right, empty, or a refusal

Be generous about form and strict about truth:

TREAT AS IDENTICAL, never penalise:
  - the same number in any unit or format: "$57,048 million" = "$57,048M" = "~$57.0B"
  - a ticker and a company name ("CL" = "Colgate-Palmolive", "NVDA" = "NVIDIA")
  - a username and a full name; a channel with or without "#"
  - capitalisation, punctuation, separators, ordering, wording, prose vs list
  - omitting derived commentary that only annotates the answer
  - omitting the bridging entity used to reach the subject, when the question asked about the
    subject
  - on "which of A or B" questions: naming the correct one. The other one's figure is not needed.
  - an unambiguous role description used in place of a proper name
  - citations, source ids and provenance lines. Ignore them completely.

DO penalise, and do not be talked out of it:
  - asserting something FALSE. An answer that names four correct companies and three wrong ones
    is worse than one naming four correct and stopping. Wrong information costs more than
    missing information.
  - getting the central fact wrong, even if surrounding detail is right
  - attributing the right fact to the wrong person or company
  - an empty answer or a refusal: 0, however reasonable declining may have been

Reply with STRICT JSON only: {{"score": <0-100 number>, "why": "<one short sentence>"}}"""


def _rubric() -> str:
    """v1 by default. Nothing silently changes scoring; the switch has to be set."""
    v = os.getenv("LLM_JUDGE_RUBRIC", "v1").lower()
    return {"v2": RUBRIC_V2, "v3": RUBRIC_V3}.get(v, RUBRIC)


def judge_one(question: str, gold: str, answer: str) -> tuple[float, str]:
    import requests
    key = os.getenv("GOOGLE_API_KEY", "")
    if not (answer or "").strip():
        return 0.0, "empty answer"          # no need to spend a call on nothing
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}"
           f":generateContent?key={key}")
    body = {"contents": [{"parts": [{"text": _rubric().format(
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
        print(f"\n{p.name}: judging {len(rows)} answers with {JUDGE_MODEL} "
              f"rubric={os.getenv('LLM_JUDGE_RUBRIC','v1')} (blind to pipeline)")
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
