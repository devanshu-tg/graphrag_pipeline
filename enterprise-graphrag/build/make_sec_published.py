"""Build the PUBLISHED SEC result: v2 rubric + two corrected gold answers.

Two changes from eval_sec_fair_prompt.jsonl, both disclosed on the page.

1. RUBRIC v2 instead of v1. v1 penalised answers for omitting DERIVED detail — the gold is
   written "$57,048M (~$57.0B), down 2% YoY" and an answer of "$57,048 million" is that fact
   exactly. v2 states that the same number in another unit is identical, that a ticker equals a
   company name, that citations are ignored, and that on "which of A or B" naming the winner
   answers the question. It does NOT forgive a missing fact: a two-part question answered in one
   part still scores ~50.

   Every arm gained, and the WEAKEST gained most: LLM-only +3.6, Basic RAG +2.1, GraphRAG +0.9.
   The gap over the baseline NARROWED, which is the evidence the rubric was not bent.

   v3 was also built and rejected. It scored two answers 100 that recite real figures from the
   WRONG fiscal year — Walmart $611,289M when the answer is $713.2B — because it credited
   naming the right company on a comparison and ignored the numbers beside it. For financial QA
   that is not a correct answer.

2. TWO GOLD ANSWERS CORRECTED. EQ25 and EQ37 omit Citigroup from the list of KPMG-audited
   companies. The corpus is explicit:

       "…has been audited by KPMG LLP, Citi's independent registered public accounting firm…"
       "/s/ KPMG LLP  We have served as the Company's auditor since 1969."

   20 Citigroup chunks mention KPMG. All three arms listed it and all three were marked down for
   being more complete than the key. The key is corrected here and every arm is re-judged
   against the corrected version — this is not applied selectively.

The frozen eval_set.jsonl is NOT edited; mastergraph-round3/ must keep verifying 428/428. The
correction lives in this script and in the output file.

    python build/make_sec_published.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))
from _portability import init_console, load_env, ropen, wopen   # noqa: E402

load_env()
RESULTS = ROOT / "results" / "enterprise"
SRC = RESULTS / "eval_sec_v2judge.jsonl"          # already judged under v2
OUT = RESULTS / "eval_sec_published.jsonl"

# question_id -> corrected gold. Evidence is in this module's docstring.
GOLD_FIX = {
    "EQ25": "ACN, ADBE, BNY, C, COST, EMR, GD, HD, PEP, PFE, V, WFC (12)",
    "EQ37": "BNY Mellon (BNY), Citigroup (C), Wells Fargo (WFC); + Visa (V) if classed Financials",
}


def main() -> int:
    init_console()
    if not SRC.exists():
        print(f"ABORT: {SRC.name} missing — run llm_judge with LLM_JUDGE_RUBRIC=v2 first")
        return 1
    rows = [json.loads(l) for l in ropen(SRC) if l.strip()]

    os.environ["LLM_JUDGE_RUBRIC"] = "v2"          # same rubric as the rest of the file
    import llm_judge                                # noqa: E402

    n = 0
    for r in rows:
        qid = r["question_id"]
        if qid not in GOLD_FIX:
            continue
        r["gold_original"] = r["gold"]
        r["gold"] = GOLD_FIX[qid]
        r["gold_corrected"] = 1
        score, why = llm_judge.judge_one(r["question"], r["gold"], r.get("answer", ""))
        if score >= 0:
            r["judge_pct"] = round(score, 1)
            r["judge_score"] = round(score / 100.0, 4)
            r["judge_why"] = why
            r["f1"] = round(score / 100.0, 4)       # SEC has no set scorer; the judge IS f1
            r["exact"] = int(score >= 99.9)
        n += 1
        print(f"  {qid} [{r['arm']}]  {r.get('judge_pct')}  {why[:60]}")

    # llm_judge only rewrites judge_pct/judge_score/judge_why — it leaves `exact` and `f1`
    # holding whatever the ORIGINAL run computed. On SEC those were derived from the v1 judge,
    # so publishing without this loop mixes v1 exact flags with v2 judge scores in one file.
    # SEC has no deterministic scorer: the judge IS the score, so both fields are derived here.
    for r in rows:
        j = r.get("judge_pct")
        if j is None:
            continue
        r["f1"] = round(j / 100.0, 4)
        r["exact"] = int(j >= 99.9)

    with wopen(OUT) as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  re-judged {n} rows against the corrected key")
    print(f"  wrote {OUT.name}")

    import statistics as st
    print(f"\n  {'arm':12s}{'strict':>10}{'judge':>9}")
    print("  " + "-" * 31)
    for a, name in (("p1", "LLM only"), ("p2", "Basic RAG"), ("p3", "GraphRAG")):
        R = [r for r in rows if r["arm"] == a]
        print(f"  {name:12s}{sum(r['exact'] for r in R):>7}/{len(R):<2}"
              f"{st.mean([r.get('judge_pct') or 0 for r in R]):>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
