"""Score MasterGraph's recorded SEC answers, so the demo can show only answered questions.

The SEC evaluation set carries a gold answer for all 50 questions ("$416,161M (~$416.2B)"),
and _evidence/theirs.jsonl carries what each of their three pipelines actually replied. Those
were never scored per question in a form this repo can read — the round reported a blended
judge score only. Without per-question scores the dashboard has to show all 50 chips
regardless of whether anything answered them, which makes a demo a coin flip.

Same lenient judge as the textual side, same model (gemini-2.5-pro, different from the
gemini-2.5-flash that produced the answers), blind to which pipeline it is grading.

Financial answers need one extra rule the channel-name rubric did not: "$416,161 million",
"$416,161M" and "~$416.2B" are the same number written three ways, and a judge that misses
that would mark correct answers wrong.

Writes results/enterprise/sec_judged.jsonl. Nothing under _evidence/ is modified.

    python build/judge_sec.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
from _portability import init_console, load_env, ropen, wopen   # noqa: E402

init_console()
load_env()

import llm_judge   # noqa: E402

MG = Path(__file__).resolve().parents[2] / "mastergraph-round3"
EVID = Path(__file__).resolve().parents[2] / "_evidence"
OUT = Path("results/enterprise/sec_judged.jsonl")

# override the rubric with a numeric-aware variant, then restore nothing — this module is a
# one-shot script and the judge module is reloaded fresh by any other caller
llm_judge.RUBRIC = llm_judge.RUBRIC.replace(
    "Treat these as THE SAME and never penalise them:",
    "Treat these as THE SAME and never penalise them:\n"
    "  - the same number written differently: \"$416,161 million\" = \"$416,161M\" = \"~$416.2B\"\n"
    "    = \"416161\" (units, separators, abbreviation and sensible rounding are all equivalent)\n"
    "  - a correct value followed by a Citations: block\n"
    "  - extra precision or an explicit currency where the gold omits it")

ARMS = [("mg-p1-llmonly", "p1"), ("mg-p2-rag", "p2"), ("mg-p3-graphrag", "p3")]


def main() -> int:
    gold = {}
    for line in ropen(MG / "datasets" / "round3" / "eval_set.jsonl"):
        if line.strip():
            d = json.loads(line)
            gold[d["id"]] = d
    ans: dict = {}
    for line in ropen(EVID / "theirs.jsonl"):
        if line.strip():
            r = json.loads(line)
            ans.setdefault(r["question_id"], {})[r["arm"]] = r

    rows, n = [], 0
    total = len(gold) * len(ARMS)
    for qid, q in gold.items():
        for arm, short in ARMS:
            r = ans.get(qid, {}).get(arm) or {}
            s, why = llm_judge.judge_one(q["question"], q["answer"], r.get("answer", ""))
            n += 1
            rows.append({"question_id": qid, "arm": short, "tier": q.get("tier", ""),
                         "question": q["question"], "gold": q["answer"],
                         "answer": (r.get("answer") or "").strip(),
                         "judge_score": None if s < 0 else s, "judge_why": why,
                         "prompt_tokens": r.get("prompt_tokens", 0),
                         "completion_tokens": r.get("completion_tokens", 0)})
            if n % 25 == 0:
                print(f"    {n}/{total}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wopen(OUT) as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics as st
    from collections import Counter
    print(f"\nwrote {OUT}")
    print(f"{'arm':<6}{'mean judge':>12}{'>=0.75':>9}{'answered':>10}")
    for _, short in ARMS:
        R = [r for r in rows if r["arm"] == short and r["judge_score"] is not None]
        good = sum(1 for r in R if r["judge_score"] >= 0.75)
        print(f"{short:<6}{st.mean([r['judge_score'] for r in R]):>12.3f}{good:>9}{len(R):>10}")
    best = {}
    for r in rows:
        if r["judge_score"] is not None:
            best[r["question_id"]] = max(best.get(r["question_id"], 0), r["judge_score"])
    print(f"\nquestions where SOMETHING scored >=0.75: "
          f"{sum(1 for v in best.values() if v >= 0.75)}/{len(best)}")
    print(f"per tier: {dict(Counter(gold[q]['tier'] for q, v in best.items() if v >= 0.75))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
