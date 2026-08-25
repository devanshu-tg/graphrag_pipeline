"""Judge my arms with the SAME judge the published column used, so the score is comparable.
Judging the PLANNER arm too is the calibration: if it lands on the published 100%, the rubric
is the right one and the regex arm's score can sit in the same table."""
import json, os, sys, pathlib
sys.path.insert(0, "."); sys.path.insert(0, "build")
for l in open(".env"):
    if "=" in l and not l.strip().startswith("#"):
        k, v = l.strip().split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
os.environ.setdefault("LLM_JUDGE_RUBRIC", os.getenv("RUBRIC", "v1"))
import llm_judge
print(f"judge={llm_judge.JUDGE_MODEL} rubric={os.environ['LLM_JUDGE_RUBRIC']}\n")
R = pathlib.Path("_experiments/regex_router/results")
for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(R / f"{name}.jsonl") if l.strip()]
    out, scores = [], []
    for i, r in enumerate(rows, 1):
        try:
            s, why = llm_judge.judge_one(r["question"], r["gold"], r["answer"])
        except Exception as e:
            s, why = -1, f"judge failed: {e}"
        if s >= 0: scores.append(s)
        r["judge_pct"] = s; r["judge_why"] = why
        out.append(r)
        print(f"  [{i:>2}/{len(rows)}] {r['question_id']} {s:>6.1f}", flush=True)
    with open(R / f"{name}_judged.jsonl", "w") as fh:
        for r in out: fh.write(json.dumps(r) + "\n")
    print(f"\n{name}: judged {len(scores)}/{len(rows)}  mean = "
          f"{sum(scores)/max(1,len(scores)):.1f}%  full-credit(100) = "
          f"{sum(1 for s in scores if s >= 100)}/{len(scores)}\n")
