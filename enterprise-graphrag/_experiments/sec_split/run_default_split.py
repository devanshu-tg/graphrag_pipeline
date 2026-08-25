"""Re-measure ONLY the 6 questions in the SEC matrix's `default` bucket, at the two budgets
sec_budgets.json covers, so the dropdown can show `compare` and `temporal` separately.

WHY A RE-RUN AT ALL. sec_budgets.json merged compare+temporal into `default` when it was
generated, and its generator does not exist anywhere in the repo (only build_report.py reads the
file). An average cannot be un-merged, so the two tiers have to be measured. Scope is kept to
the 6 affected questions rather than all 50: the other tiers in that file are untouched.

mastergraph-round3/ is READ-ONLY - its file hashes must keep verifying. This imports and calls,
writes nothing there. Same approach demo/server_factual.py already uses.

  .venv/bin/python _experiments/sec_split/run_default_split.py --chunks 10
  .venv/bin/python _experiments/sec_split/run_default_split.py --chunks 6
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ENT = pathlib.Path("/Users/devanshu/Desktop/GraphRAG_Demo/enterprise-graphrag")
MG = pathlib.Path("/Users/devanshu/Desktop/GraphRAG_Demo/mastergraph-round3")

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", type=int, required=True)
args = ap.parse_args()

for _l in open(ENT / ".env"):
    if "=" in _l and not _l.strip().startswith("#"):
        _k, _v = _l.strip().split("=", 1)
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

# Budget must be set BEFORE importing their pipelines - both read it into module constants at
# import time, exactly like the enterprise side.
os.environ.update({"CONTEXT_NUM_CHUNKS": str(args.chunks),
                   "MAX_CONTEXT_CHUNKS": str(args.chunks),
                   "DATASET": "round3"})

QUESTIONS = ["EQ13", "EQ14", "EQ15", "EQ45", "EQ46", "EQ47"]

# pull the question text and the route from the one per-question SEC file that records both
src = [json.loads(l) for l in open(ENT / "results/enterprise/eval_sec_norefusal.jsonl")]
meta = {r["question_id"]: r for r in src if r["arm"] == "p3"}
gold = {r["question_id"]: r.get("gold", "") for r in src if r["arm"] == "p3"}

os.chdir(MG)
sys.path.insert(0, str(MG))
sys.path.insert(0, str(MG / "pipeline3_round3"))

import pipeline2_basic_rag as p2            # noqa: E402
import pipeline3_round3.pipeline as p3      # noqa: E402

out = HERE / f"default_split_k{args.chunks}.jsonl"
fh = open(out, "w")
print(f"\nbudget={args.chunks} chunks, {len(QUESTIONS)} questions, arms p2 + p3\n")
for qid in QUESTIONS:
    q = meta[qid]["question"]
    for arm, mod in (("p2", p2), ("p3", p3)):
        t0 = time.time()
        try:
            m = mod.run_single(q)
            ans, err = (m.answer or "").strip(), ""
        except Exception as e:
            m, ans, err = None, "", f"{type(e).__name__}: {e}"
        row = {"question_id": qid, "arm": arm, "budget": args.chunks,
               "route": meta[qid].get("route"), "question": q, "gold": gold[qid],
               "answer": ans, "error": err,
               "prompt_tokens": getattr(m, "prompt_tokens", 0) if m else 0,
               "completion_tokens": getattr(m, "completion_tokens", 0) if m else 0,
               "latency_s": round(time.time() - t0, 2)}
        row["tokens"] = row["prompt_tokens"] + row["completion_tokens"]
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(f"  {qid} {row['route']:9s} {arm}  tokens={row['tokens']:>6,}  "
              f"{row['latency_s']:>5.1f}s  {ans[:44]!r}" + (f"  ERR {err[:50]}" if err else ""))
fh.close()
print(f"\nwrote {out}")
