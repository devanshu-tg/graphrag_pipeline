"""EXPERIMENT harness — measures the regex router against the LLM planner at three budgets.

Touches no production file. The regex router is injected by monkeypatching
`schema_router.plan` at runtime, the same technique demo/server_factual.py already uses to
repair MasterGraph's router without editing it.

  python _experiments/regex_router/run_bench.py --arm graph-regex --chunks 3
  python _experiments/regex_router/run_bench.py --arm graph-planner --chunks 10
  python _experiments/regex_router/run_bench.py --arm basic --chunks 10

Writes one JSONL per (arm, chunks) into _experiments/regex_router/results/, appending as it
goes so a killed run can be resumed and partial progress is never lost.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, ".")
sys.path.insert(0, "pipeline3_round3")
sys.path.insert(0, str(HERE))

for _line in open(".env"):
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["graph-regex", "graph-planner", "basic"])
ap.add_argument("--chunks", type=int, required=True)
ap.add_argument("--n", type=int, default=50)
# RR_POOL changes WHICH chunks the cross-encoder can pick, not how many, so it moves
# accuracy while leaving the token count almost unchanged. budget50_judge.json records
# rr_pool=500; the code default is 50. That is the whole difference between a baseline
# that scores 23-24/50 and one that scores 31/50.
ap.add_argument("--rr-pool", type=int, default=50)
ap.add_argument("--questions", default="datasets/enterprise/eval_mixed50.jsonl")
ap.add_argument("--tag", default="")
args = ap.parse_args()

# The chunk budget must be set BEFORE the pipeline modules are imported: they read these into
# module-level constants at import time, so setting them afterwards has no effect.
os.environ.update({
    "TIGERGRAPH_GRAPH": "enterprise",
    "CONTEXT_NUM_CHUNKS": str(args.chunks),
    "MAX_CONTEXT_CHUNKS": str(args.chunks),
    "P3_VEC_K": str(args.chunks),
    "P3_VEC_SLOTS": str(args.chunks),
    # Same split the shipped demo uses: one slot for traversal evidence, the rest for chunks,
    # scaled so a 10-chunk run is not silently a 1-prov/9-chunk run at one budget and
    # 1-prov/2-chunk at another.
    "P3_PROV_SLOTS": "1",
    "RR_POOL": str(args.rr_pool),
    "RR_KEEP": str(args.chunks),
})

from llm_client import get_llm_response          # noqa: E402
# MUST be the top-level name, not pipeline3_round3.schema_router: sys.path carries both
# '.' and 'pipeline3_round3', so the two spellings create TWO module objects and
# pipeline.py imports the top-level one. Patching the other silently does nothing while
# still reporting planner_tokens=0 from the unused copy - a fake 43% saving.
import schema_router                              # noqa: E402
import regex_router as rr                        # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def set_scores(answer: str, gold: list[str]) -> tuple[float, float, float]:
    """Copied verbatim from build/run_eval.py so scores are comparable to published runs."""
    a = norm(answer)
    hit = [g for g in gold if norm(g) and norm(g) in a]
    asserted = {t for t in re.split(r"[,\n;]|\s+and\s+|\s{2,}", answer or "") if norm(t)}
    tp = len(hit)
    p = min(tp / max(1, len(asserted)), 1.0) if asserted else 0.0
    r = tp / max(1, len(gold))
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ---------------------------------------------------------------- router injection
_real_plan = schema_router.plan
_stats = {"regex": 0, "fallback": 0}


def patched_plan(question: str, get_llm):
    """Regex first; the LLM planner only when no pattern matches OR the regex plan already
    walked to nothing. pipeline.run_single re-plans when a plan lands nothing, and returning the
    same regex plan on the retry would just fail twice — so the second call for a question
    hands over to the real planner."""
    key = question.strip()
    if patched_plan._last == key:
        _stats["fallback"] += 1
        return _real_plan(question, get_llm)
    patched_plan._last = key
    hit = rr.route(question)
    if hit is None:
        _stats["fallback"] += 1
        return _real_plan(question, get_llm)
    _stats["regex"] += 1
    return hit[2]


patched_plan._last = None

if args.arm == "graph-regex":
    schema_router.plan = patched_plan

qs = [json.loads(l) for l in open(args.questions) if l.strip()]
qs = qs[: args.n]

_suffix = args.tag or ("" if args.rr_pool == 50 else f"_pool{args.rr_pool}")
out = HERE / "results" / f"{args.arm}_k{args.chunks}{_suffix}.jsonl"
done = set()
if out.exists():
    done = {json.loads(l)["question_id"] for l in open(out) if l.strip()}
    print(f"resuming: {len(done)} rows already in {out.name}")

if args.arm == "basic":
    import pipeline2rr_rerank as arm_mod
else:
    import pipeline3_round3.pipeline as arm_mod

print(f"\narm={args.arm}  chunks={args.chunks}  questions={len(qs)}\n")
t_all = time.time()
fh = open(out, "a")
for i, q in enumerate(qs, 1):
    if q["question_id"] in done:
        continue
    schema_router.reset_token_usage()
    t0 = time.time()
    try:
        m = arm_mod.run_single(q["question"])
        ans = m.answer or ""
        err = ""
    except Exception as e:                        # a dead question must not kill the run
        ans, err = "", f"{type(e).__name__}: {e}"
        m = None
    u = schema_router.plan_token_usage()
    plan_tok = u.get("prompt", 0) + u.get("output", 0)
    gm = q.get("gold_members")
    p, r, f1 = set_scores(ans, gm) if gm else (0.0, 0.0, 0.0)
    row = {
        "question_id": q["question_id"], "arm": args.arm, "chunks_budget": args.chunks,
        # eval_final50_v3.jsonl omits `shape` on its 20 prose questions, so this must be
        # .get() - q["shape"] killed the run at F3_21 with a KeyError.
        "shape": q.get("shape"), "tier": q.get("tier", ""), "question": q["question"],
        "gold": q["answer"], "answer": ans, "error": err,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "exact": int(r == 1.0 and p >= 0.99) if gm else None,
        "has_gold_members": bool(gm),
        "prompt_tokens": getattr(m, "prompt_tokens", 0) if m else 0,
        "completion_tokens": getattr(m, "completion_tokens", 0) if m else 0,
        "planner_tokens": plan_tok,
        "route": getattr(m, "route", "") if m else "error",
        "retrieved_chunks": getattr(m, "retrieved_chunks", 0) if m else 0,
        "latency_s": round(time.time() - t0, 2),
    }
    row["tokens"] = row["prompt_tokens"] + row["completion_tokens"] + row["planner_tokens"]
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    print(f"  [{i:>2}/{len(qs)}] {q['question_id']} {row['route']:12s} "
          f"tok={row['tokens']:>6,} plan={plan_tok:>5,} chunks={row['retrieved_chunks']:>2} "
          f"exact={row['exact']} {row['latency_s']:>5.1f}s"
          + (f"  ERR {err[:60]}" if err else ""))
fh.close()
print(f"\ndone in {(time.time()-t_all)/60:.1f} min   "
      f"regex plans={_stats['regex']} planner fallbacks={_stats['fallback']}")
print(f"wrote {out}")
