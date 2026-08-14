"""Run the 50 SEC questions LIVE through MasterGraph's pipelines and score them.

Until now the factual tab replayed frozen answers from _evidence/ because the `finance` graph
had lost every installed query, so their P3 could not retrieve. The queries are back
(build/round3_tg_ingest.py --queries-only, 13 of them, verified returning rows), so the round
can actually be re-run rather than replayed.

Requires the worker: python demo/server_factual.py   (port 8001, own process — their code and
ours both define `config`/`pipeline1_llm_only`, so they cannot share an interpreter).

Scoring uses SEC gold from mastergraph-round3/datasets/round3/eval_set.jsonl. Their answers are
single facts with commentary ("$57,048M (~$57.0B), down 2% YoY") rather than name sets, so the
set-F1 scorer built for the enterprise questions does NOT transfer — a numeric answer either
matches or it does not. Correctness here is decided by the lenient judge, and `exact` is the
judge awarding 100.

    python build/run_sec_live.py --out results/enterprise/eval_sec_live.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
from _portability import init_console, load_env, ropen, wopen   # noqa: E402

init_console()
load_env()

MG = Path(__file__).resolve().parents[2] / "mastergraph-round3"
# T4 runs a SECOND worker on another port under a fair generation config, so the dashboard's
# worker on 8001 keeps serving untouched while the re-measurement runs.
WORKER = os.getenv("SEC_WORKER", "http://localhost:8001")
ARMS = {"p1": "P1 LLM-only", "p2": "P2 Basic RAG", "p3": "P3 GraphRAG"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/enterprise/eval_sec_live.jsonl")
    ap.add_argument("--n", type=int, default=50)
    # dev_set.jsonl is MasterGraph's OWN held-out split: 24 questions, same A/B/C/D tiers, with
    # gold answers, and ZERO overlap with the 50 in eval_set.jsonl (verified). It is the honest
    # way to get new SEC questions without inventing gold for a corpus of real filings, and it
    # has never been used in any measurement by either team.
    ap.add_argument("--questions", default="datasets/round3/eval_set.jsonl",
                    help="path relative to mastergraph-round3/")
    args = ap.parse_args()

    import requests
    try:
        st = requests.get(f"{WORKER}/status", timeout=5).json()
    except Exception as e:
        print(f"worker unreachable at {WORKER} — start demo/server_factual.py first ({e})")
        return 2
    if st.get("state") != "ready":
        print(f"worker not ready: {st}")
        return 2
    print(f"worker ready. running {args.n} SEC questions x 3 pipelines LIVE\n")

    qs = [json.loads(l) for l in ropen(MG / args.questions) if l.strip()][: args.n]
    print(f"  questions: {args.questions}  ({len(qs)} loaded)")

    import llm_judge
    rows, t0 = [], time.time()
    for i, q in enumerate(qs, 1):
        try:
            d = requests.post(f"{WORKER}/query", json={"question": q["question"]},
                              timeout=300).json()
        except Exception as e:
            print(f"  {q['id']} request failed: {str(e)[:70]}")
            continue
        for r in d.get("results", []):
            score, why = llm_judge.judge_one(q["question"], q["answer"], r.get("answer", ""))
            pt, ct = r.get("prompt_tokens", 0), r.get("completion_tokens", 0)
            rows.append({
                "question_id": q["id"], "arm": r["arm"], "tier": q.get("tier", ""),
                "shape": q.get("hop", ""), "question": q["question"], "gold": q["answer"],
                "answer": r.get("answer", ""), "route": r.get("route", ""),
                "prompt_tokens": pt, "completion_tokens": ct, "planner_tokens": 0,
                "latency_s": r.get("latency", 0),
                "judge_pct": None if score < 0 else round(score, 1),
                "judge_score": None if score < 0 else round(score / 100.0, 4),
                "judge_why": why,
                # a numeric fact is right or wrong; the judge decides, and 100 means exact
                "f1": 0.0 if score < 0 else round(score / 100.0, 4),
                "precision": 0.0 if score < 0 else round(score / 100.0, 4),
                "recall": 0.0 if score < 0 else round(score / 100.0, 4),
                "exact": int(score >= 99.9)})
        if i % 5 == 0:
            print(f"    {i}/{len(qs)}  ({time.time()-t0:.0f}s elapsed)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wopen(out) as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics as stx
    print(f"\nwrote {out}  ({len(rows)} rows, {time.time()-t0:.0f}s)")
    print(f"\n{'pipeline':<16}{'full':>6}{'partial':>9}{'missed':>8}{'judge':>8}{'tok/q':>9}")
    print("-" * 58)
    for a, name in ARMS.items():
        R = [r for r in rows if r["arm"] == a and r["judge_pct"] is not None]
        if not R:
            continue
        full = sum(1 for r in R if r["judge_pct"] >= 99.9)
        part = sum(1 for r in R if 0 < r["judge_pct"] < 99.9)
        zero = sum(1 for r in R if r["judge_pct"] == 0)
        tk = stx.mean([r["prompt_tokens"] + r["completion_tokens"] for r in R])
        print(f"{name:<16}{full:>4}/{len(R):<2}{part:>9}{zero:>8}"
              f"{stx.mean([r['judge_pct'] for r in R]):>7.1f}%{tk:>9,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
