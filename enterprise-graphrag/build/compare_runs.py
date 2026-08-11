"""Side-by-side report: two graph builds x three pipelines, one answer key.

Both runs are scored against the SAME gold (verified identical under v1 and v2 by
build/regold.py), so every difference below comes from the graph, not from the benchmark.

    python build/compare_runs.py --a flat --b typed --labels "v1 flat" "v2 typed"
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen   # noqa: E402

init_console()

OUT = Path("results/enterprise")
NAMES = {"p1": "P1 LLM-only", "p2": "P2 Basic RAG", "p3": "P3 GraphRAG"}


def load(tag: str) -> dict:
    rows = [json.loads(l) for l in ropen(OUT / f"eval_{tag}.jsonl") if l.strip()]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    return by


def agg(rows: list[dict]) -> dict:
    pl = st.mean([r.get("planner_tokens", 0) for r in rows])
    return {
        "n": len(rows),
        "exact": sum(r["exact"] for r in rows),
        "recall": st.mean([r["recall"] for r in rows]),
        "prec": st.mean([r["precision"] for r in rows]),
        "f1": st.mean([r["f1"] for r in rows]),
        "prompt": st.mean([r["prompt_tokens"] for r in rows]),
        "compl": st.mean([r["completion_tokens"] for r in rows]),
        "plan": pl,
        "tok": st.mean([r["prompt_tokens"] + r["completion_tokens"] for r in rows]) + pl,
        "bert": st.mean([r.get("bertscore", 0) or 0 for r in rows]),
        "graph_route": sum(1 for r in rows if str(r.get("route", "")).startswith("graph")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="flat")
    ap.add_argument("--b", default="typed")
    ap.add_argument("--labels", nargs=2, default=["v1 flat", "v2 typed"])
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    la, lb = args.labels

    for title, run, lab in (("RUN 1", A, la), ("RUN 2", B, lb)):
        print(f"\n{'='*100}\n{title} — {lab}\n{'='*100}")
        print(f"{'pipeline':<16}{'exact':>8}{'recall':>9}{'prec':>8}{'setF1':>8}"
              f"{'BERTScore':>11}{'prompt':>9}{'compl':>8}{'plan':>7}{'tok/q':>9}{'graph':>7}")
        print("-" * 100)
        for arm in ("p1", "p2", "p3"):
            if arm not in run:
                continue
            s = agg(run[arm])
            g = f"{s['graph_route']}/{s['n']}" if arm == "p3" else "-"
            print(f"{NAMES[arm]:<16}{s['exact']:>5}/{s['n']:<2}{s['recall']:>9.3f}{s['prec']:>8.3f}"
                  f"{s['f1']:>8.3f}{s['bert']:>11.4f}{s['prompt']:>9,.0f}{s['compl']:>8,.0f}"
                  f"{s['plan']:>7,.0f}{s['tok']:>9,.0f}{g:>7}")

    print(f"\n{'='*100}\nDELTA — {lb} minus {la}\n{'='*100}")
    print(f"{'pipeline':<16}{'setF1':>12}{'recall':>12}{'BERTScore':>12}{'tok/q':>12}")
    print("-" * 100)
    for arm in ("p1", "p2", "p3"):
        if arm not in A or arm not in B:
            continue
        a, b = agg(A[arm]), agg(B[arm])
        print(f"{NAMES[arm]:<16}{b['f1']-a['f1']:>+12.3f}{b['recall']-a['recall']:>+12.3f}"
              f"{b['bert']-a['bert']:>+12.4f}{b['tok']-a['tok']:>+12,.0f}")

    # the headline the whole exercise exists to produce
    for run, lab in ((A, la), (B, lb)):
        if "p2" in run and "p3" in run:
            p2, p3 = agg(run["p2"]), agg(run["p3"])
            d = p3["f1"] - p2["f1"]
            t = p3["tok"] - p2["tok"]
            print(f"\n{lab:<12} GraphRAG vs Basic RAG:  set-F1 {d:+.3f}   "
                  f"tokens {t:+,.0f}/q ({100*t/max(1,p2['tok']):+.0f}%)   "
                  f"{'GRAPH WINS' if d > 0.06 else 'inside the 0.060 noise floor' if abs(d) <= 0.06 else 'BASIC RAG WINS'}")

    # per-question, so a mean is never the only thing on offer
    print(f"\n{'='*100}\nPER QUESTION — P3 set-F1\n{'='*100}")
    print(f"{'id':<7}{'shape':<18}{la:>12}{lb:>12}   {'question':<48}")
    print("-" * 100)
    a3 = {r["question_id"]: r for r in A.get("p3", [])}
    b3 = {r["question_id"]: r for r in B.get("p3", [])}
    for qid in sorted(set(a3) | set(b3)):
        ra, rb = a3.get(qid), b3.get(qid)
        fa = f"{ra['f1']:.2f}" if ra else "-"
        fb = f"{rb['f1']:.2f}" if rb else "-"
        row = ra or rb
        print(f"{qid:<7}{row['shape']:<18}{fa:>12}{fb:>12}   {row['question'][:48]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
