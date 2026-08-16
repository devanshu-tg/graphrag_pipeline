"""Run P1 / P2 / P3 over the generated multi-hop questions and score them.

Three arms, identical question set, identical context budget:
  P1  LLM-only     no retrieval    -> the LEAK GATE. Must score ~0 or the benchmark is
                                      measuring memory, not retrieval.
  P2  Basic RAG    vector top-k    -> the thing GraphRAG must beat
  P3  GraphRAG     schema-driven router + graph-first + provenance compression

Scoring is DETERMINISTIC — every gold answer is a set of names (channels or people) computed
from the graph, so set precision/recall/F1 needs no judge. On SEC the judge flipped a
question between two passes over identical answers; here that variance is designed out.
BERTScore is reported alongside as an independent, judge-free control.

    python build/run_eval.py --arms 1 2 3 --n 20 --tag flat
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline3_round3"))
from _portability import init_console, load_env, ropen, wopen   # noqa: E402

init_console()
load_env()

import config                                    # noqa: E402
from llm_client import get_llm_response          # noqa: E402

QUESTIONS = Path("datasets/enterprise/eval_questions.jsonl")
OUTDIR = Path("results/enterprise")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def set_scores(answer: str, gold: list[str]) -> tuple[float, float, float]:
    """Set P/R/F1 by membership. Gold items are channel names or person names; an answer
    counts as containing one when its normalised form appears in the normalised answer."""
    a = norm(answer)
    hit = [g for g in gold if norm(g) and norm(g) in a]
    # crude precision proxy: how many items the answer asserts. Split on whitespace too —
    # an answer written "#a #b #c #d" has no commas, so it used to count as ONE asserted item
    # and precision came out at 4.0. That was clamped on return but NOT before computing F1,
    # which could report F1 = 1.60. It never fired (all 370 measured rows answered with
    # commas, 0 inflated) but the clamp belongs before the harmonic mean, not after it.
    asserted = {t for t in re.split(r"[,\n;]|\s+and\s+|\s{2,}", answer or "") if norm(t)}
    tp = len(hit)
    p = min(tp / max(1, len(asserted)), 1.0) if asserted else 0.0
    r = tp / max(1, len(gold))
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def run_arm(arm: str, questions: list[dict], vs=None) -> list[dict]:
    rows = []
    if arm == "1":
        import pipeline1_llm_only as p1
        for q in questions:
            m = p1.run_single(q["question"])
            rows.append(_row(q, m, "p1"))
    elif arm == "2":
        import pipeline2_basic_rag as p2
        for q in questions:
            m = p2.run_single(q["question"], vs)
            rows.append(_row(q, m, "p2"))
    elif arm == "2xl":
        # Basic RAG with the 10-chunk fairness cap lifted — measures whether the answer being
        # deep in the ranking (rather than absent) is enough for the model to use it.
        import pipeline2xl_bigcontext as pxl
        for q in questions:
            m = pxl.run_single(q["question"])
            r = _row(q, m, "p2xl")
            r["route"] = getattr(m, "route", "")
            rows.append(r)
    elif arm == "2rr":
        # Basic RAG + cross-encoder reranking: 50 candidates scored, best 10 kept
        import pipeline2rr_rerank as prr
        for q in questions:
            m = prr.run_single(q["question"])
            r = _row(q, m, "p2rr")
            r["route"] = getattr(m, "route", "")
            rows.append(r)
    elif arm == "2p":
        # the steelman baseline — iterative retrieval, deliberately given more budget than P3
        import pipeline2plus_iterative as p2p
        for q in questions:
            m = p2p.run_single(q["question"])
            r = _row(q, m, "p2p")
            r["route"] = getattr(m, "route", "")
            rows.append(r)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("p3", "pipeline3_round3/pipeline.py")
        p3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p3)
        p3._load(); p3._connect()
        import schema_router
        schema_router.reset_token_usage()
        for q in questions:
            m = p3.run_single(q["question"])
            r = _row(q, m, "p3")
            r["route"] = getattr(m, "route", "")
            rows.append(r)
        pt = schema_router.plan_token_usage()
        for r in rows:
            r["planner_tokens"] = round((pt["prompt"] + pt["output"]) / max(1, len(rows)))
        # With P3_TG_TRAVERSE=1 every hop is an installed GSQL query on the typed graph. Print
        # the tally, because "the graph did the traversal" is a claim that needs a count behind
        # it — including how often it FELL BACK to the local table, and whether the two agreed.
        if getattr(p3, "TG_TRAVERSE", False):
            hs = {k: v for k, v in p3._hop_stats.items() if k != "diff_examples"}
            print(f"  hops via {p3.TG_V2_GRAPH}: {hs}")
            for d in p3._hop_stats.get("diff_examples", [])[:5]:
                print(f"    DISAGREEMENT {d}")
    return rows


def _row(q: dict, m, arm: str) -> dict:
    ans = m.answer or ""
    p, r, f1 = set_scores(ans, q["gold_members"])
    return {"question_id": q["question_id"], "arm": arm, "shape": q["shape"],
            "tier": q.get("tier", "multi_hop"),
            "question": q["question"], "gold": q["answer"], "answer": ans,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "exact": int(r == 1.0 and p >= 0.99),
            "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens,
            "latency_s": getattr(m, "latency_seconds", 0), "planner_tokens": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--questions", default="",
                    help="override the question file (used by the single-hop control)")
    args = ap.parse_args()

    qs = [json.loads(l) for l in ropen(Path(args.questions) if args.questions else QUESTIONS)
          if l.strip()][: args.n]
    print(f"{len(qs)} questions, arms={args.arms}, tag={args.tag}\n")

    # P2 now retrieves through TigerGraph (same backend as P3), so no Chroma index is
    # needed and the two arms differ only in whether the graph is consulted.
    vs = None

    all_rows = []
    for arm in args.arms:
        t0 = time.time()
        rows = run_arm(arm, qs, vs)
        all_rows += rows
        print(f"  arm p{arm}: {len(rows)} answers in {time.time()-t0:.0f}s")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"eval_{args.tag}.jsonl"
    with wopen(out) as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report ----
    print(f"\n{'='*94}\nRESULTS  ({args.tag})  n={len(qs)} multi-hop questions\n{'='*94}")
    print(f"{'arm':<22}{'exact':>7}{'recall':>9}{'prec':>8}{'setF1':>8}"
          f"{'prompt':>9}{'compl':>8}{'plan':>7}{'tok/q':>9}")
    print("-" * 94)
    names = {"p1": "P1 LLM-only", "p2": "P2 Basic RAG", "p2xl": "P2-XL Big context", "p2rr": "P2-RR Reranked",
             "p2p": "P2+ Iterative RAG", "p3": "P3 GraphRAG"}
    summary = {}
    for arm in ("p1", "p2", "p2xl", "p2rr", "p2p", "p3"):
        rows = [r for r in all_rows if r["arm"] == arm]
        if not rows:
            continue
        pl = st.mean([r["planner_tokens"] for r in rows])
        tot = st.mean([r["prompt_tokens"] + r["completion_tokens"] for r in rows]) + pl
        summary[arm] = {
            "exact": sum(r["exact"] for r in rows), "n": len(rows),
            "recall": st.mean([r["recall"] for r in rows]),
            "precision": st.mean([r["precision"] for r in rows]),
            "f1": st.mean([r["f1"] for r in rows]),
            "prompt": st.mean([r["prompt_tokens"] for r in rows]),
            "completion": st.mean([r["completion_tokens"] for r in rows]),
            "planner": pl, "total": tot,
        }
        s = summary[arm]
        print(f"{names[arm]:<22}{s['exact']:>4}/{s['n']:<2}{s['recall']:>9.3f}"
              f"{s['precision']:>8.3f}{s['f1']:>8.3f}{s['prompt']:>9,.0f}"
              f"{s['completion']:>8,.0f}{s['planner']:>7,.0f}{s['total']:>9,.0f}")

    if "p1" in summary:
        leak = summary["p1"]["recall"]
        print(f"\nLEAK GATE — P1 (no retrieval) recall = {leak:.3f}   "
              f"{'PASS (<0.15)' if leak < 0.15 else 'FAIL: questions leak, regenerate'}")
    if "p2" in summary and "p3" in summary:
        d = summary["p3"]["f1"] - summary["p2"]["f1"]
        t = summary["p3"]["total"] - summary["p2"]["total"]
        print(f"P3 vs P2: set-F1 {d:+.3f}   tokens {t:+,.0f}/question "
              f"({100*t/max(1,summary['p2']['total']):+.0f}%)")

    if args.bertscore:
        from accuracy import AccuracyEvaluator
        ev = AccuracyEvaluator()
        print("\nBERTScore F1:")
        for arm in ("p1", "p2", "p2xl", "p2rr", "p2p", "p3"):
            rows = [r for r in all_rows if r["arm"] == arm]
            if not rows:
                continue
            scored = [(i, r) for i, r in enumerate(rows) if (r["answer"] or "").strip()]
            f1 = [0.0] * len(rows)
            if scored:
                bs = ev.bertscore([r["answer"] for _, r in scored],
                                  [r["gold"] for _, r in scored])
                for (i, _), v in zip(scored, bs):
                    f1[i] = v
            print(f"  {names[arm]:<22}{st.mean(f1):>8.4f}")

    # per-tier breakdown — the whole reason the mixed set exists. A single average hides
    # that Basic RAG WINS the single-hop tier, which is the most credible thing we can show.
    tiers = [t for t in ("single_doc", "single_dir", "multi_hop")
             if any(r.get("tier") == t for r in all_rows)]
    if len(tiers) > 1:
        print("\n" + "=" * 94)
        print("BY TIER — setF1 (and tokens/question)")
        print("=" * 94)
        print(f"{'tier':<16}{'n':>4}" + "".join(f"{names[a]:>22}" for a in
              ("p1", "p2", "p2p", "p3") if a in summary))
        print("-" * 94)
        for t in tiers:
            n_t = len({r["question_id"] for r in all_rows if r.get("tier") == t})
            line = f"{t:<16}{n_t:>4}"
            for a in ("p1", "p2", "p2p", "p3"):
                if a not in summary:
                    continue
                R = [r for r in all_rows if r.get("tier") == t and r["arm"] == a]
                if not R:
                    line += f"{'-':>22}"
                    continue
                tk = st.mean([r["prompt_tokens"] + r["completion_tokens"]
                              + r.get("planner_tokens", 0) for r in R])
                line += f"{st.mean([r['f1'] for r in R]):>13.3f} {tk:>7,.0f}"
            print(line)

    json.dump(summary, wopen(OUTDIR / f"summary_{args.tag}.json"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
