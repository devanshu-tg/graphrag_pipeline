"""Aggregate the eval_final50_v3 arms into the three budget comparisons.

Two accuracy numbers, because the set matters. 30 of the 50 questions carry `gold_members`
(name lists) and get the strict set match; the other 20 are prose and have no member list, so
strict match cannot score them at all. The blind LLM judge (rubric v2) is therefore the number
that covers all 50, and exact set match is reported on its 30-question subset only.
"""
import json
import pathlib
import statistics
import sys

R = pathlib.Path(__file__).resolve().parent / "results"
TAG = "_v3_judged"


def load(arm, k):
    f = R / f"{arm}_k{k}{TAG}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    return list({r["question_id"]: r for r in rows}.values())   # resumed files can duplicate


def agg(rows):
    n = len(rows)
    js = [r["judge_pct"] for r in rows if r.get("judge_pct", -1) >= 0]
    sub = [r for r in rows if r["has_gold_members"]]
    return {
        "n": n,
        "judge": sum(js) / max(1, len(js)),
        "full": sum(1 for s in js if s >= 100),
        "judged": len(js),
        "exact": sum(1 for r in sub if r["exact"] == 1),
        "n_sub": len(sub),
        "f1": sum(r["f1"] for r in sub) / max(1, len(sub)),
        "tok": sum(r["tokens"] for r in rows) / n,
        "prompt": sum(r["prompt_tokens"] for r in rows) / n,
        "out": sum(r["completion_tokens"] for r in rows) / n,
        "plan": sum(r["planner_tokens"] for r in rows) / n,
        "chunks": sum(r["retrieved_chunks"] for r in rows) / n,
        "lat": statistics.median(r["latency_s"] for r in rows),
        "lat_mean": sum(r["latency_s"] for r in rows) / n,
        "lat_max": max(r["latency_s"] for r in rows),
        "errors": sum(1 for r in rows if r["error"]),
        "rows": rows,
    }


A = {}
for arm in ("basic", "graph-regex"):
    for k in (3, 10):
        rows = load(arm, k)
        A[(arm, k)] = agg(rows) if rows else None

miss = [f"{a}_k{k}" for (a, k), v in A.items() if v is None]
if miss:
    print(f"MISSING (not yet judged): {', '.join(miss)}")
    sys.exit(1)

W = 92
print("=" * W)
print("eval_final50_v3.jsonl  —  50 questions  —  answering model gemini-3.7-flash")
print("judge: gemini-3.1-pro-preview, rubric v2, blind   |   rr_pool 500 both arms")
print("=" * W)
print(f"{'arm':22s} {'budget':>7s} {'judge%':>7s} {'full':>7s} {'exact':>8s} {'F1':>6s} "
      f"{'tokens':>8s} {'plan':>6s} {'lat':>6s}")
NAME = {"basic": "Basic RAG (rerank)", "graph-regex": "GraphRAG (regex rtr)"}
for (arm, k), a in A.items():
    print(f"{NAME[arm]:22s} {'k'+str(k):>7s} {a['judge']:>6.1f}% {a['full']:>4d}/{a['n']:<2d} "
          f"{a['exact']:>5d}/{a['n_sub']:<2d} {a['f1']:>6.3f} {a['tok']:>8,.0f} "
          f"{a['plan']:>6,.0f} {a['lat']:>5.1f}s"
          + (f"  [{a['errors']} err]" if a["errors"] else ""))

CONFIGS = [("TEST 1 — equal budget, 10 vs 10", 10, 10),
           ("TEST 2 — equal budget,  3 vs  3", 3, 3),
           ("TEST 3 — as shipped, 10 basic vs 3 graph", 10, 3)]
print()
print("=" * W)
print("THE THREE TESTS")
print("=" * W)
for label, bk, gk in CONFIGS:
    b, g = A[("basic", bk)], A[("graph-regex", gk)]
    cut = 100 * (b["tok"] - g["tok"]) / b["tok"]
    pcut = 100 * (b["prompt"] - g["prompt"]) / b["prompt"]
    print(f"\n{label}")
    print(f"  {'':22s} {'judge%':>8s} {'full/50':>8s} {'exact/30':>9s} {'tokens':>9s} {'prompt':>8s} {'lat':>7s}")
    for nm, a in ((f"Basic RAG  k={bk}", b), (f"GraphRAG   k={gk}", g)):
        print(f"  {nm:22s} {a['judge']:>7.1f}% {a['full']:>5d}/50 {a['exact']:>6d}/30 "
              f"{a['tok']:>9,.0f} {a['prompt']:>8,.0f} {a['lat']:>6.1f}s")
    d = g["judge"] - b["judge"]
    print(f"  -> accuracy {d:+.1f} pts judge ({g['full']-b['full']:+d} full-credit, "
          f"{g['exact']-b['exact']:+d} exact)")
    print(f"  -> tokens   {cut:+.1f}% total  ({pcut:+.1f}% prompt)  "
          f"= {'GraphRAG cheaper' if cut > 0 else 'GraphRAG dearer'} by "
          f"{abs(b['tok']-g['tok']):,.0f} tok/question")

print()
print("=" * W)
print("BY TIER  (judge %, all 50 questions)")
print("=" * W)
tiers = {}
for r in A[("basic", 10)]["rows"]:
    tiers.setdefault(r["tier"], 0)
    tiers[r["tier"]] += 1
hdr = f"{'tier':12s} {'n':>3s}"
for arm, k in (("basic", 3), ("basic", 10), ("graph-regex", 3), ("graph-regex", 10)):
    hdr += f" {arm.split('-')[0][:5]+'/k'+str(k):>11s}"
print(hdr)
for t, n in sorted(tiers.items(), key=lambda kv: -kv[1]):
    line = f"{t:12s} {n:>3d}"
    for arm, k in (("basic", 3), ("basic", 10), ("graph-regex", 3), ("graph-regex", 10)):
        rs = [r for r in A[(arm, k)]["rows"] if r["tier"] == t and r.get("judge_pct", -1) >= 0]
        line += f" {sum(r['judge_pct'] for r in rs)/max(1,len(rs)):>10.1f}%"
    print(line)
