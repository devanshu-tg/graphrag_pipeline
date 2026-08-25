"""Aggregate the six benchmark arms into the three budget comparisons that were asked for."""
import json
import pathlib

R = pathlib.Path(__file__).resolve().parent / "results"


def load(arm, k):
    f = R / f"{arm}_k{k}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    # a resumed file can hold duplicates; last row per question wins
    dedup = {r["question_id"]: r for r in rows}
    return list(dedup.values())


def agg(rows):
    n = len(rows)
    if not n:
        return None
    return {
        "n": n,
        "exact": sum(r["exact"] for r in rows),
        "f1": sum(r["f1"] for r in rows) / n,
        "tok": sum(r["tokens"] for r in rows) / n,
        "prompt": sum(r["prompt_tokens"] for r in rows) / n,
        "plan": sum(r["planner_tokens"] for r in rows) / n,
        "chunks": sum(r["retrieved_chunks"] for r in rows) / n,
        "lat": sum(r["latency_s"] for r in rows) / n,
        "errors": sum(1 for r in rows if r["error"]),
    }


A = {}
for arm in ("basic", "graph-planner", "graph-regex"):
    for k in (3, 10):
        rows = load(arm, k)
        A[(arm, k)] = agg(rows) if rows else None

print("=" * 84)
print("PER-ARM RESULTS  (50 questions, gemini-3.6-flash, exact = strict set match)")
print("=" * 84)
print(f"{'arm':16s} {'budget':>7s} {'exact':>7s} {'F1':>6s} {'tokens':>8s} "
      f"{'prompt':>8s} {'planner':>8s} {'chunks':>7s} {'lat':>6s}")
for (arm, k), a in A.items():
    if not a:
        print(f"{arm:16s} {'k'+str(k):>7s}   -- not run --")
        continue
    print(f"{arm:16s} {'k'+str(k):>7s} {a['exact']:>4d}/{a['n']:<2d} {a['f1']:>6.3f} "
          f"{a['tok']:>8,.0f} {a['prompt']:>8,.0f} {a['plan']:>8,.0f} "
          f"{a['chunks']:>7.1f} {a['lat']:>5.1f}s"
          + (f"   [{a['errors']} errors]" if a["errors"] else ""))

print()
print("=" * 84)
print("TOKEN REDUCTION FROM THE REGEX ROUTER  (same budget, planner swapped out)")
print("=" * 84)
for k in (3, 10):
    p, g = A[("graph-planner", k)], A[("graph-regex", k)]
    if not (p and g):
        continue
    cut = 100 * (p["tok"] - g["tok"]) / p["tok"]
    print(f"  {k:>2} chunks: planner {p['tok']:>6,.0f} -> regex {g['tok']:>6,.0f} tokens "
          f"= {cut:>5.1f}% cheaper   |  accuracy {p['exact']}/{p['n']} -> {g['exact']}/{g['n']}"
          f"   |  latency {p['lat']:.1f}s -> {g['lat']:.1f}s")

print()
print("=" * 84)
print("THE THREE REQUESTED COMPARISONS  (GraphRAG using the regex router)")
print("=" * 84)
CONFIGS = [("equal budget - 10 vs 10", 10, 10),
           ("equal budget -  3 vs 3 ", 3, 3),
           ("as shipped  - 10 vs 3  ", 10, 3)]
for label, bk, gk in CONFIGS:
    b, g = A[("basic", bk)], A[("graph-regex", gk)]
    gp = A[("graph-planner", gk)]
    if not (b and g):
        print(f"  {label}: incomplete")
        continue
    cut = 100 * (b["tok"] - g["tok"]) / b["tok"]
    print(f"\n  {label}")
    print(f"    Basic RAG  {bk:>2} chunks : {b['exact']:>2}/{b['n']} exact  "
          f"{b['tok']:>6,.0f} tokens  {b['lat']:>5.1f}s")
    print(f"    GraphRAG   {gk:>2} chunks : {g['exact']:>2}/{g['n']} exact  "
          f"{g['tok']:>6,.0f} tokens  {g['lat']:>5.1f}s   <- regex router")
    if gp:
        old = 100 * (b["tok"] - gp["tok"]) / b["tok"]
        print(f"    (with the LLM planner it was {gp['tok']:,.0f} tokens = {old:.1f}% cheaper)")
    print(f"    => GraphRAG is {cut:.1f}% cheaper than Basic RAG at this budget")

print()
print("=" * 84)
print("BY QUESTION SHAPE  (regex router, 3-chunk budget)")
print("=" * 84)
rows = load("graph-regex", 3) or []
byshape = {}
for r in rows:
    byshape.setdefault(r["shape"], []).append(r)
for shape, rs in sorted(byshape.items(), key=lambda kv: -len(kv[1])):
    a = agg(rs)
    print(f"  {shape:16s} n={a['n']:>2}  exact {a['exact']:>2}/{a['n']:<2}  "
          f"{a['tok']:>6,.0f} tokens  routes: "
          f"{sorted({x['route'] for x in rs})}")
