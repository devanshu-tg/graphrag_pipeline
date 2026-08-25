"""The three pairings on eval_mixed50, all four arms from the SAME question set.

  equal budget  3 v 3    basic_k3       vs graph-regex_k3
  equal budget 10 v 10   basic_k10      vs graph-regex_k10
  head to head 10 v 3    basic_k10      vs graph-regex_k3   (the page's claim)

Exact match is the headline: it is deterministic. The blind LLM judge (rubric v2) sits beside
it because a judge changes its mind between passes and must never be the only number.
"""
import json, pathlib, statistics, sys

R = pathlib.Path(__file__).resolve().parent / "results"
TIERS = ["single_doc", "single_dir", "multi_hop"]


def load(stem):
    """Prefer the judged file; fall back to raw so the table still builds unjudged."""
    for name in (f"{stem}_judged.jsonl", f"{stem}.jsonl"):
        p = R / name
        if p.exists():
            return [json.loads(l) for l in open(p) if l.strip()], name
    sys.exit(f"missing: {stem}")


def stats(rows):
    j = [r["judge_pct"] for r in rows if r.get("judge_pct") is not None and r["judge_pct"] >= 0]
    return {
        "n": len(rows),
        "exact": sum(1 for r in rows if r.get("exact")),
        "f1": statistics.mean(r["f1"] for r in rows),
        "tok": statistics.mean(r["tokens"] for r in rows),
        "lat": statistics.mean(r["latency_s"] for r in rows),
        "judge": statistics.mean(j) if j else None,
        "full": sum(1 for s in j if s >= 100) if j else None,
        "part": sum(1 for s in j if 0 < s < 100) if j else None,
        "zero": sum(1 for s in j if s == 0) if j else None,
        "by_tier": {t: (sum(1 for r in rows if r["tier"] == t and r.get("exact")),
                        sum(1 for r in rows if r["tier"] == t)) for t in TIERS},
    }


def line(label, s):
    j = f"{s['judge']:>6.1f}%" if s["judge"] is not None else "     --"
    fz = f"{s['full']:>2d}/{s['part']:>2d}/{s['zero']:<2d}" if s["full"] is not None else "  --   "
    tiers = "  ".join(f"{s['by_tier'][t][0]:>2d}/{s['by_tier'][t][1]:<2d}" for t in TIERS)
    print(f"  {label:<26s} {s['exact']:>2d}/{s['n']:<3d} {s['f1']:>6.3f} {j} {fz} "
          f"{s['tok']:>7,.0f} {s['lat']:>6.1f}s   {tiers}")


ARMS = {k: load(k) for k in ("basic_k3", "basic_k10", "graph-regex_k3", "graph-regex_k10")}
S = {k: stats(v[0]) for k, v in ARMS.items()}

print("\neval_mixed50 (MX01-MX50) — GraphRAG uses the PATTERN router, not the LLM planner")
print("files:", ", ".join(sorted(v[1] for v in ARMS.values())))
hdr = (f"\n  {'arm':<26s} {'exact':>6s} {'meanF1':>6s} {'judge':>7s} {'f/p/z':>7s} "
       f"{'tok/q':>7s} {'lat':>7s}   {'doc':>5s} {'dir':>6s} {'multi':>5s}")

PAIRS = [("Equal budget — 3 vs 3", "basic_k3", "graph-regex_k3"),
         ("Equal budget — 10 vs 10", "basic_k10", "graph-regex_k10"),
         ("Head to head — 10 vs 3", "basic_k10", "graph-regex_k3")]

for title, a, b in PAIRS:
    print(f"\n{'='*118}\n{title}\n{'='*118}{hdr}")
    line(f"Basic RAG  {a.split('_k')[1]} chunks", S[a])
    line(f"GraphRAG   {b.split('_k')[1]} chunks", S[b])
    da, db = S[a], S[b]
    tok = (db["tok"] / da["tok"] - 1) * 100
    print(f"  {'-> delta':<26s} {db['exact']-da['exact']:>+3d}    "
          f"{db['f1']-da['f1']:>+6.3f} "
          + (f"{db['judge']-da['judge']:>+6.1f}p" if None not in (da['judge'], db['judge']) else "      --")
          + f"         {tok:>+6.1f}% tokens")

print(f"\n{'='*118}")
print("exact = deterministic set match (all gold items present, nothing extra)")
print("f/p/z = judge full-credit / partial / zero;  judge = gemini-3.1-pro-preview, rubric v2, blind")
print("tier columns are exact-match:  doc = single_doc (20), dir = single_dir (10), multi = multi_hop (20)")
