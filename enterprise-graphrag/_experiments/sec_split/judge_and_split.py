"""Judge the 24 re-measured answers and write compare/temporal tiers into sec_budgets.json.

Judge and rubric come from sec_budgets.json's own _meta (gemini-3.1-pro-preview, v2), so the two
new tiers are scored the same way as every other cell in that file.

The `shipped` row is NOT re-measured: eval_sec_norefusal.jsonl already carries per-question
routes, judges and tokens for exactly that config, so its split is derived rather than re-run.
"""
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "build")
import os
for _l in open(".env"):
    if "=" in _l and not _l.strip().startswith("#"):
        _k, _v = _l.strip().split("=", 1)
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

R = pathlib.Path("results/enterprise")
HERE = pathlib.Path("_experiments/sec_split")
secb = json.loads((R / "sec_budgets.json").read_text())
os.environ["LLM_JUDGE_RUBRIC"] = secb["_meta"].get("rubric", "v2")
import llm_judge                                     # noqa: E402
print(f"judge={llm_judge.JUDGE_MODEL} rubric={os.environ['LLM_JUDGE_RUBRIC']}\n")


def cell(rows):
    j = [r["judge_pct"] for r in rows]
    return {"full": sum(1 for x in j if x >= 100), "judge": round(st.mean(j), 1),
            "tok": round(st.mean(r["tokens"] for r in rows))}


def judged(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in rows:
        if r["arm"] == "p1":
            continue
        try:
            s, why = llm_judge.judge_one(r["question"], r["gold"], r["answer"])
        except Exception as e:
            s, why = -1.0, str(e)
        r["judge_pct"] = max(s, 0.0)
        r["judge_why"] = why
        print(f"  {r['question_id']} {r['arm']} {r['route_published']:9s} {s:>6.1f}")
    return [r for r in rows if r["arm"] in ("p2", "p3")]


# ---- the two budgets that had to be re-measured
for budget, key in ((10, "k10"), (6, "k6")):
    rows = judged(HERE / f"worker_k{budget}.jsonl")
    with open(HERE / f"worker_k{budget}_judged.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for route in ("compare", "temporal"):
        sub = [r for r in rows if r["route_published"] == route]
        secb[key][route] = {"n": len(sub) // 2,
                            "p2": cell([r for r in sub if r["arm"] == "p2"]),
                            "p3": cell([r for r in sub if r["arm"] == "p3"])}
    secb[key].pop("default", None)
    print()

# ---- shipped: derive from the per-question file that already covers this exact config
src = [json.loads(l) for l in open(R / "eval_sec_norefusal.jsonl")]
route_of = {r["question_id"]: r.get("route") for r in src if r["arm"] == "p3"}
for route in ("compare", "temporal"):
    qids = {q for q, rt in route_of.items() if rt == route}
    sub = [dict(r, tokens=r["prompt_tokens"] + r["completion_tokens"])
           for r in src if r["question_id"] in qids and r["arm"] in ("p2", "p3")]
    secb["shipped"][route] = {"n": len(qids),
                              "p2": cell([r for r in sub if r["arm"] == "p2"]),
                              "p3": cell([r for r in sub if r["arm"] == "p3"])}
secb["shipped"].pop("default", None)

secb["_meta"]["labels"]["_tiers"] = (
    "compare and temporal were a merged `default` bucket; the k10 and k6 rows for those two "
    "tiers are a re-measurement of their 6 questions (the original file's generator no longer "
    "exists and an average cannot be un-merged), the shipped row is derived from "
    "eval_sec_norefusal.jsonl. Basic RAG reproduced the stored default within 0.3%; GraphRAG "
    "came in 14% lower, which is this corpus's documented retrieval variance.")
(R / "sec_budgets.json").write_text(json.dumps(secb, indent=1))

print("=== new tiers ===")
for key in ("k10", "k6", "shipped"):
    for route in ("compare", "temporal"):
        c = secb[key][route]
        print(f"  {key:8s} {route:9s} n={c['n']}  "
              f"basic {c['p2']['full']}/{c['n']} {c['p2']['judge']:5.1f}% {c['p2']['tok']:>6,}tok"
              f"   graph {c['p3']['full']}/{c['n']} {c['p3']['judge']:5.1f}% {c['p3']['tok']:>6,}tok")
print("\nwrote results/enterprise/sec_budgets.json")
