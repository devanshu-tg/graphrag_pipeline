"""Pipeline-2 (Basic RAG) ablation sweep — find the best P2 config on the eval set.

One-factor-at-a-time from a baseline, exploring the retrieval knobs the user asked about:
  - retrieval mode: mmr_rerank | mmr | similarity (cosine)   [reranker on/off, MMR vs cosine]
  - chunk size (CONTEXT_CHUNK_CHARS): 1000 / 2000 / 3000
  - top-k (CONTEXT_NUM_CHUNKS): 5 / 10 / 15
  - MMR diversity (P2_MMR_LAMBDA): 0.3 / 0.5 / 0.7
  - ticker metadata filter: on / off

Each arm runs P2 on the 50 held-out questions, evaluates (strict/graded/BERTScore, skip cite
judge for speed), and records tokens/latency/cost. Reports a ranked table + the best config.
Single-run per arm (exploration) — deltas within ~±0.02 are noise; the top arm is confirmed
with the frozen 3-run protocol afterward.

Usage: python "build/p2_ablation.py"
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
BASE_ENV = {
    "DATASET": "round3", "LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-2.5-flash",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "MAX_TOKENS": "512",
    "CONTEXT_NUM_CHUNKS": "10", "CONTEXT_CHUNK_CHARS": "2000",
    "P2_SEARCH": "mmr_rerank", "P2_RERANK_POOL": "40", "P2_MMR_LAMBDA": "0.5",
    "P2_TICKER_FILTER": "1",
}

# each arm = (name, {env overrides})
ARMS = [
    ("baseline (mmr+rerank, k10, 2000ch, filter)", {}),
    # retrieval mode
    ("mmr_no_rerank",                 {"P2_SEARCH": "mmr"}),
    ("similarity_cosine",            {"P2_SEARCH": "similarity"}),
    # chunk size
    ("chunk_1000",                   {"CONTEXT_CHUNK_CHARS": "1000"}),
    ("chunk_3000",                   {"CONTEXT_CHUNK_CHARS": "3000"}),
    # top-k
    ("topk_5",                       {"CONTEXT_NUM_CHUNKS": "5"}),
    ("topk_15",                      {"CONTEXT_NUM_CHUNKS": "15"}),
    # MMR diversity
    ("mmr_lambda_0.3",               {"P2_MMR_LAMBDA": "0.3"}),
    ("mmr_lambda_0.7",               {"P2_MMR_LAMBDA": "0.7"}),
    # ticker filter off
    ("no_ticker_filter",             {"P2_TICKER_FILTER": "0"}),
]

OUT = Path("results/round3/p2_ablation")
OUT.mkdir(parents=True, exist_ok=True)
ESET = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open("datasets/round3/eval_set.jsonl")}


def run_arm(name: str, overrides: dict) -> dict:
    env = {**os.environ, **BASE_ENV, **overrides}
    tag = name.split()[0].replace("(", "").replace(",", "")
    bench = OUT / f"bench_{tag}.json"
    ev = OUT / f"eval_{tag}.json"
    print(f"\n=== ARM: {name}  ({overrides}) ===", flush=True)
    subprocess.run([PY, "build/run_round3_benchmark.py", "--split", "test", "--runs", "1",
                    "--pipelines", "2", "--tag", f"p2abl_{tag}", "--out-dir", str(OUT)],
                   env=env, check=True)
    src = OUT / f"bench_test_p2abl_{tag}_run1.json"
    d = json.load(open(src))
    for r in d["pipelines"]["basic_rag"]:
        if r.get("id") in ESET:
            r["correct_answer"] = ESET[r["id"]]
    json.dump(d, open(bench, "w"), indent=2)
    subprocess.run([PY, "build/evaluate_round3.py", str(bench), "--out", str(ev),
                    "--skip-cite-judge"], env=env, check=True)
    a = json.load(open(ev))["pipelines"]["basic_rag"]["aggregate"]
    b = d["aggregates"]["basic_rag"]
    return {"arm": name, "strict": a["strict_pass_rate"], "graded": a["graded_quality_mean"],
            "bert": a["bertscore_f1_mean"], "tokens": a["avg_total_inference_tokens"],
            "latency": b.get("avg_latency_seconds"), "cost": b.get("avg_cost_usd")}


def main():
    results = []
    for name, ov in ARMS:
        try:
            results.append(run_arm(name, ov))
        except Exception as e:
            print(f"  arm {name} failed: {str(e)[:120]}")
    results.sort(key=lambda r: (-r["strict"], -r["graded"]))
    print("\n" + "=" * 96)
    print(f"{'arm':44} {'strict':>7} {'graded':>7} {'bert':>7} {'tokens':>7} {'lat(s)':>7} {'cost$':>9}")
    print("-" * 96)
    for r in results:
        print(f"{r['arm']:44} {r['strict']:>7.3f} {r['graded']:>7.3f} {r['bert']:>7.3f} "
              f"{r['tokens']:>7.0f} {str(r['latency']):>7} {r['cost']:>9.6f}")
    json.dump(results, open(OUT / "summary.json", "w"), indent=2)
    if results:
        best = results[0]
        print(f"\nBEST by strict pass: {best['arm']}  (strict={best['strict']:.3f}, tokens={best['tokens']:.0f})")
    print(f"[p2-ablation] wrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
