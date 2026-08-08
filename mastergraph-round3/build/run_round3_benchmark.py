"""Round 3 three-pipeline benchmark harness.

Runs P1 (LLM-only), P2 (Basic RAG), P3 (TigerGraph GraphRAG) over a question set,
records EVERY PS-required per-question field, and supports repeated runs (reproducibility).

Records per (pipeline, question): answer, retrieved chunk IDs, citations, the full token
breakdown (system/question/context/optimizer/output/total-inference, actual vs estimated),
latency, route, and P3 graph provenance (tg_live, graph_doc_ids).

Usage:
  DATASET=round3 LLM_PROVIDER=gemini GEMINI_MODEL=gemini-2.5-flash EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    python "build/run_round3_benchmark.py" --split dev --runs 1 --pipelines 1,2,3
  # held-out final: --split test --runs 3
Ablations (P3 only): pass --p3-env "P3_GRAPH_OFF=1" etc. to tag a variant run.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")


def load_questions(split: str) -> list[dict]:
    path = Path("datasets/round3") / ("eval_set.jsonl" if split == "test" else "dev_set.jsonl")
    return [json.loads(l) for l in open(path)]


def run_pipeline(which: str, questions: list[dict], vectorstore):
    import config
    rows = []
    if which == "1":
        import pipeline1_llm_only as p1
        for qd in questions:
            m = p1.run_single(qd["question"])
            rows.append(_row(qd, m))
    elif which == "2":
        import pipeline2_basic_rag as p2
        for qd in questions:
            m = p2.run_single(qd["question"], vectorstore)
            rows.append(_row(qd, m))
    elif which == "3":
        import importlib.util
        spec = importlib.util.spec_from_file_location("p3r3", "pipeline3_round3/pipeline.py")
        p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)
        p3._load(); p3._connect()
        for qd in questions:
            m = p3.run_single(qd["question"])
            rows.append(_row(qd, m))
    return rows


def _row(qd: dict, m) -> dict:
    d = m.to_dict()
    d["id"] = qd.get("id", "")
    d["tier"] = qd.get("tier", "")
    d["hop"] = qd.get("hop", "")
    d["correct_answer"] = qd.get("answer", "")
    d["source_docs"] = qd.get("source_docs", "")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--pipelines", default="1,2,3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/round3")
    args = ap.parse_args()

    import config
    from metrics import aggregate_metrics, CallMetrics
    questions = load_questions(args.split)
    pipes = args.pipelines.split(",")
    print(f"[bench] split={args.split} runs={args.runs} pipelines={pipes} n_questions={len(questions)}")

    vectorstore = None
    if "2" in pipes:
        from chroma_store import load_vectorstore
        vectorstore = load_vectorstore()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name_map = {"1": "llm_only", "2": "basic_rag", "3": "graphrag"}

    for run_i in range(1, args.runs + 1):
        result = {"meta": {"split": args.split, "run": run_i, "runs": args.runs, "tag": args.tag,
                           "generator": f"{config.LLM_PROVIDER}:{config.GEMINI_MODEL if config.LLM_PROVIDER=='gemini' else config.OPENAI_MODEL}",
                           "embedder": config.EMBEDDING_MODEL, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  "pipelines": {}, "aggregates": {}}
        for which in pipes:
            print(f"\n[bench] run {run_i}/{args.runs} — pipeline {which} ({name_map[which]})")
            rows = run_pipeline(which, questions, vectorstore)
            result["pipelines"][name_map[which]] = rows
            # rebuild CallMetrics-lite objects for aggregation
            metrics = [CallMetrics(**{k: v for k, v in r.items() if k in CallMetrics.__annotations__}) for r in rows]
            result["aggregates"][name_map[which]] = aggregate_metrics(metrics)

        suffix = f"_{args.tag}" if args.tag else ""
        out = Path(args.out_dir) / f"bench_{args.split}{suffix}_run{run_i}.json"
        json.dump(result, open(out, "w"), indent=2)
        print(f"[bench] wrote {out}")
        # quick token comparison
        aggs = result["aggregates"]
        if "basic_rag" in aggs and "graphrag" in aggs:
            r2 = aggs["basic_rag"]["avg_total_inference_tokens"]
            r3 = aggs["graphrag"]["avg_total_inference_tokens"]
            print(f"[bench] avg inference tokens — RAG={r2} GraphRAG={r3} "
                  f"reduction={(1-r3/r2)*100:.1f}%" if r2 else "")


if __name__ == "__main__":
    main()
