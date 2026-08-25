"""Which questions retrieve a different chunk set on a re-run? Zero LLM calls.

The instability recorded in demo/server_factual.py is in Savanna's ANN vector search, not in
generation — so it can be measured by asking for the chunks twice and diffing the ids. Costs
nothing and takes minutes instead of 200 LLM calls.

Pass 1 writes ids to a json; run the script AGAIN in a fresh process and it diffs against that
file, which is what "two identical runs" actually means — within one process an index or
embedding cache can hide the variation.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "pipeline3_round3")
for _l in open(".env"):
    if "=" in _l and not _l.strip().startswith("#"):
        _k, _v = _l.strip().split("=", 1)
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))
os.environ.update({"TIGERGRAPH_GRAPH": "enterprise",
                   "TG_SCHEMA": "datasets/enterprise/schema_v2.json",
                   "CONTEXT_NUM_CHUNKS": "10", "MAX_CONTEXT_CHUNKS": "3", "P3_VEC_K": "3",
                   "P3_VEC_SLOTS": "3", "P3_PROV_SLOTS": "1", "RR_POOL": "200", "RR_KEEP": "10"})

import pipeline3_round3.pipeline as p3          # noqa: E402
import retrieval_common                          # noqa: E402

p3._load()

OUT = pathlib.Path("_experiments/regex_router/results/retrieval_ids.json")
qs = [json.loads(l) for l in open("datasets/enterprise/eval_mixed50.jsonl") if l.strip()]


def basic_ids(q):
    """What pipeline2rr_rerank feeds the model: RR_POOL candidates, reranked, keep RR_KEEP."""
    pool = p3._tg_vector_search(q, k=retrieval_common.RR_POOL)
    kept = retrieval_common.rerank(q, pool, lambda c: p3._chunk_text.get(c, ""),
                                   keep=retrieval_common.RR_KEEP)
    return list(kept or [])


def graph_vec_ids(q):
    """The vector slots GraphRAG fills alongside its traversal evidence."""
    return list(p3._tg_vector_search(q, k=3) or [])


now = {}
for i, d in enumerate(qs, 1):
    q = d["question"]
    now[d["question_id"]] = {"basic": basic_ids(q), "graph": graph_vec_ids(q),
                             "basic_2": basic_ids(q), "graph_2": graph_vec_ids(q)}
    if i % 10 == 0:
        print(f"  ...{i}/50", flush=True)

within = [k for k, v in now.items()
          if v["basic"] != v["basic_2"] or v["graph"] != v["graph_2"]]
print("\nWITHIN one process, asked twice back to back:")
print(f"  questions whose chunk set changed: {len(within)}/50  {within}")

if OUT.exists():
    prev = json.loads(OUT.read_text())
    diff_b = [k for k in now if k in prev and now[k]["basic"] != prev[k]["basic"]]
    diff_g = [k for k in now if k in prev and now[k]["graph"] != prev[k]["graph"]]
    print("\nACROSS processes (this run vs the saved earlier run):")
    print(f"  Basic RAG chunk set changed  : {len(diff_b)}/50  {diff_b}")
    print(f"  GraphRAG vector slots changed: {len(diff_g)}/50  {diff_g}")
    for k in (diff_b or diff_g)[:3]:
        print(f"\n  {k}:")
        print(f"    before basic: {[c[-28:] for c in prev[k]['basic'][:3]]}")
        print(f"    now    basic: {[c[-28:] for c in now[k]['basic'][:3]]}")
else:
    print("\n(no earlier run saved — run this script again for the across-process diff)")

OUT.write_text(json.dumps(now, indent=1))
print(f"\nsaved ids to {OUT}")
