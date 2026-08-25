"""Ask a running SEC worker the 6 `default`-bucket questions and record every arm.

Uses the worker rather than importing MasterGraph directly: server_factual.py applies the
router repair and the prompt patches the shipped SEC numbers were measured under, and a bare
import skips all of that — measured, that shortcut returned "not supported by the provided
context" on 4 of these 6 questions.

  .venv/bin/python _experiments/sec_split/ask_worker.py 8011 10
"""
import json
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ENT = pathlib.Path("/Users/devanshu/Desktop/GraphRAG_Demo/enterprise-graphrag")

port, budget = sys.argv[1], int(sys.argv[2])
QIDS = ["EQ13", "EQ14", "EQ15", "EQ45", "EQ46", "EQ47"]

src = [json.loads(l) for l in open(ENT / "results/enterprise/eval_sec_norefusal.jsonl")]
meta = {r["question_id"]: r for r in src if r["arm"] == "p3"}

out = HERE / f"worker_k{budget}.jsonl"
fh = open(out, "w")
print(f"\nport {port}, budget {budget} chunks, {len(QIDS)} questions\n")
for qid in QIDS:
    q = meta[qid]["question"]
    body = json.dumps({"question": q}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"http://localhost:{port}/query", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    for res in d.get("results", []):
        row = {"question_id": qid, "arm": res.get("arm"), "budget": budget,
               "route_published": meta[qid].get("route"), "route": res.get("route"),
               "question": q, "gold": meta[qid].get("gold", ""),
               "answer": (res.get("answer") or "").strip(),
               "prompt_tokens": res.get("prompt_tokens", 0),
               "completion_tokens": res.get("completion_tokens", 0),
               "tokens": res.get("tokens", 0), "chunks": res.get("chunks", 0),
               "latency_s": res.get("latency", 0)}
        fh.write(json.dumps(row) + "\n")
        print(f"  {qid} {row['route_published']:9s} {row['arm']:3s} "
              f"route={str(row['route'])[:14]:14s} tokens={row['tokens']:>6,} "
              f"chunks={row['chunks']:>2}  {row['answer'][:40]!r}")
    fh.flush()
    print(f"     ({time.time()-t0:.1f}s for all arms)")
fh.close()
print(f"\nwrote {out}")
