"""Execute each regex plan against the graph and compare the landed set to gold_members.
Graph only - no LLM, no generation. This is the test that decides whether the regex plans are
as good as the planner's before any tokens are spent."""
import json, os, sys, pathlib, time
sys.path.insert(0, ".")
sys.path.insert(0, "pipeline3_round3")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
for line in open(".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))
# the demo's shipped config, so this measures what the demo would ship
os.environ.update({"TIGERGRAPH_GRAPH": "enterprise", "CONTEXT_NUM_CHUNKS": "10",
                   "RR_POOL": "50", "RR_KEEP": "10", "MAX_CONTEXT_CHUNKS": "3",
                   "P3_VEC_K": "3", "P3_VEC_SLOTS": "3", "P3_PROV_SLOTS": "1"})
import regex_router as rr
t0 = time.time()
import pipeline3_round3.pipeline as p3
p3._load()
print(f"pipeline loaded in {time.time()-t0:.0f}s\n")

ds = [json.loads(l) for l in open("datasets/enterprise/eval_mixed50.jsonl")]
ok = part = bad = empty = 0
rows = []
for d in ds:
    route, shape, plan = rr.route(d["question"])
    landed, trail = p3._walk(plan)
    norm = lambda s: " ".join(s.replace("_"," ").lower().split())
    gold = {norm(g) for g in d["gold_members"]}
    got = {norm(x) for x in landed}
    exact = got == gold
    if not landed:   empty += 1; verdict = "EMPTY"
    elif exact:      ok += 1;    verdict = "exact"
    elif gold & got: part += 1;  verdict = "partial"
    else:            bad += 1;   verdict = "WRONG"
    rows.append({"qid": d["question_id"], "shape": shape, "route": route,
                 "verdict": verdict, "n_landed": len(landed),
                 "missed": sorted(gold - got)[:4], "extra": sorted(got - gold)[:4]})
    if verdict != "exact":
        print(f"  {verdict:8s} {d['question_id']} {shape:16s} landed={len(landed)} "
              f"missed={sorted(gold-got)[:3]} extra={sorted(got-gold)[:3]}")

print(f"\nTRAVERSAL RESULT over {len(ds)} questions")
print(f"  exact set match : {ok}/{len(ds)}")
print(f"  partial         : {part}")
print(f"  wrong           : {bad}")
print(f"  landed nothing  : {empty}")
json.dump(rows, open("_experiments/regex_router/results/walk_check.json", "w"), indent=1)
