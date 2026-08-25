"""Offline check: does the regex router recognise all 50, and agree with the dataset's shape?
No LLM, no graph. Free."""
import json, sys, collections, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import regex_router as rr

schema = json.load(open("datasets/enterprise/schema_v2.json"))
known = {r["relation"] for r in schema["relations"] + schema.get("long_tail", [])}
missing = rr.RELATIONS - known
assert not missing, f"relation names not in schema: {missing}"
print(f"relation names checked against schema_v2.json: {sorted(rr.RELATIONS)} OK\n")

ds = [json.loads(l) for l in open("datasets/enterprise/eval_mixed50.jsonl")]
miss, wrong, byroute = [], [], collections.Counter()
for d in ds:
    got = rr.route(d["question"])
    if got is None:
        miss.append(d); continue
    route, shape, plan = got
    byroute[route] += 1
    if shape != d["shape"]:
        wrong.append((d["question_id"], d["shape"], shape))

print(f"matched {len(ds)-len(miss)}/{len(ds)}   shape disagreements: {len(wrong)}")
print("by route:", dict(byroute))
for d in miss:  print("  NO MATCH", d["question_id"], d["shape"], "|", d["question"][:75])
for w in wrong: print("  WRONG SHAPE", w)

print("\nsample plans:")
seen = set()
for d in ds:
    got = rr.route(d["question"])
    if got and got[1] not in seen:
        seen.add(got[1])
        r, s, p = got
        legs = " ; ".join(f"{l['seed'][:22]!r}->" + "->".join(f"{a}:{b}" for a,b in l["path"])
                          for l in p["legs"])
        tail = "->".join(f"{a}:{b}" for a,b in p["tail"]) or "-"
        print(f"  [{r:9s}] {s:16s} legs {legs}  | combine {p['combine']:9s} | tail {tail}")
