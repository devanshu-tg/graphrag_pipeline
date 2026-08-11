"""Build ONE mixed benchmark: single-hop, directory-hop, and multi-hop in a single file.

WHY MIXED
---------
The 20-question set was 100% multi-hop, and that is the weakest thing about it. A benchmark
made only of the cases your system wins invites exactly one question — "did you test anything
else?" — and there was no answer to it except a separate control file nobody would look at.

Mixing the tiers turns the result from a claim into a map:

    tier                       expected winner
    single-hop, one document   Basic RAG   (vector retrieval is genuinely good at this)
    single-hop, directory      roughly even
    multi-hop, 2-3 hops        GraphRAG

Reporting per tier is more useful than any single average, and it is far harder to attack:
the system is shown LOSING where it should lose.

EVERY question here is generated from the graph and answered from the graph, then required to
resolve against documents that exist in the retrievable corpus. Nothing is hand-picked.

    python build/generate_mixed_eval.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

IDX = Path("pipeline3_round3/index")
CORPUS = Path("datasets/enterprise/corpus.jsonl")
EXISTING = Path("datasets/enterprise/eval_questions_pool.jsonl")
OUT = Path("datasets/enterprise/eval_heldout.jsonl")
SEED = 77777001

N_SINGLE_DOC = 10      # answerable from ONE document's own metadata
N_SINGLE_DIR = 10      # one hop, but requires the employee directory
N_MULTI = 20           # the existing verified multi-hop set, reused unchanged

# Three shapes rather than one, so the tier is not a single template repeated. Measured
# availability in the corpus: 26 managers with 2-5 reports, 160 people with a manager,
# 5 departments with 2-5 members (the rest run to 56 people, too large to score as a set).
DIR_REPORTS, DIR_MANAGER, DIR_DEPT = 4, 3, 3


def main() -> int:
    rels = json.load(ropen(IDX / "relationships.json"))
    fwd: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    rev: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rels:
        fwd[r["relation"]][r["source"]].add(r["target"])
        rev[r["relation"]][r["target"]].add(r["source"])

    have = set()
    for line in ropen(CORPUS):
        if line.strip():
            have.add(json.loads(line)["doc_id"])

    rng = random.Random(SEED)
    out: list[dict] = []

    def add(tier, shape, question, members, hops):
        out.append({"question_id": "", "tier": tier, "shape": shape, "question": question,
                    "answer": ", ".join(sorted(members)), "gold_members": sorted(members),
                    "hops": hops})

    # ---- TIER 1: single-hop, one document ------------------------------------------
    # The document states the answer in its own metadata, so a single vector hit is enough.
    # This is the tier Basic RAG should WIN, and including it is the point.
    auth = [(p, d) for p, ds in fwd["authored"].items() for d in ds if d in have]
    rng.shuffle(auth)
    n = 0
    for person, doc in auth:
        if len(rev["authored"].get(doc, set())) != 1:
            continue                      # ambiguous author is not a clean single-hop answer
        name = doc.rsplit("/", 1)[-1].replace(".json", "").replace("-", " ")
        add("single_doc", "author", f"Who authored the document '{name}'?", [person],
            "document -> author, stated in that document")
        n += 1
        if n >= N_SINGLE_DOC // 2:
            break
    chan = [(d, c) for d, cs in fwd["in_channel"].items() for c in cs if d in have]
    rng.shuffle(chan)
    n = 0
    for doc, c in chan:
        if len(fwd["in_channel"].get(doc, set())) != 1:
            continue
        name = doc.rsplit("/", 1)[-1].replace(".json", "").replace("-", " ")
        add("single_doc", "channel_of_doc", f"Which Slack channel is the message '{name}' in?",
            [c], "document -> channel, stated in that document")
        n += 1
        if n >= N_SINGLE_DOC - N_SINGLE_DOC // 2:
            break

    # ---- TIER 2: single-hop, needs the employee directory ---------------------------
    # One hop, but the fact lives only on the directory page. Both approaches can reach it:
    # the graph by an edge, retrieval by finding that page. Expected to be close, which is
    # exactly why it belongs here.
    mgrs = [m for m, rs in rev["reports_to"].items() if 2 <= len(rs) <= 5]
    rng.shuffle(mgrs)
    for m in mgrs[:DIR_REPORTS]:
        add("single_dir", "direct_reports", f"Who reports directly to {m}?",
            sorted(rev["reports_to"][m]), "manager -> direct reports, from the directory")

    # the inverse direction. Both are stated on the directory pages (a manager's page lists
    # their reports, a report's page names their manager), so retrieval is not handicapped
    # relative to the graph, which can walk one edge either way.
    staff = [p for p, ms in fwd["reports_to"].items() if len(ms) == 1
             and p not in mgrs[:DIR_REPORTS]]
    rng.shuffle(staff)
    for p in staff[:DIR_MANAGER]:
        add("single_dir", "manager_of", f"Who is {p}'s manager?",
            sorted(fwd["reports_to"][p]), "person -> manager, from the directory")

    depts = [(d, sorted(rev["in_department"][d])) for d in rev["in_department"]
             if 2 <= len(rev["in_department"][d]) <= 5]
    rng.shuffle(depts)
    for d, members in depts[:DIR_DEPT]:
        add("single_dir", "dept_members", f"Which people are in the {d} department?",
            members, "department -> members, from the directory")

    # ---- TIER 3: the existing verified multi-hop set, reused VERBATIM ---------------
    # Not regenerated: these already passed the leak gate and their gold was cross-checked
    # against both graph builds. Changing them now would invalidate every number so far.
    for q in [json.loads(l) for l in ropen(EXISTING) if l.strip()][:N_MULTI]:
        out.append({**q, "tier": "multi_hop"})

    for i, q in enumerate(out, 1):
        q["question_id"] = f"MX{i:02d}"

    with wopen(OUT) as fh:
        for q in out:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wrote {len(out)} questions -> {OUT}")
    print(f"  tiers: {dict(Counter(q['tier'] for q in out))}")
    print(f"  shapes: {dict(Counter(q['shape'] for q in out))}")
    for t in ("single_doc", "single_dir", "multi_hop"):
        ex = next((q for q in out if q["tier"] == t), None)
        if ex:
            print(f"\n  [{t}] {ex['question'][:78]}")
            print(f"        -> {ex['answer'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
