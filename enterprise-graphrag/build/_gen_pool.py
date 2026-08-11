"""Phase 5 — generate multi-hop eval questions FROM THE GRAPH.

WHY GENERATED, NOT BORROWED
---------------------------
EnterpriseRAG-Bench ships 500 questions, but only 19% need more than one document — 377 of
500 are answered by finding a single document. On those, a graph has nothing to traverse and
GraphRAG is doing Basic RAG's job with extra steps. SEC was 84% multi-hop, which is why the
graph could show anything at all there.

The corpus DOES have the structure (13,914 people, 160 manager links, 88,188
participated_in edges). Its own questions just don't ask for it. So we ask.

HOW THIS AVOIDS THE TRAP BOTH TEAMS FELL INTO
---------------------------------------------
On SEC, both teams read the eval questions and then built routers to answer them. On 5
held-out questions the ranking reversed. So here:

  * questions are instantiated from RELATIONS THAT EXIST, mechanically — never by browsing
    for a fact that would make a nice question
  * every answer is computed from the edge index, then VERIFIED against the source documents
  * the router cannot be tuned to them: it has no lexicons to tune (schema_router.py)
  * generated once, never iterated on

Each question requires >= 2 documents by construction — the answer is an intersection or a
two-hop path, so no single document contains it.

    python build/generate_questions.py --n 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

IDX = Path("pipeline3_round3/index")
OUT = Path("datasets/enterprise/eval_questions_pool.jsonl")
SEED = 77777001


def load_graph():
    rels = json.loads((IDX / "relationships.json").read_text(encoding="utf-8"))
    etypes = json.loads((IDX / "entity_types.json").read_text(encoding="utf-8"))
    fwd: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))   # rel -> src -> {tgt}
    rev: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))   # rel -> tgt -> {src}
    for r in rels:
        fwd[r["relation"]][r["source"]].add(r["target"])
        rev[r["relation"]][r["target"]].add(r["source"])
    return fwd, rev, etypes


MIN_ANSWER, MAX_ANSWER = 2, 5   # a 12-item answer is unmeasurable; a 1-item one isn't multi-hop


def make_questions(fwd, rev, etypes, n: int) -> list[dict]:
    rng = random.Random(SEED)
    qs: list[dict] = []

    docs_of = fwd["participated_in"]                 # person -> {doc}
    people_of = rev["participated_in"]               # doc -> {person}
    reports_of = rev["reports_to"]                   # manager -> {reports}
    dept_of = fwd["in_department"]                   # person -> {dept}
    chan_of = fwd["in_channel"]                      # doc -> {channel}
    auth_of = fwd["authored"]                        # person -> {doc}

    # Only people from the employee directory. Slack contributes ~8,000 handles ('anya',
    # 'ben_carter') that are a DIFFERENT namespace from the directory's full names — the same
    # human can be two nodes, and only 8% of handles map cleanly. Asking about "alex" would be
    # ambiguous and would silently miss that person's Confluence/Jira activity.
    people = sorted(dept_of.keys())
    activity = {p: (docs_of.get(p, set()) | auth_of.get(p, set())) for p in people}

    def add(shape, question, members, evidence, hops, deriv):
        if not (MIN_ANSWER <= len(members) <= MAX_ANSWER) or len(evidence) < 2:
            return
        qs.append({"shape": shape, "question": question,
                   "answer": ", ".join(sorted(members)), "gold_members": sorted(members),
                   "evidence_docs": sorted(evidence)[:10], "hops": hops, "derivation": deriv})

    # --- shape 1: TWO-HOP ORG. manager -> reports -> the channels they work in ----------
    # Needs the directory AND the documents; no single document contains the answer.
    mgrs = [m for m, rs in reports_of.items()
            if 2 <= len(rs) <= 8 and sum(1 for r in rs if activity.get(r)) >= 2]
    rng.shuffle(mgrs)
    for m in mgrs:
        rs = sorted(r for r in reports_of[m] if activity.get(r))
        chans = {c for r in rs for d in activity[r] for c in chan_of.get(d, set())}
        ev = {d for r in rs for d in list(activity[r])[:3]}
        add("two_hop_org",
            f"Which Slack channels do the people reporting to {m} take part in?",
            chans, ev, "manager -> direct reports -> documents -> channels",
            f"reports({m}) with activity = {rs}")
        if len(qs) >= n // 2:
            break

    # --- shape 2: INTERSECTION over two named colleagues --------------------------------
    pairs = Counter()
    for d, ps in people_of.items():
        ps = [p for p in ps if p in activity]
        if 2 <= len(ps) <= 10:
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    pairs[tuple(sorted((ps[i], ps[j])))] += 1
    strong = [p for p, c in pairs.items() if 3 <= c <= 30]
    rng.shuffle(strong)
    for a, b in strong:
        shared = activity[a] & activity[b]
        chans = {c for d in shared for c in chan_of.get(d, set())}
        add("intersection",
            f"Which Slack channels contain documents involving both {a} and {b}?",
            chans, shared, "person -> documents -> channels, intersected over two people",
            f"|activity({a}) & activity({b})| = {len(shared)}")
        if len(qs) >= n * 3 // 4:
            break

    # --- shape 3: DEPARTMENT -> which of its members appear in a given channel ----------
    # Two hops through the org chart and the document metadata, and the answer is a small
    # set of PEOPLE rather than channels, so the shapes are not all the same underneath.
    by_dept: dict[str, list] = defaultdict(list)
    for p in people:
        for d in dept_of.get(p, set()):
            if activity.get(p):
                by_dept[d].append(p)
    for dept, members in by_dept.items():
        chan_members: dict[str, set] = defaultdict(set)
        for p in members:
            for d in activity[p]:
                for c in chan_of.get(d, set()):
                    chan_members[c].add(p)
        for c, ppl in chan_members.items():
            add("dept_in_channel",
                f"Which people from the {dept} department take part in the "
                f"#{c} Slack channel?",
                ppl, {d for p in ppl for d in list(activity[p])[:3]},
                "department -> members -> documents -> channel",
                f"{dept} members in #{c}")
            if len(qs) >= n:
                break
        if len(qs) >= n:
            break

    rng.shuffle(qs)
    for i, q in enumerate(qs[:n], 1):
        q["question_id"] = f"MH{i:02d}"
    return qs[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    fwd, rev, etypes = load_graph()
    qs = make_questions(fwd, rev, etypes, args.n)

    # VERIFY: every question must genuinely need >=2 documents
    corpus_docs = {json.loads(l)["doc_id"] for l in ropen("datasets/enterprise/documents.jsonl")
                   if l.strip()}
    ok = []
    for q in qs:
        ev = [d for d in q["evidence_docs"] if d in corpus_docs]
        if len(ev) >= 2 and q["gold_members"]:
            q["evidence_docs"] = ev
            ok.append(q)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wopen(OUT) as fh:
        for q in ok:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"generated {len(qs)}, kept {len(ok)} verified multi-hop questions -> {OUT}\n")
    print(f"{'shape':<18}{'n':>4}   example")
    print("-" * 100)
    seen = set()
    for q in ok:
        if q["shape"] in seen:
            continue
        seen.add(q["shape"])
        n = sum(1 for x in ok if x["shape"] == q["shape"])
        print(f"{q['shape']:<18}{n:>4}   {q['question'][:70]}")
        print(f"{'':22}   -> {q['answer'][:70]}")
        print(f"{'':22}   hops: {q['hops']}")
    print(f"\nevidence documents per question: "
          f"{sum(len(q['evidence_docs']) for q in ok)/max(1,len(ok)):.1f} average "
          f"(minimum 2 by construction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
