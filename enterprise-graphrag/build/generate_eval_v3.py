"""A genuinely HELD-OUT enterprise eval set: 20 single-doc / 10 directory / 20 multi-hop.

WHY A NEW SET
-------------
Every enterprise number so far comes from `eval_mixed50.jsonl`. Three audit findings apply to
it and none can be answered by re-running it:

  I1  the gold answer IS the traversal output — the graph is asked to reproduce a computation
      it performs by definition
  I3  50/50 questions contain a word matching the relation they need
  I4  it is 8 templates with names substituted, so the effective n is ~8, not 50

This set fixes ONE of those: the entities are disjoint from the old set, so nothing that was
seen while debugging can be scored again. It deliberately does NOT fix I1 (gold still comes
from the graph — that is what makes verification possible) or I3 (same phrasing, so the two
sets stay comparable; `eval_paraphrase50.jsonl` is the test for I3).

So this measures GENERALISATION TO NEW ENTITIES, and nothing more. Stating that plainly matters
more than the number it produces.

DISJOINTNESS
------------
Every seed used by eval_mixed50 — documents, managers, staff, departments, channels and the
people named in intersection questions — is excluded. A question is only emitted if none of its
seeds appear in that set.

SELF-CHECK GATE
---------------
Before emitting anything, the generator RE-DERIVES the gold for the existing 50 questions using
the same code path and requires an exact match on all of them. If it cannot reproduce the old
answer key from the old questions, its own new answer key is not trustworthy either. This is
the same gate `build/regold.py` applies.

    python build/generate_eval_v3.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

IDX = Path("pipeline3_round3/index")
CORPUS = Path("datasets/enterprise/corpus.jsonl")
OLD = Path("datasets/enterprise/eval_mixed50.jsonl")
OUT = Path("datasets/enterprise/eval_v3_50.jsonl")
SEED = 20260812

N_AUTHOR, N_CHANNEL = 10, 10          # single_doc  = 20
N_REPORTS, N_MANAGER, N_DEPT = 4, 3, 3   # single_dir = 10
N_INTERSECT, N_TWOHOP, N_DEPTCHAN = 9, 6, 5   # multi_hop = 20

fwd: dict = defaultdict(lambda: defaultdict(set))
rev: dict = defaultdict(lambda: defaultdict(set))


def load() -> set:
    for r in json.load(ropen(IDX / "relationships.json")):
        fwd[r["relation"]][r["source"]].add(r["target"])
        rev[r["relation"]][r["target"]].add(r["source"])
    return {json.loads(l)["doc_id"] for l in ropen(CORPUS) if l.strip()}


# ---- gold derivations. ONE definition, used for both the self-check and the new questions ----
def gold_author(doc):        return sorted(rev["authored"].get(doc, set()))
def gold_channel(doc):       return sorted(fwd["in_channel"].get(doc, set()))
def gold_reports(mgr):       return sorted(rev["reports_to"].get(mgr, set()))
def gold_manager(person):    return sorted(fwd["reports_to"].get(person, set()))
def gold_dept(dept):         return sorted(rev["in_department"].get(dept, set()))


def gold_intersection(a, b):
    docs = fwd["participated_in"].get(a, set()) & fwd["participated_in"].get(b, set())
    return sorted({c for d in docs for c in fwd["in_channel"].get(d, set())})


def gold_two_hop_org(mgr):
    people = rev["reports_to"].get(mgr, set())
    docs = set().union(*[fwd["participated_in"].get(p, set()) for p in people]) if people else set()
    return sorted({c for d in docs for c in fwd["in_channel"].get(d, set())})


def gold_dept_in_channel(dept, chan):
    members = rev["in_department"].get(dept, set())
    docs = rev["in_channel"].get(chan, set())
    return sorted({p for p in members if fwd["participated_in"].get(p, set()) & docs})


def self_check() -> tuple[int, int]:
    """Reproduce the EXISTING answer key from the existing questions. Must be 50/50."""
    ok = tot = 0
    for q in (json.loads(l) for l in ropen(OLD) if l.strip()):
        t, sh = q["question"], q["shape"]
        got = None
        if sh == "author":
            m = re.fullmatch(r"Who authored the document '(.+)'\?", t)
            got = gold_author(_doc_by_title(m.group(1))) if m else None
        elif sh == "channel_of_doc":
            m = re.fullmatch(r"Which Slack channel is the message '(.+)' in\?", t)
            got = gold_channel(_doc_by_title(m.group(1))) if m else None
        elif sh == "direct_reports":
            got = gold_reports(re.fullmatch(r"Who reports directly to (.+)\?", t).group(1))
        elif sh == "manager_of":
            got = gold_manager(re.fullmatch(r"Who is (.+)'s manager\?", t).group(1))
        elif sh == "dept_members":
            got = gold_dept(re.fullmatch(r"Which people are in the (.+) department\?", t).group(1))
        elif sh == "intersection":
            m = re.fullmatch(r"Which Slack channels contain documents involving both (.+) and (.+)\?", t)
            got = gold_intersection(m.group(1), m.group(2))
        elif sh == "two_hop_org":
            m = re.fullmatch(r"Which Slack channels do the people reporting to (.+) take part in\?", t)
            got = gold_two_hop_org(m.group(1))
        elif sh == "dept_in_channel":
            m = re.fullmatch(r"Which people from the (.+) department take part in the #(\S+) Slack channel\?", t)
            got = gold_dept_in_channel(m.group(1), m.group(2))
        tot += 1
        if got is not None and sorted(got) == sorted(q["gold_members"]):
            ok += 1
        else:
            print(f"    MISMATCH [{q['question_id']}] {sh}: got {got} want {q['gold_members']}")
    return ok, tot


_title2doc: dict = {}


def _doc_by_title(title: str) -> str:
    return _title2doc.get(title, "")


def _title(doc: str) -> str:
    return doc.rsplit("/", 1)[-1].replace(".json", "").replace("-", " ")


def used_seeds() -> set:
    """Everything eval_mixed50 built a question from. Nothing here may be reused."""
    used = set()
    for q in (json.loads(l) for l in ropen(OLD) if l.strip()):
        t = q["question"]
        for pat in (r"the document '(.+)'\?", r"the message '(.+)' in\?",
                    r"reports directly to (.+)\?", r"Who is (.+)'s manager\?",
                    r"in the (.+) department\?", r"reporting to (.+) take part in\?"):
            m = re.search(pat, t)
            if m:
                used.add(m.group(1).strip())
        m = re.search(r"both (.+?) and (.+?)\?", t)
        if m:
            used.update({m.group(1).strip(), m.group(2).strip()})
        m = re.search(r"from the (.+?) department take part in the #(\S+)", t)
        if m:
            used.update({m.group(1).strip(), m.group(2).strip()})
        used.update(q["gold_members"])          # answers too — no recycling an old answer as a seed
    return used


def main() -> int:
    have = load()
    for d in list(fwd["in_channel"]) + list(rev["authored"]):
        _title2doc[_title(d)] = d

    print("SELF-CHECK — re-deriving the EXISTING answer key from the existing questions")
    ok, tot = self_check()
    print(f"  {ok}/{tot} reproduced exactly")
    if ok != tot:
        print("  GATE FAILED — the derivations do not reproduce the old key, so the new key\n"
              "  cannot be trusted either. Nothing written.")
        return 1

    used = used_seeds()
    print(f"\nexcluding {len(used)} seeds/answers already used by eval_mixed50")
    rng = random.Random(SEED)
    out: list[dict] = []

    def add(tier, shape, q, members, hops):
        out.append({"question_id": "", "tier": tier, "shape": shape, "question": q,
                    "answer": ", ".join(sorted(members)), "gold_members": sorted(members),
                    "hops": hops})

    # ---- single_doc (20) ----
    auth = [(p, d) for p, ds in fwd["authored"].items() for d in ds
            if d in have and len(rev["authored"].get(d, set())) == 1
            and p not in used and _title(d) not in used]
    rng.shuffle(auth)
    for p, d in auth[:N_AUTHOR]:
        add("single_doc", "author", f"Who authored the document '{_title(d)}'?", [p],
            "document -> author")

    chan = [(d, c) for d, cs in fwd["in_channel"].items() for c in cs
            if d in have and len(fwd["in_channel"].get(d, set())) == 1 and _title(d) not in used]
    rng.shuffle(chan)
    for d, c in chan[:N_CHANNEL]:
        add("single_doc", "channel_of_doc", f"Which Slack channel is the message '{_title(d)}' in?",
            [c], "document -> channel")

    # ---- single_dir (10) ----
    mgrs = [m for m, rs in rev["reports_to"].items() if 2 <= len(rs) <= 5 and m not in used]
    rng.shuffle(mgrs)
    for m in mgrs[:N_REPORTS]:
        add("single_dir", "direct_reports", f"Who reports directly to {m}?",
            gold_reports(m), "manager -> direct reports")

    staff = [p for p, ms in fwd["reports_to"].items()
             if len(ms) == 1 and p not in used and p not in mgrs[:N_REPORTS]]
    rng.shuffle(staff)
    for p in staff[:N_MANAGER]:
        add("single_dir", "manager_of", f"Who is {p}'s manager?", gold_manager(p),
            "person -> manager")

    depts = [d for d in rev["in_department"]
             if 2 <= len(rev["in_department"][d]) <= 5 and d not in used]
    rng.shuffle(depts)
    for d in depts[:N_DEPT]:
        add("single_dir", "dept_members", f"Which people are in the {d} department?",
            gold_dept(d), "department -> members")

    # Only 5 departments have 2-5 members and the old set consumed 3, so dept_members cannot
    # always supply its full quota. Backfill the TIER from manager_of, which has 160 candidates,
    # rather than pad with a department whose answer set is too large to score as a set.
    # The tier total is what the benchmark design calls for; the split inside it is not load
    # bearing, and the shortfall is printed either way.
    short = 10 - sum(1 for q in out if q["tier"] == "single_dir")
    if short > 0:
        extra = [p for p in staff[N_MANAGER:] if p not in used]
        for p in extra[:short]:
            add("single_dir", "manager_of", f"Who is {p}'s manager?", gold_manager(p),
                "person -> manager")
        print(f"  note: dept_members short by {short}; backfilled with manager_of")

    # ---- multi_hop (20) ----
    people = sorted({p for p in fwd["participated_in"]
                     if 3 <= len(fwd["participated_in"][p]) <= 200 and p not in used})
    rng.shuffle(people)
    seen_pairs = set()
    for i in range(len(people)):
        if sum(1 for q in out if q["shape"] == "intersection") >= N_INTERSECT:
            break
        for j in range(i + 1, min(i + 60, len(people))):
            a, b = people[i], people[j]
            if (a, b) in seen_pairs:
                continue
            g = gold_intersection(a, b)
            if 2 <= len(g) <= 6:
                seen_pairs.add((a, b))
                add("multi_hop", "intersection",
                    f"Which Slack channels contain documents involving both {a} and {b}?", g,
                    "2 people -> shared documents -> channels")
                break

    mg2 = [m for m in rev["reports_to"] if m not in used and m not in mgrs[:N_REPORTS]]
    rng.shuffle(mg2)
    for m in mg2:
        if sum(1 for q in out if q["shape"] == "two_hop_org") >= N_TWOHOP:
            break
        g = gold_two_hop_org(m)
        if 2 <= len(g) <= 6:
            add("multi_hop", "two_hop_org",
                f"Which Slack channels do the people reporting to {m} take part in?", g,
                "manager -> reports -> documents -> channels")

    dc = [(d, c) for d in rev["in_department"] for c in rev["in_channel"]
          if d not in used and c not in used]
    rng.shuffle(dc)
    for d, c in dc:
        if sum(1 for q in out if q["shape"] == "dept_in_channel") >= N_DEPTCHAN:
            break
        g = gold_dept_in_channel(d, c)
        if 2 <= len(g) <= 6:
            add("multi_hop", "dept_in_channel",
                f"Which people from the {d} department take part in the #{c} Slack channel?", g,
                "department + channel -> people in both")

    for i, q in enumerate(out, 1):
        q["question_id"] = f"V3{i:02d}"

    from collections import Counter
    tiers = Counter(q["tier"] for q in out)
    print(f"\ngenerated {len(out)}: {dict(tiers)}")
    print(f"  shapes: {dict(Counter(q['shape'] for q in out))}")
    if tiers["single_doc"] != 20 or tiers["single_dir"] != 10 or tiers["multi_hop"] != 20:
        print("  WARNING: tier counts differ from the requested 20/10/20 — not enough\n"
              "  unused entities of some shape. Reported as-is rather than padded.")

    with wopen(OUT) as fh:
        for q in out:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")

    old_ids = {q["question"] for q in (json.loads(l) for l in ropen(OLD) if l.strip())}
    dup = [q for q in out if q["question"] in old_ids]
    print(f"\noverlap with eval_mixed50: {len(dup)} questions (must be 0)")
    for t in ("single_doc", "single_dir", "multi_hop"):
        ex = next((q for q in out if q["tier"] == t), None)
        if ex:
            print(f"  [{t}] {ex['question'][:82]}")
            print(f"        -> {ex['answer'][:66]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
