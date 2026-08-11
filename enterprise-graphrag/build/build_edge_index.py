"""Phase 2 — build the KG index from the schema, deterministically.

Emits exactly the files build/round3_tg_ingest.py already reads, so the loader needs no
changes beyond its CURATED_EDGES list:

    entity_to_docs.json   entity -> [doc_id, ...]     (drives mention_count)
    entity_types.json     entity -> PERSON|TEAM|...
    relationships.json    [{source, relation, target, count}]
    edges_prov.json       [{source, relation, target, evidence, chunk_id, doc_id}]

Every edge comes from a JSON field that already exists in the document, or from
employee_directory.yaml. Nothing is inferred, so there is no extraction error to audit —
the Phase-2 gate on the SEC run (hand-check 20 edges against their evidence sentence) becomes
a check that the field was read correctly, not that a model guessed correctly.

`evidence` is the field it came from, rendered as a sentence, so provenance-sentence
substitution in P3 still has something short and true to feed the model.

    python build/build_edge_index.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

DOCS = Path("datasets/enterprise/documents.jsonl")
CORPUS = Path("datasets/enterprise/corpus.jsonl")
EMPLOYEES = Path("datasets/enterprise/employee_directory.yaml")
SCHEMA = Path("datasets/enterprise/schema.json")
OUT = Path("pipeline3_round3/index")

# Same field map the schema derivation measured coverage on. Kept in one place: the schema
# says WHICH relations are worth having, this says WHERE each one comes from.
FIELD_RELATIONS = [
    ("authored",        "PERSON",   "DOC",      ["author", "creator", "mailbox_owner"]),
    ("participated_in", "PERSON",   "DOC",      ["participants", "participants_internal",
                                                 "redwood_attendees", "collaborators", "reviewers"]),
    ("assigned_to",     "DOC",      "PERSON",   ["assignee"]),
    ("owned_by",        "DOC",      "PERSON",   ["owner", "redwood_owner"]),
    ("in_team",         "DOC",      "TEAM",     ["owner_team", "team", "project", "space", "drive_area"]),
    ("in_channel",      "DOC",      "CHANNEL",  ["channel"]),
    ("in_repo",         "DOC",      "REPO",     ["repo"]),
    ("about_customer",  "DOC",      "CUSTOMER", ["customer_company", "company_name"]),
    ("has_label",       "DOC",      "LABEL",    ["labels"]),
    ("has_status",      "DOC",      "STATUS",   ["status", "state", "stage"]),
    ("has_priority",    "DOC",      "PRIORITY", ["priority", "severity"]),
    ("doc_type",        "DOC",      "TYPE",     ["issue_type", "doc_type", "call_type", "thread_type"]),
]
FIELD_PHRASE = {"author": "authored by", "creator": "created by", "mailbox_owner": "mailbox of",
                "participants": "participants include", "participants_internal": "internal participants include",
                "redwood_attendees": "attendees include", "collaborators": "collaborators include",
                "reviewers": "reviewers include", "assignee": "assigned to", "owner": "owned by",
                "redwood_owner": "owned by", "owner_team": "owned by team", "team": "belongs to team",
                "project": "in project", "space": "in space", "drive_area": "in drive area",
                "channel": "posted in channel", "repo": "in repository",
                "customer_company": "concerns customer", "company_name": "concerns account",
                "labels": "labelled", "status": "status is", "state": "state is", "stage": "stage is",
                "priority": "priority is", "severity": "severity is", "issue_type": "issue type is",
                "doc_type": "document type is", "call_type": "call type is", "thread_type": "thread type is"}


def clean(v) -> str:
    s = re.sub(r"\s*<[^>]+>$", "", str(v).strip())
    return re.sub(r"\s*\([^)]*\)$", "", s).strip()


def main() -> int:
    keep = {r["relation"] for r in json.loads(SCHEMA.read_text(encoding="utf-8"))["relations"]}
    tail = {r["relation"] for r in json.loads(SCHEMA.read_text(encoding="utf-8"))["long_tail"]}
    print(f"schema: {len(keep)} curated relations {sorted(keep)}")
    print(f"        {len(tail)} long-tail relations -> related_to\n")

    # first chunk id per document, so every edge carries a citable provenance target
    first_chunk: dict[str, str] = {}
    for line in ropen(CORPUS):
        if line.strip():
            c = json.loads(line)
            first_chunk.setdefault(c["doc_id"], c["title"])

    e2d: dict[str, set] = defaultdict(set)
    etypes: dict[str, str] = {}
    counts: Counter = Counter()
    prov: dict[tuple, dict] = {}
    n_docs = 0

    def add(src, rel, tgt, kind_s, kind_t, doc_id, phrase):
        if not src or not tgt:
            return
        counts[(src, rel, tgt)] += 1
        etypes.setdefault(src, kind_s)
        etypes.setdefault(tgt, kind_t)
        e2d[src].add(doc_id)
        e2d[tgt].add(doc_id)
        prov.setdefault((src, rel, tgt), {
            "source": src, "relation": rel, "target": tgt,
            "evidence": f"{src} — {phrase} — {tgt}.",
            "chunk_id": first_chunk.get(doc_id, ""), "doc_id": doc_id})

    for line in ropen(DOCS):
        if not line.strip():
            continue
        d = json.loads(line)
        n_docs += 1
        try:
            body = json.loads(d["text"])
            if not isinstance(body, dict):
                continue
        except Exception:
            continue
        doc_id = d["doc_id"]
        etypes.setdefault(doc_id, "DOC")
        e2d[doc_id].add(doc_id)

        for rel, s_kind, o_kind, fields in FIELD_RELATIONS:
            if rel not in keep and rel not in tail:
                continue
            for f in fields:
                if f not in body or body[f] in (None, "", [], {}):
                    continue
                for v in (body[f] if isinstance(body[f], list) else [body[f]]):
                    if not isinstance(v, (str, int, float)):
                        continue
                    name = clean(v)
                    if not name or len(name) > 120:
                        continue
                    phrase = FIELD_PHRASE.get(f, f)
                    if s_kind == "PERSON":
                        add(name, rel, doc_id, "PERSON", "DOC", doc_id, phrase)
                    else:
                        add(doc_id, rel, name, "DOC", o_kind, doc_id, phrase)

    # org hierarchy from the employee directory — structured, exact
    txt = EMPLOYEES.read_text(encoding="utf-8")
    dept, n_mgr, n_dept = None, 0, 0
    for line in txt.splitlines():
        dm = re.match(r"^  ([\w /&-]+):\s*$", line)
        if dm:
            dept = dm.group(1)
        nm = re.search(r'-\s*name:\s*"([^"]+)"', line)
        if nm:
            cur_person = nm.group(1)
            etypes.setdefault(cur_person, "PERSON")
            if dept:
                add(cur_person, "in_department", dept, "PERSON", "DEPARTMENT",
                    "employee_directory.yaml", "works in department")
                n_dept += 1
    for block in re.split(r'\n\s*-\s+name:', txt)[1:]:
        nm = re.match(r'\s*"([^"]+)"', block)
        mg = re.search(r'manager:\s*"([^"]+)"', block)
        if nm and mg:
            add(nm.group(1), "reports_to", mg.group(1), "PERSON", "PERSON",
                "employee_directory.yaml", "reports to")
            n_mgr += 1

    OUT.mkdir(parents=True, exist_ok=True)
    rels = [{"source": s, "relation": r, "target": t, "count": c}
            for (s, r, t), c in counts.items()]
    json.dump({k: sorted(v) for k, v in e2d.items()}, wopen(OUT / "entity_to_docs.json"))
    json.dump(etypes, wopen(OUT / "entity_types.json"))
    json.dump(rels, wopen(OUT / "relationships.json"))
    json.dump(list(prov.values()), wopen(OUT / "edges_prov.json"))
    json.dump({}, wopen(OUT / "company_facts.json"))
    json.dump([], wopen(OUT / "events.json"))

    print(f"documents processed : {n_docs:,}")
    print(f"reports_to edges    : {n_mgr}")
    print(f"in_department edges : {n_dept}")
    print(f"\nentities            : {len(e2d):,}")
    print(f"edges (distinct)    : {len(rels):,}")
    print(f"provenance records  : {len(prov):,}")
    print(f"\n{'relation':<20}{'edges':>10}")
    print("-" * 30)
    per = Counter(r["relation"] for r in rels)
    for r, c in per.most_common():
        print(f"{r:<20}{c:>10,}  {'curated' if r in keep else 'related_to'}")
    print(f"\nentity types: {dict(Counter(etypes.values()).most_common())}")
    print(f"\nwrote 6 index files -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
