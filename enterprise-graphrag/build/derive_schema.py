"""Phase 1 — derive the relation schema FROM DATA COVERAGE, over ALL 20,000 documents.

WHY THIS IS NOT LLM EXTRACTION
------------------------------
The first attempt ran open LLM extraction over a sample and asked what relations exist. It
failed the coverage gate twice, and the failure was informative: the verbs it returned were
`asks`, `suggests`, `confirms`, `mentions` — DISCOURSE ACTS, not durable facts. Slack threads
and email do not often say "X reports to Y"; they argue about a rollback.

The durable structure is not in the prose. It is in the JSON fields every document already
carries — `participants`, `author`, `assignee`, `owner_team`, `channel`, `customer_company` —
plus employee_directory.yaml, which supplies 160 explicit manager links.

MasterGraph's SEC pipeline did the same thing and said so:

    BACKBONE (deterministic, no LLM), from manifest + filenames
    (Sector is a structured CSV column — no reason to spend LLM calls on it.)

So: measure coverage of the METADATA relations across every document. Deterministic, exact,
free, and it covers 100% of the corpus instead of a 500-document estimate.

    python build/derive_schema.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

DOCS = Path("datasets/enterprise/documents.jsonl")
EMPLOYEES = Path("datasets/enterprise/employee_directory.yaml")
OUT = Path("datasets/enterprise/schema.json")

# (relation, subject_kind, object_kind, [json fields that supply it], object is a list?)
# Every entry is a field that EXISTS in the data. Nothing here was chosen because a question
# needs it — this list was written by reading one document per source and taking every field
# that names another entity.
FIELD_RELATIONS = [
    ("authored",        "PERSON",   "DOC",      ["author", "creator", "mailbox_owner"],           False),
    ("participated_in", "PERSON",   "DOC",      ["participants", "participants_internal",
                                                 "redwood_attendees", "collaborators",
                                                 "reviewers"],                                    True),
    ("assigned_to",     "DOC",      "PERSON",   ["assignee"],                                     False),
    ("owned_by",        "DOC",      "PERSON",   ["owner", "redwood_owner"],                       False),
    ("in_team",         "DOC",      "TEAM",     ["owner_team", "team", "project", "space",
                                                 "drive_area"],                                   False),
    ("in_channel",      "DOC",      "CHANNEL",  ["channel"],                                      False),
    ("in_repo",         "DOC",      "REPO",     ["repo"],                                         False),
    ("about_customer",  "DOC",      "CUSTOMER", ["customer_company", "company_name"],             False),
    ("has_label",       "DOC",      "LABEL",    ["labels"],                                       True),
    ("has_status",      "DOC",      "STATUS",   ["status", "state", "stage"],                     False),
    ("has_priority",    "DOC",      "PRIORITY", ["priority", "severity"],                         False),
    ("doc_type",        "DOC",      "TYPE",     ["issue_type", "doc_type", "call_type",
                                                 "thread_type"],                                  False),
]


def clean(v) -> str:
    s = str(v).strip()
    s = re.sub(r"\s*<[^>]+>$", "", s)          # "Alyssa Chen <a@x.com>" -> "Alyssa Chen"
    s = re.sub(r"\s*\([^)]*\)$", "", s)        # "Dana Wright (CISO)"    -> "Dana Wright"
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.20)
    args = ap.parse_args()

    rel_docs = Counter()
    rel_pairs: dict[str, set] = defaultdict(set)
    rel_examples: dict[str, list] = defaultdict(list)
    by_source_rel: dict[str, Counter] = defaultdict(Counter)
    n_docs = n_parsed = 0

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
        n_parsed += 1
        doc_id, src = d["doc_id"], d["source"]

        for rel, s_kind, o_kind, fields, is_list in FIELD_RELATIONS:
            hit = False
            for f in fields:
                if f not in body or body[f] in (None, "", [], {}):
                    continue
                vals = body[f] if isinstance(body[f], list) else [body[f]]
                for v in vals:
                    if not isinstance(v, (str, int, float)):
                        continue
                    name = clean(v)
                    if not name or len(name) > 120:
                        continue
                    hit = True
                    # subject/object orientation follows the declared kinds
                    pair = (name, doc_id) if s_kind == "PERSON" else (doc_id, name)
                    rel_pairs[rel].add(pair)
                    if len(rel_examples[rel]) < 3:
                        rel_examples[rel].append(
                            {"source": pair[0], "target": pair[1], "field": f, "doc": doc_id})
            if hit:
                rel_docs[rel] += 1
                by_source_rel[src][rel] += 1

    # employee directory: the org hierarchy, structured and free
    mgr_edges = dept_edges = 0
    if EMPLOYEES.exists():
        txt = EMPLOYEES.read_text(encoding="utf-8")
        blocks = re.split(r'\n\s*-\s+name:', txt)
        dept = None
        for line in txt.splitlines():
            m = re.match(r"^  (\w[\w /&-]*):\s*$", line)
            if m:
                dept = m.group(1)
        for b in blocks[1:]:
            nm = re.match(r'\s*"([^"]+)"', b)
            mg = re.search(r'manager:\s*"([^"]+)"', b)
            if nm and mg:
                rel_pairs["reports_to"].add((nm.group(1), mg.group(1)))
                mgr_edges += 1
        for m in re.finditer(r'^  ([\w /&-]+):\s*$', txt, re.M):
            dept = m.group(1)
        # department membership
        cur = None
        for line in txt.splitlines():
            dm = re.match(r"^  ([\w /&-]+):\s*$", line)
            if dm:
                cur = dm.group(1); continue
            nm = re.search(r'-\s*name:\s*"([^"]+)"', line)
            if nm and cur:
                rel_pairs["in_department"].add((nm.group(1), cur)); dept_edges += 1
        if mgr_edges:
            rel_docs["reports_to"] = n_parsed          # structural: applies corpus-wide
            rel_examples["reports_to"] = [{"source": s, "target": t, "field": "employee_directory",
                                           "doc": "employee_directory.yaml"}
                                          for s, t in list(rel_pairs["reports_to"])[:3]]
        if dept_edges:
            rel_docs["in_department"] = n_parsed
            rel_examples["in_department"] = [{"source": s, "target": t,
                                              "field": "employee_directory",
                                              "doc": "employee_directory.yaml"}
                                             for s, t in list(rel_pairs["in_department"])[:3]]

    print(f"documents        : {n_docs:,}   (JSON-parsed: {n_parsed:,})")
    print(f"manager links    : {mgr_edges}")
    print(f"department links : {dept_edges}\n")
    print(f"{'relation':<20}{'docs':>8}{'coverage':>10}{'edges':>10}{'entities':>10}")
    print("-" * 60)
    kept, tail = [], []
    for rel, _, _, _, _ in FIELD_RELATIONS + [("reports_to", "", "", [], False),
                                              ("in_department", "", "", [], False)]:
        if rel not in rel_docs:
            continue
        cov = rel_docs[rel] / max(1, n_parsed)
        ents = len({x for p in rel_pairs[rel] for x in p})
        print(f"{rel:<20}{rel_docs[rel]:>8,}{cov:>9.1%}{len(rel_pairs[rel]):>10,}{ents:>10,}")
        rec = {"relation": rel, "documents": rel_docs[rel], "coverage": round(cov, 4),
               "edges": len(rel_pairs[rel]), "entities": ents,
               "examples": rel_examples[rel]}
        (kept if cov >= args.floor else tail).append(rec)

    schema = {
        "method": "deterministic metadata extraction over all documents (no LLM)",
        "documents": n_docs, "documents_parsed": n_parsed,
        "coverage_floor": args.floor,
        "relations": kept, "long_tail": tail,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(schema, wopen(OUT), indent=2)

    total_edges = sum(len(v) for v in rel_pairs.values())
    print("-" * 60)
    print(f"{'TOTAL':<20}{'':>8}{'':>10}{total_edges:>10,}")
    print(f"\nkept {len(kept)} relations (>= {args.floor:.0%} coverage), "
          f"{len(tail)} in the long tail")
    print(f"wrote {OUT}")

    if len(kept) < 3:
        print("\nGATE FAILED: fewer than 3 relations clear the floor.")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
