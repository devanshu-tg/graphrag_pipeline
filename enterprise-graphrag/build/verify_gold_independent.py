"""Verify the answer key against the RAW DOCUMENTS, without touching the graph.

THE ATTACK THIS ANSWERS
-----------------------
Gold answers were computed by walking the graph: reports_to -> participated_in -> in_channel.
P3 answers by walking reports_to -> participated_in -> in_channel. Same index, same relations.

So if the graph is wrong, the answer key is wrong IN THE SAME DIRECTION and P3 still scores
1.00. That is circular, and it is the first thing a sharp reviewer will say.

The defence has always been "the graph is built deterministically from document metadata
fields, so it is traceable to the documents" — but traceable is not the same as traced. This
script actually traces it: for each question it re-derives the answer by READING THE RAW
DOCUMENT JSON, parsing the same fields a human would look at, and never consulting
relationships.json, edges_prov.json or TigerGraph.

If the independently-derived answer matches the stored gold, the key is sound and the
circularity objection is answered with evidence. If it does not, the benchmark is broken and
better to know.

    python build/verify_gold_independent.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen   # noqa: E402

init_console()

DOCS = Path("datasets/enterprise/documents.jsonl")
EMPLOYEES = Path("datasets/enterprise/employee_directory.yaml")
QUESTIONS = Path("datasets/enterprise/eval_mixed40.jsonl")

PARTICIPANT_FIELDS = ("participants", "participants_internal", "redwood_attendees",
                      "collaborators", "reviewers")
AUTHOR_FIELDS = ("author", "creator", "mailbox_owner")


def load_documents() -> dict:
    """doc_id -> parsed JSON body. The BODY, not the graph's summary of it."""
    out = {}
    for line in ropen(DOCS):
        if not line.strip():
            continue
        d = json.loads(line)
        body = d.get("text", "")
        try:
            # every source in this corpus stores its metadata as a JSON object at the top of
            # the document text; that is the field a human would read
            out[d["doc_id"]] = json.loads(body) if body.lstrip().startswith("{") else {}
        except Exception:
            out[d["doc_id"]] = {}
    return out


def load_directory() -> tuple[dict, dict]:
    """(manager_of, department_of) straight from the YAML — no graph involved."""
    import yaml
    depts = yaml.safe_load(ropen(EMPLOYEES))["departments"]
    mgr, dept = {}, {}
    for dname, members in depts.items():
        for p in members:
            dept[p["name"]] = dname
            if p.get("manager"):
                mgr[p["name"]] = p["manager"]
    return mgr, dept


def channel_of(doc_id: str, body: dict) -> str | None:
    ch = body.get("channel")
    if ch:
        return str(ch).lstrip("#")
    # slack paths are slack/<channel>/<file>.json — the channel is in the path itself
    parts = doc_id.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "slack" else None


def people_on(body: dict) -> set:
    out = set()
    for f in PARTICIPANT_FIELDS:
        v = body.get(f)
        if isinstance(v, list):
            out |= {str(x).strip() for x in v if str(x).strip()}
        elif v:
            out.add(str(v).strip())
    for f in AUTHOR_FIELDS:
        if body.get(f):
            out.add(str(body[f]).strip())
    return out


def main() -> int:
    docs = load_documents()
    mgr_of, dept_of = load_directory()
    reports_of: dict = {}
    for person, m in mgr_of.items():
        reports_of.setdefault(m, set()).add(person)
    print(f"read {len(docs):,} raw documents and {len(dept_of)} directory entries "
          f"(graph files NOT opened)\n")

    qs = [json.loads(l) for l in ropen(QUESTIONS) if l.strip()]
    checked = agree = 0
    disagreements = []

    for q in qs:
        text, gold = q["question"], set(q["gold_members"])
        derived: set | None = None

        if (m := re.match(r"^Who reports directly to (.+)\?$", text)):
            derived = set(reports_of.get(m.group(1), set()))
        elif (m := re.match(r"^Who is (.+)'s manager\?$", text)):
            derived = {mgr_of[m.group(1)]} if m.group(1) in mgr_of else set()
        elif (m := re.match(r"^Which people are in the (.+) department\?$", text)):
            derived = {p for p, d in dept_of.items() if d == m.group(1)}
        elif (m := re.match(r"^Which Slack channels do the people reporting to (.+) take part in\?$", text)):
            people = reports_of.get(m.group(1), set())
            derived = {c for did, b in docs.items()
                       if people & people_on(b) and (c := channel_of(did, b))}
        elif (m := re.match(r"^Which Slack channels contain documents involving both (.+) and (.+)\?$", text)):
            a, b_ = m.group(1), m.group(2)
            derived = {c for did, b in docs.items()
                       if {a, b_} <= people_on(b) and (c := channel_of(did, b))}
        elif (m := re.match(r"^Which people from the (.+) department take part in the #(.+) Slack channel\?$", text)):
            dept, chan = m.group(1), m.group(2)
            members = {p for p, d in dept_of.items() if d == dept}
            derived = {p for did, b in docs.items() if channel_of(did, b) == chan
                       for p in (members & people_on(b))}

        if derived is None:
            continue                      # single-document shapes: answer IS the field, nothing to cross-check
        checked += 1
        if derived == gold:
            agree += 1
        else:
            disagreements.append((q["question_id"], q["tier"], sorted(gold),
                                  sorted(derived), text))

    print(f"{'question shapes cross-checked':<44}{checked}")
    print(f"{'independently derived answer MATCHES gold':<44}{agree}/{checked}"
          f"   ({100*agree/max(1,checked):.0f}%)\n")
    if disagreements:
        print("DISAGREEMENTS — gold vs derived-from-raw-documents:")
        for qid, tier, g, d, t in disagreements[:10]:
            print(f"\n  {qid} [{tier}] {t[:72]}")
            print(f"     gold    : {g}")
            print(f"     derived : {d}")
            print(f"     only in gold   : {sorted(set(g)-set(d))}")
            print(f"     only in derived: {sorted(set(d)-set(g))}")
    else:
        print("The answer key is reproducible from the raw documents alone.\n"
              "The graph is a convenient index over those facts, not the source of them,\n"
              "so the circularity objection does not hold.")
    return 0 if not disagreements else 1


if __name__ == "__main__":
    raise SystemExit(main())
