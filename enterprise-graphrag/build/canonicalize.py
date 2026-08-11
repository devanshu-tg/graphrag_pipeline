"""Rebuild v2 — merge the person namespace, type the vertices, fold filters into attributes.

THREE THINGS WRONG WITH v1, IN ORDER OF HOW MUCH THEY COST
----------------------------------------------------------
1. PERSON NAMESPACE SPLIT — the expensive one, and invisible until measured.
   The corpus records the same human two ways: 'Laura Bennett' in Confluence/email/meeting
   metadata, 'laura_bennett' in Slack. v1 built them as two unrelated vertices. Measured:

       participated_in reachable from the full name : 5,068
       hidden behind the username                   : 13,603

   So a walk that starts at a directory person sees 27% of what they actually did. Laura
   Bennett: 169 edges under the full name, 900 under the username. Every multi-hop question
   about people was answering from a quarter of the graph — and the GOLD ANSWERS were
   computed from the same broken graph, so the benchmark could not see its own error.

2. EVERY VERTEX WAS TYPE `Entity`. A 38,899-vertex graph with two node types is not a
   knowledge graph, it is an edge list with a label column. Typing lets the walk constrain a
   hop to the kind of thing being asked for, and it is what a graph database is for.

3. status / priority / doc_type WERE VERTICES. 'open', 'P1' and 'bug' are ATTRIBUTES of a
   document, not entities that a question traverses through. As vertices they added 11,939
   edges, three fake relations for the router to choose wrongly from, and hub nodes with
   thousands of neighbours that blow up any fan-out. They become Document attributes.

WHAT IS NOT CHANGED: the extraction method (still deterministic, still from metadata fields,
still $0), the relation set's derivation (still by data coverage, never by question), and the
router (still schema-driven, still zero hardcoded entity names).

    python build/canonicalize.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

IN = Path("pipeline3_round3/index")
OUT = Path("pipeline3_round3/index_v2")
SCHEMA_IN = Path("datasets/enterprise/schema.json")
SCHEMA_OUT = Path("datasets/enterprise/schema_v2.json")

# 'open'/'P1'/'bug' describe a document; they are not things you traverse THROUGH.
ATTR_RELATIONS = {"has_status": "status", "has_priority": "priority", "doc_type": "doc_type"}

# The nine real vertex types. STATUS/PRIORITY/TYPE are gone — folded above.
TYPE_OF_RELATION = {
    "authored":        ("PERSON", "DOC"),
    "participated_in": ("PERSON", "DOC"),
    "assigned_to":     ("DOC", "PERSON"),
    "owned_by":        ("DOC", "PERSON"),
    "in_team":         ("DOC", "TEAM"),
    "in_channel":      ("DOC", "CHANNEL"),
    "in_repo":         ("DOC", "REPO"),
    "about_customer":  ("DOC", "CUSTOMER"),
    "has_label":       ("DOC", "LABEL"),
    "reports_to":      ("PERSON", "PERSON"),
    "in_department":   ("PERSON", "DEPARTMENT"),
    "related_to":      ("DOC", "DOC"),
}


def alias_forms(full: str) -> set[str]:
    """The username conventions this corpus actually uses, derived by inspection of the data.

    Deliberately conservative: no fuzzy matching, no edit distance. A wrong merge fuses two
    real people into one and is far more damaging than a missed merge, which only leaves v1's
    existing behaviour in place.
    """
    parts = full.split()
    if len(parts) < 2:
        return set()
    a, b = parts[0].lower(), parts[-1].lower()
    if len(a) < 2 or len(b) < 2:
        return set()
    return {f"{a}.{b}", f"{a}_{b}", f"{a}-{b}", f"{a}{b}", f"{a[0]}{b}", f"{a}{b[0]}"}


# Slots whose occupant is structurally NOT a person, whatever v1's type table says.
# v1 typed by first-writer-wins, so an entity that is both a channel and a poster
# ('eng-oncall' posts automated alerts AND is the channel) came out as PERSON.
STRUCTURAL_SLOT = {"in_channel": "target", "in_team": "target", "in_repo": "target",
                   "about_customer": "target", "has_label": "target",
                   "in_department": "target"}


def structural_entities(rels: list[dict]) -> set[str]:
    """Everything that occupies a channel/team/repo/customer/label/department slot."""
    return {e[slot] for e in rels
            if (slot := STRUCTURAL_SLOT.get(e["relation"])) and e.get(slot)}


def build_alias_map(rels: list[dict], types: dict, structural: set[str]) -> dict[str, str]:
    """username -> canonical full name.

    Anchored on FULL NAMES THAT EXIST AS VERTICES, so the merge target is always a person the
    graph already knows. A username claimed by two different full names is dropped rather than
    guessed — ambiguity here silently corrupts answers.

    Structural entities are excluded from BOTH sides. Without that, 'eng-oncall' (a channel
    that also posts automated alerts, so v1 typed it PERSON) matched a person-shaped
    'Eng Oncall' and the merge RENAMED THE CHANNEL — every question whose answer contained
    eng-oncall silently changed its gold. Caught by the regold diff, not by any type check.
    """
    people = {e for e, t in types.items() if t == "PERSON"} - structural
    full = {p for p in people if " " in p and p[:1].isupper()}
    usernames = {p for p in people if " " not in p}
    ul = {u.lower(): u for u in usernames}

    claims: dict[str, set] = defaultdict(set)
    for f in full:
        for form in alias_forms(f):
            if form in ul:
                claims[ul[form]].add(f)

    amap, dropped = {}, 0
    for user, owners in claims.items():
        if len(owners) == 1:
            amap[user] = next(iter(owners))
        else:
            dropped += 1                 # e.g. 'jsmith' claimed by John Smith AND Jane Smith
    print(f"  alias map: {len(amap):,} usernames merged, {dropped} ambiguous dropped")
    return amap


def main() -> int:
    print("loading v1 index...")
    types = json.load(ropen(IN / "entity_types.json"))
    rels = json.load(ropen(IN / "relationships.json"))
    prov = json.load(ropen(IN / "edges_prov.json"))
    e2d = json.load(ropen(IN / "entity_to_docs.json"))
    print(f"  {len(types):,} entities, {len(rels):,} edges, {len(prov):,} provenance records")

    # ---- 1. canonicalise -------------------------------------------------------------
    print("\n[1] merging the person namespace")
    structural = structural_entities(rels)
    amap = build_alias_map(rels, types, structural)
    canon = lambda n: amap.get(n, n)                                      # noqa: E731

    # ---- 2. fold attribute relations out of the edge set -----------------------------
    print("\n[2] folding status / priority / doc_type into Document attributes")
    doc_attrs: dict[str, dict] = defaultdict(dict)
    for e in rels:
        if e["relation"] in ATTR_RELATIONS:
            doc_attrs[e["source"]][ATTR_RELATIONS[e["relation"]]] = e["target"]
    folded = sum(1 for e in rels if e["relation"] in ATTR_RELATIONS)
    print(f"  {folded:,} edges became attributes on {len(doc_attrs):,} documents")

    # ---- 3. rewrite edges ------------------------------------------------------------
    merged: dict[tuple, int] = Counter()
    for e in rels:
        if e["relation"] in ATTR_RELATIONS:
            continue
        merged[(canon(e["source"]), e["relation"], canon(e["target"]))] += e.get("count", 1)
    rels2 = [{"source": s, "relation": r, "target": t, "count": c}
             for (s, r, t), c in merged.items()]

    prov2, seen = [], set()
    for p in prov:
        if p["relation"] in ATTR_RELATIONS:
            continue
        s, t = canon(p["source"]), canon(p["target"])
        k = (s, p["relation"], t, p.get("chunk_id", ""))
        if k in seen:
            continue                          # merging creates exact duplicate provenance
        seen.add(k)
        prov2.append({**p, "source": s, "target": t})

    e2d2: dict[str, list] = defaultdict(list)
    for ent, docs in e2d.items():
        e2d2[canon(ent)].extend(docs)
    e2d2 = {k: sorted(set(v)) for k, v in e2d2.items()}

    # ---- 4. type every vertex from the relation it participates in -------------------
    print("\n[3] typing vertices")
    types2: dict[str, str] = {}
    for e in rels2:
        st, tt = TYPE_OF_RELATION.get(e["relation"], ("ENTITY", "ENTITY"))
        types2.setdefault(e["source"], st)
        types2.setdefault(e["target"], tt)
    # a PERSON identified anywhere stays a PERSON — authored says DOC for its target only
    for e in rels2:
        if e["relation"] in ("reports_to", "in_department", "authored", "participated_in"):
            types2[e["source"]] = "PERSON"
    # ...but a structural slot ALWAYS wins. 'eng-oncall' both posts and IS a channel; calling
    # it a person makes it a merge candidate and corrupts the channel namespace.
    STRUCT_TYPE = {"in_channel": "CHANNEL", "in_team": "TEAM", "in_repo": "REPO",
                   "about_customer": "CUSTOMER", "has_label": "LABEL",
                   "in_department": "DEPARTMENT"}
    for e in rels2:
        if (t := STRUCT_TYPE.get(e["relation"])):
            types2[e["target"]] = t

    print(f"\n  entities {len(types):,} -> {len(types2):,}   "
          f"({len(types)-len(types2):,} removed by merge + attribute folding)")
    print(f"  edges    {len(rels):,} -> {len(rels2):,}")
    print(f"  types    {dict(Counter(types2.values()).most_common())}")

    # ---- 5. schema v2 ----------------------------------------------------------------
    sch = json.load(ropen(SCHEMA_IN))
    keep = [r for r in sch["relations"] if r["relation"] not in ATTR_RELATIONS]
    tail = [r for r in sch.get("long_tail", []) if r["relation"] not in ATTR_RELATIONS]
    # coverage is unchanged by merging (it counts documents, not entities), but the person
    # relations now reach more of the graph, so recompute the edge counts that the router shows
    counts = Counter(e["relation"] for e in rels2)
    for r in keep + tail:
        r["edges"] = counts.get(r["relation"], 0)
    sch2 = {**sch, "relations": keep, "long_tail": tail,
            "doc_attributes": sorted(ATTR_RELATIONS.values()),
            "vertex_types": sorted(set(types2.values()))}

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(types2, wopen(OUT / "entity_types.json"))
    json.dump(rels2, wopen(OUT / "relationships.json"))
    json.dump(prov2, wopen(OUT / "edges_prov.json"))
    json.dump(e2d2, wopen(OUT / "entity_to_docs.json"))
    json.dump(dict(doc_attrs), wopen(OUT / "doc_attributes.json"))
    json.dump(amap, wopen(OUT / "alias_map.json"))
    json.dump({}, wopen(OUT / "company_facts.json"))
    json.dump({}, wopen(OUT / "events.json"))
    json.dump(sch2, wopen(SCHEMA_OUT), indent=2)

    # ---- 6. the number that justifies the rebuild ------------------------------------
    dirp = {e["source"] for e in rels2 if e["relation"] in ("reports_to", "in_department")}
    before = sum(1 for e in rels if e["relation"] == "participated_in" and e["source"] in dirp)
    after = sum(e["count"] for e in rels2
                if e["relation"] == "participated_in" and e["source"] in dirp)
    print(f"\n  participated_in reachable from a directory person: "
          f"{before:,} -> {after:,}  ({after/max(1,before):.1f}x)")
    print(f"\nwrote {OUT}/ and {SCHEMA_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
