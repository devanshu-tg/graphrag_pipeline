"""Load the v2 index into TigerGraph as a genuinely TYPED graph.

WHY A SEPARATE GRAPH
--------------------
`enterprise` (v1) holds one vertex type, Entity, for all 38,899 nodes plus Chunk. That is an
edge list with a label column, not a knowledge graph: nothing can be constrained by kind, the
schema tells a reader nothing, and a demo of it shows two boxes.

`enterprise_v2` gives each kind its own vertex type — Person, Document, Team, Channel,
Customer, Label, Repo, Department — and each relation its own directed edge type with the
correct FROM/TO. One edge type per relation is not cosmetic: TigerGraph keys edges by
(from, to), so a single generic edge type silently collapses two distinct relations between
the same pair into one. That cost real data on the SEC run.

CHUNKS ARE NOT COPIED. The 73,613 chunk vectors stay in `enterprise`, which is where both
P2 and P3 do vector search. Re-embedding and re-uploading them into a second graph would take
about an hour, double the instance's memory footprint, and change nothing about the
measurement — the entity graph is what is typed here.

    python build/ingest_typed_v2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyTigerGraph as tg
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, load_env, ropen   # noqa: E402

init_console()
load_env()

TG_HOST = os.getenv("TIGERGRAPH_HOST", "")
TG_USER = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASS = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET = os.getenv("TIGERGRAPH_SECRET", "")
GRAPH = os.getenv("TG_V2_GRAPH", "enterprise_v2")
INDEX = Path(os.getenv("TG_INDEX_DIR", "pipeline3_round3/index_v2"))

VERTEX_BATCH, EDGE_BATCH, RETRIES = 2000, 1000, 8

# vertex type -> TigerGraph vertex name
VTYPES = ["PERSON", "DOC", "TEAM", "CHANNEL", "CUSTOMER", "LABEL", "REPO", "DEPARTMENT"]
VNAME = {t: f"{t.capitalize()}_{GRAPH}" for t in VTYPES}

# relation -> (FROM type, TO type). Mirrors build/canonicalize.py exactly.
EDGE_ENDS = {
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


def connect(graphname: str = ""):
    conn = tg.TigerGraphConnection(host=TG_HOST, graphname=graphname or GRAPH,
                                   gsqlSecret=TG_SECRET or None, username=TG_USER,
                                   password=TG_PASS)
    for _ in range(5):
        try:
            conn.getToken(TG_SECRET)
            break
        except Exception:
            time.sleep(3)
    return conn


def _gsql(conn, stmt: str, what: str = "") -> str:
    for attempt in range(4):
        try:
            return str(conn.gsql(stmt))
        except Exception as e:
            if attempt == 3:
                print(f"  gsql {what} error: {str(e)[:160]}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def build_schema(conn) -> None:
    """Person/Document/Team/... plus one directed edge type per relation.

    `status`, `priority` and `doc_type` are ATTRIBUTES of Document here, not vertices. As
    vertices in v1 they created hub nodes with thousands of neighbours (every open ticket
    pointing at one 'open' node), which is both meaningless as a fact and ruinous for fan-out.
    """
    verts = "\n".join(
        f'CREATE VERTEX {VNAME[t]} (PRIMARY_ID entity_id STRING, name STRING DEFAULT "", '
        f'mention_count INT DEFAULT 0'
        + (', status STRING DEFAULT "", priority STRING DEFAULT "", doc_type STRING DEFAULT ""'
           if t == "DOC" else "")
        + ') WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"'
        for t in VTYPES)
    edges = "\n".join(
        f'CREATE DIRECTED EDGE {rel}_{GRAPH} (FROM {VNAME[a]}, TO {VNAME[b]}, '
        f'rel_count INT DEFAULT 1, chunk_id STRING DEFAULT "") WITH REVERSE_EDGE="{rel}_rev_{GRAPH}"'
        for rel, (a, b) in EDGE_ENDS.items())
    names = ", ".join([VNAME[t] for t in VTYPES]
                      + [f"{r}_{GRAPH}" for r in EDGE_ENDS]
                      + [f"{r}_rev_{GRAPH}" for r in EDGE_ENDS])
    stmt = f"USE GLOBAL\n{verts}\n{edges}\nCREATE GRAPH {GRAPH}({names})"
    print(f"creating graph {GRAPH}: {len(VTYPES)} vertex types, {len(EDGE_ENDS)} edge types")
    print(_gsql(conn, stmt, "schema")[:400])


def _post(url, payload, headers, what) -> int:
    for attempt in range(RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=90)
            if r.status_code == 200 and not r.json().get("error", True):
                res = r.json().get("results", [{}])[0]
                return res.get("accepted_vertices", 0) + res.get("accepted_edges", 0)
            if attempt == RETRIES - 1:
                print(f"    {what} failed: {str(r.text)[:140]}")
        except Exception as e:
            if attempt == RETRIES - 1:
                print(f"    {what} exc: {str(e)[:100]}")
        time.sleep(min(2 ** attempt, 20))
    return 0


def main() -> int:
    types = json.load(ropen(INDEX / "entity_types.json"))
    rels = json.load(ropen(INDEX / "relationships.json"))
    e2d = json.load(ropen(INDEX / "entity_to_docs.json"))
    attrs = json.load(ropen(INDEX / "doc_attributes.json"))
    print(f"{len(types):,} entities, {len(rels):,} edges, {len(attrs):,} documents with attributes")

    conn = connect("")
    build_schema(conn)
    conn = connect(GRAPH)
    url = f"{TG_HOST}/restpp/graph/{GRAPH}"
    headers = {"Authorization": f"Bearer {conn.getToken(TG_SECRET)[0]}"}

    # ---- vertices, bucketed by type ----
    by_type: dict[str, list] = defaultdict(list)
    for name, t in types.items():
        by_type[t if t in VNAME else "DOC"].append(name)
    total_v = 0
    for t, names in by_type.items():
        for i in range(0, len(names), VERTEX_BATCH):
            batch = names[i:i + VERTEX_BATCH]
            verts = {}
            for n in batch:
                a = {"name": {"value": n[:500]},
                     "mention_count": {"value": len(e2d.get(n, []))}}
                if t == "DOC" and n in attrs:
                    for k in ("status", "priority", "doc_type"):
                        if attrs[n].get(k):
                            a[k] = {"value": str(attrs[n][k])[:120]}
                verts[n] = a
            total_v += _post(url, {"vertices": {VNAME[t]: verts}}, headers, f"{t} verts")
        print(f"  {VNAME[t]:<24} {len(names):>7,}")
    print(f"  vertices accepted: {total_v:,}")

    # ---- edges, bucketed by relation ----
    by_rel: dict[str, list] = defaultdict(list)
    for e in rels:
        if e["relation"] in EDGE_ENDS:
            by_rel[e["relation"]].append(e)
    total_e = 0
    for rel, items in by_rel.items():
        a, b = EDGE_ENDS[rel]
        for i in range(0, len(items), EDGE_BATCH):
            payload = {"edges": {VNAME[a]: {}}}
            bucket = payload["edges"][VNAME[a]]
            for e in items[i:i + EDGE_BATCH]:
                bucket.setdefault(e["source"], {}).setdefault(f"{rel}_{GRAPH}", {}) \
                      .setdefault(VNAME[b], {})[e["target"]] = {
                          "rel_count": {"value": e.get("count", 1)}}
            total_e += _post(url, payload, headers, f"{rel} edges")
        print(f"  {rel:<20} {len(items):>7,}")
    print(f"  edges accepted: {total_e:,}")

    # ---- gate: a query that RETURNS ROWS, not an echo ----
    print("\nGATE — typed neighbour lookup")
    person = next((e["source"] for e in rels if e["relation"] == "reports_to"), "")
    q = (f'INTERPRET QUERY () FOR GRAPH {GRAPH} {{\n'
         f'  s = {{{VNAME["PERSON"]}.*}};\n'
         f'  r = SELECT t FROM s:v -(reports_to_{GRAPH}:e)- {VNAME["PERSON"]}:t '
         f'WHERE v.name == "{person}";\n'
         f'  PRINT r;\n}}')
    out = _gsql(conn, q, "gate")
    print(f"  reports_to from {person!r}: {out[:300]}")
    print(f"\ntypes loaded: {dict(Counter(types.values()).most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
