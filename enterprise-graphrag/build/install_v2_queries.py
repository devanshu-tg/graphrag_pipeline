"""Install GSQL queries on `enterprise_v2` so the typed graph is queryable, not decorative.

WHY THIS EXISTS
---------------
`build/ingest_typed_v2.py` created enterprise_v2 with 8 vertex types and 24 edge types (12
relations + their reverses) and loaded 38,397 vertices — then stopped. It verified itself with
an INTERPRET QUERY and installed nothing. Measured before this script ran:

    enterprise      9 installed queries    Entity + Chunk        73,613 chunk vectors
    enterprise_v2   0 installed queries    8 typed vertex types  no Chunk vertex
    finance        13 installed queries

Zero installed queries means nothing outside a GSQL shell can traverse it, so the typed graph
was demo-visual only. This installs the traversal primitives and gates each one on RETURNING
ROWS — the same gate as _evidence/CONTEXT.md 3, which exists because a previous load answered
metadata endpoints happily while every query returned nothing.

WHAT IS INSTALLED
-----------------
hop_<rel>_out / hop_<rel>_in for all 12 relations (24 queries). These are exactly the
primitive `pipeline3_round3/pipeline.py::_walk.step()` needs: given a SET of vertex names and
a relation + direction, return the neighbour names. Two directions rather than one undirected
query because direction carries meaning here — reports_to out is "my manager", in is
"my reports", and collapsing them merges an org chart into a blob.

Plus two queries that exist to show what the TYPING buys, because they cannot be written
against v1 at all:
  people_in_dept_in_channel  Person constrained BY KIND on both ends of a 3-hop path
  org_reach                  transitive closure down reports_to, Person-only by construction
In v1 every node is `Entity` with a label attribute, so "people in Product who post in
#incidents" cannot be constrained to people — it would also match documents and labels.

WHAT IS NOT INSTALLED
---------------------
vector_search. It requires `Chunk.emb`, and v2 has no Chunk vertex by design — copying 73,613
embeddings would take about an hour, double the instance's memory, and change no measurement.
Vector retrieval stays on `enterprise` for both P2 and P3. So v2 can serve TRAVERSAL only, and
P3_TG_TRAVERSE=1 wires exactly that: structure from v2, vectors from v1.

    python build/install_v2_queries.py            # install + gate
    python build/install_v2_queries.py --gate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pyTigerGraph as tg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, load_env, ropen   # noqa: E402

init_console()
load_env()

TG_HOST = os.getenv("TIGERGRAPH_HOST", "")
TG_USER = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASS = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET = os.getenv("TIGERGRAPH_SECRET", "")
GRAPH = os.getenv("TG_V2_GRAPH", "enterprise_v2")
INDEX = Path(os.getenv("TG_INDEX_V2_DIR", "pipeline3_round3/index_v2"))

VTYPES = ["PERSON", "DOC", "TEAM", "CHANNEL", "CUSTOMER", "LABEL", "REPO", "DEPARTMENT"]
VNAME = {t: f"{t.capitalize()}_{GRAPH}" for t in VTYPES}

# Mirrors build/ingest_typed_v2.py::EDGE_ENDS exactly. If these drift apart the queries compile
# and return nothing, which is the failure mode the gate below is here to catch.
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
    last = None
    for _ in range(5):
        try:
            conn.getToken(TG_SECRET)
            return conn
        except Exception as e:
            last = e
            time.sleep(3)
    raise SystemExit(f"cannot authenticate to TigerGraph: {str(last)[:160]}")


def _gsql(conn, stmt: str, what: str = "") -> str:
    for attempt in range(4):
        try:
            return str(conn.gsql(stmt))
        except Exception as e:
            if attempt == 3:
                return f"__ERROR__ {str(e)[:200]}"
            time.sleep(2 ** attempt)
    return ""


# --------------------------------------------------------------------- query text
def hop_queries() -> dict:
    """One query per (relation, direction). Matching is on entity_id, not name.

    entity_id is the PRIMARY_ID and therefore the exact string the local edge index uses;
    `name` was loaded truncated to 500 chars, so matching on it would silently miss the
    longest document paths.
    """
    out = {}
    for rel, (a, b) in EDGE_ENDS.items():
        fwd, rev = f"{rel}_{GRAPH}", f"{rel}_rev_{GRAPH}"
        out[f"hop_{rel}_out"] = f"""CREATE OR REPLACE QUERY hop_{rel}_out(SET<STRING> seeds) SYNTAX v3 {{
  SetAccum<STRING> @@nodes;
  SetAccum<STRING> @@pairs;
  r = SELECT t FROM (s:{VNAME[a]})-[e:{fwd}]->(t:{VNAME[b]})
      WHERE s.entity_id IN seeds
      ACCUM @@nodes += t.entity_id, @@pairs += (s.entity_id + " >> " + t.entity_id);
  PRINT @@nodes AS nodes, @@pairs AS pairs;
}}"""
        # the reverse edge type, not a re-query of the forward one: the whole point of
        # REVERSE_EDGE is that the backward hop is an index lookup rather than a scan
        out[f"hop_{rel}_in"] = f"""CREATE OR REPLACE QUERY hop_{rel}_in(SET<STRING> seeds) SYNTAX v3 {{
  SetAccum<STRING> @@nodes;
  SetAccum<STRING> @@pairs;
  r = SELECT t FROM (s:{VNAME[b]})-[e:{rev}]->(t:{VNAME[a]})
      WHERE s.entity_id IN seeds
      ACCUM @@nodes += t.entity_id, @@pairs += (t.entity_id + " >> " + s.entity_id);
  PRINT @@nodes AS nodes, @@pairs AS pairs;
}}"""
    return out


def showcase_queries() -> dict:
    """Two queries that are impossible to express on v1, where everything is one Entity type."""
    P, D, DEP, CH = VNAME["PERSON"], VNAME["DOC"], VNAME["DEPARTMENT"], VNAME["CHANNEL"]
    return {
        # Person -in_department-> Department  AND  Person -participated_in-> Doc -in_channel-> Channel
        # Both endpoints are constrained BY VERTEX TYPE. On v1 the same pattern also matches
        # labels, statuses and documents that happen to share a name with a person.
        "people_in_dept_in_channel": f"""CREATE OR REPLACE QUERY people_in_dept_in_channel(STRING dept, STRING channel) SYNTAX v3 {{
  SetAccum<STRING> @@people;
  SetAccum<STRING> @@docs;
  in_dept = SELECT p FROM (p:{P})-[:in_department_{GRAPH}]->(d:{DEP})
            WHERE lower(d.entity_id) == lower(dept);
  docs = SELECT dd FROM (dd:{D})-[:in_channel_{GRAPH}]->(c:{CH})
         WHERE lower(c.entity_id) == lower(channel)
         ACCUM @@docs += dd.entity_id;
  res = SELECT p FROM (p:in_dept)-[:participated_in_{GRAPH}]->(dd:docs)
        ACCUM @@people += p.entity_id;
  PRINT @@people AS people, @@people.size() AS n, @@docs.size() AS docs_in_channel;
}}""",
        # transitive closure down reports_to. Person-only by construction, so it cannot wander
        # into a document the way an untyped multi-hop on v1 does.
        "org_reach": f"""CREATE OR REPLACE QUERY org_reach(STRING manager, INT max_depth=6) SYNTAX v3 {{
  SetAccum<STRING> @@reach;
  seed = SELECT p FROM (p:{P}) WHERE lower(p.entity_id) == lower(manager);
  frontier = seed;
  WHILE frontier.size() > 0 LIMIT max_depth DO
    frontier = SELECT t FROM (s:frontier)-[:reports_to_rev_{GRAPH}]->(t:{P})
               WHERE t.entity_id NOT IN @@reach
               ACCUM @@reach += t.entity_id;
  END;
  PRINT @@reach AS reports, @@reach.size() AS n;
}}""",
    }


# --------------------------------------------------------------------- install
def install(conn, queries: dict) -> None:
    print(f"creating {len(queries)} queries on {GRAPH}...")
    body = "\n".join(queries.values())
    r = _gsql(conn, f"USE GRAPH {GRAPH}\n{body}", "create all")
    fails = [ln for ln in r.splitlines() if "error" in ln.lower() or "fail" in ln.lower()]
    print(f"  create: {len(queries) - len(fails)} ok" + (f", {len(fails)} problem lines" if fails else ""))
    for ln in fails[:12]:
        print(f"    {ln.strip()[:200]}")

    # One batch INSTALL rather than 26 separate ones: each INSTALL on Savanna costs ~30-60s of
    # codegen, so per-query installs would take ~20 minutes for the same result.
    print("installing (single batch — this takes a few minutes)...")
    t0 = time.time()
    r = _gsql(conn, f"USE GRAPH {GRAPH}\nINSTALL QUERY ALL", "install all")
    print(f"  install finished in {time.time() - t0:.0f}s")
    for ln in r.splitlines()[-6:]:
        if ln.strip():
            print(f"    {ln.strip()[:200]}")


# --------------------------------------------------------------------- gate
def gate(conn) -> int:
    """Every relation must return ROWS in both directions, using seeds taken from the index.

    A query that compiles and returns nothing is the exact failure this project has already
    been bitten by, so 'installed' is not the gate — 'returns rows' is.
    """
    rels = json.load(ropen(INDEX / "relationships.json"))
    by_rel: dict[str, list] = {}
    for e in rels:
        by_rel.setdefault(e["relation"], []).append(e)

    print(f"\nGATE — {len(EDGE_ENDS)} relations x 2 directions, seeds from {INDEX}")
    print(f"{'relation':<18}{'edges':>8}   {'out':>18}   {'in':>18}")
    print("-" * 70)
    ok = bad = skipped = 0
    for rel in EDGE_ENDS:
        items = by_rel.get(rel, [])
        if not items:
            print(f"{rel:<18}{0:>8}   {'(no edges loaded)':>18}")
            skipped += 1
            continue
        sample = items[: 8]
        cells = []
        for direction, key in (("out", "source"), ("in", "target")):
            seeds = sorted({e[key] for e in sample})
            try:
                r = conn.runInstalledQuery(f"hop_{rel}_{direction}", {"seeds": seeds})
                n = len((r[0].get("nodes") or []) if r else [])
            except Exception as ex:
                cells.append(f"ERR {str(ex)[:12]}")
                bad += 1
                continue
            cells.append(f"{n} nodes")
            if n:
                ok += 1
            else:
                bad += 1
        print(f"{rel:<18}{len(items):>8}   {cells[0]:>18}   {cells[1] if len(cells) > 1 else '':>18}")

    print(f"\n  {ok} directions returned rows, {bad} empty/error, {skipped} relations have no edges")

    # showcase queries
    print("\nSHOWCASE — typed constraints that v1 cannot express")
    try:
        r = conn.runInstalledQuery("people_in_dept_in_channel",
                                   {"dept": "Engineering", "channel": "eng-infra"})
        d = r[0] if r else {}
        print(f"  people_in_dept_in_channel(Engineering, eng-infra): "
              f"n={d.get('n')} docs_in_channel={d.get('docs_in_channel')} "
              f"sample={ (d.get('people') or [])[:4] }")
    except Exception as e:
        print(f"  people_in_dept_in_channel ERROR: {str(e)[:180]}")
        bad += 1
    try:
        mgr = next((e["target"] for e in by_rel.get("reports_to", [])), "")
        r = conn.runInstalledQuery("org_reach", {"manager": mgr, "max_depth": 6})
        d = r[0] if r else {}
        print(f"  org_reach({mgr!r}): n={d.get('n')} sample={ (d.get('reports') or [])[:4] }")
    except Exception as e:
        print(f"  org_reach ERROR: {str(e)[:180]}")
        bad += 1

    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true", help="skip install, just verify")
    args = ap.parse_args()

    conn = connect(GRAPH)
    if not args.gate_only:
        qs = {**hop_queries(), **showcase_queries()}
        install(conn, qs)
    rc = gate(conn)
    print("\nGATE PASSED" if rc == 0 else "\nGATE FAILED — do not use v2 for traversal yet")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
