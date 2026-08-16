"""Pipeline 3 — GraphRAG over TigerGraph, with SCHEMA-DRIVEN routing.

This is MasterGraph's Round-3 pipeline with exactly one layer replaced.

KEPT VERBATIM — all domain-free, all measured to work on the SEC run:
  * TigerGraph as both graph and vector store
  * graph-first answering for set questions        (SEC: Tier-C set-F1 0.843 vs 0.414 vector-only)
  * provenance-sentence substitution — ONE evidence sentence per edge instead of a whole
    chunk. This is where their 48% token reduction came from.
  * a hard context budget identical to P2's, so the token comparison stays fair
  * retry/backoff around Savanna, which throws transient 500s under load

REPLACED — the hand-written routing layer:
  Their router is ~400 lines of regex and lexicons naming the entities that appear in their
  eval questions ('kpmg', 'tsmc', 'name nvidia', 'climate change'). It routed 20/20 Tier-C
  questions on the set it was written against; on 5 held-out questions the ranking reversed.
  It is also the single biggest porting cost — worthless the moment the domain changes.

  Here the GRAPH publishes its relations and one LLM call maps the question onto them
  (schema_router.py). Nothing in this file names a company, person, channel or topic.

Ablation flags: P3_ROUTER_OFF=1 (vector only, no graph planning), P3_GRAPH_OFF=1,
  P3_COMPRESS_OFF=1 (full chunks instead of evidence sentences), P3_REQUIRE_TG=1 (default).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyTigerGraph as tg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import retrieval_common
import schema_router
from _portability import init_console, load_env, ropen
from llm_client import get_llm_response
from metrics import CallMetrics, MetricsTimer

init_console()
load_env()

TG_HOST = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPH = os.getenv("TIGERGRAPH_GRAPH", "enterprise")
TG_USER = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASS = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET = os.getenv("TIGERGRAPH_SECRET", "")

INDEX_DIR = Path(os.getenv("P3_INDEX_DIR", "pipeline3_round3/index"))
CORPUS = Path(os.getenv("P3_CORPUS", "datasets/enterprise/corpus.jsonl"))

ROUTER_OFF = os.getenv("P3_ROUTER_OFF", "0") != "0"
GRAPH_OFF = os.getenv("P3_GRAPH_OFF", "0") != "0"
COMPRESS_OFF = os.getenv("P3_COMPRESS_OFF", "0") != "0"
REQUIRE_TG = (os.getenv("P3_REQUIRE_TG", "1") != "0") and not GRAPH_OFF

MAX_CTX_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", str(config.CONTEXT_NUM_CHUNKS)))
MAX_CTX_CHARS = int(os.getenv("MAX_CHUNK_CHARS", str(config.CONTEXT_CHUNK_CHARS)))
# Must equal CONTEXT_NUM_CHUNKS, which is what P2 retrieves. It was 12 against P2's 10:
# both then cap at MAX_CTX_CHUNKS=10 AFTER dedup, so P3 had two spare candidates to
# refill from when duplicates were dropped and P2 had none. A small asymmetry, but it
# favoured the system under test, which is the one direction that is never acceptable.
VEC_K = int(os.getenv("P3_VEC_K", str(config.CONTEXT_NUM_CHUNKS)))
# How many of the 10 context slots traversal evidence may occupy when a graph route fires.
# The rest go to reranked vector chunks, so a WRONG plan can never crowd out the text that
# holds the answer — the failure that cost 26 content questions their answer entirely.
# Half is a deliberate default rather than a tuned one; the aggregation tier is re-run to
# confirm it does not cost anything there, since on those questions the answer travels in the
# `facts` traversal line rather than in the provenance sentences.
PROV_SLOTS = int(os.getenv("P3_PROV_SLOTS", str(max(1, config.CONTEXT_NUM_CHUNKS // 2))))
# Hard cap on how many VECTOR chunks join the context. Without it the context simply fills to
# MAX_CTX_CHUNKS, so a question where the graph found little gets the MOST chunks — which is why
# single_doc was the most expensive tier at 6,962 tokens while multi_hop cost 4,750. Measured
# retrieval coverage on the official content questions: top-3 keeps 91% of top-10's coverage for
# 30% of the context tokens, so capping trades a little recall for a lot of budget.
# Default equals MAX_CTX_CHUNKS, i.e. exactly the previous behaviour, so stored results stay
# reproducible and this knob has to be set deliberately.
VEC_SLOTS = int(os.getenv("P3_VEC_SLOTS", str(config.CONTEXT_NUM_CHUNKS)))

# --- graph-backed traversal (P3_TG_TRAVERSE=1) -------------------------------------------
# Default OFF, because the stored 50-question result was produced with the local adjacency and
# changing the default would silently invalidate it.
#
# With it ON, every hop of the walk becomes an installed GSQL query against the TYPED graph
# `enterprise_v2` (build/install_v2_queries.py) instead of a Python dict lookup. Vector search
# still runs on `enterprise` — v2 has no Chunk vertex, so it cannot serve vectors.
#
# Honest boundary, because it matters for what may be claimed: the graph supplies the
# STRUCTURE (which nodes a hop reaches). The evidence SENTENCE for each edge still comes from
# the local edges_prov index, because ingest_typed_v2.py loaded edges without their chunk_id —
# v2 holds no provenance. So this proves traversal runs in TigerGraph; it does not make the
# provenance graph-resident.
TG_TRAVERSE = os.getenv("P3_TG_TRAVERSE", "0") != "0"
TG_V2_GRAPH = os.getenv("TG_V2_GRAPH", "enterprise_v2")
# The same 800-node frontier cap the local walk uses, so a capped hop truncates identically in
# both modes. HOP_BATCH keeps one POST body a sane size.
HOP_CAP, HOP_BATCH = 800, 400
# Relations v2 has an edge type for. doc_type / has_status / has_priority are ATTRIBUTES of
# Document in v2, not vertices, so they have no edge to traverse — those hops fall back to the
# local table and are counted in _hop_stats so the fallback can never go unreported.
V2_RELS = {"authored", "participated_in", "assigned_to", "owned_by", "in_team", "in_channel",
           "in_repo", "about_customer", "has_label", "reports_to", "in_department", "related_to"}
_hop_stats = {"graph": 0, "local_fallback": 0, "capped": 0, "errors": 0}

_conn: tg.TigerGraphConnection | None = None
_conn_v2: tg.TigerGraphConnection | None = None
_embedder = None
_chunk_text: dict | None = None
_chunk_meta: dict | None = None
_edges_prov: list | None = None
_rel_index: dict | None = None
_tg_live = True


def _load() -> None:
    global _chunk_text, _chunk_meta, _edges_prov, _rel_index, _embedder
    if _chunk_text is not None:
        return
    _chunk_text, _chunk_meta = {}, {}
    for line in ropen(CORPUS):
        if not line.strip():
            continue
        c = json.loads(line)
        _chunk_text[c["title"]] = c["text"]
        _chunk_meta[c["title"]] = {k: c.get(k, "") for k in
                                   ("source", "company", "form", "section", "doc_id")}
    _edges_prov = json.load(ropen(INDEX_DIR / "edges_prov.json")) \
        if (INDEX_DIR / "edges_prov.json").exists() else []
    _rel_index = {}
    for e in _edges_prov:
        _rel_index.setdefault(e["relation"], []).append(e)
    from embeddings.embedder import get_embedder
    _embedder = get_embedder()
    print(f"[p3] loaded {len(_chunk_text):,} chunks, {len(_edges_prov):,} prov edges, "
          f"{len(_rel_index):,} relation types")


def _connect() -> tg.TigerGraphConnection | None:
    """Savanna throws transient 500s under load; retry before giving up."""
    global _conn, _tg_live
    if not _tg_live:
        return None
    if _conn is not None:
        return _conn
    last = None
    for attempt in range(4):
        try:
            conn = tg.TigerGraphConnection(host=TG_HOST, graphname=TG_GRAPH,
                                           gsqlSecret=TG_SECRET or None,
                                           username=TG_USER, password=TG_PASS)
            conn.getToken(TG_SECRET)
            conn.echo()
            _conn = conn
            return _conn
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + 1)
    if REQUIRE_TG:
        raise RuntimeError(f"[p3] live TigerGraph REQUIRED but unreachable: {str(last)[:140]}")
    _tg_live = False
    return None


# Savanna drops large responses mid-read under load. The original list caught 500/timeout/reset
# but not the urllib3 "Connection broken: IncompleteRead(...)" that a big payload actually
# produces, so a single truncated read aborted a whole 50-question run.
_TRANSIENT = ("500", "502", "503", "504", "timeout", "reset", "incompleteread",
              "connection broken", "connection aborted", "protocolerror", "remote end closed",
              "max retries", "temporarily unavailable")


def _run_query(conn, name, params, retries=4):
    last = None
    for attempt in range(retries):
        try:
            return conn.runInstalledQuery(name, params)
        except Exception as e:
            last = e
            if any(t in str(e).lower() for t in _TRANSIENT):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last


# ---------------------------------------------------------------- typed-graph traversal
def _connect_v2() -> tg.TigerGraphConnection | None:
    """Second connection, to the TYPED graph. Separate from _conn because the two graphs hold
    different things: enterprise has the chunk vectors, enterprise_v2 has the typed entities."""
    global _conn_v2
    if _conn_v2 is not None:
        return _conn_v2
    last = None
    for attempt in range(4):
        try:
            conn = tg.TigerGraphConnection(host=TG_HOST, graphname=TG_V2_GRAPH,
                                           gsqlSecret=TG_SECRET or None,
                                           username=TG_USER, password=TG_PASS)
            conn.getToken(TG_SECRET)
            conn.echo()
            _conn_v2 = conn
            return _conn_v2
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + 1)
    raise RuntimeError(f"[p3] P3_TG_TRAVERSE=1 but {TG_V2_GRAPH} unreachable: {str(last)[:140]}")


def _tg_hop(frontier: set, rel: str, direction: str) -> tuple[set, list[tuple[str, str]]]:
    """One hop, executed by TigerGraph: hop_<rel>_<out|in>(SET<STRING> seeds).

    usePost=True is not optional. pyTigerGraph serialises a SET parameter into the query STRING
    for GET, joining values with commas — and these seeds are document paths, so any value
    containing a comma would be split into two bogus ids. POST sends them as a JSON list.
    """
    conn = _connect_v2()
    seeds = sorted(frontier)
    if len(seeds) > HOP_CAP:
        _hop_stats["capped"] += 1
        seeds = seeds[:HOP_CAP]
    nodes: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(seeds), HOP_BATCH):
        batch = seeds[i:i + HOP_BATCH]
        try:
            r = conn.runInstalledQuery(f"hop_{rel}_{direction}", {"seeds": batch}, usePost=True)
        except Exception as e:
            _hop_stats["errors"] += 1
            if REQUIRE_TG:
                raise RuntimeError(f"[p3] hop_{rel}_{direction} failed: {str(e)[:140]}")
            return set(), []
        for row in (r or []):
            nodes.update(row.get("nodes") or [])
            for p in (row.get("pairs") or []):
                if " >> " in p:
                    a, b = p.split(" >> ", 1)
                    pairs.append((a, b))
    _hop_stats["graph"] += 1
    return nodes, pairs


# ---------------------------------------------------------------- retrieval
def _tg_vector_search(question: str, k: int = VEC_K) -> list[str]:
    """Vector search over Chunk.emb in Savanna — TigerGraph as the vector DB.

    Stays ACTIVE under GRAPH_OFF: that ablation disables graph TRAVERSAL, leaving a
    vector-only P3, so the ablation isolates graph structure rather than vector retrieval.
    """
    conn = _connect()
    if conn is None:
        return []
    try:
        qv = _embedder.embed_query(question)
        # vector_search_lite returns chunk IDS ONLY. The original vector_search ends in
        # `PRINT vs WITH VECTOR`, so every hit carries its full 384-float embedding — ~40KB at
        # k=10 but ~1.3MB at the reranker's k=200, and Savanna dropped the connection mid-read
        # (IncompleteRead: 838KB of 1.27MB). The pipeline only ever read v_id, so the vectors
        # were pure payload waste. Falls back to the original query if lite is not installed,
        # because this file has to keep working against a graph built before it existed.
        ids: list[str] = []
        try:
            r = _run_query(conn, "vector_search_lite", {"query_vec": qv, "k": k})
            for row in (r or []):
                ids.extend([c for c in (row.get("ids") or []) if c])
        except Exception:
            ids = []
        if not ids:
            r = _run_query(conn, "vector_search", {"query_vec": qv, "k": k})
            if r and isinstance(r, list):
                for row in r:
                    for vs in (row.get("vs") or []) if isinstance(row.get("vs"), list) else []:
                        cid = vs.get("v_id") or vs.get("attributes", {}).get("chunk_id")
                        if cid:
                            ids.append(cid)
        return ids
    except Exception as e:
        if REQUIRE_TG:
            raise RuntimeError(f"[p3] vector_search failed: {str(e)[:140]}")
        return []


def _curated_agg() -> set:
    return {r["relation"] for r in schema_router.graph_relations() if r["curated"]}


_fwd: dict | None = None      # rel -> source -> {targets}
_rev: dict | None = None      # rel -> target -> {sources}


def _adjacency():
    """Adjacency built once from the same edge records that were loaded into TigerGraph.

    A path walk is many small neighbour lookups; doing each as a REST round-trip to Savanna
    would be dominated by latency and would hit the transient-500 behaviour we already have
    retry logic for. The edges are identical to the graph's — the graph is still what
    supplies them.
    """
    global _fwd, _rev
    if _fwd is None:
        from collections import defaultdict
        _fwd, _rev = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
        for e in (_edges_prov or []):
            _fwd[e["relation"]][e["source"]].add(e["target"])
            _rev[e["relation"]][e["target"]].add(e["source"])
    return _fwd, _rev


_known_cache: set | None = None
_norm_index: dict | None = None     # punctuation-free form -> entity ids
_base_index: dict | None = None     # punctuation-free FILE NAME of a path -> entity ids

_SUFFIXES = (" slack channel", " channel", " department", " team", " dept",
             " document", " doc", " message", " thread", " page", " ticket")


def _nrm(s: str) -> str:
    """Strip everything that is not a letter or digit. 'eng-sre', 'eng sre' and 'Eng_SRE' all
    collapse to 'engsre', which is what makes loose mentions resolvable."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _build_resolver() -> None:
    global _known_cache, _norm_index, _base_index
    fwd, rev = _adjacency()
    _known_cache = {n for r in fwd for n in fwd[r]} | {n for r in rev for n in rev[r]}
    _norm_index, _base_index = {}, {}
    for n in _known_cache:
        _norm_index.setdefault(_nrm(n), []).append(n)
        # document ids are paths — index the bare file name too, because that is the part a
        # question ever names: "the document 'eval harness ci determinism'"
        base = str(n).rsplit("/", 1)[-1]
        if base != n:
            for ext in (".json", ".md", ".yaml", ".txt"):
                if base.endswith(ext):
                    base = base[: -len(ext)]
                    break
            _base_index.setdefault(_nrm(base), []).append(n)


def _resolve_seed(name: str) -> list[str]:
    """Match a seed the question spells loosely against entities the graph actually holds.

    Seeds arrive in whatever form the question used — '#incidents Slack channel' for the
    entity 'incidents', or the readable document title
    '20270425 selfsignup sdr handoff timezone compact'
    for the entity
    'slack/growth/20270425-selfsignup-sdr-handoff-timezone-compact.json'.

    The earlier version compared lowercased strings directly, so PUNCTUATION decided whether a
    seed resolved. Spaces in the question versus dashes in the filename meant every
    document-titled seed resolved to nothing, the walk returned empty, and P3 fell back to
    vector search. Measured cost: all 10 single-document questions routed to 'vector' — the
    graph was never consulted on the one tier where a single edge answers the question.

    Matching is ordered most-specific first so a loose substring can never outrank an exact
    name, and ties break toward the SHORTEST candidate (an exact channel over a longer one
    that merely contains it).
    """
    if _norm_index is None:
        _build_resolver()
    if name in _known_cache:
        return [name]

    s = name.strip().strip("'\"").lstrip("#")
    low = s.lower()
    for suf in _SUFFIXES:
        if low.endswith(suf):
            s = s[: -len(suf)].strip()
            break
    key = _nrm(s)
    if not key:
        return []

    # A mention may carry MORE path context than the entity id does. The planner copies the
    # path style it sees in the relation examples and emits
    #     'slack/1719877777 int8 kernel branching cpu lull probe'
    # for the entity
    #     slack/eng-runtime/1719877777-int8-kernel-branching-cpu-lull-probe.json
    # The invented 'slack/' makes the seed LONGER than the file name, so testing only
    # "is the seed inside the entity" fails. Measured before this: the planner added a prefix
    # on 4 of 4 attempts for one question and ~2 of 4 for two others, which is precisely the
    # 2-question run-to-run wobble on the single-document tier. Try the last path segment too,
    # and test containment in BOTH directions.
    keys = [key]
    if "/" in s:
        tail = _nrm(s.rsplit("/", 1)[-1])
        if tail and tail != key:
            keys.append(tail)

    # The two containment directions need OPPOSITE tie-breaks, and conflating them cost real
    # accuracy: with one shared "shortest first" rule, a 6-character entity name sitting inside
    # a long seed outranked the correct 40-character one, and 'name.json' fell from 100% to 83%.
    #   seed INSIDE entity  (partial title)  -> prefer the SHORTEST entity: least extra text
    #   entity INSIDE seed  (seed has extra) -> prefer the LONGEST entity: explains most of it
    # Alphabetical is the final tie-break so a result never depends on set iteration order.
    def short_first(hits):
        return sorted(set(hits), key=lambda n: (len(n), n))[:2]

    def long_first(hits):
        return sorted(set(hits), key=lambda n: (-len(n), n))[:2]

    for k in keys:                                    # 1. exact name / 2. exact file name
        if k in _norm_index:
            return short_first(_norm_index[k])
        if k in _base_index:
            return short_first(_base_index[k])
    for k in keys:                                    # 3. seed is part of a file name
        hits = [n for bk, v in _base_index.items() if k in bk for n in v]
        if hits:
            return short_first(hits)
    for k in keys:                                    # 4. file name is part of the seed
        hits = [n for bk, v in _base_index.items() if len(bk) > 3 and bk in k for n in v]
        if hits:
            return long_first(hits)
    for k in keys:                                    # 5. last resort: any entity name
        hits = [n for nk, v in _norm_index.items() if k in nk for n in v]
        if hits:
            return short_first(hits)
    return []


def _walk(plan: dict) -> tuple[list[str], list[dict]]:
    """Execute a plan of LEGS -> COMBINE -> TAIL.

    Two measured failures killed the previous {seeds, path} model, where one path was applied
    to every seed:

    1. Seeds can need DIFFERENT routes to the same destination. "Which people from the Product
       department take part in #incidents?" reaches PEOPLE from 'Product' via in_department:in
       but from '#incidents' via in_channel:in -> participated_in:in. No single path serves
       both, so the planner returned an empty plan on 2 of 5 such questions and P3 silently
       degraded to Basic RAG — 0/5 on that whole shape.

    2. The intersection belongs at the hop the question names, not at the end. "Channels
       containing documents involving BOTH A and B" means shared DOCUMENTS; intersecting the
       final channel sets also accepts a channel where A and B each have their own separate
       document. Measured on MH01: that admits eng-releases, which the gold excludes.

    So each leg walks its own path, the legs are combined, and the tail runs afterwards —
    which places the intersection exactly where the question puts it.

    A leg that resolves but walks to nothing is DROPPED, not intersected to nothing: an empty
    leg means the planned route does not apply to that seed, and treating it as "reached
    nothing" would erase a correct answer the other legs found.
    """
    fwd, rev = _adjacency()
    trail: list[dict] = []

    def step(frontier: set, path: list) -> set:
        for rel, direction in path:
            table = fwd[rel] if direction == "out" else rev[rel]
            # sorted() only in traverse mode: the local default must stay byte-identical to the
            # run that produced results/enterprise/eval_ent50.jsonl, but the cross-check below
            # is only meaningful if both modes start from the SAME capped subset.
            seeds = (sorted(frontier) if TG_TRAVERSE else list(frontier))[:HOP_CAP]
            local: set[str] = set()
            for node in seeds:
                local.update(table.get(node, ()))
            if TG_TRAVERSE and rel in V2_RELS:
                nxt, _ = _tg_hop(frontier, rel, direction)
                # The typed graph and this local adjacency were loaded from the SAME edge
                # records, so a disagreement is a load or query bug, not a difference of
                # opinion. Counted rather than raised, so one flaky hop cannot abort a
                # 50-question run — and reported at the end so it cannot go unnoticed either.
                k = "agree" if nxt == local else "disagree"
                _hop_stats[k] = _hop_stats.get(k, 0) + 1
                if nxt != local:
                    _hop_stats.setdefault("diff_examples", []).append(
                        {"rel": rel, "dir": direction, "seeds": len(seeds),
                         "graph_only": sorted(nxt - local)[:3],
                         "local_only": sorted(local - nxt)[:3]})
            else:
                if TG_TRAVERSE:
                    _hop_stats["local_fallback"] += 1   # relation has no edge type in v2
                nxt = local
            for e in _rel_index.get(rel, []):
                a, b = ((e["source"], e["target"]) if direction == "out"
                        else (e["target"], e["source"]))
                if a in frontier and b in nxt and len(trail) < MAX_CTX_CHUNKS * 4:
                    trail.append(e)
            frontier = nxt
            if not frontier:
                break
        return frontier

    landings: list[set] = []
    for leg in plan.get("legs", []):
        start = set(_resolve_seed(leg["seed"]))
        if not start:
            continue
        got = step(start, leg["path"])
        if got:
            landings.append(got)
    if not landings:
        return [], []

    combined = (set.intersection(*landings)
                if plan.get("combine") == "intersect" and len(landings) > 1
                else set().union(*landings))
    final = step(combined, plan.get("tail") or []) if plan.get("tail") else combined
    # keep provenance that actually supports the surviving answer
    trail = [e for e in trail if e["source"] in final or e["target"] in final] or trail
    return sorted(final), trail


# ---------------------------------------------------------------- context
def _dedup(ids: list[str]) -> list[str]:
    seen, out = set(), []
    for cid in ids:
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _context_from_chunks(chunk_ids: list[str]) -> tuple[str, list[str]]:
    ids = _dedup(chunk_ids)[:MAX_CTX_CHUNKS]
    blocks = []
    for cid in ids:
        txt = (_chunk_text.get(cid) or "")[:MAX_CTX_CHARS]
        if txt:
            blocks.append(f"[{cid}]\n{txt}")
    return "\n\n---\n\n".join(blocks), ids


def _context_from_prov(prov: list[dict], extra: list[str] | None = None) -> tuple[str, list[str]]:
    """Compressed context: one evidence sentence per graph edge. THE TOKEN WIN."""
    blocks, ids = [], []
    for e in prov[:MAX_CTX_CHUNKS]:
        ev = (e.get("evidence") or "").strip()
        if not ev:
            continue
        cid = e.get("chunk_id") or e.get("doc_id") or "graph"
        blocks.append(f"[{cid}] {ev}")
        ids.append(cid)
    for cid in (extra or []):
        if cid not in ids and len(ids) < MAX_CTX_CHUNKS:
            txt = (_chunk_text.get(cid) or "")[:MAX_CTX_CHARS]
            if txt:
                blocks.append(f"[{cid}]\n{txt}")
                ids.append(cid)
    return "\n\n---\n\n".join(blocks), ids


def _build_prompt(question: str, context: str, facts: str = "") -> str:
    gf = f"Knowledge-graph facts:\n{facts}\n\n" if facts else ""
    return f"{config.SYSTEM_PROMPT}\n{gf}Context:\n{context}\n\nQuestion: {question}\nAnswer:"


# ---------------------------------------------------------------- main entry
def run_single(question: str, vectorstore=None, num_hops: int = 2) -> CallMetrics:
    _load()
    t0 = time.perf_counter()
    facts, graph_doc_ids, route_name = "", [], "vector"

    plan = {"legs": [], "combine": "union", "tail": [], "intent": "lookup"}
    if not ROUTER_OFF and not GRAPH_OFF:
        plan = schema_router.plan(question, get_llm_response)

    ids: list[str] = []
    context = ""
    trail: list[dict] = []          # stays empty when the planner returns no legs
    if plan["legs"]:
        landed, trail = _walk(plan)
        if landed:
            legs = " ; ".join(
                f"{leg['seed']} -> " + " -> ".join(f"{r}({d})" for r, d in leg["path"])
                for leg in plan["legs"])
            tail = " -> ".join(f"{r}({d})" for r, d in plan["tail"])
            depth = max(len(leg["path"]) for leg in plan["legs"]) + len(plan["tail"])
            route_name = f"graph:{depth}hop"
            facts = (f"Graph traversal [{legs}]"
                     + (f" {plan['combine']} then {tail}" if tail else
                        f" {plan['combine']}" if len(plan["legs"]) > 1 else "")
                     + f" ({len(landed)} results): {', '.join(landed[:60])}")
            graph_doc_ids = [e.get("chunk_id", "") for e in trail[:MAX_CTX_CHUNKS]]

    # UNION, never replacement — and now the code actually does that.
    #
    # It used to read `if not ids:`, so vector search ran ONLY when the graph returned nothing.
    # When the graph returned something WRONG the answering text was never fetched at all.
    # Measured on the official 200 content questions: 26 were routed to the graph, and on those
    # retrieval hit@k collapsed to 22.9% and the score to 9.6%, against Basic RAG's 46.8% on the
    # same questions. The comment above has claimed "union, never replacement" since the design
    # document; the implementation did the opposite.
    #
    # Vector search now ALWAYS runs, and is reranked with the same cross-encoder and the same
    # RR_POOL/RR_KEEP the baseline uses (retrieval_common), because the fallback previously took
    # plain top-10 while the arm it is compared against reranked 50 -> 10. That asymmetry cost
    # 8.4 points on the 166 questions that had already fallen back, and it favoured the
    # BASELINE, which is the only direction that is never acceptable to leave in place.
    #
    # Budget stays identical to P2's: at most MAX_CTX_CHUNKS items total. Traversal evidence is
    # capped at PROV_SLOTS so a wrong plan can never crowd out the chunk holding the answer;
    # the remaining slots are filled with reranked chunks. On aggregation questions the answer
    # travels in the `facts` line, not in the provenance sentences, so capping them is safe —
    # and both benchmarks are re-run to confirm that rather than assume it.
    pool = _tg_vector_search(question, k=max(VEC_K, retrieval_common.RR_POOL))
    vec_ids = retrieval_common.rerank(question, pool, _chunk_text.get,
                                      keep=VEC_SLOTS, max_chars=MAX_CTX_CHARS)
    if facts:
        keep_prov = trail[:PROV_SLOTS]
        if COMPRESS_OFF:
            context, ids = _context_from_chunks(
                [e.get("chunk_id", "") for e in trail] + vec_ids)
        else:
            context, ids = _context_from_prov(keep_prov, extra=vec_ids)
    else:
        context, ids = _context_from_chunks(vec_ids)
    if not graph_doc_ids:
        graph_doc_ids = ids

    prompt = _build_prompt(question, context, facts)
    with MetricsTimer("Pipeline 3: GraphRAG (schema-driven)", question) as timer:
        answer, raw = get_llm_response(prompt)

    accounted = (facts + "\n\n" + context) if facts else context
    m = timer.record(answer, raw.get("usage", {}), question=question, context_text=accounted,
                     retrieved_chunks=len(ids), retrieved_ids=ids,
                     optimizer_tokens=0, num_llm_calls=1,
                     preserve_answer=True)
    m.latency_seconds = round(time.perf_counter() - t0, 3)
    m.route = route_name
    m.tg_live = bool(_tg_live) and not GRAPH_OFF
    m.tg_host = TG_HOST if m.tg_live else ""
    m.graph_doc_ids = graph_doc_ids
    m.graph_entity_count = len(graph_doc_ids)
    return m


def run_batch(questions: list[str], vectorstore=None, num_hops: int = 2,
              verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    _load()
    _connect()
    out = []
    for q in tqdm(questions, desc="Pipeline 3 (schema-driven GraphRAG)"):
        m = run_single(q)
        out.append(m)
        if verbose:
            print(f"  [{m.route}] {q[:60]} -> {m.answer[:70]} ({m.total_tokens} tok)")
    return out
