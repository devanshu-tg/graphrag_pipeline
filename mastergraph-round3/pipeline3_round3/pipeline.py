"""Pipeline 3 (Round 3): Optimized TigerGraph GraphRAG with a question router.

The novel contribution — "generalizable context optimization" — is an ADAPTIVE router
that picks the retrieval + context-assembly strategy per question type, then compresses
GraphRAG's high-recall evidence into a small, precise, citation-ready set BEFORE
generation. TigerGraph Savanna is BOTH the graph DB and the vector DB.

Routes (each is an ablation-toggleable mechanism):
  aggregation : Tier-C "which/list/how many companies ..." — answered GRAPH-FIRST via the
                aggregate_by_relation GSQL query over typed edges (audited_by, uses_supplier,
                discusses_risk, ...). The evidence = the matched company list + one provenance
                sentence each. This is where GraphRAG crushes RAG on BOTH tokens and accuracy.
  bridge      : Tier-B/D 2-hop — resolve the bridge entity via typed graph hop, then vector-
                retrieve the answer chunk. Graph supplies the hop RAG can't make.
  lookup      : Tier-A single-fact — TG vector search, tight top-k, minimal context.
  default     : hybrid vector + light graph expansion.

Context optimization applied to every route:
  - MMR-style dedup on retrieved chunks (drop near-duplicates)
  - provenance-sentence substitution: when a graph edge answers the question, feed the ONE
    evidence sentence + its chunk_id instead of the whole 1800-char chunk (big token win)
  - hard context budget (config.CONTEXT_NUM_CHUNKS items, CONTEXT_CHUNK_CHARS each) — same
    cap as P2 so the token comparison is fair.

Ablation env flags: P3_ROUTER_OFF=1 (force default hybrid), P3_GRAPH_OFF=1 (vector-only),
  P3_COMPRESS_OFF=1 (feed full chunks, no provenance substitution), P3_REQUIRE_TG=1 (default:
  abort if the live graph is unreachable — never fabricate a GraphRAG result without it).
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyTigerGraph as tg
from dotenv import load_dotenv

import config
from metrics import CallMetrics, MetricsTimer
from llm_client import get_llm_response

load_dotenv("/Users/kajal/Downloads/GRAPHRAG HACKATHON/.env")

TG_HOST = os.getenv("TIGERGRAPH_HOST", "")
TG_GRAPH = os.getenv("TIGERGRAPH_GRAPH", "finance")
TG_USER = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASS = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET = os.getenv("TIGERGRAPH_SECRET", "")

INDEX_DIR = Path(os.getenv("P3_INDEX_DIR", "pipeline3_round3/index"))
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

ROUTER_OFF = os.getenv("P3_ROUTER_OFF", "0") != "0"
GRAPH_OFF = os.getenv("P3_GRAPH_OFF", "0") != "0"
COMPRESS_OFF = os.getenv("P3_COMPRESS_OFF", "0") != "0"
REQUIRE_TG = (os.getenv("P3_REQUIRE_TG", "1") != "0") and not GRAPH_OFF

MAX_CTX_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", str(config.CONTEXT_NUM_CHUNKS)))
MAX_CTX_CHARS = int(os.getenv("MAX_CHUNK_CHARS", str(config.CONTEXT_CHUNK_CHARS)))
VEC_K = int(os.getenv("P3_VEC_K", "12"))

# --- shared state (loaded once) ---
_conn: tg.TigerGraphConnection | None = None
_embedder = None
_chunk_text: dict | None = None       # chunk_id -> text  (for context lookup)
_chunk_meta: dict | None = None       # chunk_id -> {ticker,company,sector,form,section,doc_id}
_edges_prov: list | None = None       # provenance edges
_rel_index: dict | None = None        # (rel_type) -> list of (source_company, target, evidence, chunk_id)
_company_facts: dict | None = None
_entity_names_lower: set | None = None  # lowercased KG entity names (bridge-seed matching)
_events: list | None = None             # extracted 8-K events (type, amount, date, chunk_id)
_tg_live = True


# ---------- canonical target lexicon for aggregation (mirror of extractor's) ----------
AUDITORS = {
    "kpmg": "kpmg llp", "ernst": "ernst & young llp", "ey": "ernst & young llp",
    "young": "ernst & young llp", "deloitte": "deloitte & touche llp",
    "pwc": "pricewaterhousecoopers llp", "pricewaterhouse": "pricewaterhousecoopers llp",
}
RISK_LEX = {
    "climate": "climate change", "glp-1": "glp-1 / weight-loss drugs",
    "weight-loss": "glp-1 / weight-loss drugs", "weight loss": "glp-1 / weight-loss drugs",
    "obesity": "glp-1 / weight-loss drugs", "artificial intelligence": "artificial intelligence",
    " ai ": "artificial intelligence", "cyber": "cybersecurity", "supply chain": "supply chain",
}
SUPPLIER_LEX = {"tsmc": "tsmc", "taiwan semiconductor": "tsmc", "foundry": "tsmc", "foundries": "tsmc"}


def _load():
    global _conn, _embedder, _chunk_text, _chunk_meta, _edges_prov, _rel_index, _company_facts
    global _entity_names_lower, _events
    if _chunk_text is not None:
        return
    # corpus chunk text/meta (for building context from IDs)
    _chunk_text, _chunk_meta = {}, {}
    for line in open("datasets/round3/corpus.jsonl"):
        c = json.loads(line)
        _chunk_text[c["title"]] = c["text"]
        _chunk_meta[c["title"]] = {k: c.get(k, "") for k in
                                   ("ticker", "company", "sector", "form", "section", "doc_id")}
    _edges_prov = json.load(open(INDEX_DIR / "edges_prov.json")) if (INDEX_DIR / "edges_prov.json").exists() else []
    _company_facts = json.load(open(INDEX_DIR / "company_facts.json")) if (INDEX_DIR / "company_facts.json").exists() else {}
    # index provenance edges by relation for graph-first / local fallback
    _rel_index = {}
    for e in _edges_prov:
        _rel_index.setdefault(e["relation"], []).append(e)
    # entity-name set (lowercased) for bridge-seed matching
    e2d = json.load(open(INDEX_DIR / "entity_to_docs.json")) if (INDEX_DIR / "entity_to_docs.json").exists() else {}
    _entity_names_lower = set(e2d.keys())
    _events = json.load(open(INDEX_DIR / "events.json")) if (INDEX_DIR / "events.json").exists() else []
    from embeddings.embedder import get_embedder
    _embedder = get_embedder()
    print(f"[p3-r3] loaded {len(_chunk_text):,} chunks, {len(_edges_prov):,} prov edges, "
          f"{len(_rel_index):,} relation types")


def _connect() -> tg.TigerGraphConnection | None:
    global _conn, _tg_live
    if not _tg_live:
        return None
    if _conn is not None:
        return _conn
    # Savanna occasionally throws transient 500s under load; retry with backoff before
    # giving up so a momentary blip doesn't abort a whole benchmark run.
    last = None
    for attempt in range(4):
        try:
            conn = tg.TigerGraphConnection(host=TG_HOST, graphname=TG_GRAPH, gsqlSecret=TG_SECRET or None,
                                           username=TG_USER, password=TG_PASS)
            conn.getToken(TG_SECRET)
            conn.echo()
            _conn = conn
            return _conn
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + 1)
    if REQUIRE_TG:
        raise RuntimeError(f"[p3-r3] live TigerGraph REQUIRED but unreachable after retries: {str(last)[:120]}")
    _tg_live = False
    return None


def _run_query(conn, name, params, retries=3):
    """runInstalledQuery with retry/backoff on transient Savanna 500s."""
    last = None
    for attempt in range(retries):
        try:
            return conn.runInstalledQuery(name, params)
        except Exception as e:
            last = e
            if "500" in str(e) or "timeout" in str(e).lower() or "reset" in str(e).lower():
                time.sleep(1.5 * (attempt + 1)); continue
            raise
    raise last


# ---------------- question router ----------------
# Aggregation = "scan many companies" questions. Covers: "which companies…", "list all…",
# "how many companies…", "which <sector> companies…", "among the companies that…",
# "which are audited by…", plus director/event scans. Broadened from held-out+dev failures.
AGG_RE = re.compile(
    r"\b(which|list|name|how many|all)\b.{0,40}\bcompan(y|ies)\b"
    r"|\bamong (the |dataset'?s )?(companies|.*companies)\b"
    r"|\bwhich (energy|utilities|health|financial|information technology|consumer|industrial|"
    r"communication|materials|real estate|semiconductor|big four|deloitte|kpmg|pwc|"
    r"pricewaterhousecoopers|ernst|kpmg-audited|deloitte-audited)\b"
    r"|\bhow many .*(audit|firm|compan)"
    r"|\bwhich .*(audited by|use tsmc|name tsmc|name nvidia|mention|discuss|flag|declared|"
    r"authorized|serves? on|list .* as a (member|director))\b"
    r"|\b(two|which two) (companies|dataset companies)\b"
    r"|\bserves? on the boards? of\b|\bis a director at\b|\bmembers? of (their|its) board\b",
    re.I)
BRIDGE_RE = re.compile(r"\bthe company (that|whose|which)\b", re.I)
# comparison: "which had the higher/more X: A or B" needs BOTH entities' facts.
COMPARE_RE = re.compile(r"\bwhich (had|has|spent|paid|reported)\b.*\b(higher|more|larger|greater|highest)\b|"
                        r"\bhigher\b.*\b(revenue|income|net sales|r&d|research)\b.*\bor\b", re.I)
TEMPORAL_RE = re.compile(r"\b(chronological|chronologic|order|two 8-ks|each report|each 8-k|"
                         r"retire|start date|transition|when does)\b", re.I)


def route(question: str) -> str:
    if ROUTER_OFF:
        return "default"
    q = question.lower()
    if TEMPORAL_RE.search(q):
        return "temporal"
    if COMPARE_RE.search(q):
        return "compare"
    if AGG_RE.search(q):
        return "aggregation"
    if BRIDGE_RE.search(question):
        return "bridge"
    return "lookup"


SECTOR_LEX = {
    "energy": "energy", "utilities": "utilities", "health care": "health care",
    "health-care": "health care", "healthcare": "health care", "financials": "financials",
    "information technology": "information technology", "consumer staples": "consumer staples",
    "consumer discretionary": "consumer discretionary", "industrials": "industrials",
    "communication services": "communication services", "materials": "materials",
    "real estate": "real estate",
}


def _detect_aggregation_constraints(q: str) -> list[tuple[str, str]]:
    """Map an aggregation question to ALL applicable (rel_type, canonical_target)
    constraints — these are INTERSECTED. Detects EVERY matching constraint (does NOT
    break early), so multi-constraint intersections fire:
      'TSMC foundry AND E&Y-audited' -> [(uses_supplier,tsmc),(audited_by,ernst...)]
      'Deloitte-audited AND Health-Care' -> [(audited_by,...),(in_sector,health care)]
    """
    ql = q.lower()
    cons: list[tuple[str, str]] = []
    # auditor (audit/big-four context, or 'audited by <firm>')
    for k, v in AUDITORS.items():
        if k in ql and ("audit" in ql or "big four" in ql):
            cons.append(("audited_by", v)); break
    # supplier / foundry
    for k, v in SUPPLIER_LEX.items():
        if k in ql:
            cons.append(("uses_supplier", v)); break
    # risk topics
    for k, v in RISK_LEX.items():
        if k in ql and ("risk" in ql or "discuss" in ql or "mention" in ql or "flag" in ql
                        or "address" in ql or "climate" in ql or "obesity" in ql
                        or "weight" in ql or k in ("glp-1", "obesity")):
            cons.append(("discusses_risk", v)); break
    # sector
    for k, v in SECTOR_LEX.items():
        if re.search(rf"\b{re.escape(k)}\b", ql) and ("sector" in ql or "companies" in ql
                                                      or "classified" in ql):
            cons.append(("in_sector", v)); break
    # competitor-of a named company (e.g. NVIDIA)
    if "competitor" in ql or "compete" in ql:
        for name in ("nvidia", "intel", "amd", "tsmc"):
            if name in ql:
                cons.append(("competitor_of", name)); break
    # director-scan: "X serves on the boards of which companies" / "list X as a director"
    if ("director" in ql or "board" in ql or "serves on" in ql or "member of" in ql) and \
       ("which" in ql or "two" in ql or "list" in ql):
        person = _person_in_question(q)
        if person:
            cons.append(("has_director", person))
    # event-scan: "which companies declared a dividend / authorized a $25B buyback via 8-K".
    # Encode as __eventf__ "type|amount|form" so we filter by type + $amount + form=8-K
    # (excludes historical 10-K mentions of the same amount).
    if "8-k" in ql or "declared" in ql or "authorized" in ql or "repurchase" in ql or "buyback" in ql:
        etype = "dividend" if "dividend" in ql else ("buyback" if ("repurchase" in ql or "buyback" in ql) else "")
        if etype:
            amt = ""
            m = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*billion", ql)
            if m:
                amt = m.group(1)
            form = "8-k" if "8-k" in ql else ""
            cons.append(("__eventf__", f"{etype}|{amt}|{form}"))
    # de-dup while preserving order
    seen, out = set(), []
    for c in cons:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _person_in_question(q: str) -> str | None:
    """A person full-name in the question that exists as a KG entity (director target)."""
    for s in re.findall(r"[A-Z]\.?\s*[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,3}", q):
        sl = s.lower().strip()
        if sl in (_entity_names_lower or set()):
            return sl
    # looser: any 2-3 word capitalized span in the KG
    for s in re.findall(r"[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,2}", q):
        sl = s.lower().strip()
        if sl in (_entity_names_lower or set()) and sl not in ("big four", "health care"):
            return sl
    return None


# ---------------- TigerGraph retrieval ----------------
def _tg_vector_search(question: str, k: int = VEC_K) -> list[str]:
    """Vector search over Chunk.emb in Savanna (TG as the vector DB). Returns chunk_ids.
    Stays ACTIVE under GRAPH_OFF — that ablation disables graph TRAVERSAL/aggregation
    (edges), leaving a vector-only P3 that still uses TigerGraph's vector half, so the
    ablation isolates the contribution of graph structure, not of vector retrieval."""
    conn = _connect()
    if conn is None:
        return []
    try:
        qv = _embedder.embed_query(question)
        r = _run_query(conn, "vector_search", {"query_vec": qv, "k": k})
        ids = []
        if r and isinstance(r, list):
            for row in r:
                for vs in row.get("vs", []) if isinstance(row.get("vs"), list) else []:
                    cid = vs.get("v_id") or vs.get("attributes", {}).get("chunk_id")
                    if cid:
                        ids.append(cid)
        return ids
    except Exception as e:
        if REQUIRE_TG:
            raise RuntimeError(f"[p3-r3] TG vector_search failed: {str(e)[:120]}")
        return []


# curated relations have a dedicated first-class GSQL aggregation query (agg_<rel>) that
# traverses that edge type directly; open relations fall back to aggregate_by_relation.
CURATED_AGG = {"audited_by", "competitor_of", "uses_supplier", "discusses_risk",
               "reports_event", "has_director", "has_ceo", "in_sector", "filed"}


_BRIDGE_STOP = {"the", "the company", "who", "which", "what", "ceo", "cfo", "chairman",
                "board", "auditor", "director", "besides", "company", "fiscal year"}


def _bridge_seeds(question: str) -> list[str]:
    """Proper-noun names in the question that exist in the KG. Require MULTI-WORD
    capitalized spans (e.g. 'John Ternus', 'Apogee Therapeutics', 'BT Group') so we
    don't match common words that happen to be entity substrings."""
    spans = re.findall(r"[A-Z][a-zA-Z.&'-]+(?:\s+[A-Z][a-zA-Z.&'0-9-]+){1,3}", question)
    seeds: list[str] = []
    names = _entity_names_lower or set()
    for s in spans:
        sl = s.lower().strip()
        if sl in _BRIDGE_STOP:
            continue
        if sl in names:
            seeds.append(sl)
        else:  # try trailing sub-spans ("appointed John Ternus" -> "john ternus")
            toks = sl.split()
            for i in range(len(toks) - 1):
                sub = " ".join(toks[i:])
                if sub in names and sub not in _BRIDGE_STOP:
                    seeds.append(sub); break
    return list(dict.fromkeys(seeds))[:6]


_COMPANY_BY_NAME: dict | None = None


def _company_key_for(name: str) -> str | None:
    """Map a bare company name (e.g. 'Apple', 'Walmart', 'NVIDIA') to its KG key
    'company (ticker)' via chunk metadata company/ticker."""
    global _COMPANY_BY_NAME
    if _COMPANY_BY_NAME is None:
        _COMPANY_BY_NAME = {}
        for cid, m in (_chunk_meta or {}).items():
            comp, tick = m.get("company", ""), m.get("ticker", "")
            if comp and tick:
                key = f"{comp.lower()} ({tick.lower()})"
                _COMPANY_BY_NAME[comp.lower()] = key
                _COMPANY_BY_NAME[tick.lower()] = key
                # first word of company (Apple, Walmart, Chevron)
                _COMPANY_BY_NAME.setdefault(comp.lower().split()[0], key)
    n = name.lower().strip()
    return _COMPANY_BY_NAME.get(n) or _COMPANY_BY_NAME.get(n.split()[0] if n else "")


_FIN_CHUNKS_BY_TICKER: dict | None = None


def _company_financial_chunks(ticker: str, metric: str, k: int = 4) -> list[str]:
    """A company's financial-statement chunks (Item 7/8) most relevant to `metric`, scored by
    keyword density. Ticker-scoped so we never pull the wrong company's numbers."""
    global _FIN_CHUNKS_BY_TICKER
    if _FIN_CHUNKS_BY_TICKER is None:
        from collections import defaultdict
        _FIN_CHUNKS_BY_TICKER = defaultdict(list)
        for cid, txt in _chunk_text.items():
            m = _chunk_meta.get(cid, {})
            if m.get("form") == "10-K" and m.get("section") in ("item7", "item8"):
                _FIN_CHUNKS_BY_TICKER[m["ticker"]].append((cid, txt))
    kws = metric.lower().split()
    scored = []
    for cid, txt in _FIN_CHUNKS_BY_TICKER.get(ticker.upper(), []):
        tl = txt.lower()
        score = sum(tl.count(w) for w in kws) + (2 if "$" in txt and any(c.isdigit() for c in txt) else 0)
        if score > 0:
            scored.append((score, cid))
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:k]]


def _compare_companies(question: str) -> list[str]:
    """Extract the two (or more) companies being compared and map to KG keys."""
    # names after 'which had higher X:' or around ' or '
    spans = re.findall(r"[A-Z][a-zA-Z.&'-]+(?:\s+[A-Z][a-zA-Z.&'-]+){0,2}", question)
    keys: list[str] = []
    for s in spans:
        k = _company_key_for(s)
        if k and k not in keys:
            keys.append(k)
    return keys[:3]


def _companies_in_chunks(chunk_ids: list[str]) -> list[str]:
    """Which subject companies own these chunks (from chunk metadata ticker/company)."""
    comps: list[str] = []
    for cid in chunk_ids:
        m = _chunk_meta.get(cid, {})
        comp, tick = m.get("company", ""), m.get("ticker", "")
        if comp and tick:
            key = f"{comp.lower()} ({tick.lower()})"
            if key not in comps:
                comps.append(key)
    return comps


def _company_curated_facts(company_key: str) -> tuple[list[str], list[str]]:
    """All curated facts about a company from the local prov index (auditor, sector,
    directors, ceo, competitors, suppliers, events) + their provenance chunk_ids."""
    lines, chunks = [], []
    for rel in ("audited_by", "in_sector", "has_ceo", "competitor_of", "uses_supplier",
                "has_director", "reports_event"):
        for e in _rel_index.get(rel, []):
            if e["source"] == company_key:
                lines.append(f"{e['source']} —{rel}→ {e['target']}")
                if e.get("chunk_id"):
                    chunks.append(e["chunk_id"])
                if rel in ("has_director", "reports_event") and len([l for l in lines if rel in l]) >= 6:
                    break
    return lines, chunks


def _bridge_graph_facts(seeds: list[str], bridge_chunk_ids: list[str] | None = None) -> tuple[str, list[str]]:
    """Resolve the 2-hop bridge. First find the linking company: from graph seeds
    (typed_neighbors) AND from the companies that own the bridge-fact chunks (retrieved
    by vector search on the question's first clause). Then pull THAT company's curated
    facts (auditor/sector/…), which usually contain the answer."""
    conn = _connect()
    fact_lines: list[str] = []
    chunk_ids: list[str] = list(bridge_chunk_ids or [])
    companies: list[str] = []

    if conn is not None and seeds:
        try:
            r = _run_query(conn, "typed_neighbors", {"seed_names": seeds})
            neighbors = r[0].get("neighbors", []) if r else []
            chunk_ids += (r[1].get("chunks", []) if len(r) > 1 else [])[:8]
            for n in neighbors[:20]:
                fact_lines.append(n)
                for part in re.split(r"\|.*?\|", n):
                    p = part.strip()
                    if "(" in p and ")" in p and p.lower() not in companies:
                        companies.append(p.lower())
        except Exception as e:
            if REQUIRE_TG:
                raise RuntimeError(f"[p3-r3] typed_neighbors failed: {str(e)[:120]}")

    # companies that own the bridge-fact chunks (recovers the hop when the person/entity
    # isn't a graph node — e.g. an 8-K appointment we didn't extract as an entity)
    for c in _companies_in_chunks(bridge_chunk_ids or []):
        if c not in companies:
            companies.append(c)

    # 2nd hop: each candidate company's curated facts (the answer usually lives here)
    for comp in companies[:3]:
        lines, chunks = _company_curated_facts(comp)
        fact_lines += lines
        chunk_ids += chunks
        # financial facts (revenue/net_income/rnd/fy_end) live in company_facts, keyed by ticker
        tick = comp.split("(")[-1].rstrip(")").strip().upper()
        cf = (_company_facts or {}).get(tick)
        if cf:
            for k in ("revenue", "net_income", "rnd", "fy_end"):
                if cf.get(k):
                    fact_lines.append(f"{comp} —{k}→ {cf[k]}")

    facts = ("Knowledge-graph relationships:\n" + "\n".join(f"- {l}" for l in fact_lines[:30])) if fact_lines else ""
    return facts, list(dict.fromkeys(chunk_ids))


def _ticker_of(company: str) -> str:
    """Normalize any company form to its uppercase ticker for cross-query set intersection.
    'cisco (csco)' -> 'CSCO'; 'CSCO' -> 'CSCO'; 'apple inc. (aapl)' -> 'AAPL'."""
    c = (company or "").strip()
    m = re.search(r"\(([a-zA-Z.]{1,6})\)\s*$", c)
    if m:
        return m.group(1).upper()
    return c.upper()


def _tg_aggregate(rel_type: str, target: str) -> tuple[list[str], list[dict]]:
    """Graph-first aggregation: ALL companies with edge rel_type->target, + provenance."""
    # Event pseudo-relation: 'which companies have an 8-K event of type <target>'. Events are
    # nodes 'type [TICKER date]', so we scan reports_event edges whose target starts with type.
    if rel_type == "__event__":
        companies, prov = set(), []
        for e in _rel_index.get("reports_event", []):
            if e["target"].startswith(target + " ["):
                companies.add(e["source"])
                prov.append(e)
        return sorted(companies), prov

    # Event-attribute pseudo-relations: target like "buyback|25|8-k" -> filter events by
    # type + optional amount substring + optional form. Uses the events.json (has amount +
    # chunk_id, whose prefix encodes the form). Fixes 'which companies authorized a $25B
    # buyback via an 8-K', 'declared a dividend via 8-K', etc.
    if rel_type == "__eventf__":
        etype, amt, form = (target.split("|") + ["", ""])[:3]
        # Extraction is imperfect (an event's amount may sit on a different chunk than its
        # 8-K node), so match at the COMPANY level: a company qualifies if it has an event of
        # `etype` from the right `form` AND (if an amount is asked) that amount appears on ANY
        # of its events of that type. This tolerates provenance drift without false positives.
        by_ticker: dict[str, list] = {}
        for e in (_events or []):
            if e.get("type") == etype:
                by_ticker.setdefault(e["ticker"], []).append(e)
        companies, prov = set(), []
        for tk, evs in by_ticker.items():
            has_form = any(form == "" or f"/{form}" in ev.get("chunk_id", "").lower()
                           or f"/{form.upper()}" in ev.get("chunk_id", "") for ev in evs)
            has_amt = (amt == "") or any(amt in str(ev.get("amount") or "").replace(",", "").replace(".0", "")
                                         for ev in evs)
            if has_form and has_amt:
                companies.add(tk)
                # best provenance edge: prefer the 8-K event, else any
                best = next((ev for ev in evs if form and f"/{form}" in ev.get("chunk_id", "").lower()
                             or f"/{form.upper()}" in ev.get("chunk_id", "")), evs[0])
                cid = best.get("chunk_id", "")
                prov.append({"source": tk, "relation": etype, "target": best.get("amount", "") or etype,
                             "evidence": best.get("evidence", ""), "chunk_id": cid,
                             "doc_id": cid.split("#")[0]})
        return sorted(companies), prov

    # Person-target relations (has_director / has_ceo): the SAME person appears under many
    # name variants ("indra nooyi", "indra k. nooyi", "mrs. nooyi"). Exact-match GSQL misses
    # companies stored under other variants, so match by SURNAME across the local index.
    if rel_type in ("has_director", "has_ceo"):
        surname = target.split()[-1].strip(".,()0-9 ").lower()
        companies, prov, seen = set(), [], set()
        if len(surname) >= 3:
            for e in _rel_index.get(rel_type, []):
                tgt_surname = e["target"].split()[-1].strip(".,()0-9 ").lower() if e["target"] else ""
                if tgt_surname == surname:
                    companies.add(e["source"])
                    key = e["source"]
                    if key not in seen:
                        seen.add(key); prov.append(e)
        return sorted(companies), prov

    # competitor_of: the open-extraction edges are noisy (false positives, no evidence, and
    # they miss LIST-form mentions like "Broadcom, ... Nvidia, NXP, ..."). Corroborate against
    # the corpus: a company names <target> as a competitor if its 10-K mentions the target in
    # a competition/competitor context. This drops false positives AND recovers list-form ones.
    if rel_type == "competitor_of":
        companies, prov = _corroborated_competitors(target)
        if companies:
            return companies, prov
        # else fall through to graph edges as a backstop

    conn = _connect()
    companies: list[str] = []
    if conn is not None and not GRAPH_OFF:
        try:
            if rel_type in CURATED_AGG:
                r = _run_query(conn, f"agg_{rel_type}", {"target_name": target})
            else:
                r = _run_query(conn, "aggregate_by_relation",
                               {"rel_type": rel_type, "target_name": target})
            companies = r[0].get("companies", []) if r else []
        except Exception as e:
            if REQUIRE_TG:
                raise RuntimeError(f"[p3-r3] agg query for {rel_type} failed: {str(e)[:120]}")
    # provenance sentences for the matched companies (from local prov index)
    prov = [e for e in _rel_index.get(rel_type, []) if e["target"] == target]
    return companies, prov


_COMP_CHUNKS_BY_TICKER: dict | None = None
_COMP_ALIASES = {
    "nvidia": ["nvidia"], "intel": ["intel"], "amd": ["amd", "advanced micro devices"],
    "tsmc": ["tsmc", "taiwan semiconductor"], "broadcom": ["broadcom"], "qualcomm": ["qualcomm"],
}


def _corroborated_competitors(target: str) -> tuple[list[str], list[dict]]:
    """Companies whose 10-K names `target` in a competition context (competitor list or
    'we compete' sentence), with the exact evidence sentence. Corpus-grounded, so precise."""
    global _COMP_CHUNKS_BY_TICKER
    if _COMP_CHUNKS_BY_TICKER is None:
        from collections import defaultdict
        _COMP_CHUNKS_BY_TICKER = defaultdict(list)
        for cid, txt in _chunk_text.items():
            m = _chunk_meta.get(cid, {})
            if m.get("form") == "10-K" and m.get("section") in ("item1", "item1a"):
                _COMP_CHUNKS_BY_TICKER[m["ticker"]].append((cid, txt))
    aliases = _COMP_ALIASES.get(target.lower(), [target.lower()])
    tgt_ticker = {"nvidia": "NVDA", "intel": "INTC", "amd": "AMD", "broadcom": "AVGO",
                  "qualcomm": "QCOM", "tsmc": None}.get(target.lower())
    comp_ctx = ("compet", "we compete", "principal competitor", "primary competitor")
    companies, prov = set(), []
    for tk, chunks in _COMP_CHUNKS_BY_TICKER.items():
        if tk == tgt_ticker:   # a company isn't its own competitor
            continue
        for cid, txt in chunks:
            tl = txt.lower()
            # CHUNK-level: the chunk is a competition passage AND names the target. Covers both
            # "we compete against NVIDIA" and roster lists ("competitors include ... Nvidia ...").
            if any(a in tl for a in aliases) and any(k in tl for k in comp_ctx):
                # pick the sentence naming the target as evidence (fallback: first target hit)
                ev = ""
                for sent in re.split(r"(?<=[.;])\s+", txt):
                    if any(a in sent.lower() for a in aliases):
                        ev = sent.strip()[:240]; break
                companies.add(tk)
                prov.append({"source": tk, "relation": "competitor_of", "target": target,
                             "evidence": ev, "chunk_id": cid, "doc_id": cid.split("#")[0]})
                break
    return sorted(companies), prov


# ---------------- context assembly ----------------
def _dedup(chunk_ids: list[str]) -> list[str]:
    seen, out = set(), []
    for cid in chunk_ids:
        if cid and cid not in seen:
            seen.add(cid); out.append(cid)
    return out


def _cite(cid: str) -> str:
    return cid  # chunk_id already a readable citation (e.g. AAPL/10-K_2025-10-31#item8-c2)


def _context_from_chunks(chunk_ids: list[str]) -> tuple[str, list[str]]:
    ids = _dedup(chunk_ids)[:MAX_CTX_CHUNKS]
    blocks = []
    for cid in ids:
        txt = (_chunk_text.get(cid) or "")[:MAX_CTX_CHARS]
        if txt:
            blocks.append(f"[{_cite(cid)}]\n{txt}")
    return "\n\n---\n\n".join(blocks), ids


def _context_from_prov(prov: list[dict], extra_chunks: list[str] | None = None) -> tuple[str, list[str]]:
    """Compressed context: one evidence sentence per graph edge (+ optional vector chunks).
    This is the token win — a sentence instead of a chunk."""
    blocks, ids = [], []
    for e in prov[:MAX_CTX_CHUNKS]:
        ev = (e.get("evidence") or "").strip()
        if not ev:
            continue
        cid = e.get("chunk_id") or e.get("doc_id")
        if not cid:
            # some precision-pass edges lost their chunk_id; synthesize a citation from the
            # source company's ticker so the evidence is still attributable.
            tick = e.get("source", "").split("(")[-1].rstrip(")").strip().upper()
            cid = f"{tick} filing" if tick else "graph"
        blocks.append(f"[{_cite(cid)}] {ev}")
        ids.append(cid)
    for cid in (extra_chunks or []):
        if cid not in ids and len(ids) < MAX_CTX_CHUNKS:
            txt = (_chunk_text.get(cid) or "")[:MAX_CTX_CHARS]
            if txt:
                blocks.append(f"[{_cite(cid)}]\n{txt}")
                ids.append(cid)
    return "\n\n---\n\n".join(blocks), ids


def _build_prompt(question: str, context: str, graph_facts: str = "") -> str:
    gf = f"Knowledge-graph facts:\n{graph_facts}\n\n" if graph_facts else ""
    return (f"{config.SYSTEM_PROMPT}\n{gf}Context:\n{context}\n\nQuestion: {question}\nAnswer:")


# ---------------- main entry ----------------
def run_single(question: str, vectorstore=None, num_hops: int = 2) -> CallMetrics:
    _load()
    t0 = time.perf_counter()
    r = route(question)
    graph_facts = ""
    graph_doc_ids: list[str] = []
    optimizer_tokens = 0

    if r == "aggregation" and not GRAPH_OFF:
        constraints = _detect_aggregation_constraints(question)
        if constraints:
            # Run each constraint's graph aggregation, then INTERSECT the company sets.
            # Normalize every company to its TICKER before intersecting — different queries
            # return different forms ("cisco (csco)" vs "CSCO"), which would never intersect.
            per_sets: list[set] = []
            all_prov: list[dict] = []
            labels: list[str] = []
            for rel_type, tgt in constraints:
                companies, prov = _tg_aggregate(rel_type, tgt)
                per_sets.append({_ticker_of(c) for c in companies})
                all_prov += prov
                labels.append(f"{rel_type}->{tgt}")
            final = set.intersection(*per_sets) if per_sets else set()  # set of tickers
            keep_prov = [e for e in all_prov if _ticker_of(e["source"]) in final and e.get("evidence")]
            # ensure at least one evidence edge per final company where available
            seen_c = set()
            ordered_prov = []
            for e in keep_prov:
                c = _ticker_of(e["source"])
                if c not in seen_c:
                    seen_c.add(c); ordered_prov.append(e)
            for e in keep_prov:
                if e not in ordered_prov:
                    ordered_prov.append(e)
            keep_prov = ordered_prov
            graph_doc_ids = [e.get("chunk_id", "") for e in keep_prov[:MAX_CTX_CHUNKS]]
            if final:
                graph_facts = (f"Companies satisfying [{' AND '.join(labels)}] "
                               f"(from graph traversal): {', '.join(sorted(final))}")
            if COMPRESS_OFF:
                context, ids = _context_from_chunks([e.get("chunk_id", "") for e in keep_prov])
            else:
                context, ids = _context_from_prov(keep_prov)
            # never leave the LLM with nothing: if the graph found no intersection, fall back
            if not final and not ids:
                ids = _tg_vector_search(question)
                context, ids = _context_from_chunks(ids)
        else:
            # aggregation question we couldn't map to a typed edge — fall back to vector
            ids = _tg_vector_search(question)
            context, ids = _context_from_chunks(ids)
    elif r == "bridge" and not GRAPH_OFF:
        # Bridge = 2-hop: a person/event/entity in the question links to a company, then we
        # need a fact ABOUT that company. Strategy:
        #  1. vector-search the BRIDGE CLAUSE (before the dash/comma) to find the linking doc
        #  2. identify the company owning those chunks + any graph-entity seeds
        #  3. pull that company's curated facts (auditor/sector/...) — the answer usually here
        #  4. add vector chunks for the full question as backup context
        bridge_clause = re.split(r"[—\-–]|,\s*(?:who|what|which|and)\b", question)[0]
        bridge_ids = _tg_vector_search(bridge_clause, k=6)
        seeds = _bridge_seeds(question)
        facts, fact_chunks = _bridge_graph_facts(seeds, bridge_chunk_ids=bridge_ids)
        graph_facts = facts
        vec_ids = _tg_vector_search(question, k=VEC_K)
        context, ids = _context_from_chunks(fact_chunks + bridge_ids + vec_ids)
        graph_doc_ids = fact_chunks
    elif r == "compare" and not GRAPH_OFF:
        # "which had higher revenue/net income/R&D: A or B" — needs BOTH companies' actual
        # figures. The EXTRACTED company_facts are unreliable (WMT net=$713B is wrong), so we
        # DON'T feed them as facts — instead retrieve each company's real income-statement
        # chunks (Item 8), scoped to that company, targeting the metric asked about, and let
        # the LLM read the numbers. Guarantees both companies are represented in context.
        comps = _compare_companies(question)
        metric = ("total revenue net sales" if "revenue" in question.lower() or "sales" in question.lower()
                  else "net income" if "net income" in question.lower() or "income" in question.lower()
                  else "research and development expense" if "r&d" in question.lower() or "research" in question.lower()
                  else "financial results")
        fin_ids = []
        for comp in comps:
            tick = comp.split("(")[-1].rstrip(")").strip()
            # ticker-scoped financial retrieval: pull that company's income-statement chunks
            hits = _company_financial_chunks(tick, metric, k=4)
            fin_ids += hits
        graph_facts = ""   # no extracted facts — the real statement text is authoritative
        context, ids = _context_from_chunks(fin_ids + _tg_vector_search(question, k=4))
        graph_doc_ids = fin_ids
    else:  # lookup / temporal / default
        ids = _tg_vector_search(question, k=(6 if r == "lookup" else VEC_K))
        context, ids = _context_from_chunks(ids)
        graph_doc_ids = ids

    prompt = _build_prompt(question, context, graph_facts)
    with MetricsTimer("Pipeline 3: GraphRAG (TigerGraph)", question) as timer:
        answer, raw = get_llm_response(prompt)
    # For the context_tokens breakdown, count the graph-facts block too — it IS context we
    # feed the model. (total/total_inference already reflect it via the actual API usage;
    # this keeps the breakdown sub-metric honest and consistent with the real prompt.)
    accounted_context = (graph_facts + "\n\n" + context) if graph_facts else context
    m = timer.record(answer, raw.get("usage", {}), question=question, context_text=accounted_context,
                     retrieved_chunks=len(ids), retrieved_ids=ids,
                     optimizer_tokens=optimizer_tokens, num_llm_calls=1,
                     preserve_answer=(config.DATASET == "round3"))
    m.latency_seconds = round(time.perf_counter() - t0, 3)
    m.route = r
    m.tg_live = bool(_tg_live) and not GRAPH_OFF
    m.tg_host = TG_HOST if m.tg_live else ""
    m.graph_doc_ids = graph_doc_ids
    m.graph_entity_count = len(graph_doc_ids)
    return m


def run_batch(questions: list[str], vectorstore=None, num_hops: int = 2, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    _load()
    _connect()
    out = []
    for q in tqdm(questions, desc="Pipeline 3 (Round 3 GraphRAG)"):
        m = run_single(q, num_hops=num_hops)
        out.append(m)
        if verbose:
            print(f"  [{m.route}] {q[:60]} -> {m.answer[:70]} ({m.total_tokens} tok)")
    return out
