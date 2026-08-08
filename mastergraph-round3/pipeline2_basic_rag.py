"""Pipeline 2: Basic RAG — embed question → retrieve chunks → LLM. Primary baseline."""

from __future__ import annotations
import os
import time

from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from metrics import CallMetrics, MetricsTimer
from llm_client import get_llm_response
import config

# Retrieval mode default: "similarity" (pure cosine top-k, no MMR, no reranker) — adopted
# per the P2 ablation study (Round 3/P2_ABLATION.md): beats mmr_rerank on strict pass,
# graded quality, AND BERTScore simultaneously, while dropping the cross-encoder dependency
# and cutting latency ~12%. All retrieval knobs are compute-only (no LLM tokens) and the
# final context stays capped at CONTEXT_NUM_CHUNKS × CONTEXT_CHUNK_CHARS — so switching modes
# keeps the P2/P3 token comparison fair.
P2_RERANK        = os.getenv("P2_RERANK", "0") != "0"
RERANK_POOL      = int(os.getenv("P2_RERANK_POOL", "40"))    # MMR candidates before rerank
MMR_LAMBDA       = float(os.getenv("P2_MMR_LAMBDA", "0.5"))  # lower = more diverse
# Retrieval mode (for ablation): "mmr_rerank" (MMR pool -> cross-encoder rerank -> top-k),
# "mmr" (MMR, no rerank), "similarity" (pure cosine top-k, no MMR, no rerank; default/best).
P2_SEARCH        = os.getenv("P2_SEARCH", "similarity")
# Ticker pre-filter: when the question names companies in the corpus, restrict retrieval to
# those companies' chunks first (fixes wrong-doc retrieval). Fair, standard metadata filtering.
P2_TICKER_FILTER = os.getenv("P2_TICKER_FILTER", "1") != "0"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_INPUT_CHARS = 600                                    # model has a 512-token limit

_cross_encoder: CrossEncoder | None = None
_ticker_index: dict | None = None   # {company_name_lower / ticker_lower: TICKER}


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[p2] Loading cross-encoder ({RERANKER_MODEL})…")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


import re as _re

# hand aliases for companies whose common name differs from the manifest form
_TICKER_ALIASES = {
    "exxon": "XOM", "exxonmobil": "XOM", "jpmorgan": "JPM", "jp morgan": "JPM",
    "j.p. morgan": "JPM", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "berkshire": "BRK.B", "nvidia": "NVDA", "amd": "AMD",
    "home depot": "HD", "p&g": "PG", "procter": "PG", "coca-cola": "KO", "coca cola": "KO",
    "netflix": "NFLX", "adobe": "ADBE", "abbvie": "ABBV", "verizon": "VZ", "amgen": "AMGN",
    "chevron": "CVX", "colgate": "CL", "cisco": "CSCO", "caterpillar": "CAT", "costco": "COST",
    "oracle": "ORCL", "accenture": "ACN", "morgan stanley": "MS",
}


def _get_ticker_index() -> dict:
    """Map company name tokens / ticker -> TICKER (manifest + aliases), for detecting which
    companies a question is about so retrieval can pre-filter to their chunks."""
    global _ticker_index
    if _ticker_index is None:
        import json
        from pathlib import Path
        _ticker_index = dict(_TICKER_ALIASES)
        mf = Path("datasets/round3/manifest_clean.json")
        if mf.exists():
            for tk, info in json.load(open(mf)).items():
                name = (info.get("company", "") or "").lower()
                # strip corporate suffixes / parentheticals for cleaner matching
                name = _re.sub(r"\s*\(.*?\)|\s+(inc\.?|corporation|corp\.?|company|co\.?|"
                               r"plc|ltd\.?|group|holdings|&\s*co\.?)\b\.?", "", name).strip()
                _ticker_index[tk.lower()] = tk
                if name and len(name) >= 3:
                    _ticker_index.setdefault(name, tk)
    return _ticker_index


def _is_corpus_scan(question: str) -> bool:
    """Aggregation / 'scan-many-companies' questions where the answer companies are NOT named
    in the question — ticker filtering would wrongly narrow retrieval, so skip it."""
    ql = question.lower()
    return bool(_re.search(
        r"\bwhich (companies|dataset companies|.*-sector companies|.*companies)\b"
        r"|\blist (all|the)\b|\bhow many .*(compan|firm|audit)"
        r"|\bamong (the |dataset'?s )?.*companies\b"
        r"|\bwhich (energy|utilities|health|financial|consumer|industrial|communication|"
        r"materials|real estate|semiconductor|kpmg|deloitte|pwc|ernst)\b"
        r"|\bserves? on the boards?\b|\bdirector at which\b|\btwo companies\b", ql))


def _tickers_in_question(question: str) -> list[str]:
    idx = _get_ticker_index()
    ql = " " + question.lower() + " "
    found = []
    for key, tk in idx.items():
        if len(key) < 3:
            continue
        # match key as a word (allowing trailing possessive/punctuation incl. ? !)
        if _re.search(r"[\s(]" + _re.escape(key) + r"(?:'s|['.,)\s?!:;])", ql) and tk not in found:
            found.append(tk)
    return found[:4]


def build_prompt(question: str, context_chunks: list[str], chunk_ids: list[str] | None = None) -> str:
    # Label each passage with its chunk ID so the model can cite the evidence it used
    # (Round 3 citation requirement). Falls back to positional labels if no IDs given.
    labels = chunk_ids or [f"Passage {i+1}" for i in range(len(context_chunks))]
    context_block = "\n\n---\n\n".join(
        f"[{labels[i] if i < len(labels) else f'Passage {i+1}'}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )
    return (
        f"{config.SYSTEM_PROMPT}\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def _maybe_ticker_filter(question: str) -> dict | None:
    if P2_TICKER_FILTER and not _is_corpus_scan(question):
        tks = _tickers_in_question(question)
        if tks:
            return {"ticker": {"$in": tks}} if len(tks) > 1 else {"ticker": tks[0]}
    return None


def run_single(question: str, vectorstore: Chroma) -> CallMetrics:
    t_start = time.perf_counter()
    k = config.CONTEXT_NUM_CHUNKS
    filt = _maybe_ticker_filter(question)

    if P2_SEARCH == "similarity":
        # Pure cosine similarity top-k — no MMR, no reranker.
        sk = {"k": k}
        if filt:
            sk["filter"] = filt
        selected = vectorstore.as_retriever(search_type="similarity", search_kwargs=sk).invoke(question)
        if not selected and filt:
            selected = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k}).invoke(question)
    else:
        # MMR candidate pool (diverse), optionally cross-encoder reranked.
        sk = {"k": RERANK_POOL, "fetch_k": RERANK_POOL * 3, "lambda_mult": MMR_LAMBDA}
        if filt:
            sk["filter"] = filt
        candidates = vectorstore.as_retriever(search_type="mmr", search_kwargs=sk).invoke(question)
        if not candidates and filt:
            sk.pop("filter")
            candidates = vectorstore.as_retriever(search_type="mmr", search_kwargs=sk).invoke(question)
        if P2_SEARCH == "mmr_rerank" and candidates:
            ce = _get_cross_encoder()
            scores = ce.predict([(question, d.page_content[:RERANK_INPUT_CHARS]) for d in candidates])
            candidates = [d for _, d in sorted(zip(scores, candidates), key=lambda x: -x[0])]
        selected = candidates[:k]

    context_chunks = [d.page_content[:config.CONTEXT_CHUNK_CHARS] for d in selected]
    retrieved_ids = [getattr(d, "metadata", {}).get("title", "") for d in selected]

    prompt = build_prompt(question, context_chunks, chunk_ids=retrieved_ids)
    context_text = "\n\n".join(context_chunks)
    with MetricsTimer("Pipeline 2: Basic RAG", question) as timer:
        answer, raw = get_llm_response(prompt)

    metrics = timer.record(
        answer, raw.get("usage", {}),
        question=question, context_text=context_text,
        retrieved_chunks=len(context_chunks),
        retrieved_ids=retrieved_ids,
        preserve_answer=(config.DATASET == "round3"),
    )
    metrics.route = "vector_rag"
    metrics.latency_seconds = round(time.perf_counter() - t_start, 3)
    return metrics


def run_batch(questions: list[str], vectorstore: Chroma, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    results = []
    for q in tqdm(questions, desc="Pipeline 2 (Basic RAG)"):
        m = run_single(q, vectorstore)
        results.append(m)
        if verbose:
            print(f"  Q: {q[:60]}...")
            print(f"  A: {m.answer[:80]}...")
            print(f"  Chunks: {m.retrieved_chunks} | Tokens: {m.total_tokens} | Latency: {m.latency_seconds}s")
    return results
