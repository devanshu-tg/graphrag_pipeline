"""Cross-encoder reranking, shared so the baseline and GraphRAG's fallback cannot drift apart.

WHY THIS EXISTS
---------------
Measured on the official 200 content questions: GraphRAG scored 45.9% against reranked Basic
RAG's 58.4%. Splitting by route showed 166 of those questions had ALREADY fallen back to vector
search, and GraphRAG still lost 8.4 points on them.

The cause was not the graph. `pipeline2rr_rerank.py` retrieves RR_POOL=50 candidates and
reranks to RR_KEEP=10 with a cross-encoder. GraphRAG's fallback called
`_tg_vector_search(k=10)` — plain top-10, no reranking. The reranker was added to the BASELINE
as a fairness fix, correctly, and nobody noticed it left the system under test with a weaker
retriever than the arm it is measured against.

That asymmetry stayed invisible for as long as the only benchmark was the generated metadata
questions, where GraphRAG routed to the graph on 50/50 questions and never touched the fallback
at all.

Both callers read the SAME RR_POOL / RR_KEEP / RR_MODEL environment variables, so the two arms
cannot be configured differently by accident.
"""
from __future__ import annotations

import os

RR_POOL = int(os.getenv("RR_POOL", "50"))
RR_KEEP = int(os.getenv("RR_KEEP", "10"))
RR_MODEL = os.getenv("RR_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_ce = None
# The demo server calls this from request threads, and a CrossEncoder takes ~18s to build on a
# 2-vCPU box. Without the lock two concurrent first-queries each build their own copy: double
# the wait and double the memory. THIS is the single instance for the whole process -
# pipeline2rr_rerank delegates here rather than keeping a second one.
_ce_lock = __import__("threading").Lock()


def reranker():
    global _ce
    if _ce is not None:
        return _ce
    with _ce_lock:
        if _ce is None:                       # re-check: another thread may have finished
            from sentence_transformers import CrossEncoder
            print(f"[rerank] loading {RR_MODEL}")
            _ce = CrossEncoder(RR_MODEL)
    return _ce


def rerank(question: str, chunk_ids: list[str], text_of, keep: int = 0,
           max_chars: int = 2000) -> list[str]:
    """Score (question, chunk) pairs together and keep the best `keep`.

    A bi-encoder scores query and passage independently, so it cannot notice that a passage
    contains BOTH people named in the question, or the exact identifier the question asks about.
    A cross-encoder reads them together and can.

    Returns the input order truncated to `keep` if anything goes wrong — degrading to plain
    top-k is always better than returning nothing.
    """
    keep = keep or RR_KEEP
    seen, pool = set(), []
    for cid in chunk_ids:
        if cid and cid not in seen:
            seen.add(cid)
            pool.append(cid)
    if not pool:
        return []

    texts = [(text_of(c) or "")[:max_chars] for c in pool]
    live = [(c, t) for c, t in zip(pool, texts) if t]
    if not live:
        return pool[:keep]
    try:
        scores = reranker().predict([(question, t) for _, t in live], show_progress_bar=False)
    except Exception as e:
        print(f"[rerank] failed ({str(e)[:80]}) — falling back to retrieval order")
        return pool[:keep]
    order = sorted(range(len(live)), key=lambda i: -float(scores[i]))[:keep]
    return [live[i][0] for i in order]
