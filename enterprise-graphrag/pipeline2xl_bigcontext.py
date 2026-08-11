"""Pipeline 2-XL : Basic RAG with the retrieval budget taken off entirely.

WHY THIS EXISTS — AND A CORRECTION IT FORCED
--------------------------------------------
For most of this project the claim was "vector search cannot find the documents that answer a
multi-hop question". That claim was too strong, and measuring recall@k is what exposed it:

    k=10   answer content present in retrieved text : 0.340   <- the budget everything used
    k=50                                            : 0.801
    k=100                                           : 0.854
    k=200                                           : 0.938

The content IS retrievable. It simply is not in the top 10, because the documents holding the
answer do not resemble the question — "which channels do Marcus Lin's reports use" is answered
by Kevin Osei's Slack messages, which never mention Marcus Lin. Similarity puts them deep in
the ranking, not nowhere in it.

So the correct question is no longer "can retrieval find it" but "does having it in the context
window let the model USE it". That is what this arm measures: same retriever, same model, same
prompt, and a context budget an order of magnitude larger.

If it answers well, the honest headline changes from "GraphRAG can do what retrieval cannot"
to "GraphRAG does it for a fraction of the tokens" — a weaker but true claim, and far better
to discover here than on stage.

Set XL_K to trade cost against coverage:
    XL_K=50   ~25k prompt tokens/question   (coverage 0.80)
    XL_K=100  ~50k prompt tokens/question   (coverage 0.85)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline3_round3"))

import config
from llm_client import get_llm_response
from metrics import CallMetrics, MetricsTimer

XL_K = int(os.getenv("XL_K", "50"))
XL_CHARS = int(os.getenv("XL_CHARS", str(config.CONTEXT_CHUNK_CHARS)))

_p3 = None


def _pipeline():
    """Same loader and vector search as P2/P3. The graph is never consulted from here."""
    global _p3
    if _p3 is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "p3mod", str(Path(__file__).resolve().parent / "pipeline3_round3" / "pipeline.py"))
        _p3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_p3)
        _p3._load()
        _p3._connect()
    return _p3


def run_single(question: str, vectorstore=None) -> CallMetrics:
    t0 = time.perf_counter()
    p3 = _pipeline()

    # NOT p3._context_from_chunks — that helper caps at MAX_CTX_CHUNKS (10) by design, which is
    # the fairness budget every other arm shares. This arm exists precisely to lift that cap,
    # so it assembles its own context and dedups by hand.
    ids, seen = [], set()
    for cid in p3._tg_vector_search(question, k=XL_K):
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)

    chunks = [(p3._chunk_text.get(c) or "")[:XL_CHARS] for c in ids]
    chunks = [c for c in chunks if c]
    block = "\n\n---\n\n".join(f"[{ids[i]}]\n{c}" for i, c in enumerate(chunks))
    prompt = f"{config.SYSTEM_PROMPT}\nContext:\n{block}\n\nQuestion: {question}\nAnswer:"

    with MetricsTimer("Pipeline 2-XL: Basic RAG, large context", question) as timer:
        answer, raw = get_llm_response(prompt)

    m = timer.record(answer, raw.get("usage", {}), question=question,
                     context_text="\n\n".join(chunks), retrieved_chunks=len(ids),
                     retrieved_ids=ids, preserve_answer=True)
    m.route = f"vector_k{XL_K}"
    m.latency_seconds = round(time.perf_counter() - t0, 3)
    return m


def run_batch(questions: list[str], vectorstore=None, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    return [run_single(q) for q in tqdm(questions, desc="Pipeline 2-XL")]
