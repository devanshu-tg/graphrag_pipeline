"""Pipeline 2: Basic RAG — vector top-k, no graph. The baseline GraphRAG must beat.

DIFFERENCE FROM MASTERGRAPH'S P2, AND WHY
-----------------------------------------
Theirs retrieves from Chroma while their P3 retrieves from TigerGraph. So their
"GraphRAG vs Basic RAG" comparison varies TWO things at once — the graph AND the vector
store (different index, different similarity implementation).

This one uses the SAME TigerGraph vector_search P3 uses, with the same k and the same
context budget. The only remaining difference between P2 and P3 is whether the graph is
consulted. That is the variable under test.

It also avoids re-embedding 73,425 chunks into a second store for no analytical gain.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline3_round3"))

import config
from llm_client import get_llm_response
from metrics import CallMetrics, MetricsTimer

_p3 = None


def _pipeline():
    """Reuse P3's loader/connection/vector search — identical retrieval, minus the graph."""
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


def build_prompt(question: str, chunks: list[str], chunk_ids: list[str] | None = None) -> str:
    labels = chunk_ids or [f"Passage {i+1}" for i in range(len(chunks))]
    block = "\n\n---\n\n".join(
        f"[{labels[i] if i < len(labels) else f'Passage {i+1}'}]\n{c}"
        for i, c in enumerate(chunks))
    return f"{config.SYSTEM_PROMPT}\nContext:\n{block}\n\nQuestion: {question}\nAnswer:"


def run_single(question: str, vectorstore=None) -> CallMetrics:
    t0 = time.perf_counter()
    p3 = _pipeline()
    k = config.CONTEXT_NUM_CHUNKS
    ids = p3._tg_vector_search(question, k=k)
    context, ids = p3._context_from_chunks(ids)
    chunks = [(p3._chunk_text.get(c) or "")[:config.CONTEXT_CHUNK_CHARS] for c in ids]

    prompt = build_prompt(question, chunks, chunk_ids=ids)
    with MetricsTimer("Pipeline 2: Basic RAG", question) as timer:
        answer, raw = get_llm_response(prompt)

    m = timer.record(answer, raw.get("usage", {}), question=question,
                     context_text="\n\n".join(chunks), retrieved_chunks=len(ids),
                     retrieved_ids=ids, preserve_answer=True)
    m.route = "vector_rag"
    m.latency_seconds = round(time.perf_counter() - t0, 3)
    return m


def run_batch(questions: list[str], vectorstore=None, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    return [run_single(q) for q in tqdm(questions, desc="Pipeline 2 (Basic RAG)")]
