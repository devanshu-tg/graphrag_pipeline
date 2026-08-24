"""Pipeline 2-RR : Basic RAG with a cross-encoder reranker — the last idea a sceptic has.

WHY
---
Measured recall on the multi-hop questions:

    coverage @ top-10   0.34     <- what plain Basic RAG sees
    coverage @ top-50   0.80
    coverage @ top-200  0.94

The answer text is not missing, it is ranked 10th-50th. A bi-encoder scores query and passage
independently, so it cannot notice that a passage happens to contain BOTH people named in the
question. A cross-encoder reads query and passage TOGETHER and can.

So this retrieves 50 candidates, reranks them, and keeps the best 10. Same 10-chunk prompt cost
as plain Basic RAG, but chosen from a pool five times deeper. If ranking is what stands between
retrieval and the answer, this is what fixes it — and it is the obvious question from anyone in
the audience ("did you try reranking?"), so it should be answered with a number.

Honest prediction, recorded before running it: this should help the single-document and
directory tiers, where the right document exists and is merely ranked low. It should barely
move multi-hop, because there the documents holding the answer never mention the person in the
question at all — a reranker cannot promote what is not in the candidate pool for the right
reason. Recorded so the result cannot be retro-fitted to whatever happens.

    XL_K style knobs:  RR_POOL=50  (candidates)   RR_KEEP=10  (kept after reranking)
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

RR_POOL = int(os.getenv("RR_POOL", "50"))
RR_KEEP = int(os.getenv("RR_KEEP", str(config.CONTEXT_NUM_CHUNKS)))
RR_MODEL = os.getenv("RR_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_p3 = None
_ce = None
# Both loaders are called from the demo server's request threads, so two concurrent queries can
# enter them at once. Without the lock the previous code published the module to the global
# BEFORE exec_module() populated it, so the second thread got a half-built module and failed
# with "module 'p3mod' has no attribute '_tg_vector_search'". Measured under 4-way concurrency:
# every Basic RAG answer in a 50-question sweep errored that way. Build into a local, then
# publish — and hold the lock so the second caller waits rather than racing.
import threading as _threading
_load_lock = _threading.Lock()


def _pipeline():
    global _p3
    if _p3 is not None:
        return _p3
    with _load_lock:
        if _p3 is None:                       # re-check: another thread may have finished
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "p3mod", str(Path(__file__).resolve().parent / "pipeline3_round3" / "pipeline.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._load()
            mod._connect()
            _p3 = mod                         # publish only once it is fully usable
    return _p3


def _reranker():
    global _ce
    if _ce is not None:
        return _ce
    with _load_lock:                          # same race: a cross-encoder takes seconds to load
        if _ce is None:
            from sentence_transformers import CrossEncoder
            print(f"[rerank] loading {RR_MODEL}")
            _ce = CrossEncoder(RR_MODEL)
    return _ce


def run_single(question: str, vectorstore=None) -> CallMetrics:
    t0 = time.perf_counter()
    p3 = _pipeline()

    # deep candidate pool, deduped by hand — p3._context_from_chunks would cap it at 10,
    # which is exactly the cap this arm exists to rank within rather than obey blindly
    pool, seen = [], set()
    for cid in p3._tg_vector_search(question, k=RR_POOL):
        if cid not in seen:
            seen.add(cid)
            pool.append(cid)

    texts = [(p3._chunk_text.get(c) or "")[:config.CONTEXT_CHUNK_CHARS] for c in pool]
    pairs = [(question, t) for t in texts if t]
    live = [c for c, t in zip(pool, texts) if t]

    if pairs:
        scores = _reranker().predict(pairs, show_progress_bar=False)
        order = sorted(range(len(live)), key=lambda i: -float(scores[i]))[:RR_KEEP]
        ids = [live[i] for i in order]
    else:
        ids = pool[:RR_KEEP]

    chunks = [(p3._chunk_text.get(c) or "")[:config.CONTEXT_CHUNK_CHARS] for c in ids]
    block = "\n\n---\n\n".join(f"[{ids[i]}]\n{c}" for i, c in enumerate(chunks))
    prompt = f"{config.SYSTEM_PROMPT}\nContext:\n{block}\n\nQuestion: {question}\nAnswer:"

    with MetricsTimer("Pipeline 2-RR: Basic RAG + cross-encoder rerank", question) as timer:
        answer, raw = get_llm_response(prompt)

    m = timer.record(answer, raw.get("usage", {}), question=question,
                     context_text="\n\n".join(chunks), retrieved_chunks=len(ids),
                     retrieved_ids=ids, preserve_answer=True)
    m.route = f"rerank:{RR_POOL}->{RR_KEEP}"
    m.latency_seconds = round(time.perf_counter() - t0, 3)
    return m


def run_batch(questions: list[str], vectorstore=None, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    return [run_single(q) for q in tqdm(questions, desc="Pipeline 2-RR")]
