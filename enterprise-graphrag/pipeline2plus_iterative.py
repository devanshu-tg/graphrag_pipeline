"""Pipeline 2+ : Iterative RAG — the STEELMAN baseline, built to be as strong as possible.

WHY THIS EXISTS
---------------
Standard single-shot RAG (pipeline2_basic_rag.py) scores 0/20 on these multi-hop questions.
A gap that wide invites the obvious objection: "you didn't try very hard with the baseline."

This is trying hard. It is the published iterative-retrieval pattern (self-ask / IRCoT
family): retrieve, notice which entities you now need, retrieve again for those, then answer.
It is given MORE than GraphRAG gets — three LLM calls instead of one, several retrieval
rounds, and a much larger context budget. The comparison is deliberately tilted against the
system under test, because a win under those conditions is the only kind worth reporting.

Measured justification for the design (not guesses):
  single-shot retrieval           -> answer text present in 0.307 of cases
  naive 2-round (decompose first) -> no better; round 1 never surfaces the directory page, so
                                     round 2 searches for names it never learned, and on one
                                     question invented names off irrelevant chunks
  entity-anchored round 1         -> "Who is Marcus Lin's manager?" returns 10/10 directory
                                     pages, so anchoring on the ENTITY rather than the whole
                                     question is what makes hop 1 work
  perfect hop-1 + round 2         -> 0.71
So the ceiling is real, and the way to reach it is entity-anchored retrieval first.

FAIRNESS RULES — the whole point is that this is a HONEST baseline
-----------------------------------------------------------------
  * NO graph access of any kind. Seeds come from reading the QUESTION TEXT. The moment this
    resolves a name against the graph's entity table it stops being a retrieval baseline and
    silently becomes GraphRAG.
  * NO hardcoded entity names, no question patterns — the same rule the router lives under.
  * NO tuning against the 20 eval questions.
  * NO gold answers or evidence documents. Those were diagnostics, not inputs.
  * Same model, temperature, corpus, embeddings, retriever and final answer prompt as P2/P3.

Every token across all three calls is counted and reported.
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

# Deliberately generous. P3 answers from a traversal line inside ~1.7K prompt tokens; this is
# allowed roughly an order of magnitude more context, because the question being asked is
# "can retrieval get there if we stop rationing it?"
SEED_K = 8            # chunks per named entity in round 1
HOP_K = 8             # chunks per discovered entity in round 2
MAX_SEEDS = 4
MAX_HOP_ENTITIES = 8
MAX_CONTEXT_CHUNKS = 40

_p3 = None


def _pipeline():
    """Reuse P2/P3's loader and vector search — identical retrieval stack, no graph calls.

    Only _tg_vector_search and _chunk_text are touched. _adjacency/_walk (the graph) are never
    invoked from here; that separation is what keeps this a retrieval baseline.
    """
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


_usage = {"prompt": 0, "completion": 0, "calls": 0}


def _ask(prompt: str) -> str:
    text, raw = get_llm_response(prompt)
    u = raw.get("usage", {})
    _usage["prompt"] += u.get("prompt_tokens", 0)
    _usage["completion"] += u.get("completion_tokens", 0)
    _usage["calls"] += 1
    return text or ""


def _lines(text: str, cap: int) -> list[str]:
    out = []
    for ln in (text or "").split("\n"):
        s = ln.strip().strip("-*•0123456789. ").strip().strip('"')
        if s and len(s) < 80 and s.lower() not in ("none", "n/a", "unknown"):
            out.append(s)
    return out[:cap]


SEED_PROMPT = (
    "List the specific named entities in this question — people, teams, departments, "
    "channels, companies, products. One per line, exactly as written in the question. "
    "No explanation, no numbering. If none, output NONE.\n\nQuestion: {q}\n\nEntities:"
)

HOP_PROMPT = (
    "You are answering a question that needs TWO steps of lookup.\n\n"
    "Question: {q}\n\n"
    "Passages retrieved so far:\n{ctx}\n\n"
    "These passages may identify intermediate entities (for example the specific PEOPLE who "
    "report to someone, or the members of a team) that you must look up next in order to "
    "answer.\n"
    "List those intermediate entities, one name per line, exactly as spelled in the passages. "
    "Only list entities actually named in the passages — never invent one. "
    "If the passages already fully answer the question, output NONE.\n\nEntities to look up:"
)


def build_prompt(question: str, chunks: list[str], ids: list[str]) -> str:
    block = "\n\n---\n\n".join(
        f"[{ids[i] if i < len(ids) else f'Passage {i+1}'}]\n{c}" for i, c in enumerate(chunks))
    return f"{config.SYSTEM_PROMPT}\nContext:\n{block}\n\nQuestion: {question}\nAnswer:"


def run_single(question: str, vectorstore=None) -> CallMetrics:
    t0 = time.perf_counter()
    p3 = _pipeline()
    before = dict(_usage)

    def fetch(query: str, k: int) -> list[str]:
        ids = p3._tg_vector_search(query, k=k)
        _, ids = p3._context_from_chunks(ids)
        return ids

    # ---- round 0: entities named in the QUESTION (never from the graph) ----------------
    seeds = _lines(_ask(SEED_PROMPT.format(q=question)), MAX_SEEDS)

    ids: list[str] = []
    for s in seeds:
        for c in fetch(s, SEED_K):
            if c not in ids:
                ids.append(c)
    if not ids:                                  # no entity resolved -> plain single-shot RAG
        ids = fetch(question, config.CONTEXT_NUM_CHUNKS)

    # ---- round 1: what do we now know we still need? ---------------------------------
    ctx1 = "\n\n".join((p3._chunk_text.get(c) or "")[:1200] for c in ids[:20])
    hop = _lines(_ask(HOP_PROMPT.format(q=question, ctx=ctx1[:14000])), MAX_HOP_ENTITIES)

    # ---- round 2: retrieve for each discovered entity ---------------------------------
    for h in hop:
        for c in fetch(f"{h} channel message document", HOP_K):
            if c not in ids:
                ids.append(c)
    # always keep the plain question's own results underneath, so this can only add
    for c in fetch(question, config.CONTEXT_NUM_CHUNKS):
        if c not in ids:
            ids.append(c)

    ids = ids[:MAX_CONTEXT_CHUNKS]
    chunks = [(p3._chunk_text.get(c) or "")[:config.CONTEXT_CHUNK_CHARS] for c in ids]

    prompt = build_prompt(question, chunks, ids)
    with MetricsTimer("Pipeline 2+: Iterative RAG", question) as timer:
        answer, raw = get_llm_response(prompt)

    u = raw.get("usage", {})
    # the two planning calls are real cost and are reported, not hidden
    extra_p = _usage["prompt"] - before["prompt"]
    extra_c = _usage["completion"] - before["completion"]
    merged = {"prompt_tokens": u.get("prompt_tokens", 0) + extra_p,
              "completion_tokens": u.get("completion_tokens", 0) + extra_c,
              "total_tokens": u.get("total_tokens", 0) + extra_p + extra_c}

    m = timer.record(answer, merged, question=question, context_text="\n\n".join(chunks),
                     retrieved_chunks=len(ids), retrieved_ids=ids, preserve_answer=True)
    m.route = f"iterative:{len(seeds)}seed+{len(hop)}hop"
    m.latency_seconds = round(time.perf_counter() - t0, 3)
    return m


def run_batch(questions: list[str], vectorstore=None, verbose: bool = False) -> list[CallMetrics]:
    from tqdm import tqdm
    return [run_single(q) for q in tqdm(questions, desc="Pipeline 2+ (Iterative RAG)")]
