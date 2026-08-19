"""Chunk-budget matrix — Basic RAG vs GraphRAG at EQUAL and UNEQUAL context budgets.

THE QUESTION
------------
"GraphRAG beats Basic RAG" is uninteresting if GraphRAG is simply fed more text. So this runs
the same 50 enterprise questions at three budget pairings, where a "chunk" means one context
item in the prompt:

    P2=10 vs P3=10    equal budget, the fair comparison
    P2=10 vs P3=3     GraphRAG HANDICAPPED — baseline gets 3x the context
    P2=3  vs P3=3     equal budget, both starved

Four distinct arms cover all three pairings (P2@10 is shared by the first two).

WHAT IS HELD IDENTICAL
----------------------
Same question set, same corpus, same TigerGraph vector index, same embedder, same generator
(config.GEMINI_MODEL) at temperature 0, same SYSTEM_PROMPT, same 2000-char cap per item, and
the same 50-candidate pool reranked by the same cross-encoder. The ONLY differences between the
arms are the graph and the item count.

WHY THE BASELINE IS P2-RR AND NOT PLAIN P2
------------------------------------------
P3's vector fallback reranks 50 candidates down to VEC_SLOTS with the cross-encoder
(retrieval_common). Plain pipeline2_basic_rag takes raw top-k with NO reranking. Measuring a
reranked GraphRAG against an unreranked baseline puts a thumb on the scale in favour of the
system under test, which this repo treats as never acceptable. So the baseline is
pipeline2rr_rerank at the same pool and the same keep. This also matches the published
50-question run (results/enterprise/summary_ent50.json), which used p2rr.

ONE THING P3 CARRIES THAT IS NOT A CHUNK
----------------------------------------
When a graph route fires, P3's prompt also holds a `Graph traversal (...): a, b, c` line. It is
NOT one of the N context items, so at "3 vs 3" P3 still has the traversal line and the baseline
does not. That is the mechanism under test rather than a leak, but it is why the two arms are
not token-identical at the same item count — so tokens are reported per arm and every pairing
is stated in BOTH accuracy and tokens.

PARALLELISM
-----------
Questions inside an arm run on a thread pool (the cost is Gemini + TigerGraph round trips, not
CPU). Arms run one after another because the budget knobs are module-level ints that
run_single reads at call time; two budgets cannot be in flight in one process.

Every lazy cache in pipeline.py (_adjacency, _build_resolver) assigns an EMPTY container to its
global and then fills it, so a second thread arriving mid-build would see a non-None half-built
index and silently answer from it. They are therefore ALL warmed before the pool starts.

    python build/run_chunk_matrix.py --n 50 --workers 6 --tag chunkmatrix
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline3_round3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
from _portability import init_console, load_env, ropen, wopen   # noqa: E402

init_console()
load_env()

import config                                    # noqa: E402
# identical scorer to the published runs — imported, never re-implemented, so these numbers
# stay comparable with results/enterprise/summary_ent50.json
from run_eval import set_scores                  # noqa: E402

OUTDIR = Path("results/enterprise")
QUESTIONS = Path("datasets/enterprise/eval_mixed50.jsonl")

_print_lock = threading.Lock()
_tls = threading.local()


def _say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _load_p3():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p3", str(Path("pipeline3_round3/pipeline.py").resolve()))
    p3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p3)
    return p3


def warm(p3, prr) -> None:
    """Build every lazy global BEFORE any thread runs. See module docstring."""
    t0 = time.time()
    p3._load()
    _say(f"  [warm] corpus+index {time.time()-t0:.0f}s")
    p3._connect()
    t0 = time.time()
    p3._adjacency()
    p3._build_resolver()
    _say(f"  [warm] adjacency+resolver {time.time()-t0:.0f}s "
         f"({len(p3._known_cache):,} entities)")
    import schema_router
    schema_router.graph_relations()
    import retrieval_common
    retrieval_common.reranker()
    # p2rr keeps its OWN reranker global; point it at the same loaded object so the two arms
    # cannot drift to different models and so the weights are held once.
    prr._ce = retrieval_common._ce
    prr._p3 = p3
    _say("  [warm] router schema + cross-encoder ready")


def patch_thread_local_conn(p3) -> None:
    """One TigerGraph connection PER THREAD.

    _connect() caches a single global conn, whose underlying requests session and auth token
    would then be shared by every worker. Savanna already throws transient 500s under load;
    sharing one session across six threads adds a failure mode for no gain.
    """
    def _tl_connect():
        conn = getattr(_tls, "conn", None)
        if conn is not None:
            return conn
        last = None
        for attempt in range(4):
            try:
                c = p3.tg.TigerGraphConnection(
                    host=p3.TG_HOST, graphname=p3.TG_GRAPH,
                    gsqlSecret=p3.TG_SECRET or None,
                    username=p3.TG_USER, password=p3.TG_PASS)
                c.getToken(p3.TG_SECRET)
                c.echo()
                _tls.conn = c
                return c
            except Exception as e:      # noqa: BLE001
                last = e
                time.sleep(2 ** attempt + 1)
        raise RuntimeError(f"per-thread TG connect failed: {str(last)[:140]}")
    p3._connect = _tl_connect


def set_budget(p3, prr, kind: str, budget: int) -> dict:
    """Point one arm at a context budget of `budget` items and return what was set."""
    if kind == "p2rr":
        # RR_POOL stays 50 — the candidate pool is identical to P3's, only the KEEP differs
        prr.RR_KEEP = budget
        config.CONTEXT_NUM_CHUNKS = budget
        return {"RR_POOL": prr.RR_POOL, "RR_KEEP": prr.RR_KEEP,
                "chars_per_item": config.CONTEXT_CHUNK_CHARS}
    p3.MAX_CTX_CHUNKS = budget           # hard cap on total context items
    p3.VEC_SLOTS = budget                # reranked chunks allowed in
    p3.VEC_K = budget                    # pool is max(VEC_K, RR_POOL) = 50 either way
    p3.PROV_SLOTS = max(1, budget // 2)  # same formula config.py uses at its default
    return {"MAX_CTX_CHUNKS": p3.MAX_CTX_CHUNKS, "VEC_SLOTS": p3.VEC_SLOTS,
            "PROV_SLOTS": p3.PROV_SLOTS, "RR_POOL": 50,
            "chars_per_item": p3.MAX_CTX_CHARS}


def run_one(kind: str, q: dict, p3, prr) -> dict:
    """One question, one arm. Retried once; a hard failure is RECORDED, never dropped."""
    last = None
    for attempt in range(2):
        try:
            m = prr.run_single(q["question"]) if kind == "p2rr" else p3.run_single(q["question"])
            ans = m.answer or ""
            p, r, f1 = set_scores(ans, q["gold_members"])
            return {"question_id": q["question_id"], "shape": q["shape"],
                    "tier": q.get("tier", ""), "question": q["question"],
                    "gold": q["answer"], "answer": ans,
                    "route": getattr(m, "route", ""),
                    "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                    "exact": int(r == 1.0 and p >= 0.99),
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "retrieved_chunks": getattr(m, "retrieved_chunks", 0),
                    "latency_s": getattr(m, "latency_seconds", 0),
                    "planner_tokens": 0, "error": ""}
        except Exception as e:           # noqa: BLE001
            last = e
            if attempt == 0:
                time.sleep(3)
    _say(f"    !! {q['question_id']} {kind} FAILED: {str(last)[:100]}")
    return {"question_id": q["question_id"], "shape": q["shape"],
            "tier": q.get("tier", ""), "question": q["question"], "gold": q["answer"],
            "answer": "", "route": "error", "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "exact": 0, "prompt_tokens": 0, "completion_tokens": 0, "retrieved_chunks": 0,
            "latency_s": 0, "planner_tokens": 0, "error": str(last)[:200]}


def run_arm(name: str, kind: str, budget: int, qs: list[dict], p3, prr,
            workers: int) -> tuple[list[dict], dict]:
    import schema_router
    knobs = set_budget(p3, prr, kind, budget)
    schema_router.reset_token_usage()
    _say(f"\n=== {name}  ({kind}, budget={budget} items)  {knobs}")
    t0 = time.time()
    done = [0]

    def task(q):
        row = run_one(kind, q, p3, prr)
        with _print_lock:
            done[0] += 1
            if done[0] % 10 == 0:
                print(f"    {done[0]}/{len(qs)}  ({time.time()-t0:.0f}s)", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(task, qs))
    # stable output order regardless of completion order
    order = {q["question_id"]: i for i, q in enumerate(qs)}
    rows.sort(key=lambda r: order[r["question_id"]])

    if kind == "p3":
        # Planner tokens are accumulated by schema_router with a plain `+=`, which is not atomic
        # under threads, so a concurrent increment can be lost. This is an ESTIMATE of the
        # router's per-question cost, and it can only ever UNDER-report — it never flatters P3
        # in the token comparison, which is the direction that would matter.
        pt = schema_router.plan_token_usage()
        per_q = round((pt["prompt"] + pt["output"]) / max(1, len(rows)))
        for r in rows:
            r["planner_tokens"] = per_q
    for r in rows:
        r["arm"] = name
        r["arm_kind"] = kind
        r["budget"] = budget
    _say(f"  {name}: {len(rows)} answers in {time.time()-t0:.0f}s "
         f"({sum(1 for r in rows if r['error'])} errors)")
    return rows, knobs


def agg(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    tot = [r["prompt_tokens"] + r["completion_tokens"] + r["planner_tokens"] for r in ok]
    graph = sum(1 for r in ok if str(r["route"]).startswith("graph"))
    return {
        "n": len(rows), "scored": len(ok), "errors": len(rows) - len(ok),
        "exact": sum(r["exact"] for r in ok),
        "recall": st.mean([r["recall"] for r in ok]) if ok else 0.0,
        "precision": st.mean([r["precision"] for r in ok]) if ok else 0.0,
        "f1": st.mean([r["f1"] for r in ok]) if ok else 0.0,
        "prompt": st.mean([r["prompt_tokens"] for r in ok]) if ok else 0.0,
        "completion": st.mean([r["completion_tokens"] for r in ok]) if ok else 0.0,
        "planner": st.mean([r["planner_tokens"] for r in ok]) if ok else 0.0,
        "total_tokens": st.mean(tot) if tot else 0.0,
        "items_in_context": st.mean([r["retrieved_chunks"] for r in ok]) if ok else 0.0,
        "latency_s": st.mean([r["latency_s"] for r in ok]) if ok else 0.0,
        "graph_routed": graph,
    }


def by_tier(rows: list[dict]) -> dict:
    out = {}
    for t in ("single_doc", "single_dir", "multi_hop"):
        R = [r for r in rows if r.get("tier") == t and not r["error"]]
        if R:
            out[t] = {"n": len(R), "f1": st.mean([r["f1"] for r in R]),
                      "exact": sum(r["exact"] for r in R),
                      "total_tokens": st.mean([r["prompt_tokens"] + r["completion_tokens"]
                                               + r["planner_tokens"] for r in R])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--tag", default="chunkmatrix")
    ap.add_argument("--budgets", default="10,3")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    qs = [json.loads(l) for l in ropen(Path(args.questions)) if l.strip()][: args.n]
    print(f"chunk-budget matrix: {len(qs)} questions, budgets={budgets}, "
          f"workers={args.workers}, model={config.GEMINI_MODEL}\n")

    p3 = _load_p3()
    import pipeline2rr_rerank as prr
    warm(p3, prr)
    # the per-item character cap must match on both sides or "N items" is not one budget
    assert p3.MAX_CTX_CHARS == config.CONTEXT_CHUNK_CHARS, \
        f"per-item char cap differs: p3={p3.MAX_CTX_CHARS} p2rr={config.CONTEXT_CHUNK_CHARS}"
    patch_thread_local_conn(p3)

    arms = []
    for b in budgets:
        arms.append((f"p2rr_k{b}", "p2rr", b))
        arms.append((f"p3_k{b}", "p3", b))

    all_rows, summary, knobs_used = [], {}, {}
    for name, kind, b in arms:
        rows, knobs = run_arm(name, kind, b, qs, p3, prr, args.workers)
        all_rows += rows
        summary[name] = {**agg(rows), "kind": kind, "budget": b, "by_tier": by_tier(rows)}
        knobs_used[name] = knobs

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"eval_{args.tag}.jsonl"
    with wopen(out) as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*100}\nPER-ARM  ({args.tag})  n={len(qs)}  set-F1 scorer, gold = member sets"
          f"\n{'='*100}")
    print(f"{'arm':<12}{'items':>6}{'exact':>8}{'recall':>8}{'prec':>8}{'setF1':>8}"
          f"{'prompt':>9}{'compl':>7}{'plan':>7}{'tok/q':>9}{'graph':>8}{'lat':>7}")
    print("-" * 100)
    for name, _, _ in arms:
        s = summary[name]
        print(f"{name:<12}{s['budget']:>6}{s['exact']:>5}/{s['scored']:<2}{s['recall']:>8.3f}"
              f"{s['precision']:>8.3f}{s['f1']:>8.3f}{s['prompt']:>9,.0f}"
              f"{s['completion']:>7,.0f}{s['planner']:>7,.0f}{s['total_tokens']:>9,.0f}"
              f"{s['graph_routed']:>6}/{s['scored']:<2}{s['latency_s']:>7.1f}")

    pairings = [(f"p2rr_k{a}", f"p3_k{c}") for a, c in
                [(budgets[0], budgets[0]), (budgets[0], budgets[-1]),
                 (budgets[-1], budgets[-1])]]
    print(f"\n{'='*100}\nPAIRINGS — Basic RAG (reranked) vs GraphRAG\n{'='*100}")
    print(f"{'comparison':<34}{'P2 setF1':>10}{'P3 setF1':>10}{'delta':>9}"
          f"{'P2 tok/q':>11}{'P3 tok/q':>11}{'tok delta':>16}")
    print("-" * 100)
    for pa, pb in pairings:
        if pa not in summary or pb not in summary:
            continue
        A, B = summary[pa], summary[pb]
        label = f"P2={A['budget']} chunks vs P3={B['budget']} chunks"
        d = B["f1"] - A["f1"]
        tk = B["total_tokens"] - A["total_tokens"]
        pct = 100 * tk / max(1.0, A["total_tokens"])
        print(f"{label:<34}{A['f1']:>10.3f}{B['f1']:>10.3f}{d:>+9.3f}"
              f"{A['total_tokens']:>11,.0f}{B['total_tokens']:>11,.0f}"
              f"{tk:>+11,.0f} ({pct:+.0f}%)")

    print(f"\n{'='*100}\nBY TIER — set-F1 (tokens/question)\n{'='*100}")
    tiers = ("single_doc", "single_dir", "multi_hop")
    print(f"{'arm':<12}" + "".join(f"{t:>26}" for t in tiers))
    print("-" * 100)
    for name, _, _ in arms:
        line = f"{name:<12}"
        for t in tiers:
            d = summary[name]["by_tier"].get(t)
            line += f"{'-':>26}" if not d else \
                f"{d['f1']:>15.3f} ({d['total_tokens']:>6,.0f})"
        print(line)

    meta = {"tag": args.tag, "n": len(qs), "questions": args.questions,
            "model": config.GEMINI_MODEL, "temperature": config.TEMPERATURE,
            "budgets": budgets, "workers": args.workers, "knobs": knobs_used,
            "baseline": "pipeline2rr_rerank (cross-encoder, pool 50)",
            "summary": summary}
    json.dump(meta, wopen(OUTDIR / f"summary_{args.tag}.json"), indent=2)
    print(f"\nwrote {out}\nwrote {OUTDIR / f'summary_{args.tag}.json'}")
    errs = sum(summary[n]["errors"] for n, _, _ in arms)
    if errs:
        print(f"\nWARNING: {errs} question-arm cells failed after one retry and are scored 0. "
              f"They are in the jsonl with an `error` field — do not read them as model misses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
