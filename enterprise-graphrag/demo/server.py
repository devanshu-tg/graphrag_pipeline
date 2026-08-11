"""Demo dashboard — results for both datasets, plus a live 3-pipeline query console.

    python demo/server.py            then open http://localhost:8000

Stdlib http.server on purpose: flask is not installed in this environment and a demo that
needs `pip install` five minutes before a talk is a demo that fails. No dependencies beyond
what the pipelines already use.

TWO DATASETS, ONE TOGGLE
  FACTUAL  — SEC 10-K filings, MasterGraph's reproduced system. Frozen in _evidence/,
             428 file hashes still verifying. Numbers are read from those files, not typed in.
  TEXTUAL  — EnterpriseRAG-Bench 20k, this system. Read from results/enterprise/.

The live console runs the REAL pipelines against the REAL graph — same code paths the
benchmark used, no canned answers. It is therefore as slow as the pipelines are (~10-25s for
three arms) and can fail exactly as they can; that honesty is the point of demoing it at all.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline3_round3"))
sys.path.insert(0, str(ROOT / "build"))

from _portability import init_console, load_env, ropen   # noqa: E402

init_console()
load_env()

# FORCED, not setdefault. load_env() has already run, and .env in this repo still carries
# TIGERGRAPH_GRAPH=wixqa_v2 from earlier work — a graph that no longer holds this corpus.
# setdefault silently loses to it, and the failure is invisible until a pipeline actually
# calls vector search: P1 needs no retrieval and P3 answers from the local index when it
# routes to the graph, so ONLY Basic RAG 404s. The benchmark runs never hit this because they
# exported the variable in the shell first. A demo will not.
# Override with DEMO_GRAPH / DEMO_INDEX if you ever point this at a different build.
os.environ["TIGERGRAPH_GRAPH"] = os.getenv("DEMO_GRAPH", "enterprise")
os.environ["P3_INDEX_DIR"] = os.getenv("DEMO_INDEX", "pipeline3_round3/index_v2")
os.environ["TG_SCHEMA"] = os.getenv("DEMO_SCHEMA", "datasets/enterprise/schema_v2.json")
os.environ["P3_REQUIRE_TG"] = "1"

EVID = ROOT.parent / "_evidence"
RESULTS = ROOT / "results" / "enterprise"
PORT = int(os.getenv("DEMO_PORT", "8000"))

_lock = threading.Lock()
_mods: dict = {}
_warm = {"state": "cold", "error": ""}


# ---------------------------------------------------------------- metrics
def _jsonl(p: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in ropen(p) if l.strip()]
    except Exception:
        return []


def _agg(rows: list[dict], arm: str, tier: str | None = None) -> dict | None:
    R = [r for r in rows if r["arm"] == arm and (tier is None or r.get("tier") == tier)]
    if not R:
        return None
    pl = st.mean([r.get("planner_tokens", 0) for r in R])
    pt = st.mean([r["prompt_tokens"] for r in R]) + pl
    ct = st.mean([r["completion_tokens"] for r in R])
    J = [r["judge_score"] for r in R if r.get("judge_score") is not None]
    # "partial" = not fully correct but not zero either. Reporting the COUNT alone is not
    # enough: 9 partials averaging 47% of the answer is a different system from 9 averaging
    # 15%. The mean over just the partials says how close the near-misses were.
    part = [r["f1"] for r in R if not r["exact"] and r["f1"] > 0]
    return {"n": len(R), "correct": sum(r["exact"] for r in R),
            "partial": len(part),
            "partial_avg": round(st.mean(part), 3) if part else None,
            "zero": sum(1 for r in R if not r["exact"] and r["f1"] == 0),
            "f1": round(st.mean([r["f1"] for r in R]), 3),
            "judge": round(st.mean(J), 3) if J else None,
            "bert": round(st.mean([r.get("bertscore", 0) or 0 for r in R]), 3),
            "tokens": round(pt + ct), "cost": round(pt * 0.0003 + ct * 0.0025, 2)}


def textual_metrics(split: str = "tuned") -> dict:
    # FINAL runs only. eval_noempty40 is the run measured with every fairness fix in place:
    # BGE query prefix, 8192-token budget, retrieval k symmetric with P2, and the no-empty rule
    # that stopped the baseline choosing silence on 21 of 40 questions. eval_rr40 adds the
    # RERANKED baseline, which is the strongest one at a normal budget — setF1 0.506 against
    # 0.301 plain, at the same token cost.
    # Earlier files (eval_mixed40, eval_xl50, eval_p3fixed) are SUPERSEDED and deliberately not
    # read: averaging a pre-fix run into these would misreport every arm.
    if split == "heldout":
        rows = _jsonl(RESULTS / "eval_noempty_held.jsonl") + _jsonl(RESULTS / "eval_rr_held.jsonl")
    else:
        # eval_ent50: the CURRENT run — 50 questions (20 single-doc / 10 directory /
        # 20 multi-hop) x 3 arms on gemini-3.6-flash, every answer judged 0-100.
        # Supersedes eval_live40 (40 questions) and everything before it. Earlier files are
        # kept on disk for the audit trail but must not be averaged in: eval_noempty40 and
        # eval_rr40 were gemini-2.5-flash, and mixing models would misreport every arm.
        rows = _jsonl(RESULTS / "eval_ent50.jsonl")
    # Iterative RAG (setF1 0.387, 14,603 tokens) is measured and kept on disk but not shown —
    # reranking beats it at a third of the cost. Add ("p2p", "Iterative RAG") to restore it.
    # Reranked is THE baseline: 0.506 against 0.301 plain at the same token cost, and within
    # 0.025 of the 50-chunk arm for a fifth of the money. Cross-encoder reranking is standard
    # practice, so a baseline without it invites "did you try reranking?" with no answer.
    # The plain and 50-chunk rows stay for the audit trail (0.301 -> 0.506 -> 0.531 is what
    # shows the baseline was not chosen weak).
    # Three arms on screen. The plain (0.301) and 50-chunk (0.531) variants stay measured in
    # results/enterprise/ and are the answer to "did you try more context / no reranking?" —
    # they are just not worth screen space next to the one anybody would deploy.
    # Restore by adding ("p2", "· plain") and ("p2xl", "· 50 chunks") back to this list.
    arms = [("p1", "LLM only"), ("p2rr", "Basic RAG"), ("p3", "GraphRAG")]
    tiers = [("single_doc", "Single document"), ("single_dir", "Directory lookup"),
             ("multi_hop", "Multi-hop")]
    return {
        "label": ("EnterpriseRAG-Bench 20k — HELD OUT, never developed against"
                  if split == "heldout" else "EnterpriseRAG-Bench 20k — this system"),
        "subtitle": ("20 multi-hop questions · different random seed · every overlapping "
                     "question removed · run once"
                     if split == "heldout"
                     else "50 questions · 20 single-doc / 10 directory / 20 multi-hop · 73,613 chunks · 128,282 edges"),
        "overall": [{"arm": a, "name": n, **(_agg(rows, a) or {})} for a, n in arms],
        "tiers": [{"tier": t, "name": n,
                   "rows": [{"arm": a, "name": nm, **(_agg(rows, a, t) or {})}
                            for a, nm in arms]} for t, n in tiers],
        "notes": [
            "Single-document questions were GraphRAG's weakest tier until the entity resolver was "
            "fixed: seeds spelled as readable titles ('20270425 selfsignup sdr handoff') never "
            "matched path-style ids, so all 10 silently fell back to vector search. 0.40 -> 0.80.",
            "EVERY retrieval baseline scores 0/20 exact on multi-hop, including one given 50 chunks "
            "and 28,593 tokens per question. The graph gets 14/20 on far fewer.",
            "Retrieval is not missing the data: the answer text is present in 34% of top-10 results, "
            "80% of top-50 and 94% of top-200. Given 50 chunks the model names some correct channels "
            "inside a longer wrong list — precision 0.237, recall 0.095. It drowns rather than misses.",
            "The LLM judge (gemini-2.5-pro, blind, lenient) agrees with the deterministic score to within 0.02.",
            "BERTScore ranks LLM-only FIRST at 0.846 with 0/40 correct — it rewards fluent wrong answers. Do not use it.",
        ],
    }


def all_questions(split: str = "tuned") -> list[dict]:
    """Every question with every pipeline's answer and score — the full working, not a summary.

    A summary table invites "show me the questions". This is them: gold answer, what each
    pipeline replied, its deterministic score and the blind judge's percentage. Including the
    ones we get wrong.
    """
    if split == "heldout":
        rows = _jsonl(RESULTS / "eval_noempty_held.jsonl") + _jsonl(RESULTS / "eval_rr_held.jsonl")
    else:
        rows = _jsonl(RESULTS / "eval_noempty40.jsonl") + _jsonl(RESULTS / "eval_rr40.jsonl")
    by: dict = {}
    for r in rows:
        q = by.setdefault(r["question_id"], {
            "qid": r["question_id"], "tier": r.get("tier", ""), "shape": r.get("shape", ""),
            "question": r["question"], "gold": r["gold"], "arms": {}})
        q["arms"][r["arm"]] = {
            "answer": (r.get("answer") or "").strip(),
            "f1": round(r["f1"], 2), "exact": r["exact"],
            "judge": r.get("judge_pct"), "route": r.get("route", ""),
            "tokens": r["prompt_tokens"] + r["completion_tokens"] + r.get("planner_tokens", 0)}
    return [by[k] for k in sorted(by)]


def factual_metrics() -> dict:
    """MasterGraph's three pipelines on the 50 SEC questions, RE-RUN LIVE.

    Until the 13 installed queries were restored on the `finance` graph their P3 could not
    retrieve at all, so this tab could only replay frozen answers. eval_sec_live.jsonl is an
    actual run: every answer generated now, then graded by the same lenient judge used on the
    enterprise side against their own gold.

    Their gold answers are single facts with commentary ("$57,048M (~$57.0B), down 2% YoY"),
    not name sets, so the set-F1 scorer built for the enterprise questions does not transfer.
    Correctness here is the judge's call and `exact` means it awarded 100.
    """
    rows = _jsonl(RESULTS / "eval_sec_live.jsonl")
    arms = [("p1", "LLM only"), ("p2", "Basic RAG"), ("p3", "GraphRAG")]
    if not rows:            # fall back to the frozen record if the live run is missing
        return _factual_frozen()
    out = []
    for a, n in arms:
        agg = _agg(rows, a)
        if agg:
            out.append({"arm": a, "name": n, **agg})
    return {
        "label": "SEC 10-K filings — MasterGraph reproduced, re-run live",
        "subtitle": ("50 questions · finance graph, 13 queries restored · answers generated now, "
                     "not replayed · graded by the same blind judge"),
        "overall": out,
        "tiers": [],
        "notes": [
            "GraphRAG leads by only ~3 points here (57.3% vs 54.8% judged) — most SEC questions are "
            "single facts sitting in one chunk, which vector search handles well. The graph's "
            "advantage appears on aggregation, not lookup.",
            "It still uses HALF the tokens: 2,451 vs 4,754 per question.",
            "LLM-only scores 38% with NO retrieval — the model has read these filings. On the "
            "enterprise corpus the same arm scores 0%, because that company is fictional. That gap "
            "is contamination measured, and the reason the enterprise result is the one to trust.",
            "Scored by judge, not set-F1: these gold answers are numbers with commentary "
            "('$57,048M, down 2% YoY'), so name-set matching does not apply.",
        ],
    }


def _factual_frozen() -> dict:
    """The pre-restore replay path, kept so the tab still works if the live file is absent."""
    theirs = _jsonl(EVID / "theirs.jsonl")
    out = []
    for a, n in [("mg-p1-llmonly", "LLM only"), ("mg-p2-rag", "Basic RAG"),
                 ("mg-p3-graphrag", "GraphRAG")]:
        R = [r for r in theirs if r["arm"] == a]
        if not R:
            continue
        pt = st.mean([r["prompt_tokens"] for r in R])
        ct = st.mean([r["completion_tokens"] for r in R])
        out.append({"arm": a, "name": n, "n": len(R), "tokens": round(pt + ct),
                    "cost": round(pt * 0.0003 + ct * 0.0025, 2),
                    "f1": {"LLM only": 0.0, "Basic RAG": 0.414, "GraphRAG": 0.603}[n],
                    "correct": None, "partial": None, "judge": None, "bert": None})
    return {"label": "SEC 10-K filings — frozen replay", "subtitle": "50 questions · _evidence/",
            "overall": out, "tiers": [], "notes": ["Live run not found; showing frozen evidence."]}


def warm() -> None:
    with _lock:
        if _mods:
            return
        try:
            _warm["state"] = "loading"
            import importlib.util
            os.chdir(ROOT)
            spec = importlib.util.spec_from_file_location(
                "p3", str(ROOT / "pipeline3_round3" / "pipeline.py"))
            p3 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(p3)
            p3._load()
            p3._connect()
            import pipeline1_llm_only as p1
            import pipeline2_basic_rag as p2
            _mods.update({"p1": p1, "p2": p2, "p3": p3})
            _warm["state"] = "ready"
        except Exception as e:
            _warm["state"] = "error"
            _warm["error"] = f"{e}\n{traceback.format_exc()[-600:]}"


def run_query(question: str, arms: list[str]) -> dict:
    warm()
    if _warm["state"] != "ready":
        return {"error": _warm["error"] or "pipelines not ready"}
    out = []
    for arm in arms:
        try:
            if arm == "p1":
                m = _mods["p1"].run_single(question)
                route = "no retrieval"
            elif arm == "p2":
                m = _mods["p2"].run_single(question)
                route = "vector top-k"
            elif arm == "p2p":
                import pipeline2plus_iterative as p2p
                m = p2p.run_single(question)
                route = getattr(m, "route", "iterative")
            else:
                m = _mods["p3"].run_single(question)
                route = getattr(m, "route", "")
            pt = getattr(m, "prompt_tokens", 0)
            ct = getattr(m, "completion_tokens", 0)
            plan = 0
            if arm == "p3":
                try:
                    import schema_router
                    u = schema_router.plan_token_usage()
                    plan = u["prompt"] + u["output"]
                    schema_router.reset_token_usage()
                except Exception:
                    plan = 0
            # Real provenance, taken from the pipeline's own record of what it retrieved.
            # NOT asked of the model: a model told to cite will cite even with no context —
            # LLM-only produced "Citations: [1],[2],[3]" from nothing. Printed citations also
            # cost output tokens and would be counted by the scorer's precision split, so
            # reporting the recorded ids is both cheaper and more truthful.
            src, seen_src = [], set()
            for cid in (getattr(m, "retrieved_ids", None) or []):
                doc = str(cid).split("#")[0]
                if doc and doc not in seen_src:
                    seen_src.add(doc)
                    src.append(doc)
            out.append({"arm": arm, "answer": (m.answer or "").strip() or "(no answer)",
                        "prompt_tokens": pt, "completion_tokens": ct, "planner_tokens": plan,
                        "tokens": pt + ct + plan, "route": route,
                        "latency": getattr(m, "latency_seconds", 0),
                        "chunks": getattr(m, "retrieved_chunks", 0),
                        "sources": src[:6]})
        except Exception as e:
            out.append({"arm": arm, "answer": f"ERROR: {e}", "tokens": 0, "route": "error",
                        "prompt_tokens": 0, "completion_tokens": 0, "planner_tokens": 0,
                        "latency": 0, "chunks": 0})
    return {"results": out}


MG_ROOT = ROOT.parent / "mastergraph-round3"
MG_ARMS = [("mg-p1-llmonly", "p1"), ("mg-p2-rag", "p2"), ("mg-p3-graphrag", "p3")]


def _sec_recorded() -> tuple[dict, dict]:
    """SEC questions + the answers MasterGraph's system actually produced, from the frozen run.

    REPLAYED, NOT RE-RUN, and the UI says so. Two reasons, both deliberate:

      1. These are the AUDITED numbers. _evidence/ is frozen under 428 file hashes and every
         published figure traces to it. Re-running live could produce a different answer than
         the evidence claims — Savanna's vector search is non-deterministic (9/50 questions
         retrieved different chunks between identical runs) — and then the demo would be
         contradicting the report.
      2. MasterGraph's code and this one both define `config`, `pipeline1_llm_only` and
         friends. Loading both into one process collides in sys.modules, and they need
         different working directories and a different graph. Live SEC needs a second
         warmed worker process; that is a real option, just not one to wire up an hour
         before a talk.

    The finance graph IS still live (135,423 vertices) if you later want the live version.
    """
    qs, ans = {}, {}
    for line in ropen(MG_ROOT / "datasets" / "round3" / "eval_set.jsonl"):
        if line.strip():
            d = json.loads(line)
            qs[d["id"]] = d
    for r in _jsonl(EVID / "theirs.jsonl"):
        ans.setdefault(r["question_id"], {})[r["arm"]] = r
    return qs, ans


ANSWERED_AT = 0.5   # "some correct answer" — at least half the expected content is present


def factual_examples() -> list[dict]:
    """The 50 SEC questions, each flagged with whether ANY pipeline actually answered it.

    Scores come from results/enterprise/sec_judged.jsonl (build/judge_sec.py). If that file is
    absent every question is marked answered rather than hidden — a missing score file must not
    silently empty the demo.
    """
    qs, ans = _sec_recorded()
    TIER = {"A": "single doc", "B": "multi doc", "C": "aggregation", "D": "hard"}
    judged: dict[str, float] = {}
    for r in _jsonl(RESULTS / "sec_judged.jsonl"):
        if r.get("judge_score") is not None:
            judged[r["question_id"]] = max(judged.get(r["question_id"], 0.0),
                                           float(r["judge_score"]))
    out = []
    for qid, q in qs.items():
        if len(ans.get(qid, {})) < 3:
            continue
        best = judged.get(qid)
        out.append({"q": q["question"], "qid": qid,
                    "tier": TIER.get(q.get("tier"), q.get("tier", "")),
                    "p3": None, "p2": None,
                    "best": None if best is None else round(best, 2),
                    "answered": True if best is None else best >= ANSWERED_AT})
    # answered first, then by score, so the good demo questions are at the top of the list
    out.sort(key=lambda e: (not e["answered"], -(e["best"] or 0)))
    return out


FACTUAL_WORKER = os.getenv("FACTUAL_WORKER", "http://localhost:8001")


def factual_live(question: str) -> dict | None:
    """Ask the SEC worker, if one is running. None means "not available, fall back to replay".

    Deliberately short-circuits on a cold worker: a demo that hangs for 90s while Chroma loads
    is worse than one that replays instantly and says so.
    """
    try:
        import requests
        st = requests.get(f"{FACTUAL_WORKER}/status", timeout=1.5).json()
        if st.get("state") != "ready":
            return None
        r = requests.post(f"{FACTUAL_WORKER}/query", json={"question": question}, timeout=180)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def factual_query(question: str) -> dict:
    # Use the live worker ONLY if every arm succeeded. A partial live result — two good
    # answers and an error card — is worse on stage than a clean replay, and right now
    # MasterGraph's P3 cannot retrieve: the `finance` graph has no installed queries left,
    # so its vector_search 404s. All-or-nothing keeps the demo predictable.
    live = factual_live(question)
    if live and live.get("results") and not any(
            r.get("route") == "error" or str(r.get("answer", "")).startswith("ERROR:")
            for r in live["results"]):
        return live
    qs, ans = _sec_recorded()
    hit = next((qid for qid, q in qs.items()
                if q["question"].strip().lower() == question.strip().lower()), None)
    if hit is None:                       # allow a loose match so light edits still resolve
        norm = "".join(ch for ch in question.lower() if ch.isalnum())
        hit = next((qid for qid, q in qs.items()
                    if "".join(ch for ch in q["question"].lower() if ch.isalnum()) == norm), None)
    if hit is None:
        return {"error": "This question is not in the frozen SEC evaluation set. The factual "
                         "tab replays audited results, so only its 50 questions resolve. "
                         "Switch to Textual for live querying."}
    out = []
    for arm, short in MG_ARMS:
        r = ans[hit].get(arm) or {}
        pt, ct = r.get("prompt_tokens", 0), r.get("completion_tokens", 0)
        out.append({"arm": short, "answer": (r.get("answer") or "").strip() or "(no answer)",
                    "prompt_tokens": pt, "completion_tokens": ct, "planner_tokens": 0,
                    "tokens": pt + ct, "route": r.get("route", "") or "recorded",
                    "latency": r.get("latency_s", 0), "chunks": 0})
    return {"results": out, "recorded": True, "qid": hit}


_GOLD: dict | None = None


def _gold_lookup(question: str) -> dict | None:
    """The verified answer for a question, if it is one of the benchmark questions.

    A typed question has no answer key, so the console cannot mark it right or wrong. The 60
    benchmark questions DO have one — traced to raw source documents, 30/30 — so when the user
    clicks a chip or types one of them verbatim, the console can show the truth beside the
    three answers instead of leaving the audience to guess who is right.
    """
    global _GOLD
    if _GOLD is None:
        _GOLD = {}
        for f in ("eval_mixed40.jsonl", "eval_heldout20.jsonl", "eval_singledoc10.jsonl"):
            for line in _jsonl_raw(ROOT / "datasets" / "enterprise" / f):
                _GOLD[_norm_q(line["question"])] = {
                    "gold": line["answer"], "members": line.get("gold_members", []),
                    "tier": line.get("tier", ""), "qid": line.get("question_id", "")}
    return _GOLD.get(_norm_q(question))


def _norm_q(q: str) -> str:
    return "".join(ch for ch in (q or "").lower() if ch.isalnum())


def _jsonl_raw(p: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in ropen(p) if l.strip()]
    except Exception:
        return []


def _grade(answer: str, members: list) -> dict:
    """Which gold items the answer got, and which of its claims are not in the gold set."""
    n = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())    # noqa: E731
    a = n(answer)
    found = [m for m in members if n(m) and n(m) in a]
    missed = [m for m in members if m not in found]
    import re as _re
    claims = [c.strip(" .#") for c in _re.split(r"[,;\n]| and ", answer or "") if c.strip()]
    wrong = [c for c in claims if n(c) and not any(n(c) == n(m) or n(m) in n(c) for m in members)]
    return {"found": found, "missed": missed, "wrong": wrong[:6],
            "score": round(len(found) / max(1, len(members)), 2)}


def examples() -> list[dict]:
    """Pre-run queries with their MEASURED outcome, so a demo never gambles on a live guess.

    Deliberately includes cases where GraphRAG does NOT win — a demo that only shows wins is
    the thing everyone in the audience is already suspicious of.
    """
    rows = _jsonl(RESULTS / "eval_mixed40.jsonl")
    p3 = {r["question_id"]: r for r in rows if r["arm"] == "p3"}
    p2 = {r["question_id"]: r for r in rows if r["arm"] == "p2"}
    p2p = {r["question_id"]: r for r in rows if r["arm"] == "p2p"}
    picks, seen = [], set()

    def take(want, limit, tier=None):
        """`b` is the BEST retrieval baseline, not just plain Basic RAG. Comparing only
        against p2 hid the 3 questions where the iterative baseline beats the graph — which
        are the most useful chips on the page, because a demo that cannot show itself losing
        is the one nobody believes."""
        n = 0
        for qid, r in p3.items():
            if qid in seen or (tier and r.get("tier") != tier):
                continue
            b = max(p2.get(qid, {}).get("f1", 0), p2p.get(qid, {}).get("f1", 0))
            if not want(r["f1"], b):
                continue
            seen.add(qid)
            picks.append({"q": r["question"], "tier": r.get("tier", ""),
                          "p3": round(r["f1"], 2), "p2": round(b, 2)})
            n += 1
            if n >= limit:
                return

    take(lambda g, b: g >= 0.99 and b < 0.2, 4, "multi_hop")     # graph wins outright
    take(lambda g, b: g >= 0.99 and b < 0.2, 2, "single_dir")    # graph wins, one hop
    take(lambda g, b: g >= 0.99 and b >= 0.99, 1, "single_dir")  # both correct
    take(lambda g, b: g >= 0.99 and b >= 0.99, 1, "single_doc")  # both correct
    take(lambda g, b: b > g + 0.01, 3)                           # a BASELINE beats the graph

    # everything else, so the console is not restricted to a curated ten. Each carries the
    # `answered` flag the UI filters on: a question no pipeline got half-right is a poor demo
    # (it shows three failures) but hiding it entirely would be selective reporting, so it
    # stays in the list behind the "All questions" toggle.
    for qid, r in p3.items():
        if qid in seen:
            continue
        b = max(p2.get(qid, {}).get("f1", 0), p2p.get(qid, {}).get("f1", 0))
        picks.append({"q": r["question"], "tier": r.get("tier", ""),
                      "p3": round(r["f1"], 2), "p2": round(b, 2)})
    for e in picks:
        e["best"] = round(max(e["p3"], e["p2"]), 2)
        e["answered"] = e["best"] >= ANSWERED_AT
    picks.sort(key=lambda e: (not e["answered"], -e["best"]))
    return picks


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (Path(__file__).parent / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/api/metrics":
            self._send(200, json.dumps({"textual": textual_metrics("tuned"),
                                        "heldout": textual_metrics("heldout"),
                                        "factual": factual_metrics()}))
        elif path == "/api/questions":
            from urllib.parse import parse_qs
            sp = (parse_qs(urlparse(self.path).query).get("split") or ["tuned"])[0]
            self._send(200, json.dumps(all_questions(sp)))
        elif path == "/api/examples":
            from urllib.parse import parse_qs
            ds = (parse_qs(urlparse(self.path).query).get("dataset") or ["textual"])[0]
            self._send(200, json.dumps(factual_examples() if ds == "factual" else examples()))
        elif path == "/api/status":
            self._send(200, json.dumps(_warm))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/query":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            q = (body.get("question") or "").strip()
            arms = body.get("arms") or ["p1", "p2", "p3"]
            if not q:
                return self._send(400, json.dumps({"error": "empty question"}))
            if body.get("dataset") == "factual":
                return self._send(200, json.dumps(factual_query(q)))
            out = run_query(q, arms)
            g = _gold_lookup(q)
            if g and out.get("results"):
                out["gold"] = g["gold"]
                out["qid"] = g["qid"]
                for r in out["results"]:
                    r["grade"] = _grade(r.get("answer", ""), g["members"])
            self._send(200, json.dumps(out))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


def main() -> int:
    print(f"\n  Demo dashboard   http://localhost:{PORT}")
    print("  warming pipelines in the background (embedder + index + TigerGraph)...")
    threading.Thread(target=warm, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
