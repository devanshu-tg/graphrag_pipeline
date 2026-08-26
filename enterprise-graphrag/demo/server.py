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

# ONE file behind the summary table AND the question list, because they used to be different
# ones: the table read eval_ent50 (50 questions, gemini-3.6-flash) while the question list read
# eval_noempty40 + eval_rr40 (40 questions, gemini-2.5-flash). Only 23 questions overlapped, so
# the percentages on screen were never computed from the questions printed underneath them —
# and the 40-set still carries a malformed question ("...the message '1778881234'..."). Two
# readers, one constant: they cannot drift again without someone editing this line.
# Built by build/make_pub50.py, which gates on all three arms covering the same 50 ids.
# The *_regex file is the published set measured with the pattern router, and it is what
# demo/build_report.py renders the page tables from. The server must read the SAME file or
# the console replays answers to questions the tables no longer contain — which is exactly
# what happened after the 2026-08-25 question swap: the tables moved to MX51-MX54 and the
# replay still served the four questions they replaced.
PUB = ("eval_pub50_regex.jsonl"
       if (RESULTS / "eval_pub50_regex.jsonl").exists() else "eval_pub50.jsonl")

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
        # eval_pub50: the CURRENT run — 50 questions (20 single-doc / 10 directory /
        # 20 multi-hop) x 3 arms on gemini-3.6-flash, every answer judged 0-100.
        # Supersedes eval_ent50, which measured GraphRAG BEFORE the union fix: vector search
        # only ran when the graph came back empty, so P3 read 513 prompt tokens and the
        # "70% fewer tokens" headline was really "reads no chunks at all". Fixed, at a 3-chunk
        # cap, it is 2,733 against the baseline's 5,312 — 49%, and 50/50 rather than 49/50.
        # Earlier files stay on disk for the audit trail but must not be averaged in:
        # eval_noempty40 and eval_rr40 were gemini-2.5-flash, and mixing models would
        # misreport every arm.
        rows = _jsonl(RESULTS / PUB)
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
            "Multi-hop is the whole story: Basic RAG scores 26% there and 100% on directory "
            "lookups — same corpus, same model, same questions. The only variable is whether the "
            "question needs a join. GraphRAG holds 100% on both. On multi-hop the baseline gets "
            "0 of 20 exactly right.",
            "GraphRAG has ZERO complete failures here — 50 perfect, 0 partial, 0 wrong. Basic RAG "
            "has 13 zeros and 11 partials. A traversal either lands on the exact set or returns "
            "nothing, so it fails visibly; retrieval fails plausibly, which is harder to catch in "
            "production.",
            "GraphRAG reads 3 chunks, Basic RAG reads 10 — and still loses. The budgets are "
            "deliberately NOT equal: 10 is where the reranked baseline scores best (dropping it "
            "to 3 costs it hit@k 82.5% -> 75.0%), so the baseline is shown at its strongest. "
            "GraphRAG at 3 chunks costs 2,733 tokens against 5,312, i.e. 49% fewer while scoring "
            "100% against 52%. NOT measured: the baseline end-to-end at a 3-chunk budget.",
            "Of GraphRAG's 2,733 tokens, 1,185 are the planner call — 43% of its spend, and a "
            "fixed cost that does not shrink with the chunk cap. That is the honest floor on the "
            "token claim, not a number to hide.",
            "The three arms come from three separate runs (LLM-only and the baseline were not "
            "re-run for the 3-chunk measurement, because neither reads that setting). Same 50 "
            "questions, same generator, same judge, and build/make_pub50.py aborts unless all "
            "three cover identical question ids and wording.",
            "LLM-only scores 0.8% with no retrieval. The company is fictional, so nothing here is "
            "answerable from model memory — that is the leak gate, and it is what makes the rest "
            "of the numbers mean anything.",
            "The answer key was re-derived from 20,182 RAW documents without opening the graph, "
            "30/30 (build/verify_gold_independent.py). Gold was originally computed BY traversal, "
            "so without this check the graph would be graded against its own representation.",
            "On 20 held-out questions generated with a different seed and never developed against, "
            "multi-hop scored 0.717 versus 0.700 on the tuned set — ABOVE, not below. That is the "
            "test that caught overfitting on the SEC round, where the ranking reversed.",
            "Every fairness fix helped the BASELINE: the BGE query prefix, an 8,192-token budget "
            "(thinking was starving the answer), retrieval k matched to P2's, a no-empty rule, and "
            "cross-encoder reranking which took Basic RAG 0.30 to 0.53 at the same token cost.",
            "KNOWN GAP: all 50 questions are relationship traversals — the shape a graph is built "
            "for. SEC has a 'bridge' tier (entity then its number) where both systems fall to ~38%. "
            "That shape has never been tested on this data.",
            "BERTScore is shown but is NOT the number of record. Here it happens to rank the "
            "arms correctly (0.852 / 0.934 / 1.000), but on the SEC set it put LLM-only ABOVE "
            "Basic RAG (0.839 vs 0.830) while LLM-only got 5 of 50 right — it rewards fluent "
            "wrong answers. A metric that only works when you already know the answer is not "
            "evidence; exact-set and the blind judge are.",
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
        rows = _jsonl(RESULTS / PUB)          # same file as the table above it — see PUB
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
    """MasterGraph's three pipelines on the 50 SEC questions, re-run under a FAIR config.

    eval_sec_live was their config as shipped, and two settings in it were breaking their own
    numbers rather than measuring their architecture:

      MAX_TOKENS = 512. gemini-2.5-flash draws thinking AND output from one budget, so answers
      were cut at ~18 visible tokens, mid-number: 'JPMorgan Chase: $57.0 billion\\nExxonMobil:
      $18'. It truncated 19 of 50 BASELINE answers against 7 of ours — a long retrieved context
      provokes longer thinking, so the setting punished the arm that retrieves most. Our
      enterprise side runs 8,192.

      Their prompt permits refusal ('if the answer is not supported by them, say so'); ours
      forbids it. 11 of 50 answers on each retrieval arm were refusals scoring zero.

    eval_sec_fair_prompt fixes both. BOTH CHANGES HELPED THEIR BASELINE — Basic RAG went 54.8%
    to 58.2% — which is the point: the deficit was config, not architecture. We modified their
    prompt to get here, and that is disclosed on the page rather than buried. The changes are
    injected at runtime; mastergraph-round3/ stays frozen under 428 file hashes.

    Their gold answers are single facts with commentary ("$57,048M (~$57.0B), down 2% YoY"),
    not name sets, so the set-F1 scorer built for the enterprise questions does not transfer.
    Correctness here is the judge's call and `exact` means it awarded 100.
    """
    rows = _jsonl(RESULTS / "eval_sec_fair_prompt.jsonl")
    arms = [("p1", "LLM only"), ("p2", "Basic RAG"), ("p3", "GraphRAG")]
    if not rows:            # fall back to the frozen record if the live run is missing
        return _factual_frozen()
    out = []
    for a, n in arms:
        agg = _agg(rows, a)
        if agg:
            out.append({"arm": a, "name": n, **agg})
    # The 57.3% average was hiding a 4-way split in which the graph wins the hardest tier and
    # loses the easiest by 38 points. That split is the finding; the average is the noise.
    tiers = [("A", "Single lookup"), ("B", "Bridge · relation then number"),
             ("C", "Aggregation"), ("D", "Multi-part")]
    return {
        "label": "SEC 10-K filings — MasterGraph reproduced, re-run under a fair config",
        "subtitle": ("50 questions · finance graph, 13 queries restored · 8,192-token budget and "
                     "no-refusal prompt applied to ALL arms · graded by the same blind judge"),
        "overall": out,
        "tiers": [{"tier": t, "name": n,
                   "rows": [{"arm": a, "name": nm, **(_agg(rows, a, t) or {})}
                            for a, nm in arms]} for t, n in tiers],
        "notes": [
            "CONFIG, NOT ARCHITECTURE: their shipped config capped answers at 512 tokens, which "
            "truncated 19 of 50 Basic RAG answers mid-number against 7 of ours — it punished the "
            "arm that retrieves most. Raising it to 8,192 and removing the refusal option took "
            "their baseline from 54.8% to 58.2% and GraphRAG from 57.3% to 65.7%. We changed "
            "their prompt to run this; their source is untouched and stays hash-frozen.",
            "GraphRAG leads by 7.5 points on the average (65.7% vs 58.2%) — but the average is "
            "the least interesting number here. See the tiers.",
            "It uses HALF the tokens: 2,545 vs 4,846 per question.",
            "LLM-only scores 38% with NO retrieval — the model has read these filings. On the "
            "enterprise corpus the same arm scores 0%, because that company is fictional. That gap "
            "is contamination measured, and the reason the enterprise result is the one to trust.",
            "The judge also inflates every arm here: it gave 75% to 'Walmart $611,289M; Apple "
            "$383,285M' when the gold was $713.2B and $416.2B — real figures from the wrong year. "
            "For financial QA a wrong number is a wrong answer, so `exact` (23 of 50) is the "
            "figure to trust, not the percentage.",
            "Scored by judge, not set-F1: these gold answers are numbers with commentary "
            "('$57,048M, down 2% YoY'), so name-set matching does not apply.",
            "BY TIER — GraphRAG wins the hardest and LOSES the easiest. Aggregation (20 q, "
            "'every company audited by KPMG'): 76.1% vs 48.0%, and 12 exact against 3. Single "
            "lookup (8 q, 'Apple's net sales'): 61.9% vs 99.4% — retrieval is simply the better "
            "tool for one number in one filing, and the planner call is wasted overhead.",
            "The BRIDGE tier is where both struggle (16 q, 54.7% vs 45.9%): 'the company that "
            "appointed John Ternus as CEO — what were its total net sales?' Hop one is a "
            "relationship, hop two is a table cell. Neither tool does both well.",
            "SEC relations are THIN: audited_by 100 edges, uses_supplier 178, has_ceo just ONE in "
            "the whole graph. Enterprise participated_in has 88,188. Same idea, ~800x the density "
            "— which is why CEO bridge questions had nothing to traverse.",
            "18 of 50 answers are a NUMBER or DATE; on the enterprise set it is 0 of 50. A graph "
            "stores relationships, so it can point at the right filing but cannot pick the right "
            "table row. Two failures were exactly that: $344,582M for $332,238M, and a filing "
            "date read as the fiscal year end.",
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
            _install_ent_facts_capture(p3)
            import pipeline1_llm_only as p1
            import pipeline2_basic_rag as p2
            # The published baseline is the RERANKED one. The console sends arm "p2rr" and
            # without this key the dispatch below fell through to else: and ran GraphRAG, so
            # the "Basic RAG" row was a second GraphRAG run wearing the baseline's label.
            import pipeline2rr_rerank as p2rr
            # Build the cross-encoder HERE, during startup, not on the first user query. It
            # takes ~18s on a 2-vCPU box, and paying that inside a request meant the browser
            # aborted at 90s and the console silently replayed the recorded run - a card
            # reading "recorded" when nothing was wrong with the pipeline.
            try:
                import retrieval_common
                retrieval_common.reranker()
                print("[warm] cross-encoder ready")
            except Exception as _e:
                print(f"[warm] cross-encoder preload failed ({_e}); "
                      f"it will load on first use instead")
            _mods.update({"p1": p1, "p2": p2, "p2rr": p2rr, "p3": p3})
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
            elif arm == "p2rr":
                m = _mods["p2rr"].run_single(question)
                route = getattr(m, "route", "rerank")
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
                        "graph_facts": (_ent_facts_count(question)[0] if arm == "p3" else 0),
                        "graph_trail": (_ent_facts_count(question)[1] if arm == "p3" else ""),
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


# ---------------------------------------------------------------- live SEC judging
# The SEC tab could not show an accuracy figure, for a concrete reason: _grade() below is a
# substring matcher over a set of names, and these gold answers are numbers written as prose.
# "$416,161M (~$416.2B)" against a correct "$416,161 million" normalises to "416161m4162b" vs
# "416161million" — no substring, score 0.0. Every right answer would be marked wrong, which is
# why demo/build_static.py passes grade=False for this tab.
#
# The fix is the rubric that produced the PUBLISHED column, not a new one. build/judge_sec.py
# patches llm_judge.RUBRIC to treat units, separators, abbreviation and sensible rounding as
# equivalent; the same patch is applied here so a live verdict and the published table are
# scored by identical rules. The judge is gemini-3.1-pro-preview — a different family from the
# generator, so it stays blind.
#
# Costs one extra call per arm (~1-3s). SEC_JUDGE=0 turns it off and restores the em-dash.
SEC_JUDGE = os.getenv("SEC_JUDGE", "1") != "0"
_judge_mod = None


_ENT_FACTS: dict = {}


def _install_ent_facts_capture(p3mod) -> None:
    """Record the traversal block per question so the console can report graph facts instead of
    a chunk count. A graph route can legitimately read ZERO chunks - the traversal returns the
    answer - and "chunks 0" reads as a failure when it is the best case."""
    orig = p3mod._build_prompt

    def build(question, context, facts=""):
        _ENT_FACTS[question.strip()] = facts or ""
        return orig(question, context, facts)

    p3mod._build_prompt = build


def _ent_facts_count(question: str) -> tuple[int, str]:
    """(entities the traversal returned, a short traversal line) for a question."""
    import re as _r
    gf = _ENT_FACTS.get(question.strip(), "")
    if not gf:
        return 0, ""
    # shape: "Graph traversal [legs] (N results): a, b, c"
    tail = gf.rsplit(":", 1)[-1]
    names = [x.strip() for x in tail.split(",") if x.strip()]
    m = _r.search(r"\[([^\]]+)\]", gf)
    trail = f"{m.group(1)} \u21d2 {', '.join(names[:8])}" if (m and names) else ""
    return len(names), trail


def _sec_judge():
    """Import llm_judge once, pinned to the v2 rubric. None if unavailable.

    v2 is the rubric the PUBLISHED SEC column was scored under, and it exists for exactly the
    complaint this tab produced: v1 penalised answers for omitting DERIVED detail. Gold is
    written "$57,048M (~$57.0B), down 2% YoY" — one fact plus two annotations of it — and the
    question only asked for net income. v1 scored the correct "$57,048 million" at 0.85/0.95 for
    missing an annotation nobody asked for; v2 states outright that a restated unit, a bracketed
    approximation and a year-on-year change are annotations, not required facts.

    Set through the environment because llm_judge._rubric() reads LLM_JUDGE_RUBRIC per call —
    patching RUBRIC (v1) instead, as this first did, leaves v1 selected and the bug in place.
    """
    global _judge_mod
    if _judge_mod is not None or not SEC_JUDGE:
        return _judge_mod
    try:
        os.environ.setdefault("LLM_JUDGE_RUBRIC", "v2")
        import llm_judge
        print(f"[sec-judge] {llm_judge.JUDGE_MODEL}, rubric="
              f"{os.getenv('LLM_JUDGE_RUBRIC')} (same rubric as the published column)")
        _judge_mod = llm_judge
    except Exception as e:
        print(f"[sec-judge] unavailable, accuracy will stay blank: {e}")
        _judge_mod = None
    return _judge_mod


def _judge_factual(out: dict, question: str, gold: str) -> None:
    """Attach a grade to each arm, in place. Fails soft: no grade means the card shows an
    em-dash, which is what it did before — never a fabricated number."""
    j = _sec_judge()
    if not j or not gold:
        return
    out["gold"] = gold
    for r in out.get("results", []):
        try:
            score, why = j.judge_one(question, gold, r.get("answer", "") or "")
            if score < 0:                      # judge itself failed; stay silent
                continue
            r["grade"] = {"score": round(score / 100.0, 2), "why": why,
                          "found": [], "missed": [], "wrong": [], "judged": True}
        except Exception as e:
            print(f"[sec-judge] {r.get('arm')}: {e}")


def factual_query(question: str) -> dict:
    # Use the live worker ONLY if every arm succeeded. A partial live result — two good
    # answers and an error card — is worse on stage than a clean replay, and right now
    # MasterGraph's P3 cannot retrieve: the `finance` graph has no installed queries left,
    # so its vector_search 404s. All-or-nothing keeps the demo predictable.
    live = factual_live(question)
    if live and live.get("results") and not any(
            r.get("route") == "error" or str(r.get("answer", "")).startswith("ERROR:")
            for r in live["results"]):
        # The worker returns answers only — no gold, no grade. Resolve the key from the frozen
        # eval set and judge here, so the live tab reports accuracy under the same rubric the
        # published column used.
        qsl, _ = _sec_recorded()
        hitl = next((qid for qid, qq in qsl.items()
                     if qq["question"].strip().lower() == question.strip().lower()), None)
        if hitl:
            live["qid"] = hitl
            _judge_factual(live, question, qsl[hitl].get("answer", ""))
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
                    "latency": r.get("latency_s", 0),
                    # not recorded in the frozen SEC run; the page shows an em dash
                    "chunks": r.get("retrieved_chunks") or 0, "recorded": True})
    res = {"results": out, "recorded": True, "qid": hit}
    _judge_factual(res, question, qs[hit].get("answer", ""))
    return res


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
        # eval_mixed50 FIRST and, until this line, missing: the published set is the 50, and
        # 27 of them appear in none of the other three files. Clicking one of those chips
        # produced no CORRECT/WRONG badge at all — the console showed three answers and left
        # the audience to work out which was right. Ordering matters only for duplicates, and
        # the 23 shared questions carry identical gold in both files.
        for f in ("eval_mixed50.jsonl", "eval_mixed40.jsonl",
                  "eval_heldout20.jsonl", "eval_singledoc10.jsonl"):
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

    THIS is the list the page actually shows, and it used to read eval_mixed40 — a third file,
    different again from the summary table and from all_questions(). 17 of its 40 questions were
    not in the benchmark at all, so the chips under the table were largely questions the
    percentages above were never computed from. Now it reads PUB like everything else.

    Kept a category for questions the BASELINE wins, because a demo that cannot show itself
    losing is the one nobody believes. On this run it returns nothing: GraphRAG is 50/50, so no
    such question exists here. That is a property of this question set — all 50 are relationship
    traversals — not evidence that the category was removed. The SEC tab is where the graph
    loses, and it loses there in public.
    """
    rows = _jsonl(RESULTS / PUB)
    p3 = {r["question_id"]: r for r in rows if r["arm"] == "p3"}
    p2 = {r["question_id"]: r for r in rows if r["arm"] == "p2rr"}
    p2p: dict = {}          # the iterative baseline was not re-run for PUB; p2rr is the strongest
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


# ---------------------------------------------------------------- access control
# Off by default. Set DEMO_TOKEN and/or DEMO_RATE_PER_HOUR to arm them; the deploy config
# sets both. Kept deliberately small — no dependency, no store, no user accounts. It is not
# authentication, it is a spend limit with a shared password in front of it.
DEMO_TOKEN = os.getenv("DEMO_TOKEN", "")
DEMO_RATE = int(os.getenv("DEMO_RATE_PER_HOUR", "0"))     # 0 = unlimited
_hits: dict[str, list[float]] = {}
_hits_lock = threading.Lock()


def _client_ip(h) -> str:
    # Behind Fly/Railway/Render the socket peer is the proxy, so the real client only appears
    # in X-Forwarded-For. Taking the FIRST entry is right for those platforms, which append
    # rather than trust: a client-supplied XFF ends up later in the list. Do not copy this to
    # a platform that passes the header through unmodified — there the first entry is a lie.
    xff = h.headers.get("X-Forwarded-For", "")
    return (xff.split(",")[0].strip() if xff else h.client_address[0]) or "?"


def _gate(h) -> tuple[int, str] | None:
    """None if the request may proceed, else (status, message)."""
    # A browser on this machine sends no token — index.html has none to send — so arming
    # DEMO_TOKEN for a tunnel would 401 the operator's own dashboard, which reads as "the
    # pipeline is broken again". Loopback is exempt.
    #
    # The test is "no X-Forwarded-For AND the peer is loopback", not "peer is loopback": with
    # a tunnel in front, cloudflared connects FROM 127.0.0.1, so peer alone would exempt the
    # entire internet. XFF present means proxied, means enforce.
    if not h.headers.get("X-Forwarded-For", "") and \
            h.client_address[0] in ("127.0.0.1", "::1", "localhost"):
        return None
    if DEMO_TOKEN:
        from urllib.parse import parse_qs
        supplied = (h.headers.get("X-Demo-Token", "")
                    or (parse_qs(urlparse(h.path).query).get("token") or [""])[0])
        # compare_digest, not ==, so the check does not leak the token's length or prefix
        # through response timing.
        import hmac
        if not hmac.compare_digest(supplied, DEMO_TOKEN):
            return 401, "this demo needs an access token"
    if DEMO_RATE > 0:
        import time as _t
        now, ip = _t.time(), _client_ip(h)
        with _hits_lock:
            seen = [t for t in _hits.get(ip, []) if now - t < 3600]
            if len(seen) >= DEMO_RATE:
                wait = int((3600 - (now - seen[0])) / 60) + 1
                _hits[ip] = seen
                return 429, (f"rate limit: {DEMO_RATE} live queries per hour. "
                             f"Try again in ~{wait} min, or browse the recorded runs below.")
            seen.append(now)
            _hits[ip] = seen
            if len(_hits) > 5000:      # unbounded dict = a slow memory leak under a scan
                for k in [k for k, v in _hits.items() if not v or now - v[-1] > 3600]:
                    _hits.pop(k, None)
    return None


# Needed only when the published page and this service are on different origins — the static
# site on Netlify calling a tunnel or a VM. Same-origin use (opening this server directly)
# never triggers CORS, so the default of "unset" keeps local development untouched.
#
# An explicit origin, never "*": this endpoint costs ~8,500 tokens and a TigerGraph round trip
# per call, and a wildcard would let ANY page on the internet drive it from a visitor's
# browser, spending the key's quota without the visitor knowing.
DEMO_CORS_ORIGIN = os.getenv("DEMO_CORS_ORIGIN", "")


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        if DEMO_CORS_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", DEMO_CORS_ORIGIN)
            self.send_header("Vary", "Origin")

    def _send(self, code, body, ctype="application/json"):
        # EVERY write here can fail once the client has gone: the console aborts at 90s and
        # end_headers() flushes as eagerly as wfile.write(). Left unguarded, do_POST's handler
        # tried to report the real error with _send(500, ...), that threw BrokenPipeError too,
        # and the ORIGINAL exception was never logged - a failed request showed up as nothing
        # but a broken pipe. Swallowing a dead socket keeps the real traceback visible.
        data = body if isinstance(body, bytes) else str(body).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[send] client gone before the {code} response was written")

    def do_OPTIONS(self):
        # The browser sends this BEFORE the POST because /api/query carries a custom header
        # (X-Demo-Token), which makes it a non-simple request. Without a handler here
        # BaseHTTPRequestHandler answers 501 and the real POST is never sent — the query box
        # would fail with nothing in this server's log to show why.
        self.send_response(204)
        self._cors()
        if DEMO_CORS_ORIGIN:
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Demo-Token")
            self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            # Prefer the BUILT report page (docs/index.html from build_report.py): the light
            # two-tab scorecard, with the live URL and token already baked in. demo/index.html
            # is the older dark dashboard and is only a fallback for a tree that has never
            # been built. Netlify blocks deploys once account credits run out, so the tunnel
            # has to be able to serve the real page by itself.
            built = Path(__file__).resolve().parents[2] / "docs" / "index.html"
            src = built if built.exists() else Path(__file__).parent / "index.html"
            self._send(200, src.read_bytes(), "text/html; charset=utf-8")
        elif path == "/architecture-sec.png":
            # the page references this by relative path, so it must be served alongside it
            img = Path(__file__).resolve().parents[2] / "docs" / "architecture-sec.png"
            if not img.exists():
                img = Path(__file__).parent / "architecture-sec.png"
            if img.exists():
                self._send(200, img.read_bytes(), "image/png")
            else:
                self._send(404, json.dumps({"error": "not found"}))
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
        elif path.startswith("/api/") and path.endswith(".json"):
            # The built page's replay fallback fetches these by RELATIVE path, so when the page
            # is served from here (tunnel or localhost) they have to be served from here too.
            # They 404'd, which is why a failed live call surfaced as "no recorded answer for
            # that question" instead of quietly replaying the measured run. basename only: the
            # path is attacker-controlled and must not be able to walk out of docs/api.
            f = (Path(__file__).resolve().parents[2] / "docs" / "api"
                 / Path(path).name)
            if f.is_file():
                self._send(200, f.read_bytes(), "application/json")
            else:
                self._send(404, json.dumps({"error": "not found"}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/query":
            return self._send(404, json.dumps({"error": "not found"}))
        # Every call past this point spends money: three LLM calls and a TigerGraph round trip,
        # ~77s and ~8,500 tokens for a full three-arm query. On a laptop that is nobody's
        # problem. On a public URL it is an anonymous Gemini endpoint funded by whoever owns
        # the key, so the gate below is REQUIRED before this is exposed. Both checks are
        # inert unless their env var is set, so local development is unchanged.
        gate = _gate(self)
        if gate:
            return self._send(gate[0], json.dumps({"error": gate[1]}))
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
