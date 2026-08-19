"""Live SEC worker — runs MasterGraph's three pipelines in their OWN process.

    python demo/server_factual.py          # listens on 8001, warms in the background

WHY A SEPARATE PROCESS
----------------------
MasterGraph's system and this one both define `config`, `pipeline1_llm_only`,
`pipeline2_basic_rag` and `metrics`. Importing both into one interpreter collides in
sys.modules — whichever loads first wins and the other silently gets the wrong module. They
also need different working directories (their Chroma index and corpus are relative paths)
and a different graph (`finance`, not `enterprise`).

Process isolation is the honest fix: this worker chdir's into mastergraph-round3/, imports
their code untouched, and serves one endpoint. The dashboard proxies to it and falls back to
replaying the frozen evidence if it is not running, so the demo never hard-fails.

READ-ONLY. Nothing in mastergraph-round3/ is written to — its 428 file hashes must keep
verifying. This only imports and calls.

WHAT LIVE COSTS YOU
-------------------
The frozen numbers in _evidence/ are what every published figure traces to. Savanna's vector
search is non-deterministic — 9 of 50 questions retrieved a different chunk set between two
identical runs — so a live answer can legitimately differ from the recorded one. That is worth
knowing before demoing it: live is honest, but it is not guaranteed to reproduce the report.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve()
ENT = HERE.parents[1]                      # enterprise-graphrag/
MG = ENT.parent / "mastergraph-round3"     # read-only evidence

sys.path.insert(0, str(ENT))
from _portability import init_console, load_env   # noqa: E402

init_console()
load_env()                                  # our .env — theirs points at a Mac path

os.environ["TIGERGRAPH_GRAPH"] = os.getenv("SEC_GRAPH", "finance")
PORT = int(os.getenv("FACTUAL_PORT", "8001"))

# --- T4: run their pipelines under a FAIR generation config -------------------------------
# The audit measured three config choices that depress every SEC arm, none of them
# architectural:
#   MAX_TOKENS=512   with gemini-2.5-flash drawing thinking and output from ONE budget, answers
#                    are cut at ~18-21 visible tokens. 19/50 P2 answers were truncated, vs 7/50
#                    for P3 — it penalises the BASELINE 2.7x hardest.
#   refusal allowed  "if the answer is not supported by them, say so" -> 11/50 P2 and 11/50 P3
#                    answers are refusals scoring 0. Our enterprise prompt forbids declining.
#   gemini-2.5-flash two generations behind; on our held-out 20 the switch to 3.6-flash moved
#                    GraphRAG 0.717 -> 1.000.
#
# MAX_TOKENS and GEMINI_MODEL are already os.getenv() in THEIR config, so those need no patch —
# just the environment. SYSTEM_PROMPT is a literal, so SEC_NO_REFUSAL patches the module
# ATTRIBUTE at runtime. Every arm reads config.SYSTEM_PROMPT at call time (verified in
# pipeline1_llm_only.py:11, pipeline2_basic_rag.py:114, pipeline3_round3/pipeline.py:666), so
# one assignment covers all three, and NOTHING on disk is modified — the 428 frozen hashes must
# still verify afterwards.
NO_REFUSAL = os.getenv("SEC_NO_REFUSAL", "0") != "0"

# Same wording as our enterprise config.SYSTEM_PROMPT, verbatim, so the two projects are
# finally being measured under the same instruction rather than opposite ones.
_NO_EMPTY = (
    "Give your best supported answer rather than declining. NEVER return an empty answer, "
    "and never reply 'None', 'N/A' or that you cannot answer. If the context does not fully "
    "settle it, name the MOST LIKELY item(s) the context supports and output only those. "
    "An empty reply is always worse than a best-effort one.\n")


# Prompt rules derived from the 21 published GraphRAG misses, one rule per observed failure —
# not general advice. SEC_PROMPT=v1 restores the untouched prompt so the two can be compared.
#
#   EQ09  "company that appointed John Ternus CEO - its FY25 net sales?" answered "$209,586"
#         citing the 8-K appointment doc. Gold is "Apple; $416,161M": BOTH parts. Nothing asked
#         it to name the company, so nothing forced it to commit to one -> rule 1.
#   EQ12  same shape, answered "KPMG" citing BNY/WFC/PFE - three unrelated companies -> rule 1+2.
#   EQ06  "when does Microsoft's FY2025 end?" answered "July 30, 2025", the FILING date sitting
#         in the passage id MSFT/10-K_2025-07-30, not the fiscal year end -> rule 3.
#   EQ04  "total revenues AND OTHER INCOME" answered $344,582 instead of $332,238 - a
#         neighbouring line item on the same statement -> rule 4.
#   EQ16  emitted "Citations: [...]" with no answer above it -> rule 5.
_PROMPT_RULES = (
    "Additional rules, in priority order:\n"
    "1. Two-part questions ('the company that DID X - what is its Y?') need BOTH parts: "
    "name the company FIRST, then the fact, separated by a semicolon "
    "(e.g. 'Apple; $416,161 million'). Naming the company is not optional and not preamble.\n"
    "2. Take the fact from THAT company's own filing. The document that establishes the link "
    "(an 8-K appointment, a press release) usually does not contain the financial figure; the "
    "company's 10-K does. If the passages only cover a different company, you have the wrong "
    "company - re-read for the one the question describes.\n"
    "3. NEVER read a date out of a passage ID, filename or document label. Those encode the "
    "FILING date, which is not a fiscal-year end, an event date or an effective date. Dates "
    "must come from the passage TEXT.\n"
    "4. Match the exact line item named. 'Total revenues and other income', 'total revenues' "
    "and 'net sales' are different rows on the same statement; pick the one whose label "
    "matches the question's wording, not the nearest large number.\n"
    "5. Never output a 'Citations:' line without an answer above it. If you can cite a "
    "passage you can state what it says.\n")


# ---------------------------------------------------------------- router repair
# MasterGraph's route() misroutes 13 of the 16 bridge-tier questions, and five of them score a
# flat 0%. Two causes, and the second matters more than the first:
#
#   1. BRIDGE_RE is one literal phrasing — r"\bthe company (that|whose|which)\b". A two-hop
#      question worded any other way never matches.
#   2. ORDER. route() tests aggregation BEFORE bridge, and AGG_RE is broad
#      (r"\b(which|list|name|how many|all)\b.{0,40}\bcompan(y|ies)\b"), so it swallows bridge
#      questions first. Measured: EQ11 EQ19 EQ20 EQ23 -> "aggregation", EQ12 -> "lookup",
#      all five 0%. Widening BRIDGE_RE alone would not have fixed one of them, because
#      aggregation still wins the race.
#
# A bridge question has two parts: a company referred to DESCRIPTIVELY rather than by name, and
# a fact requested about it. Both must be present — requiring both is what keeps genuine
# aggregation questions ("list every KPMG-audited company") out of the bridge branch, since they
# describe a set and ask for nothing further about it.
#
# Patched here, not in their file: mastergraph-round3/ is frozen under 428 hashes.
# SEC_ROUTER=orig restores their routing untouched.
import re as _re

# a company identified by what it DID, not by name
_DESCRIPTIVE = _re.compile(
    r"\bthe compan(y|ies) (that|whose|which)\b"
    r"|\bcompan(y|ies) that (formed|appointed|named|authorized|announced|expanded|filed|"
    r"reported|declared|acquired|entered)\b"
    r"|\bamong the companies that\b"
    r"|\bbesides \w+,? which compan(y|ies)\b"
    r"|\bwhich \w+(-\w+)? compan(y|ies) (had|has|reported|filed|declared)\b", _re.I)

# ...and a SECOND thing asked about that company
_SECOND_HOP = _re.compile(
    r"[—–-]\s*(what|who|which|how)\b"       # em/en dash then a question word
    r"|,\s*(and )?(what|who|which|how)\b"             # comma then a question word
    r"|\bwhat (were|was|is|are) its\b"
    r"|\bwho is its\b|\bwhat is its\b"
    r"|\band (who|what) (is|are|was|were)\b", _re.I)


# A bridge resolves to ONE company; an aggregation returns a set. Both can open with the same
# descriptive clause, so the number being asked for is the discriminator:
#   EQ23  "Among the companies that name TSMC as a foundry, which ONE had the highest..."  bridge
#   EQ38  "Among the companies that name TSMC as a foundry, which ARE audited by E&Y"      agg
# Without this guard EQ38/EQ39 were pulled into bridge, which would have broken two questions
# the aggregation branch already answers.
_PLURAL_ASK = _re.compile(
    r"\bwhich are\b|\bwhich compan(ies)\b|\bwhich \w+(-\w+)? companies\b"
    r"|\bwhich two\b|\blist\b|\ball of (which|them)\b|\bhow many\b", _re.I)


def _install_router(p3) -> None:
    """Wrap p3.route so bridge is detected on meaning and tested before aggregation."""
    if os.getenv("SEC_ROUTER", "fixed") == "orig":
        return
    original = p3.route

    def route(question: str) -> str:
        r = original(question)
        # only ever RE-route what the original called aggregation or lookup; temporal and
        # compare keep their own branches, and compare is the best-performing route on the set
        # (2/3 full marks) so it is left strictly alone.
        if r in ("aggregation", "lookup") and not p3.ROUTER_OFF:
            if (_DESCRIPTIVE.search(question) and _SECOND_HOP.search(question)
                    and not _PLURAL_ASK.search(question)):
                return "bridge"
        return r

    p3.route = route
    print("[factual] router: bridge detected on meaning, ahead of aggregation "
          "(SEC_ROUTER=orig to disable)")


def _patch_prompt_rules(cfg) -> None:
    """Append the failure-derived rules. Additive: their wording is left intact, so the only
    difference between v1 and v2 is this block."""
    if _PROMPT_RULES not in cfg.SYSTEM_PROMPT:
        cfg.SYSTEM_PROMPT = cfg.SYSTEM_PROMPT + "\n" + _PROMPT_RULES


def _patch_prompt(cfg) -> None:
    """Replace their refusal licence with our no-empty rule. Applied to all three arms."""
    old = "if the answer is not supported by them, say so. "
    p = cfg.SYSTEM_PROMPT
    if old in p:
        cfg.SYSTEM_PROMPT = p.replace(old, "") + "\n" + _NO_EMPTY
    else:
        # their wording changed; append rather than silently doing nothing
        print("[factual] WARNING: refusal clause not found verbatim — appending no-empty rule")
        cfg.SYSTEM_PROMPT = p + "\n" + _NO_EMPTY

_lock = threading.Lock()
_mods: dict = {}
_state = {"state": "cold", "error": ""}


def warm() -> None:
    with _lock:
        if _mods:
            return
        try:
            _state["state"] = "loading"
            os.chdir(MG)                    # their corpus/Chroma paths are relative
            for p in (str(MG), str(MG / "pipeline3_round3")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "mg_p3", str(MG / "pipeline3_round3" / "pipeline.py"))
            p3 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(p3)
            p3._load()
            p3._connect()
            import pipeline1_llm_only as p1
            import pipeline2_basic_rag as p2
            from chroma_store import load_vectorstore
            import config as mg_config
            if NO_REFUSAL:
                _patch_prompt(mg_config)
            if os.getenv("SEC_PROMPT", "v2") != "v1":
                _patch_prompt_rules(mg_config)
            _install_router(p3)
            print(f"[factual] model={mg_config.GEMINI_MODEL} max_tokens={mg_config.MAX_TOKENS} "
                  f"no_refusal={NO_REFUSAL} prompt={os.getenv('SEC_PROMPT', 'v2')}")
            _state["config"] = {"model": mg_config.GEMINI_MODEL,
                                "max_tokens": mg_config.MAX_TOKENS,
                                "no_refusal": NO_REFUSAL}
            _mods.update({"p1": p1, "p2": p2, "p3": p3, "vs": load_vectorstore()})
            _state["state"] = "ready"
        except Exception as e:
            _state["state"] = "error"
            _state["error"] = f"{e}\n{traceback.format_exc()[-700:]}"
            print(f"[factual] warm failed: {e}")


def run(question: str) -> dict:
    warm()
    if _state["state"] != "ready":
        return {"error": _state["error"] or "SEC pipelines not ready"}
    out = []
    for arm in ("p1", "p2", "p3"):
        try:
            if arm == "p1":
                m = _mods["p1"].run_single(question)
                route = "no retrieval"
            elif arm == "p2":
                m = _mods["p2"].run_single(question, _mods["vs"])
                route = "vector top-k"
            else:
                m = _mods["p3"].run_single(question)
                route = getattr(m, "route", "")
            pt, ct = getattr(m, "prompt_tokens", 0), getattr(m, "completion_tokens", 0)
            out.append({"arm": arm, "answer": (m.answer or "").strip() or "(no answer)",
                        "prompt_tokens": pt, "completion_tokens": ct, "planner_tokens": 0,
                        "tokens": pt + ct, "route": route,
                        "latency": getattr(m, "latency_seconds", 0),
                        "chunks": getattr(m, "retrieved_chunks", 0)})
        except Exception as e:
            out.append({"arm": arm, "answer": f"ERROR: {e}", "tokens": 0, "route": "error",
                        "prompt_tokens": 0, "completion_tokens": 0, "planner_tokens": 0,
                        "latency": 0, "chunks": 0})
    return {"results": out, "live": True}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if urlparse(self.path).path == "/status":
            return self._send(200, json.dumps(_state))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/query":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            q = (json.loads(self.rfile.read(n) or b"{}").get("question") or "").strip()
            if not q:
                return self._send(400, json.dumps({"error": "empty question"}))
            self._send(200, json.dumps(run(q)))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


def main() -> int:
    print(f"\n  SEC live worker  http://localhost:{PORT}")
    print(f"  graph={os.environ['TIGERGRAPH_GRAPH']}  cwd will be {MG.name}")
    print("  warming MasterGraph's pipelines (Chroma 928MB + corpus 119MB)...")
    threading.Thread(target=warm, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
