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
