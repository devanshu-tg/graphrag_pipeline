"""Render the dashboard to a folder of flat files — no server, no secrets, no TigerGraph.

WHY: the live app needs a 450 MB index, a cross-encoder, a warmed TigerGraph workspace and a
Gemini key, all to serve a page whose numbers are 2.8 MB of JSONL that never change. Everything
except the query console is already static; this makes that literal.

WHAT SURVIVES: the tables, the tiers, the notes, the question chips, and — because every answer
was recorded — the query console itself. Picking a question replays what the three pipelines
ACTUALLY produced on that question, with the same token counts and the same grading the live
server computes. It is a replay, and the page says "replay" on it rather than implying a call
that is not happening.

WHAT DOES NOT: free-text questions. There is no model behind a static file, so a question that
is not one of the 50 says so instead of inventing an answer.

The HTML is PATCHED, not forked. Every replacement below asserts it matched, so if index.html
is edited in a way that breaks the static build, this fails loudly at build time instead of
shipping a page whose buttons quietly do nothing.

    python demo/build_static.py [--out demo/static]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _portability import init_console   # noqa: E402


def _load_server():
    """Import demo/server.py without running it. Its module-level code only sets env vars and
    paths; the socket bind lives under __main__, so importing is safe."""
    spec = importlib.util.spec_from_file_location("demo_server", Path(__file__).parent / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- recorded answers
def build_answers(S) -> dict:
    """question text -> the exact payload /api/query would have returned.

    Arms are renamed p2rr -> p2 because the frontend asks for p1/p2/p3 and keys its cards on
    those ids. The label shown is unchanged either way (NAMES maps both to "Basic RAG").
    """
    rows = S._jsonl(S.RESULTS / S.PUB)
    ARM_OUT = {"p1": "p1", "p2rr": "p2", "p3": "p3"}
    by_q: dict[str, dict] = {}
    for r in rows:
        arm = ARM_OUT.get(r["arm"])
        if arm is None:
            continue
        q = r["question"]
        payload = by_q.setdefault(q, {"gold": r["gold"], "qid": r["question_id"], "results": []})
        entry = {
            "arm": arm,
            "answer": (r.get("answer") or "").strip() or "(no answer)",
            "prompt_tokens": r.get("prompt_tokens", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            "planner_tokens": r.get("planner_tokens", 0),
            "tokens": (r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
                       + r.get("planner_tokens", 0)),
            "route": r.get("route", "") or "recorded",
            "latency": round(r.get("latency_s", 0) or 0, 1),
            "chunks": 0,
            "sources": [],
        }
        # Grade through the server's own function, so a replayed card and a live card can never
        # disagree about what counts as correct.
        g = S._gold_lookup(q)
        if g:
            entry["grade"] = S._grade(entry["answer"], g["members"])
        payload["results"].append(entry)
    for p in by_q.values():
        p["results"].sort(key=lambda e: ["p1", "p2", "p3"].index(e["arm"]))
    return by_q


# ---------------------------------------------------------------- html patches
STATIC_JS = """
// ---- static replay shim (added by demo/build_static.py) ----------------------
// There is no backend here. Answers come from the recorded run, keyed on the exact
// question text; anything else returns an honest error rather than a guess.
let _ANS=null;
async function STATIC_QUERY(q,arm){
  if(!_ANS) _ANS=await (await fetch("api/answers.json")).json();
  const hit=_ANS[q];
  if(!hit) return {error:"this build has no recorded answer for that question — "
                        +"pick one of the questions below"};
  const r=(hit.results||[]).find(x=>x.arm===arm);
  return {gold:hit.gold, qid:hit.qid, results:r?[r]:[]};
}
"""

PATCHES = [
    # --- data sources: URL routes become files on disk
    ('fetch("/api/examples?dataset="+view)',
     'fetch("api/examples-"+view+".json")'),
    ('fetch("/api/metrics")',
     'fetch("api/metrics.json")'),
    ('fetch("/api/status")',
     'fetch("api/status.json")'),

    # --- the query console replays instead of calling
    ('''const r=await fetch("/api/query",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({question:q,arms:[a],dataset:view})});
      const d=await r.json();''',
     '''const d=await STATIC_QUERY(q,a);'''),

    # --- and says so. "live" is what drives every bit of console copy.
    ('const live=view==="textual";\n  document.getElementById("askHead")',
     'const live=false;\n  document.getElementById("askHead")'),
    ('const live=view==="textual";\n  const ARMS=',
     'const live=false;\n  const ARMS='),
    ('"replays the audited SEC run — not a live call"',
     '"replays the recorded run — every answer and token count is measured, not generated now"'),
    ('"Ask something about the company…" : "Pick one of the SEC questions below…"',
     '"Pick one of the questions below…" : "Pick one of the SEC questions below…"'),
    ('${live?"running three pipelines in parallel":"replaying the frozen SEC run"}',
     '${"replaying the recorded run"}'),

    # --- the warming banner is meaningless with no pipelines to warm
    ("""  const s=await (await fetch("api/status.json")).json();
  if(s.state!=="ready")
    document.getElementById("state").innerHTML=
      '<span class="sp"></span> warming the pipelines — the first live query will be slower';""",
     """  document.getElementById("state").innerHTML="";"""),
]


def patch_html(html: str) -> str:
    for old, new in PATCHES:
        if old not in html:
            raise SystemExit(
                "ABORT: index.html no longer contains a fragment this build depends on.\n"
                "Shipping anyway would produce a page with dead buttons. Fix the patch in\n"
                f"demo/build_static.py, then rebuild.\n\n--- expected ---\n{old[:300]}")
        html = html.replace(old, new, 1)
    return html.replace("</script>\n</body>", STATIC_JS + "</script>\n</body>", 1)


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser()
    # Repo-root /docs, because GitHub Pages will only serve a branch root or a folder named
    # exactly "docs" — an arbitrary path like demo/static needs a CI job to copy it, and a
    # deploy that needs CI is a deploy that can break the day you present. Netlify/Vercel take
    # any folder, so --out stays available.
    ap.add_argument("--out", default=str(ROOT.parent / "docs"))
    args = ap.parse_args()
    out = Path(args.out)
    api = out / "api"
    if out.exists():
        shutil.rmtree(out)
    api.mkdir(parents=True)

    S = _load_server()

    def dump(name: str, obj) -> None:
        p = api / name
        p.write_text(json.dumps(obj), encoding="utf-8")
        print(f"  {p.relative_to(out).as_posix():<28}{p.stat().st_size / 1024:>8.1f} KB")

    dump("metrics.json", {"textual": S.textual_metrics("tuned"),
                          "heldout": S.textual_metrics("heldout"),
                          "factual": S.factual_metrics()})
    dump("status.json", {"state": "static", "error": ""})
    dump("examples-textual.json", S.examples())
    dump("examples-factual.json", S.factual_examples())
    answers = build_answers(S)
    dump("answers.json", answers)

    html = patch_html((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"  {'index.html':<28}{(out / 'index.html').stat().st_size / 1024:>8.1f} KB")

    # GitHub Pages runs Jekyll by default, which silently drops files and folders whose names
    # begin with an underscore. Nothing here starts with one today, but the marker costs a byte
    # and removes a whole class of "it works locally" failure.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    if re.search(r'fetch\("/api/', html):
        raise SystemExit("ABORT: a /api/ call survived patching — the static page would 404")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n{len(answers)} questions replayable · {total / 1024:.0f} KB total · {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
