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
def build_answers(S, filename: str, grade: bool = True) -> dict:
    """question text -> the exact payload /api/query would have returned.

    Arms are renamed p2rr -> p2 because the frontend asks for p1/p2/p3 and keys its cards on
    those ids. The label shown is unchanged either way (NAMES maps both to "Basic RAG").

    `grade` is off for the SEC set: its gold answers are numbers with commentary, so the
    name-set grader would mark every one of them wrong. That tab is judged, not set-scored,
    and the console shows the gold answer beside the three replies instead of a badge.
    """
    rows = S._jsonl(S.RESULTS / filename)
    ARM_OUT = {"p1": "p1", "p2": "p2", "p2rr": "p2", "p3": "p3"}
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
            # retrieved_chunks was never recorded in the published eval files, so this is
            # unknown rather than zero; the page renders an em dash for it.
            "chunks": r.get("retrieved_chunks") or 0,
            "sources": [],
        }
        # Grade through the server's own function, so a replayed card and a live card can never
        # disagree about what counts as correct.
        if grade:
            g = S._gold_lookup(q)
            if g:
                entry["grade"] = S._grade(entry["answer"], g["members"])
        payload["results"].append(entry)
    for p in by_q.values():
        p["results"].sort(key=lambda e: ["p1", "p2", "p3"].index(e["arm"]))
    return by_q


# ---------------------------------------------------------------- html patches
STATIC_JS_TEMPLATE = """
// ---- query shim (added by demo/build_static.py) ------------------------------
// LIVE_URL is set at build time. When present the browser calls the real backend and falls
// back to the recorded run if that fails; when absent everything is replay.
//
// The fallback is the important part. The live backend is a laptop behind a tunnel: it goes
// away when the machine sleeps, and it rate-limits at 20 queries/hour/IP. Without a fallback
// every one of those cases shows the visitor an error. With it they get the real measured
// answer for that question and a note saying it is recorded. Degrading to true data beats
// failing.
//
// LIVE_TOKEN is visible here — it must be, this is a static page. It stops crawlers, not a
// person reading the source. The per-IP rate limit is the actual spend control.
const LIVE_URL   = __LIVE_URL__;
const LIVE_TOKEN = __LIVE_TOKEN__;
const LIVE_TIMEOUT_MS = 90000;   // Cloudflare cuts at ~100s; fall back before it 524s

const _ANS={};
async function REPLAY(q,arm,view){
  const f = view==="factual" ? "api/answers-factual.json" : "api/answers.json";
  if(!_ANS[f]) _ANS[f]=await (await fetch(f)).json();
  const hit=_ANS[f][q];
  if(!hit) return {error:"no recorded answer for that question — pick one below"};
  const r=(hit.results||[]).find(x=>x.arm===arm);
  if(r) r.replayed=true;
  return {gold:hit.gold, qid:hit.qid, results:r?[r]:[]};
}

async function STATIC_QUERY(q,arm,view){
  if(LIVE_URL){
    const ctl=new AbortController();
    const t=setTimeout(()=>ctl.abort(), LIVE_TIMEOUT_MS);
    try{
      const res=await fetch(LIVE_URL+"/api/query",{
        method:"POST", signal:ctl.signal,
        headers:{"Content-Type":"application/json","X-Demo-Token":LIVE_TOKEN},
        body:JSON.stringify({question:q,arms:[arm],dataset:view})});
      clearTimeout(t);
      if(res.ok){
        const d=await res.json();
        if(!d.error && (d.results||[]).length) return d;
      }
      // 429 (rate limited), 401, 5xx, or an error body: fall through to the recording
    }catch(e){ clearTimeout(t); }   // offline, aborted, DNS gone, tunnel closed
  }
  return REPLAY(q,arm,view);
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
     '''const d=await STATIC_QUERY(q,a,view);'''),

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


def patch_html(html: str, live_url: str = "", live_token: str = "") -> str:
    for old, new in PATCHES:
        if old not in html:
            raise SystemExit(
                "ABORT: index.html no longer contains a fragment this build depends on.\n"
                "Shipping anyway would produce a page with dead buttons. Fix the patch in\n"
                f"demo/build_static.py, then rebuild.\n\n--- expected ---\n{old[:300]}")
        html = html.replace(old, new, 1)

    shim = (STATIC_JS_TEMPLATE
            .replace("__LIVE_URL__", json.dumps(live_url.rstrip("/")))
            .replace("__LIVE_TOKEN__", json.dumps(live_token)))
    html = html.replace("</script>\n</body>", shim + "</script>\n</body>", 1)

    # With a live backend the console is genuinely live again, so undo the two copy patches
    # that told the visitor otherwise. Leaving them would understate what the page does.
    if live_url:
        html = html.replace('${"replaying the recorded run"}',
                            '${"running three pipelines live"}', 1)
        html = html.replace('"replays the recorded run — every answer and token count is '
                            'measured, not generated now"',
                            '"runs all three pipelines live against the real graph — '
                            'falls back to the recorded run if the backend is offline"', 1)
        html = html.replace('const live=false;\n  document.getElementById("askHead")',
                            'const live=true;\n  document.getElementById("askHead")', 1)
    return html


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser()
    # Repo-root /docs, because GitHub Pages will only serve a branch root or a folder named
    # exactly "docs" — an arbitrary path like demo/static needs a CI job to copy it, and a
    # deploy that needs CI is a deploy that can break the day you present. Netlify/Vercel take
    # any folder, so --out stays available.
    ap.add_argument("--out", default=str(ROOT.parent / "docs"))
    ap.add_argument("--live-url", default="",
                    help="backend base URL, e.g. https://x.trycloudflare.com. Omit for a "
                         "pure-replay build.")
    ap.add_argument("--live-token", default="",
                    help="value sent as X-Demo-Token. Ends up in the published JavaScript.")
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
    answers = build_answers(S, S.PUB)
    dump("answers.json", answers)
    # The SEC console replays from the SAME file its table reads. The live server replays it
    # from _evidence/theirs.jsonl instead — the pre-restore frozen run — which is the mismatch
    # that was just removed from the enterprise tab, still present on this one.
    sec = build_answers(S, "eval_sec_fair_prompt.jsonl", grade=False)
    dump("answers-factual.json", sec)

    html = patch_html((Path(__file__).parent / "index.html").read_text(encoding="utf-8"),
                      args.live_url, args.live_token)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"  {'mode':<28}{'LIVE -> ' + args.live_url if args.live_url else 'replay only':>8}")
    print(f"  {'index.html':<28}{(out / 'index.html').stat().st_size / 1024:>8.1f} KB")

    # GitHub Pages runs Jekyll by default, which silently drops files and folders whose names
    # begin with an underscore. Nothing here starts with one today, but the marker costs a byte
    # and removes a whole class of "it works locally" failure.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    if re.search(r'fetch\("/api/', html):
        raise SystemExit("ABORT: a /api/ call survived patching — the static page would 404")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n{len(answers)} enterprise + {len(sec)} SEC questions replayable · "
          f"{total / 1024:.0f} KB total · {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
