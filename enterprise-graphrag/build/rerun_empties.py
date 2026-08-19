"""Re-run the SEC answers that came back EMPTY, with the LLM client bug fixed at runtime.

mastergraph-round3/ is frozen evidence under 428 file hashes, so nothing there is edited. The
fix is monkey-patched into the imported module at run time; the tree on disk is untouched and
still verifies.

THE BUG (mastergraph-round3/llm_client.py, and the enterprise copy has it identically):

    content = cand[0]["content"]["parts"][0].get("text", "").strip()
    "completion_tokens": usage.get("candidatesTokenCount", 0)

  1. parts[0] only. Gemini 2.5+ with thinking returns MULTIPLE parts — commonly a thought part
     followed by the answer part. Taking parts[0] keeps the thought and discards the answer.
  2. candidatesTokenCount is VISIBLE output only. Thinking is billed as thoughtsTokenCount and
     is never read, so "completion_tokens: 0" reads as "generated nothing" when it can equally
     mean "spent the whole budget reasoning".
  3. finishReason is discarded, so MAX_TOKENS, SAFETY and RECITATION are indistinguishable.

THE FIX: join every part's text, record thoughts and finishReason, and retry once with a larger
budget when the visible answer is empty.

WHY ONLY THE EMPTIES: this bug produces empty answers specifically. Re-running the other
failures would test nothing about the fix, and keeping whichever result came out better would
be re-rolling the dice until the number improves. The empties are the population the bug
created, so they are the population that tests it.

    python build/rerun_empties.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MG = ROOT.parent / "mastergraph-round3"
sys.path.insert(0, str(ROOT))
from _portability import init_console, load_env, ropen, wopen   # noqa: E402

load_env()          # OUR .env. Without this GOOGLE_API_KEY is empty and every call 403s.

RESULTS = ROOT / "results" / "enterprise"
SRC = RESULTS / "eval_sec_fair_prompt.jsonl"
OUT = RESULTS / "eval_sec_empty_rerun.jsonl"

STATS = {"calls": 0, "empty_first_try": 0, "recovered_by_join": 0,
         "recovered_by_retry": 0, "still_empty": 0, "finish_reasons": {}}


def patched_gemini_call(prompt: str, json_mode: bool = False):
    """Replacement for llm_client._gemini_call with all three defects fixed."""
    import random
    import time

    import requests
    import config

    key = config.GOOGLE_API_KEY
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{config.GEMINI_MODEL}:generateContent?key={key}")

    def attempt(max_tokens: int):
        gen = {"temperature": config.TEMPERATURE, "maxOutputTokens": max_tokens}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
        for i in range(4):
            try:
                r = requests.post(url, json=body, timeout=120)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 500, 503):
                    time.sleep(2.0 * (2 ** i) + random.uniform(0, 1))
                    continue
                raise RuntimeError(f"Gemini {r.status_code}: {r.text[:160]}")
            except Exception:
                if i == 3:
                    raise
                time.sleep(2.0 * (2 ** i) + random.uniform(0, 1))
        raise RuntimeError("retries exhausted")

    def extract(j):
        cand = (j.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        # FIX 1: join every part, and skip parts flagged as thoughts
        texts = [p.get("text", "") for p in parts if not p.get("thought")]
        if not any(t.strip() for t in texts):          # nothing but thoughts
            texts = [p.get("text", "") for p in parts]
        u = j.get("usageMetadata", {})
        return ("\n".join(t for t in texts if t).strip(),
                cand.get("finishReason", "?"),
                {"prompt_tokens": u.get("promptTokenCount", 0),
                 # FIX 2: visible output, and thinking recorded separately
                 "completion_tokens": u.get("candidatesTokenCount", 0),
                 "thinking_tokens": u.get("thoughtsTokenCount", 0),
                 "total_tokens": u.get("totalTokenCount", 0)})

    STATS["calls"] += 1
    j = attempt(config.MAX_TOKENS)
    content, why, usage = extract(j)
    STATS["finish_reasons"][why] = STATS["finish_reasons"].get(why, 0) + 1
    if content and len(parts_count(j)) > 1:
        STATS["recovered_by_join"] += 1

    if not content:
        # FIX 3: a retry with more room, since an empty answer usually means the budget went
        # on reasoning rather than the model having nothing to say
        STATS["empty_first_try"] += 1
        j2 = attempt(max(config.MAX_TOKENS * 2, 16384))
        c2, why2, u2 = extract(j2)
        STATS["finish_reasons"][f"retry:{why2}"] = STATS["finish_reasons"].get(f"retry:{why2}", 0) + 1
        if c2:
            STATS["recovered_by_retry"] += 1
            content, usage = c2, u2
        else:
            STATS["still_empty"] += 1
    return content, {"usage": usage, "finish_reason": why}


def parts_count(j):
    cand = (j.get("candidates") or [{}])[0]
    return (cand.get("content") or {}).get("parts") or []


def main() -> int:
    init_console()
    rows = [json.loads(l) for l in ropen(SRC) if l.strip()]

    def body(s):
        return re.sub(r"\s*Citations?:.*$", "", " ".join(str(s or "").split()),
                      flags=re.I).strip()

    def is_empty(s):
        b = body(s)
        return (not b) or b.lower() in ("(no answer)", "no answer", "n/a", "none")

    targets = [r for r in rows if is_empty(r["answer"])]
    print(f"{len(targets)} empty answers to re-run "
          f"({', '.join(sorted(set(r['arm'] for r in targets)))})\n")
    for r in targets:
        print(f"  {r['question_id']} [{r['arm']}] {r['tier']}  compl={r.get('completion_tokens')}")

    # ORDER MATTERS. mastergraph-round3/.env holds an expired Gemini key and
    # TIGERGRAPH_GRAPH=wixqa_v2, a graph that never held this corpus. If we chdir first, its
    # config wins and every call fails — 401 on the LLM, 404 on the graph. demo/server_factual.py
    # already handles this correctly; this mirrors it. Our .env is loaded by _portability at
    # import time, so the values below are already in os.environ before MG's config is imported.
    os.environ["TIGERGRAPH_GRAPH"] = os.getenv("SEC_GRAPH", "finance")
    os.environ.setdefault("MAX_TOKENS", "8192")
    os.environ.setdefault("SEC_NO_REFUSAL", "1")
    key_before = os.environ.get("GOOGLE_API_KEY", "")
    print(f"  forced graph={os.environ['TIGERGRAPH_GRAPH']}  "
          f"MAX_TOKENS={os.environ['MAX_TOKENS']}  key={key_before[:6]}...{key_before[-3:]}")

    os.chdir(MG)                       # their corpus and Chroma paths are relative
    sys.path.insert(0, str(MG))
    import config                                       # noqa: E402
    # MG's config re-reads .env on import; put our values back if it clobbered them
    # only override if we actually have one — clobbering MG's key with an empty string is
    # how the previous attempt produced 403 "unregistered caller" on all nine questions
    if key_before:
        config.GOOGLE_API_KEY = key_before
    if not config.GOOGLE_API_KEY:
        print("  ABORT: no GOOGLE_API_KEY from either .env")
        return 1
    import llm_client                                   # noqa: E402
    llm_client._gemini_call = patched_gemini_call        # runtime only; disk untouched
    print("  patched llm_client._gemini_call in memory; nothing on disk changed")

    import importlib.util
    import pipeline1_llm_only as P1                      # noqa: E402
    import pipeline2_basic_rag as P2                     # noqa: E402
    spec = importlib.util.spec_from_file_location("mg_p3", MG / "pipeline3_round3" / "pipeline.py")
    P3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P3)
    P3._load()

    # p2 is Chroma-backed and takes the store as a second argument
    from chroma_store import load_vectorstore            # noqa: E402
    print("  loading Chroma (928 MB)...")
    VS = load_vectorstore()
    print("  ready\n")

    def call(arm, q):
        if arm == "p1":
            return P1.run_single(q)
        if arm == "p2":
            return P2.run_single(q, VS)
        return P3.run_single(q)

    out = []
    for i, r in enumerate(targets, 1):
        arm = r["arm"]
        try:
            m = call(arm, r["question"])
            ans = (getattr(m, "answer", "") or "").strip()
        except Exception as e:
            ans = f"ERROR: {e}"
        rec = dict(r)
        rec["answer_before"] = r["answer"]
        rec["answer"] = ans
        rec["recovered"] = int(not is_empty(ans) and not ans.startswith("ERROR"))
        out.append(rec)
        print(f"  [{i}/{len(targets)}] {r['question_id']} {arm}  "
              f"{'RECOVERED' if rec['recovered'] else 'still empty'}  {body(ans)[:60]!r}")

    os.chdir(ROOT)
    with wopen(OUT) as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {OUT.name}")
    print(f"\n  recovered {sum(r['recovered'] for r in out)} of {len(out)}")
    print(f"  calls {STATS['calls']} · empty on first try {STATS['empty_first_try']}"
          f" · saved by joining parts {STATS['recovered_by_join']}"
          f" · saved by retry {STATS['recovered_by_retry']}"
          f" · still empty {STATS['still_empty']}")
    print(f"  finishReason seen: {STATS['finish_reasons']}")
    print("\n  NOTE: recovered answers still have to be JUDGED to know if they are CORRECT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
