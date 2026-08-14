"""T1 — run the dataset's OWN 500 published questions. Content questions, never tested before.

WHY THIS IS THE MOST IMPORTANT TEST IN THE PROJECT
--------------------------------------------------
Every enterprise number so far comes from `eval_mixed50.jsonl`, whose questions were GENERATED
FROM THE GRAPH and whose gold answers ARE the traversal output. All 8 of its question shapes are
metadata traversals; not one asks what a message actually said. So GraphRAG scoring ~100% there
is close to a tautology, and Basic RAG scoring 17% on the multi-hop tier is partly because 5 of
those questions need 510-1,065 documents against a 10-chunk budget.

`official_questions.jsonl` is the opposite: 500 questions shipped WITH the dataset, with a
published `gold_answer` on 500/500 and 470/470 `expected_doc_ids` resolvable in our corpus (via
the `uuid` field). They are content questions — "what are the default size limits for file
uploads", "what is the name of the new metric added so SRE can track ...". The graph has no
relation to follow for those.

Phase 6 of the build plan promised to run these as an independent cross-check. It never
happened. Until it does, "GraphRAG 98%" is a claim about one self-generated question shape
family, not about the corpus.

THREE SCORES, BECAUSE NO SINGLE ONE IS TRUSTWORTHY HERE
-------------------------------------------------------
  judge_pct     0-100 vs gold_answer, via build/llm_judge.py (unchanged, so scores stay
                comparable with every other run in the project)
  fact_recall   fraction of the question's `answer_facts` the answer actually supports. Set-F1
                cannot be used — these answers are prose, not name sets — and this is the
                closest available thing to a decomposable score.
  hit@k         did retrieval even fetch the right document? Maps each retrieved chunk id to its
                uuid and tests membership in expected_doc_ids. This separates RETRIEVER failure
                from GENERATOR failure, which no existing metric in this project does.

info_not_found IS SCORED DIFFERENTLY, ON PURPOSE
------------------------------------------------
20 of the 500 have no answer in the corpus. The judge's rubric states "an empty answer or a
refusal scores 0" — correct for every other question and exactly WRONG for these, where
declining IS the right answer. Worse, config.SYSTEM_PROMPT tells all three arms to "NEVER return
an empty answer ... never reply that you cannot answer". So on these 20 the correct answer is
prohibited by our own prompt. They are reported as an ABSTENTION RATE, never folded into
accuracy.

    python build/run_official.py --n 200              # stratified prefix, resumable
    python build/run_official.py --n 500              # continue to the full set
    python build/run_official.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline3_round3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _portability import init_console, load_env, ropen   # noqa: E402

init_console()
load_env()

QUESTIONS = Path("datasets/enterprise/official_questions.jsonl")
CORPUS = Path("datasets/enterprise/corpus.jsonl")
OUT = Path("results/enterprise/eval_official500.jsonl")
SEED = 20260811
ARMS = ["p1", "p2rr", "p3"]
NAMES = {"p1": "P1 LLM-only", "p2rr": "P2-RR Basic RAG", "p3": "P3 GraphRAG"}
BLOCK = 10            # questions per checkpoint; scoring for a block is threaded
SCORE_WORKERS = 6

REFUSAL = re.compile(
    r"not (?:supported|available|found|present|specified|mentioned|in the)|"
    r"no information|cannot (?:be )?(?:answer|determin|find)|unable to|"
    r"does not (?:contain|provide|specify|mention)|not enough (?:information|context)|"
    r"^\s*(?:none|n/?a|unknown)\s*\.?\s*$", re.I)


# ------------------------------------------------------------------ scoring helpers
def _gemini(prompt: str, model: str = "") -> dict | None:
    """One structured-JSON call. Shares the judge's model so grading stays a different family
    from the answering model (gemini-3.6-flash), which is the project's bias control."""
    import requests
    import llm_judge
    key = os.getenv("GOOGLE_API_KEY", "")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model or llm_judge.JUDGE_MODEL}:generateContent?key={key}")
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096,
                                 "responseMimeType": "application/json"}}
    for attempt in range(4):
        try:
            r = requests.post(url, json=body, timeout=120)
            if r.status_code == 200:
                cand = (r.json().get("candidates") or [{}])[0]
                txt = "".join(p.get("text", "")
                              for p in cand.get("content", {}).get("parts", []))
                return json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M))
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt + 1)
                continue
            return None
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2 ** attempt + 1)
    return None


FACT_PROMPT = """Decide which of the listed FACTS are supported by the ANSWER.

QUESTION: {q}

FACTS (from the official answer key):
{facts}

ANSWER TO CHECK:
{a}

A fact counts as supported if the answer states it, even in different words, different units
written equivalently, or as part of a longer sentence. Paraphrase is fine. Do not require the
answer to mention facts it was not asked about.

Reply with STRICT JSON only: {{"supported": [<1-based indices of supported facts>],
"why": "<one short sentence>"}}"""

ABSTAIN_PROMPT = """This question has NO answer in the source corpus. The ideal response admits
that.

QUESTION: {q}

RESPONSE:
{a}

Did the response ADMIT it could not answer / that the information is unavailable, or did it
assert a specific factual answer anyway?

Reply with STRICT JSON only: {{"abstained": true|false, "why": "<one short sentence>"}}"""


def fact_recall(q: str, facts: list[str], answer: str) -> tuple[float, int, int]:
    if not facts:
        return -1.0, 0, 0
    if not (answer or "").strip():
        return 0.0, 0, len(facts)
    listing = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    d = _gemini(FACT_PROMPT.format(q=q, facts=listing, a=answer[:4000]))
    if not d:
        return -1.0, 0, len(facts)
    sup = {i for i in (d.get("supported") or []) if isinstance(i, int) and 1 <= i <= len(facts)}
    return len(sup) / len(facts), len(sup), len(facts)


def abstained(q: str, answer: str) -> int:
    """-1 on judge failure, so a failed call is excluded rather than counted as a hallucination."""
    if not (answer or "").strip():
        return 1
    if REFUSAL.search(answer):
        return 1
    d = _gemini(ABSTAIN_PROMPT.format(q=q, a=answer[:2000]))
    return -1 if d is None else int(bool(d.get("abstained")))


# ------------------------------------------------------------------ question ordering
def stratified(rows: list[dict]) -> list[dict]:
    """Round-robin across question_type so ANY PREFIX is proportional.

    The run takes hours; without this, stopping early would silently bias the sample toward
    whichever types happened to come first in the file.
    """
    rng = random.Random(SEED)
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r["question_type"]].append(r)
    for v in buckets.values():
        rng.shuffle(v)
    # Sort by NORMALISED RANK WITHIN TYPE: a question that is 30% of the way through its own
    # type lands at position 0.30 overall. Every prefix then holds ~the same type proportions as
    # the full set, whatever length it is stopped at. Type name breaks ties so the order is
    # deterministic rather than dependent on dict iteration.
    keyed = [((idx + 0.5) / len(v), t, r)
             for t, v in buckets.items() for idx, r in enumerate(v)]
    keyed.sort(key=lambda x: (x[0], x[1]))
    out = [r for _, _, r in keyed]
    assert len(out) == len(rows) and len({r["question_id"] for r in out}) == len(rows)
    return out


# ------------------------------------------------------------------ report
def report(rows: list[dict]) -> None:
    by_arm: dict[str, list] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    if not by_arm:
        print("no rows yet")
        return
    nq = len({r["question_id"] for r in rows})
    print(f"\n{'=' * 100}\nT1 — OFFICIAL QUESTIONS   n={nq} questions completed\n{'=' * 100}")

    # info_not_found is an abstention measurement, never accuracy
    inf = [r for r in rows if r["question_type"] == "info_not_found"]
    scored = [r for r in rows if r["question_type"] != "info_not_found"]

    print(f"\n{'pipeline':<18}{'n':>5}{'judge':>8}{'factrec':>9}{'hit@k':>8}"
          f"{'prompt':>9}{'compl':>8}{'plan':>7}{'tok/q':>9}")
    print("-" * 100)
    for a in ARMS:
        R = [r for r in scored if r["arm"] == a]
        if not R:
            continue
        j = [r["judge_pct"] for r in R if r.get("judge_pct") is not None and r["judge_pct"] >= 0]
        fr = [r["fact_recall"] for r in R if r.get("fact_recall", -1) >= 0]
        hk = [r["hit_at_k"] for r in R if r.get("hit_at_k", -1) >= 0]
        pl = st.mean([r.get("planner_tokens", 0) for r in R])
        tok = st.mean([r["prompt_tokens"] + r["completion_tokens"] for r in R]) + pl
        print(f"{NAMES[a]:<18}{len(R):>5}{st.mean(j) if j else 0:>7.1f}%"
              f"{(st.mean(fr) * 100) if fr else 0:>8.1f}%"
              f"{(st.mean(hk) * 100) if hk else 0:>7.1f}%"
              f"{st.mean([r['prompt_tokens'] for r in R]):>9,.0f}"
              f"{st.mean([r['completion_tokens'] for r in R]):>8,.0f}"
              f"{pl:>7,.0f}{tok:>9,.0f}")

    types = sorted({r["question_type"] for r in scored})
    print(f"\n{'question_type':<26}{'n':>4}" + "".join(f"{NAMES[a].split()[0]:>12}" for a in ARMS))
    print("-" * (30 + 12 * len(ARMS)))
    for t in types:
        cells = []
        n = 0
        for a in ARMS:
            R = [r for r in scored if r["arm"] == a and r["question_type"] == t
                 and r.get("judge_pct", -1) >= 0]
            n = max(n, len(R))
            cells.append(f"{st.mean([r['judge_pct'] for r in R]):.1f}%" if R else "-")
        print(f"{t:<26}{n:>4}" + "".join(f"{c:>12}" for c in cells))

    if inf:
        print(f"\n{'-' * 100}\ninfo_not_found — {len({r['question_id'] for r in inf})} questions "
              f"with NO answer in the corpus.\nconfig.SYSTEM_PROMPT forbids declining, so a "
              f"correct answer is prohibited by our own prompt.")
        print(f"\n{'pipeline':<18}{'n':>5}{'abstained':>11}{'hallucinated':>14}")
        print("-" * 50)
        for a in ARMS:
            R = [r for r in inf if r["arm"] == a and r.get("abstained", -1) >= 0]
            if not R:
                continue
            ab = sum(r["abstained"] for r in R)
            print(f"{NAMES[a]:<18}{len(R):>5}{ab:>10}"
                  f"{len(R) - ab:>13} ({100 * (len(R) - ab) / len(R):.0f}%)")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    # A fixed-pipeline re-run must NOT append to the file holding the old-pipeline rows: the
    # resume logic skips any (question_id, arm) already present, so it would silently skip
    # everything and report the OLD numbers as if they were new. Separate file, paired
    # comparison on the same stratified prefix.
    ap.add_argument("--out", default="")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-facts", action="store_true", help="skip fact recall (saves ~1 call/row)")
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out)
    done_rows = [json.loads(l) for l in ropen(OUT) if l.strip()] if OUT.exists() else []
    if args.report_only:
        report(done_rows)
        return 0

    allq = [json.loads(l) for l in ropen(QUESTIONS) if l.strip()]
    order = stratified(allq)[: args.n]
    done = {(r["question_id"], r["arm"]) for r in done_rows}
    todo = [q for q in order if any((q["question_id"], a) not in done for a in ARMS)]
    print(f"{len(allq)} official questions; running a stratified prefix of {len(order)}")
    print(f"  types in this prefix: {dict(Counter(q['question_type'] for q in order))}")
    print(f"  already done: {len(done)} rows -> {len(todo)} questions still to run\n")
    if not todo:
        report(done_rows)
        return 0

    # chunk id -> uuid, so retrieval can be scored against expected_doc_ids
    t2u = {}
    for line in ropen(CORPUS):
        if line.strip():
            c = json.loads(line)
            t2u[c["title"]] = c.get("uuid", "")
    print(f"  chunk->uuid map: {len(t2u):,} chunks\n")

    import importlib.util
    import pipeline1_llm_only as p1
    import pipeline2rr_rerank as prr
    spec = importlib.util.spec_from_file_location("p3", "pipeline3_round3/pipeline.py")
    p3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p3)
    p3._load(); p3._connect()
    import schema_router
    import llm_judge

    fh = open(OUT, "a", encoding="utf-8")
    t0, n_new = time.time(), 0
    for bstart in range(0, len(todo), BLOCK):
        block = todo[bstart: bstart + BLOCK]
        pending = []
        for q in block:
            for arm in ARMS:
                if (q["question_id"], arm) in done:
                    continue
                try:
                    schema_router.reset_token_usage()
                    if arm == "p1":
                        m = p1.run_single(q["question"])
                    elif arm == "p2rr":
                        m = prr.run_single(q["question"])
                    else:
                        m = p3.run_single(q["question"])
                    pt = schema_router.plan_token_usage() if arm == "p3" else {"prompt": 0, "output": 0}
                    ids = list(getattr(m, "retrieved_ids", []) or [])
                    uu = {t2u.get(i, "") for i in ids}
                    exp = q.get("expected_doc_ids") or []
                    hit = (sum(1 for d in exp if d in uu) / len(exp)) if exp else -1.0
                    pending.append({
                        "question_id": q["question_id"], "arm": arm,
                        "question_type": q["question_type"],
                        "source_types": q.get("source_types", []),
                        "question": q["question"], "gold": q.get("gold_answer") or "",
                        "answer_facts": q.get("answer_facts") or [],
                        "answer": (m.answer or "").strip(),
                        "route": getattr(m, "route", ""),
                        "prompt_tokens": m.prompt_tokens,
                        "completion_tokens": m.completion_tokens,
                        "planner_tokens": pt["prompt"] + pt["output"],
                        "latency_s": getattr(m, "latency_seconds", 0),
                        "retrieved_n": len(ids), "hit_at_k": round(hit, 4)})
                except Exception as e:
                    print(f"    {q['question_id']}/{arm} FAILED: {str(e)[:110]}")

        def score(row: dict) -> dict:
            if row["question_type"] == "info_not_found":
                row["abstained"] = abstained(row["question"], row["answer"])
                row["judge_pct"] = None
                row["fact_recall"] = -1.0
                return row
            s, why = llm_judge.judge_one(row["question"], row["gold"], row["answer"])
            row["judge_pct"] = None if s < 0 else round(s, 1)
            row["judge_score"] = None if s < 0 else round(s / 100.0, 4)
            row["judge_why"] = why
            if args.no_facts:
                row["fact_recall"] = -1.0
            else:
                fr, k, n = fact_recall(row["question"], row["answer_facts"], row["answer"])
                row["fact_recall"] = round(fr, 4)
                row["facts_supported"], row["facts_total"] = k, n
            row["abstained"] = -1
            return row

        with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as ex:
            scored = list(ex.map(score, pending))
        for row in scored:
            row.pop("answer_facts", None)          # keep the file small; it is in the source
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        done_rows += scored
        n_new += len(scored)
        el = time.time() - t0
        print(f"  {bstart + len(block)}/{len(todo)} questions  ({n_new} rows, {el/60:.1f} min, "
              f"{el / max(1, n_new) :.1f}s/row)")

    fh.close()
    report(done_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
