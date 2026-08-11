"""Round 3 evaluation — implements the PS 'Answer Evaluation and Judging' spec exactly.

Per answer, computes:
  1. strict pass/fail   — complete, factually correct, adequately supported (PASS/FAIL)
  2. graded quality      — 4 levels: fully / mostly / partial / incorrect  (-> 1.0/0.66/0.33/0.0)
  3. citation quality    — do the cited chunk IDs actually exist in the supplied evidence, and
                           does the evidence support the answer? (0-1)
  4. BERTScore F1        — PINNED config: bert-score==0.3.13, roberta-large, lang=en,
                           rescale_with_baseline=True, idf=False, vs the OFFICIAL reference.

Judge = OpenAI gpt-4o-mini (DIFFERENT model from the Gemini generator; blind to which
pipeline produced the answer — we pass only Q / reference / answer, never the pipeline name).

Reads a benchmark results JSON (per-pipeline per-question rows incl. retrieved_ids +
citations) and writes an evaluation JSON with per-item + aggregate scores per pipeline.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # portability: was sys.path.insert(0, ".") — only worked from the repo root
from _portability import init_console, load_env   # was: from dotenv import load_dotenv
init_console(); load_env()   # was: load_dotenv("/Users/kajal/Downloads/GRAPHRAG HACKATHON/.env")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

STRICT_PROMPT = """You are grading an answer to a question about SEC filings. You are blind to which system produced it.

Question: {q}
Reference answer (ground truth): {ref}
System answer: {ans}

Give a STRICT pass/fail: PASS only if the system answer is COMPLETE, factually correct, and fully consistent with the reference (all required companies/figures/names present; for list questions, the set must match with no major omissions or wrong additions). Otherwise FAIL.
Also give a GRADE: FULLY (equivalent to reference), MOSTLY (core correct, minor omission), PARTIAL (some correct, significant gaps/errors), INCORRECT (wrong/unsupported/refused).
Reply with exactly two lines:
PASS or FAIL
FULLY or MOSTLY or PARTIAL or INCORRECT"""

CITE_PROMPT = """A system answered a question using ONLY the evidence passages below. Judge citation/evidence quality.

Question: {q}
System answer: {ans}
Evidence passages actually supplied (id -> text):
{evidence}

Score 0-100: does the supplied evidence SUPPORT the answer, and are the answer's factual claims traceable to these passages? 100 = fully supported by the evidence; 0 = unsupported/contradicted. Reply with ONLY the integer."""

GRADE_SCORE = {"FULLY": 1.0, "MOSTLY": 0.66, "PARTIAL": 0.33, "INCORRECT": 0.0}

_CITE_BLOCK = re.compile(r"\n*\s*citations?\s*:.*$", re.IGNORECASE | re.DOTALL)


def answer_only(text: str) -> str:
    """Strip the trailing 'Citations: ...' block so BERTScore + the judge see only the
    answer, not the citation IDs (which otherwise pollute semantic similarity)."""
    return _CITE_BLOCK.sub("", text or "").strip()


def _openai(prompt: str, max_tokens: int = 10) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY, max_retries=3, timeout=60)
    r = client.chat.completions.create(model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens)
    return (r.choices[0].message.content or "").strip()


def judge_strict_and_grade(q: str, ref: str, ans: str) -> tuple[bool | None, str]:
    try:
        out = _openai(STRICT_PROMPT.format(q=q, ref=ref, ans=ans), max_tokens=12)
        lines = [l.strip().upper() for l in out.splitlines() if l.strip()]
        passed = any("PASS" in l for l in lines[:1]) if lines else None
        grade = next((g for l in lines for g in GRADE_SCORE if g in l), "INCORRECT")
        return passed, grade
    except Exception as e:
        print(f"  strict-judge error: {str(e)[:80]}")
        return None, "INCORRECT"


def judge_citation(q: str, ans: str, evidence_map: dict) -> float | None:
    if not evidence_map:
        return 0.0
    ev_text = "\n".join(f"[{cid}] {txt[:300]}" for cid, txt in list(evidence_map.items())[:12])
    try:
        out = _openai(CITE_PROMPT.format(q=q, ans=ans, evidence=ev_text), max_tokens=6)
        m = re.search(r"\d+", out)
        return (int(m.group()) / 100.0) if m else None
    except Exception as e:
        print(f"  cite-judge error: {str(e)[:80]}")
        return None


def citation_precision(cited: list[str], supplied_ids: list[str]) -> float | None:
    """Fraction of the answer's cited IDs that were actually in the supplied evidence."""
    if not cited:
        return None
    supplied = set(supplied_ids)
    hit = sum(1 for c in cited if c in supplied or any(c in s or s in c for s in supplied))
    return hit / len(cited)


_bertscorer = None


def bertscore_f1(preds: list[str], refs: list[str]) -> list[float]:
    """PINNED config per PS: roberta-large, rescale_with_baseline=True, idf=False."""
    global _bertscorer
    from bert_score import BERTScorer
    if _bertscorer is None:
        print("[eval] loading BERTScorer (roberta-large, rescale_with_baseline)…")
        _bertscorer = BERTScorer(model_type="roberta-large", lang="en",
                                 rescale_with_baseline=True, idf=False)
    P, R, F1 = _bertscorer.score(preds, refs)
    return [float(x) for x in F1]


def load_chunk_text() -> dict:
    d = {}
    for l in open("datasets/round3/corpus.jsonl"):
        c = json.loads(l); d[c["title"]] = c["text"]
    return d


def evaluate(results_path: str, out_path: str, skip_cite_judge: bool = False):
    data = json.load(open(results_path))
    chunk_text = load_chunk_text()
    report = {"judge_model": JUDGE_MODEL, "pipelines": {}}

    for pipe, rows in data.get("pipelines", data).items():
        if not isinstance(rows, list):
            continue
        print(f"\n[eval] {pipe}: {len(rows)} answers")
        # strip the Citations: block so it doesn't pollute BERTScore / the judge
        preds = [answer_only(r.get("answer", "")) for r in rows]
        refs = [r.get("correct_answer", r.get("reference", "")) for r in rows]
        f1s = bertscore_f1(preds, refs)

        per_item, n_pass, n_scored, grades, cites, cite_prec = [], 0, 0, [], [], []
        for r, f1 in zip(rows, f1s):
            q, ref = r.get("question", ""), r.get("correct_answer", r.get("reference", ""))
            ans = answer_only(r.get("answer", ""))
            passed, grade = judge_strict_and_grade(q, ref, ans)
            supplied_ids = r.get("retrieved_ids", [])
            ev_map = {cid: chunk_text.get(cid, "") for cid in supplied_ids[:12]}
            cq = None if skip_cite_judge else judge_citation(q, ans, ev_map)
            cp = citation_precision(r.get("citations", []), supplied_ids)
            if passed is not None:
                n_scored += 1; n_pass += int(passed)
            grades.append(GRADE_SCORE.get(grade, 0.0))
            if cq is not None:
                cites.append(cq)
            if cp is not None:
                cite_prec.append(cp)
            per_item.append({"id": r.get("id", ""), "question": q, "answer": ans, "reference": ref,
                             "strict_pass": passed, "grade": grade, "graded_score": GRADE_SCORE.get(grade, 0.0),
                             "bertscore_f1": round(f1, 4), "citation_quality": cq,
                             "citation_precision": cp, "route": r.get("route", ""),
                             "total_inference_tokens": r.get("total_inference_tokens", r.get("total_tokens", 0))})
            time.sleep(0.2)

        agg = {
            "num": len(rows),
            "strict_pass_rate": round(n_pass / n_scored, 4) if n_scored else None,
            "graded_quality_mean": round(sum(grades) / len(grades), 4) if grades else None,
            "bertscore_f1_mean": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "citation_quality_mean": round(sum(cites) / len(cites), 4) if cites else None,
            "citation_precision_mean": round(sum(cite_prec) / len(cite_prec), 4) if cite_prec else None,
            "avg_total_inference_tokens": round(sum(i["total_inference_tokens"] for i in per_item) / len(per_item), 1),
        }
        report["pipelines"][pipe] = {"aggregate": agg, "per_item": per_item}
        print(f"  strict_pass={agg['strict_pass_rate']} graded={agg['graded_quality_mean']} "
              f"bert_f1={agg['bertscore_f1_mean']} cite_q={agg['citation_quality_mean']} "
              f"avg_inf_tok={agg['avg_total_inference_tokens']}")

    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\n[eval] wrote {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="benchmark results JSON")
    ap.add_argument("--out", default="results/eval_round3.json")
    ap.add_argument("--skip-cite-judge", action="store_true")
    a = ap.parse_args()
    evaluate(a.results, a.out, a.skip_cite_judge)
