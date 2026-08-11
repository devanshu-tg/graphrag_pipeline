"""Recompute the gold answers for the SAME 20 questions against a chosen index.

WHY THIS IS NECESSARY, AND WHY IT IS NOT CHEATING
-------------------------------------------------
The gold answers were derived from the v1 graph. That graph could only see 27% of a directory
person's activity, because the other 73% was filed under their Slack username as an unrelated
vertex. So the gold sets were computed from the same defect the pipelines were being scored
against — the benchmark could not detect its own error, and a pipeline that somehow found the
hidden channels would have been marked WRONG for being right.

The questions are NOT touched. The wording, the seeds and the shapes are byte-identical; only
the answer key is recomputed, using the same traversal that generated it. Both runs are then
scored against the same corrected key, which is the only way the two graphs are comparable.

VALIDATION GATE: run with --index index (v1) and it must reproduce the existing gold EXACTLY.
If it cannot reproduce the old key from the old graph, it is not re-deriving the same thing
and its v2 output is not trustworthy either.

    python build/regold.py --index index      --check      # gate: must reproduce v1 gold
    python build/regold.py --index index_v2   --out datasets/enterprise/eval_questions_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

QUESTIONS = Path("datasets/enterprise/eval_questions.jsonl")

# The three generator templates, read back. Deterministic text -> deterministic seeds.
RE_ORG = re.compile(r"^Which Slack channels do the people reporting to (.+?) take part in\?$")
RE_INT = re.compile(r"^Which Slack channels contain documents involving both (.+?) and (.+?)\?$")
RE_DEP = re.compile(r"^Which people from the (.+?) department take part in the #(.+?) Slack channel\?$")


def load_graph(idx: Path):
    rels = json.load(ropen(idx / "relationships.json"))
    fwd: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    rev: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rels:
        fwd[r["relation"]][r["source"]].add(r["target"])
        rev[r["relation"]][r["target"]].add(r["source"])
    return fwd, rev


def regold(q: dict, fwd, rev) -> tuple[list[str], str] | None:
    """Return (gold_members, note) recomputed for this question, or None if unparseable."""
    docs_of, auth_of = fwd["participated_in"], fwd["authored"]
    reports_of, dept_of, chan_of = rev["reports_to"], fwd["in_department"], fwd["in_channel"]
    act = lambda p: docs_of.get(p, set()) | auth_of.get(p, set())          # noqa: E731

    text = q["question"]
    if (m := RE_ORG.match(text)):
        mgr = m.group(1)
        rs = sorted(r for r in reports_of.get(mgr, set()) if act(r))
        chans = {c for r in rs for d in act(r) for c in chan_of.get(d, set())}
        return sorted(chans), f"{len(rs)} reports with activity"
    if (m := RE_INT.match(text)):
        a, b = m.group(1), m.group(2)
        shared = act(a) & act(b)
        return sorted({c for d in shared for c in chan_of.get(d, set())}), f"{len(shared)} shared docs"
    if (m := RE_DEP.match(text)):
        dept, chan = m.group(1), m.group(2)
        members = [p for p in dept_of if dept in dept_of.get(p, set()) and act(p)]
        ppl = {p for p in members if any(chan in chan_of.get(d, set()) for d in act(p))}
        return sorted(ppl), f"{len(members)} dept members"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index_v2")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true",
                    help="assert the recomputed gold equals the stored gold (v1 gate)")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    fwd, rev = load_graph(Path("pipeline3_round3") / args.index)
    qs = [json.loads(l) for l in ropen(QUESTIONS) if l.strip()][: args.n]

    out, mismatched, grew, shrank = [], 0, 0, 0
    print(f"{'id':<6}{'shape':<18}{'old':>5}{'new':>6}   change")
    print("-" * 74)
    for q in qs:
        res = regold(q, fwd, rev)
        if res is None:
            print(f"{q['question_id']:<6}UNPARSEABLE — template changed?")
            return 2
        gold, note = res
        old = q["gold_members"]
        same = sorted(old) == gold
        mismatched += (not same)
        grew += len(gold) > len(old)
        shrank += len(gold) < len(old)
        added = [g for g in gold if g not in old]
        lost = [g for g in old if g not in gold]
        mark = "same" if same else (("+" + ",".join(added[:4])) if added else "") + \
                                   ((" -" + ",".join(lost[:3])) if lost else "")
        print(f"{q['question_id']:<6}{q['shape']:<18}{len(old):>5}{len(gold):>6}   {mark[:44]}")
        out.append({**q, "gold_members": gold, "answer": ", ".join(gold),
                    "gold_source": args.index, "derivation_note": note})

    print("-" * 74)
    print(f"{mismatched}/{len(qs)} answer keys changed   ({grew} grew, {shrank} shrank)")

    if args.check:
        if mismatched:
            print(f"\nGATE FAILED — re-deriving v1 gold from the v1 index did not reproduce it.\n"
                  f"The traversal here does not match the generator; do not trust the v2 key.")
            return 1
        print("\nGATE PASSED — v1 gold reproduced exactly from the v1 index.")
        return 0

    if args.out:
        with wopen(Path(args.out)) as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
