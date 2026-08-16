"""T2 — reword the 50 eval questions to strip relation vocabulary. Gold answers untouched.

WHY
---
Audit finding I3: all 50 questions in eval_mixed50.jsonl contain a word matching the relation
they need — "who AUTHORED" -> authored, "which Slack CHANNEL" -> in_channel, "who REPORTS to"
-> reports_to, "DEPARTMENT" -> in_department, "take part in" -> participated_in.

The project's headline claim is that routing is SCHEMA-DRIVEN: the graph publishes its relations
and one LLM call maps the question onto them, with no hand-written question patterns. V6 passes
literally — grep finds zero hardcoded entity names in the router. But if every question hands
the planner its own relation name, the planner never has to generalise, and the coupling has
simply moved from the router into the question wording.

This file removes that help and changes NOTHING else. If P3 still scores ~100%, the
schema-driven claim is real and I3 is wrong. If it collapses, the 100% was measuring vocabulary
matching.

WHAT IS AND IS NOT ALLOWED TO CHANGE
------------------------------------
Only `question`. question_id, tier, shape, answer, gold_members, hops are copied byte-for-byte,
so the answer key is identical and the two runs are directly comparable.

MEANING MUST BE PRESERVED EXACTLY, or the gold becomes wrong and the test measures nothing.
Two traps that were rejected while writing this:

  "Who signed off on 'X'?"        -- sign-off is approval, NOT authorship. Different relation.
  "Who is on M's team?"           -- collides with in_team, a DIFFERENT relation in the schema.
  "Where do both A and B show up?" -- loses the co-occurrence constraint. The original asks for
                                     channels containing documents involving BOTH; this version
                                     also admits channels where each appears separately, which
                                     is a different (larger) answer set. Replaced with
                                     "appear together".

Every rewrite asserts its input matched the expected template. A silent regex miss would leave
the question unchanged and manufacture a fake "no degradation" result, so a miss is fatal here.

    python build/make_paraphrase50.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

SRC = Path("datasets/enterprise/eval_mixed50.jsonl")
OUT = Path("datasets/enterprise/eval_paraphrase50.jsonl")

# relation words the planner could otherwise pattern-match. Checked against every output.
BANNED = ["author", "channel", "report", "manager", "department", "take part",
          "participat", "involv", "team", "label", "assign", "owner"]


def rewrite(shape: str, q: str) -> str:
    """One rewrite per shape, applied mechanically to every instance of that shape."""
    if shape == "author":
        m = re.fullmatch(r"Who authored the document '(.+)'\?", q)
        assert m, q
        return f"Whose work is '{m.group(1)}'?"

    if shape == "channel_of_doc":
        m = re.fullmatch(r"Which Slack channel is the message '(.+)' in\?", q)
        assert m, q
        return f"Where was '{m.group(1)}' posted?"

    if shape == "direct_reports":
        m = re.fullmatch(r"Who reports directly to (.+)\?", q)
        assert m, q
        return f"Who answers to {m.group(1)}?"

    if shape == "manager_of":
        m = re.fullmatch(r"Who is (.+)'s manager\?", q)
        assert m, q
        return f"Who does {m.group(1)} answer to?"

    if shape == "dept_members":
        m = re.fullmatch(r"Which people are in the (.+) department\?", q)
        assert m, q
        return f"Who works in {m.group(1)}?"

    if shape == "intersection":
        # "appear together" keeps the co-occurrence constraint the gold depends on
        m = re.fullmatch(r"Which Slack channels contain documents involving both (.+) and (.+)\?", q)
        assert m, q
        return f"Where do {m.group(1)} and {m.group(2)} appear together?"

    if shape == "two_hop_org":
        # "the people who answer to M" keeps it to DIRECT reports; "M's people" would be
        # ambiguous about transitivity and could justify a larger answer set
        m = re.fullmatch(r"Which Slack channels do the people reporting to (.+) take part in\?", q)
        assert m, q
        return f"Where do the people who answer to {m.group(1)} post?"

    if shape == "dept_in_channel":
        m = re.fullmatch(r"Which people from the (.+) department take part in the #(\S+) Slack channel\?", q)
        assert m, q
        return f"Who from {m.group(1)} shows up in #{m.group(2)}?"

    raise AssertionError(f"unhandled shape {shape!r}: {q}")


def main() -> int:
    rows = [json.loads(l) for l in ropen(SRC) if l.strip()]
    out, leaks = [], []
    for r in rows:
        new_q = rewrite(r["shape"], r["question"])
        assert new_q != r["question"], f"rewrite was a no-op: {r['question']}"
        low = new_q.lower()
        hit = [w for w in BANNED if w in low]
        if hit:
            leaks.append((r["question_id"], new_q, hit))
        out.append({**r, "question": new_q, "question_original": r["question"]})

    if leaks:
        print(f"WARNING: {len(leaks)} paraphrase(s) still contain a relation word:")
        for qid, q, hit in leaks:
            print(f"  {qid} {hit} -> {q}")

    with wopen(OUT) as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(out)} paraphrased questions -> {OUT}")
    print("gold_members / answer / tier / shape copied verbatim — answer key unchanged\n")
    seen = set()
    for r in out:
        if r["shape"] in seen:
            continue
        seen.add(r["shape"])
        print(f"  [{r['shape']}]")
        print(f"    was: {r['question_original']}")
        print(f"    now: {r['question']}")
        print(f"    gold({len(r['gold_members'])}): {', '.join(r['gold_members'][:4])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
