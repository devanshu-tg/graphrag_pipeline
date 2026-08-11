"""Generate SINGLE-HOP control questions — the check that the comparison is not rigged.

WHY THIS EXISTS
---------------
On the 20 multi-hop questions Basic RAG scored 0.017 while GraphRAG scored 0.623. A gap that
one-sided is a reason for suspicion, not celebration: it is exactly what you would see if the
baseline were simply broken, or if the benchmark had been built so only a graph could win.

Three explanations were ruled out by measurement:
  - a harsh scorer      — it gives partial credit ("eng-oncall, eng-platform" -> R=0.50)
  - a refusal-inducing prompt — an explicit "never answer empty" instruction moved the mean
                          setF1 by 0.000
  - a missing corpus    — the org chart IS indexed and IS retrieved 10/10 for a direct question

This file rules out the fourth and most important one: that P2 is broken. These questions are
answerable from ONE document — exactly what vector retrieval is good at. If Basic RAG is
healthy it should score WELL here, and near GraphRAG. If it scores ~0 here too, the pipeline
is broken and the multi-hop result means nothing.

A benchmark that only ever reports the case it wins is marketing. This is the control.

    python build/generate_singlehop.py --n 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

IDX = Path("pipeline3_round3/index")
CORPUS = Path("datasets/enterprise/corpus.jsonl")
OUT = Path("datasets/enterprise/eval_singlehop.jsonl")
SEED = 20260809


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    rels = json.load(ropen(IDX / "relationships.json"))
    fwd: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rels:
        fwd[r["relation"]][r["source"]].add(r["target"])

    # only documents whose text is actually retrievable, so a miss is a retrieval failure
    # rather than a question about something absent from the index
    have_text = set()
    titles: dict[str, str] = {}
    for line in ropen(CORPUS):
        if not line.strip():
            continue
        c = json.loads(line)
        have_text.add(c["doc_id"])
        titles.setdefault(c["doc_id"], c.get("title", ""))

    rng = random.Random(SEED)
    qs: list[dict] = []

    # ---- shape A: who authored this document?  (one hop, one document) ----------------
    authored = [(p, d) for p, ds in fwd["authored"].items() for d in ds if d in have_text]
    rng.shuffle(authored)
    for person, doc in authored:
        if len({p for p, ds in fwd["authored"].items() if doc in ds}) != 1:
            continue                      # ambiguous author -> not a clean single-hop answer
        name = doc.rsplit("/", 1)[-1].replace(".json", "").replace("-", " ")
        qs.append({"question_id": f"SH{len(qs)+1:02d}", "shape": "single_hop_author",
                   "question": f"Who authored the document '{name}'?",
                   "answer": person, "gold_members": [person],
                   "hops": "document -> author (stated in the document's own metadata)"})
        if len(qs) >= args.n // 2:
            break

    # ---- shape B: which channel is this document in? (one hop, one document) ---------
    in_chan = [(d, c) for d, cs in fwd["in_channel"].items() for c in cs if d in have_text]
    rng.shuffle(in_chan)
    for doc, chan in in_chan:
        if len(fwd["in_channel"].get(doc, set())) != 1:
            continue
        name = doc.rsplit("/", 1)[-1].replace(".json", "").replace("-", " ")
        qs.append({"question_id": f"SH{len(qs)+1:02d}", "shape": "single_hop_channel",
                   "question": f"Which Slack channel is the message '{name}' in?",
                   "answer": chan, "gold_members": [chan],
                   "hops": "document -> channel (stated in the document's own metadata)"})
        if len(qs) >= args.n:
            break

    with wopen(OUT) as fh:
        for q in qs[: args.n]:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote {len(qs[:args.n])} single-hop control questions -> {OUT}")
    for q in qs[: args.n]:
        print(f"  {q['question_id']} {q['shape']:<20} {q['question'][:62]}")
        print(f"         -> {q['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
