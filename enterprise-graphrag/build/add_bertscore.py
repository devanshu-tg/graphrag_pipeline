"""Attach a per-row BERTScore F1 to an eval file, in place and idempotently.

Kept separate from run_eval.py on purpose: BERTScore needs roberta-large (~1.4 GB) and adds
minutes to a run, while the answers it scores are already saved. Scoring is a pure function of
(answer, gold) — re-running the pipelines to obtain it would cost tokens for nothing.

Empty answers are recorded as 0.0 rather than scored: BERTScore on an empty string against a
reference returns a similarity that means nothing, and a refusal should not earn partial credit.

    python build/add_bertscore.py results/enterprise/eval_flat.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    from accuracy import AccuracyEvaluator
    ev = AccuracyEvaluator()

    for path in sys.argv[1:]:
        p = Path(path)
        rows = [json.loads(l) for l in ropen(p) if l.strip()]
        todo = [r for r in rows if (r.get("answer") or "").strip()]
        print(f"{p.name}: {len(rows)} rows, {len(todo)} with an answer to score")
        if todo:
            f1 = ev.bertscore([r["answer"] for r in todo], [r["gold"] for r in todo])
            for r, v in zip(todo, f1):
                r["bertscore"] = v
        for r in rows:
            r.setdefault("bertscore", 0.0)
        with wopen(p) as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["arm"], []).append(r["bertscore"])
        for arm in sorted(by):
            print(f"    {arm}  BERTScore F1 = {sum(by[arm])/len(by[arm]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
