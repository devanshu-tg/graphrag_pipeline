"""T8 — confidence intervals that respect the benchmark's real structure.

THE PROBLEM
-----------
Every interval in this project treats the 50 enterprise questions as 50 independent trials.
They are not. `build/generate_mixed_eval.py` instantiates EIGHT templates:

    author 10 · channel_of_doc 10 · intersection 9 · two_hop_org 6 · dept_in_channel 5
    direct_reports 4 · manager_of 3 · dept_members 3

Within a template only the names change. The planner either solves a shape or it does not — T2
proved exactly that, taking `two_hop_org` from 6/6 to 0/6 in one step when the wording changed.
So the questions inside a shape are almost perfectly correlated, and the effective sample size
is closer to 8 than 50.

Treating n=50 as independent makes every interval far too narrow, which is the statistical
version of the same error the whole audit is about: reporting more confidence than the evidence
supports.

WHAT THIS DOES
--------------
A CLUSTER bootstrap: resample SHAPES with replacement (not questions), recompute the mean over
whichever questions those shapes bring, 20,000 times. Reported beside the naive per-question
bootstrap so the difference is visible rather than asserted.

    python build/bootstrap_by_shape.py
    python build/bootstrap_by_shape.py --tags ent50 para50
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen   # noqa: E402

init_console()

OUT = Path("results/enterprise")
QUESTIONS = Path("datasets/enterprise/eval_mixed50.jsonl")
ITERS = 20_000
SEED = 20260812


def ci(samples: list[float]) -> tuple[float, float]:
    s = sorted(samples)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s))]


def boot_questions(vals: list[float], rng: random.Random) -> list[float]:
    n = len(vals)
    return [st.mean(rng.choices(vals, k=n)) for _ in range(ITERS)]


def boot_shapes(by_shape: dict[str, list[float]], rng: random.Random) -> list[float]:
    """Resample the CLUSTERS. A shape enters or leaves the sample whole, which is what makes
    the interval reflect 'we tested 8 kinds of question', not 'we ran 50 trials'."""
    shapes = list(by_shape)
    out = []
    for _ in range(ITERS):
        picked = rng.choices(shapes, k=len(shapes))
        vals = [v for s in picked for v in by_shape[s]]
        out.append(st.mean(vals))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["ent50", "para50"])
    ap.add_argument("--metric", default="judge_pct", choices=["judge_pct", "f1"])
    args = ap.parse_args()

    shape_of = {q["question_id"]: q["shape"]
                for q in (json.loads(l) for l in ropen(QUESTIONS) if l.strip())}
    rng = random.Random(SEED)

    for tag in args.tags:
        p = OUT / f"eval_{tag}.jsonl"
        if not p.exists():
            print(f"skip {tag}: no {p}")
            continue
        rows = [json.loads(l) for l in ropen(p) if l.strip()]
        arms = sorted({r["arm"] for r in rows})
        print(f"\n{'=' * 92}\n{tag}   metric={args.metric}   {ITERS:,} bootstrap iterations\n{'=' * 92}")
        print(f"{'arm':<8}{'mean':>8}{'  by QUESTION (n=50)':>26}{'  by SHAPE (n=8)':>26}{'width x':>10}")
        print("-" * 92)
        for arm in arms:
            R = [r for r in rows if r["arm"] == arm and r.get(args.metric) is not None]
            if not R:
                continue
            scale = 100.0 if args.metric == "f1" else 1.0
            vals = [r[args.metric] * scale for r in R]
            by_shape: dict[str, list[float]] = defaultdict(list)
            for r in R:
                by_shape[shape_of.get(r["question_id"], "?")].append(r[args.metric] * scale)
            lo_q, hi_q = ci(boot_questions(vals, rng))
            lo_s, hi_s = ci(boot_shapes(by_shape, rng))
            wq, ws = hi_q - lo_q, hi_s - lo_s
            print(f"{arm:<8}{st.mean(vals):>7.1f}%"
                  f"{f'[{lo_q:.1f}, {hi_q:.1f}]':>26}"
                  f"{f'[{lo_s:.1f}, {hi_s:.1f}]':>26}"
                  f"{ws / wq if wq else float('nan'):>9.1f}x")
        print(f"\n  shapes present: {len(by_shape)} "
              f"({', '.join(f'{k}={len(v)}' for k, v in sorted(by_shape.items()))})")
    print("\nThe shape-clustered interval is the honest one: the benchmark tested 8 kinds of\n"
          "question, not 50 independent ones, and T2 showed a shape flips as a unit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
