"""Did moving the traversal INTO TigerGraph change the answer? Three runs, one answer key.

THE QUESTION
------------
P3's walk was a Python dict lookup over `relationships.json`. The typed graph enterprise_v2
existed but had zero installed queries, so it was decorative. build/install_v2_queries.py
installed 26 traversal queries and P3_TG_TRAVERSE=1 routes every hop through them.

That is only worth claiming if the answers hold. Three runs isolate the two changes:

    ent50     index v1 + local walk     the stored baseline (P3 judge 99.5%, 49/50 exact)
    v2local   index v2 + local walk     isolates the INDEX change (canonicalised names)
    v2graph   index v2 + GSQL walk      isolates the TRAVERSAL change

v2graph vs v2local is the measurement that matters: same index, same gold, same prompt — the
only difference is whether a Python dict or TigerGraph resolved each hop. Identical answers
mean the graph reproduces the walk. A gap means one of them is wrong.

Why v2local exists at all: going straight from ent50 to v2graph changes TWO things at once
(index and traversal), so any movement could not be attributed to either. That is the mistake
this project already made once when a planner change and a model change landed together.

    python build/compare_traversal.py
    python build/compare_traversal.py --tags ent50 v2local v2graph
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen   # noqa: E402

init_console()
OUT = Path("results/enterprise")


def load(tag: str, arm: str = "p3") -> dict:
    p = OUT / f"eval_{tag}.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    rows = [json.loads(l) for l in ropen(p) if l.strip()]
    return {r["question_id"]: r for r in rows if r["arm"] == arm}


def norm_ans(s: str) -> str:
    """Compare answers as SETS of normalised items, not as strings. The model orders a list
    freely between runs ('a, b, c' vs 'b, a, c'), and treating that as a difference would
    report noise as a regression."""
    import re
    items = {re.sub(r"[^a-z0-9]+", "", t.lower())
             for t in re.split(r"[,\n;]|\s+and\s+", s or "")}
    return "|".join(sorted(i for i in items if i))


def table(runs: dict[str, dict]) -> None:
    print(f"\n{'run':<12}{'n':>4}{'judge':>8}{'setF1':>8}{'exact':>8}{'recall':>8}"
          f"{'prompt':>9}{'compl':>8}{'plan':>7}{'tok/q':>9}{'latency':>9}")
    print("-" * 98)
    for tag, by in runs.items():
        R = list(by.values())
        if not R:
            continue
        jr = [r["judge_pct"] for r in R if r.get("judge_pct") is not None]
        pl = st.mean([r.get("planner_tokens", 0) for r in R])
        tok = st.mean([r["prompt_tokens"] + r["completion_tokens"] for r in R]) + pl
        print(f"{tag:<12}{len(R):>4}"
              f"{(st.mean(jr) if jr else float('nan')):>7.1f}%"
              f"{st.mean([r['f1'] for r in R]) * 100:>7.1f}%"
              f"{sum(r['exact'] for r in R):>5}/{len(R):<2}"
              f"{st.mean([r['recall'] for r in R]) * 100:>7.1f}%"
              f"{st.mean([r['prompt_tokens'] for r in R]):>9,.0f}"
              f"{st.mean([r['completion_tokens'] for r in R]):>8,.0f}"
              f"{pl:>7,.0f}{tok:>9,.0f}"
              f"{st.mean([r.get('latency_s', 0) for r in R]):>8.1f}s")


def by_tier(runs: dict[str, dict]) -> None:
    tiers = sorted({r.get("tier", "") for by in runs.values() for r in by.values()})
    print(f"\n{'tier':<14}" + "".join(f"{t:>22}" for t in runs))
    print("-" * (14 + 22 * len(runs)))
    for t in tiers:
        cells = []
        for by in runs.values():
            R = [r for r in by.values() if r.get("tier") == t]
            jr = [r["judge_pct"] for r in R if r.get("judge_pct") is not None]
            cells.append(f"{(st.mean(jr) if jr else 0):.1f}% judge / "
                         f"{sum(r['exact'] for r in R)}/{len(R)}" if R else "-")
        print(f"{t:<14}" + "".join(f"{c:>22}" for c in cells))


def head_to_head(a_tag: str, b_tag: str, a: dict, b: dict) -> None:
    """The measurement of record: same index, same gold, traversal is the only variable."""
    shared = sorted(set(a) & set(b))
    same_ans = [q for q in shared if norm_ans(a[q]["answer"]) == norm_ans(b[q]["answer"])]
    same_route = [q for q in shared if a[q].get("route") == b[q].get("route")]
    same_f1 = [q for q in shared if abs(a[q]["f1"] - b[q]["f1"]) < 1e-9]
    print(f"\n{'=' * 98}\n{b_tag} vs {a_tag} — per question ({len(shared)} shared)\n{'=' * 98}")
    print(f"  identical answer set : {len(same_ans)}/{len(shared)}")
    print(f"  identical route      : {len(same_route)}/{len(shared)}")
    print(f"  identical set-F1     : {len(same_f1)}/{len(shared)}")

    diffs = [q for q in shared if q not in set(same_ans)]
    if not diffs:
        print("\n  No answer differed. The graph walk reproduced the local walk exactly.")
        return
    print(f"\n  {len(diffs)} answer(s) differed:")
    for q in diffs:
        ra, rb = a[q], b[q]
        print(f"\n  [{q}] {ra['question'][:88]}")
        print(f"      gold      {str(ra['gold'])[:88]}")
        print(f"      {a_tag:<9} F1={ra['f1']:.2f} judge={ra.get('judge_pct')} "
              f"route={ra.get('route')}  {ra['answer'][:74]}")
        print(f"      {b_tag:<9} F1={rb['f1']:.2f} judge={rb.get('judge_pct')} "
              f"route={rb.get('route')}  {rb['answer'][:74]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["ent50", "v2local", "v2graph"])
    args = ap.parse_args()

    runs = {t: load(t) for t in args.tags}
    table(runs)
    by_tier(runs)
    if len(args.tags) >= 2:
        head_to_head(args.tags[-2], args.tags[-1], runs[args.tags[-2]], runs[args.tags[-1]])
    if len(args.tags) >= 3:
        head_to_head(args.tags[0], args.tags[1], runs[args.tags[0]], runs[args.tags[1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
