# T2 — adversarial paraphrase: result

Run 2026-08-11. Predictions were frozen in `T1_T2_PREDICTIONS.md` **before** this executed.
Predicted P3 would fall to 60–85%. **It fell to 74.7%.** Inside the predicted band.

## What was changed, and what was not

Only the `question` string. `gold_members`, `answer`, `tier`, `shape` and `question_id` were
copied byte-for-byte from `eval_mixed50.jsonl` by `build/make_paraphrase50.py`, so the answer key
is identical and the two runs are directly comparable.

| shape | original | paraphrase |
|---|---|---|
| `author` | Who authored the document 'X'? | Whose work is 'X'? |
| `channel_of_doc` | Which Slack channel is the message 'X' in? | Where was 'X' posted? |
| `direct_reports` | Who reports directly to M? | Who answers to M? |
| `manager_of` | Who is P's manager? | Who does P answer to? |
| `dept_members` | Which people are in the D department? | Who works in D? |
| `intersection` | Which Slack channels contain documents involving both A and B? | Where do A and B appear together? |
| `two_hop_org` | Which Slack channels do the people reporting to M take part in? | Where do the people who answer to M post? |
| `dept_in_channel` | Which people from the D department take part in the #C Slack channel? | Who from D shows up in #C? |

Rejected during writing, for changing meaning rather than only vocabulary: *"who signed off on
X"* (approval, not authorship), *"who is on M's team"* (collides with `in_team`, a different
relation), *"where do both A and B show up"* (drops the co-occurrence constraint the gold
depends on — replaced with "appear together").

## Result

```
                 original    paraphrased    delta
P3 GraphRAG        99.5%        74.7%      -24.8   judge
                   0.997        0.703      -0.294   set-F1
                   49/50        30/50        -19    exact
P2-RR Basic RAG    53.0%        53.5%       +0.5   judge   <- THE CONTROL
                   0.548        0.523      -0.026   set-F1
                   23/50        24/50         +1    exact
```

**The control is what makes this valid.** P2-RR moved +0.5 points. Vector retrieval never
depended on the relation vocabulary, so if the paraphrases had made the questions harder in
general, P2-RR would have dropped too. It did not. P3's 24.8-point fall is attributable to the
wording change alone.

## Per shape — the finding is narrower than "the router is fake"

```
shape              n    orig    para    delta    exact
two_hop_org        6   1.000   0.000   -1.000   6/6  -> 0/6     total collapse
intersection       9   1.000   0.495   -0.505   9/9  -> 0/9     zero exact
author            10   1.000   0.600   -0.400   10/10 -> 6/10
dept_in_channel    5   0.971   0.933   -0.038   4/5  -> 4/5     held
channel_of_doc    10   1.000   1.000   +0.000   10/10 -> 10/10  held
direct_reports     4   1.000   1.000   +0.000   4/4  -> 4/4     held
manager_of         3   1.000   1.000   +0.000   3/3  -> 3/3     held
dept_members       3   1.000   1.000   +0.000   3/3  -> 3/3     held
```

**Four of eight shapes held perfectly.** "Where was 'X' posted?" still routed to `in_channel`
10/10 with the word "channel" removed; manager, reports and department all held. So the planner
does perform real semantic mapping on single-hop shapes — the schema-driven claim is not empty.

**Multi-hop planning did not survive at all.** `two_hop_org` went to absolute zero and
`intersection` to zero exact. Those are the 3-relation chains the GraphRAG story rests on, and
they were being held up by the question naming its own relations.

Caveat against my own test: `author` → "Whose work is 'X'?" is genuinely vaguer than "who
authored it", so part of that −0.400 is ambiguity introduced by the paraphrase rather than
planner failure. That row should be discounted; the `two_hop_org` and `intersection` rows should
not, because their paraphrases preserve meaning precisely.

## Routing confirms the mechanism

```
original    vector  0  ·  graph:1hop 30  ·  graph:2hop 14  ·  graph:3hop  6
paraphrase  vector 13  ·  graph:1hop 32  ·  graph:2hop  5  ·  graph:3hop  0
```

**Every 3-hop plan disappeared.** 13 questions abandoned the graph entirely and fell back to
vector search — at which point P3 *is* Basic RAG plus a wasted planner call.

## The token claim degrades with it

```
              prompt   compl   planner   tok/q
original         513     6.1     1,090   1,609
paraphrase     2,060    42.5     1,182   3,284
P2-RR (para)   5,324     6.0         0   5,330
```

The 70% token reduction becomes **38%**. The saving was never a property of the architecture
alone — it exists only while the planner succeeds and the answer can be a traversal line
instead of ten chunks.

## Conclusion for the write-up

Audit finding I3 is **confirmed, scoped to multi-hop**: the 99.5% is robust for single-hop
lookups and not robust for the multi-hop tier. The honest claim is

> *"Schema-driven routing generalises across phrasings for single-hop lookups (4/4 shapes held
> at 100% under paraphrase). Multi-hop plan construction does not: removing relation vocabulary
> took `two_hop_org` from 6/6 to 0/6 and `intersection` from 9/9 to 0/9 exact, while the Basic
> RAG control moved +0.5 points."*

Not *"GraphRAG is 99.5% accurate."*

## Artifacts

- `datasets/enterprise/eval_paraphrase50.jsonl` — questions (gold untouched, original retained
  in `question_original`)
- `results/enterprise/eval_para50.jsonl` — judged answers, both arms
- `build/make_paraphrase50.py` — generator, asserts every template matched
