# T1 — the dataset's own 500 questions: result

Run 2026-08-11/12. n=200, a stratified prefix of `official_questions.jsonl` (proportional at
every prefix length — verified exact). Predictions frozen in `T1_T2_PREDICTIONS.md` before the
run. These are the dataset's OWN published questions with its OWN gold answers; Phase 6 of the
build plan promised to run them as an independent cross-check and never did.

## Result

```
n=200 (192 scored; 8 info_not_found reported separately)

pipeline              judge   factrec   hit@k   prompt   compl   plan    tok/q
P1 LLM-only           23.5%    15.8%     0.0%      284      34      0      318
P2-RR Basic RAG       58.4%    44.3%    67.7%    5,571      42      0    5,612
P3 GraphRAG           45.9%    34.1%    50.4%    4,915      35  1,942    6,892
```

**GraphRAG is 12.5 points WORSE than reranked Basic RAG, retrieves the right document 17 points
less often, and costs 23% more tokens.**

Basic RAG wins 8 of 9 question types:

```
question_type                n       P1     P2-RR       P3     delta
conflicting_info             8     8.4%    75.4%    41.6%    P2 +33.8
completeness                 8    20.0%    64.0%    31.2%    P2 +32.8
miscellaneous                8    18.8%    74.4%    49.4%    P2 +25.0
intra_document_reasoning    16    13.9%    54.2%    33.8%    P2 +20.4
project_related             16    23.4%    57.8%    38.4%    P2 +19.4
basic                       70    24.2%    70.3%    58.5%    P2 +11.8
semantic                    50    22.9%    33.7%    29.1%    P2  +4.6
high_level                   4    46.2%    57.5%    56.2%    tie
constrained                 12    42.5%    73.2%    76.1%    P3  +2.9   <- only win, n=12
```

## The gap decomposes into TWO separate faults, not one

Splitting P3's rows by the route it chose, and comparing against P2-RR **on the same questions**:

```
route          n     P3 judge   P3 hit@k    P2 on same qs    delta
vector       166       51.5%      55.1%         59.9%        -8.4
graph:1hop    24        9.6%      22.9%         46.8%       -37.2
graph:2hop     2       17.5%       0.0%         75.0%       -57.5
```

### Fault 1 — catastrophic mis-routing (26 questions, 14%)

The planner decided 26 content questions were graph questions. On those, P3 scored **9.6%** where
Basic RAG scored **46.8%** on the identical questions, and retrieval hit@k collapsed to **22.9%**
(0% for the 2-hop ones). The mechanism is `_context_from_prov()`: once a graph route is chosen,
the context becomes one evidence *sentence* per edge instead of the chunk that actually contains
the answer. For "what are the default size limits for file uploads" that sentence is useless.

This is the same compression that produces the token win on metadata questions. It is actively
destructive on content questions.

### Fault 2 — P3's fallback is unreranked while the baseline is reranked

On the 166 questions P3 routed to vector search it still lost **8.4 points**. That is not the
graph's fault. `pipeline.py` line ~451 does `_tg_vector_search(question, k=VEC_K)` with
`VEC_K = CONTEXT_NUM_CHUNKS = 10` — plain top-10. `pipeline2rr_rerank.py` retrieves
`RR_POOL = 50` and reranks down to `RR_KEEP = 10` with a cross-encoder.

So **P3's fallback is *plain* Basic RAG, and the baseline is *reranked* Basic RAG.** The
reranker was added to the baseline as a fairness fix (correctly — it was the obvious sceptic's
question). Nobody noticed it left P3's own fallback path weaker than the arm it is compared
against. It never showed up before because on the metadata benchmark P3 routed to the graph
100% of the time and never used the fallback.

Weighting the two contributions:

```
166/192 x -8.4  = -7.3   fallback lacks the reranker   (58% of the gap)
 26/192 x -38.8 = -5.3   mis-routed to the graph       (42% of the gap)
                  -12.6  ~= the measured -12.5
```

**Neither fault is "the graph does not help on content questions".** That is true but separate —
it would predict parity, not a 12.5-point deficit. Both faults are fixable; neither is fixed
here, per the audit's terms.

## info_not_found — the hallucination cost of our own prompt

```
8 questions with NO answer in the corpus
P1        1 abstained   7 hallucinated (88%)
P2-RR     0 abstained   8 hallucinated (100%)
P3        1 abstained   7 hallucinated (88%)
```

**23 of 24 responses asserted a confident answer to an unanswerable question.**
`config.SYSTEM_PROMPT` states *"NEVER return an empty answer … never reply 'None', 'N/A' or that
you cannot answer."* It was added for a measured reason — 21 of 40 baseline answers came back
empty, which understated Basic RAG and made the comparison look rigged. This is its price, and
it applies to all three arms equally. Confirms audit finding I8.

## Scoring the frozen predictions

One was wrong, and wrong in the direction of being too generous to our own system.

| # | prediction | outcome |
|---|---|---|
| 1 | P2-RR and P3 **within noise** on basic+semantic | **WRONG** — P3 decisively worse, −12.5 overall, −11.8 on basic |
| 2 | P3 **may score lower**; mis-routing substitutes a provenance sentence for the chunk | **RIGHT** — and hit@k 22.9% on graph-routed questions is the receipt |
| 3 | P1 well above its enterprise 0.8% but still worst | **RIGHT** — 23.5% |
| 4 | retrieval hit@10 between 0.50 and 0.75 | **RIGHT** — P2-RR 67.7%, P3 50.4% |
| 5 | abstention under 25% on info_not_found | **RIGHT** — 4–12% |

## The two benchmarks side by side

Same code, same corpus, same three arms. The only variable is **who wrote the questions**.

| | our generated questions (n=50) | the dataset's questions (n=200) |
|---|---|---|
| GraphRAG | **99.5%** | **45.9%** |
| Basic RAG | 53.0% | **58.4%** |
| tokens, P3 vs P2 | 1,609 vs 5,409 = **−70%** | 6,892 vs 5,612 = **+23%** |
| retrieval hit@k | not measured | **50.4% vs 67.7%** |

## Defensible claim after T1 and T2

> The graph wins decisively on set aggregation over metadata relations, where top-k retrieval is
> structurally incapable — some questions require 1,065 documents against a 10-chunk budget. On
> content questions it is 12.5 points worse than reranked retrieval and 23% more expensive,
> because 14% of them are mis-routed into a compression that discards the answering text, and
> because its fallback path lacks the reranker the baseline has. It should be **routed to, not
> used by default**.

Not *"GraphRAG is 98% accurate."*

## Artifacts

- `results/enterprise/eval_official500.jsonl` — 600 judged rows (200 questions x 3 arms)
- `build/run_official.py` — runner; judge + fact recall + retrieval hit@k, checkpointed/resumable
- 300 questions of the 500 remain unrun; the stratified ordering means the reported prefix is
  proportionally representative, and the remaining 300 are available to extend n.
