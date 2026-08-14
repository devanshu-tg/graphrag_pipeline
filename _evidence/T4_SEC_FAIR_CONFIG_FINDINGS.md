# T4 — was SEC's 57% a config artifact? No. It is real.

Run 2026-08-12. Three runs of the same 50 SEC questions through MasterGraph's own pipelines,
changing ONE config variable at a time. **Nothing in `mastergraph-round3/` was modified** —
`MAX_TOKENS` and `GEMINI_MODEL` are already `os.getenv()` in their config, and all three arms
read `config.SYSTEM_PROMPT` at call time (`pipeline1_llm_only.py:11`,
`pipeline2_basic_rag.py:114`, `pipeline3_round3/pipeline.py:666`), so the prompt was patched as
a runtime module attribute by `demo/server_factual.py` (ours). **428/428 frozen hashes verify
after the runs.**

## The three runs

```
run                     arm    judge  EXACT  partial  zero  refusals
A  512tok, refusal ok   p1     38.0%      5       30    15       8
A  512tok, refusal ok   p2     54.8%     17       21    12      12
A  512tok, refusal ok   p3     57.3%     20       14    16      11

B  8192tok, refusal ok  p1     36.2%      5       27    18       9
B  8192tok, refusal ok  p2     52.2%     16       18    16      16
B  8192tok, refusal ok  p3     59.2%     23       12    15      15

C  8192tok, NO-refusal  p1     38.3%      5       28    17       1
C  8192tok, NO-refusal  p2     58.2%     16       24    10       0
C  8192tok, NO-refusal  p3     65.7%     23       18     9       0
```

## Attribution

```
      token budget A->B      prompt B->C          TOTAL A->C
p1     -1.8 judge / +0 exact  +2.1 judge / +0      +0.3 / +0
p2     -2.5 judge / -1 exact  +6.0 judge / +0      +3.5 / -1
p3     +1.9 judge / +3 exact  +6.5 judge / +0      +8.4 / +3
```

**Fixing every config fault the audit identified changed correctness by +3 questions for
GraphRAG and −1 for Basic RAG.** Both are inside the project's documented 6-point noise floor
(Savanna's vector search retrieves a different chunk set on 9/50 questions between identical
runs). SEC's ~40–46% exact rate is a real property of the task, not an artifact.

## Two audit findings are hereby corrected

### I11 (`MAX_TOKENS=512`) — real mechanism, wrong impact, wrong sign

The truncation was real and I measured it correctly: **19/50 P2 answers cut off at A, 0/50 at
B.** Raising the budget eliminated truncation completely and roughly doubled mean completion
tokens (33.6 → 59.6 for P2).

But scores did **not** improve — P2 fell 2.5 points. The reason is visible in the refusal
column: with room to think, the model reasons its way to *"the context does not support this"*
more often (P2 refusals 12 → 16). Their prompt permits that and the judge scores a refusal 0.

```
EQ13  A  p2  'JPMorgan Chase: $57.0 billion\nExxonMobil: $18'      judge 60
      B  p2  'The answer is not supported by the provided context.'  judge  0
```

**The truncated fragment scored better than the honest refusal.** So the 512-token budget was
quietly *inflating* SEC scores via partial credit on cut-off guesses, not depressing them. My
claim that it "invalidated the SEC vs enterprise comparison" was wrong.

### I12 (refusal-permitting prompt) — real, and it moves the judge, but buys no correctness

Replacing their refusal licence with our no-empty rule drove refusals to **zero** (P2 16 → 0,
P3 15 → 0) and moved the judge +6.0 / +6.5. **Exact answers: unchanged at 16 and 23.**

Every point gained came from the judge awarding 20–60% for a plausible-but-wrong guess where it
previously awarded 0 for an admission of ignorance. Not one additional question was answered
correctly.

## The uncomfortable implication for OUR OWN configuration

`config.SYSTEM_PROMPT` in `enterprise-graphrag` carries the same no-empty rule, applied to all
three arms. This run shows what that rule actually does:

- **it does not improve correctness** — 0 additional exact answers here
- **it inflates the judge score by ~6 points** by converting abstentions into scored guesses
- **and T1 measured its other cost**: 23 of 24 responses hallucinated a confident answer to
  questions with no answer in the corpus (`info_not_found`)

The rule was added for a defensible measured reason — 21 of 40 baseline answers came back empty,
which understated Basic RAG and made the comparison look rigged. It is applied identically to
every arm, so it does not bias the *comparison*. But it does inflate every **absolute** judge
number in this project, ours included, and it manufactures confident wrong answers on
unanswerable questions.

## What survives about the factual project

Unaffected by config, still true:

- **P1 fabricates 100% of its 85 citations** from zero context (`passage_1`, `1`, `2`, `DEF`),
  validated against their 61,777 real chunk ids. P2/P3 cite honestly (~94% verifiable).
- **Contamination**: P1 scores 36–38% with no retrieval at all, against 0.8% on enterprise.
  These are real companies the model has read about.
- **The judge rewards numerically wrong answers**: P1 answered Walmart $611,289M / Apple
  $383,285M — real FY2023 figures, wrong year, both wrong — and scored 75%.
- **The two dashboard tabs run different architectures**: SEC uses MasterGraph's hand-tuned
  router (`lookup`/`bridge`/`aggregation`/`compare`/`temporal`), enterprise uses our
  schema-driven planner.
- **Per tier is the real story**: aggregation P3 73.2% / 11 exact vs P2 47.4% / 3 exact;
  single-lookup P2 98.8% vs P3 61.9%; bridge — P1-with-no-retrieval beats both.

## Verdict on the factual project

The audit over-attributed SEC's weakness to configuration. Corrected: **the config faults are
real but immaterial to correctness.** SEC is genuinely harder than the enterprise benchmark
because its answers are numbers and dates that must be exact, its relations are thin
(`audited_by` 100 edges, `has_ceo` one) and its corpus is contaminated. MasterGraph's
implementation is not the problem.

If the fair config is adopted for reporting, the honest SEC line is:

> P1 38.3% · Basic RAG 58.2% · GraphRAG 65.7% (judge), with 23/50 vs 16/50 exact — and note
> that judge percentages here include partial credit for wrong numbers, so `exact` is the
> figure of record.

## Artifacts

- `results/enterprise/eval_sec_fair_tok.jsonl` — Run B
- `results/enterprise/eval_sec_fair_prompt.jsonl` — Run C
- `demo/server_factual.py` — `SEC_NO_REFUSAL=1` runtime prompt patch, nothing written to disk
- `build/run_sec_live.py` — `SEC_WORKER` env override so a second worker can be targeted
