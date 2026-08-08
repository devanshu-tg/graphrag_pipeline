# mastergraph-improved — what changed and why

A copy of `mastergraph-round3/` with seven fixes applied. **Nothing in the original is
modified**; the 428 source files are hashed in
`wixqa-token-efficiency/results/sp100/MG_SOURCE_FROZEN.sha256` and that manifest still
verifies. The 927 MB Chroma store and the `Round 3 Dataset` folder are directory junctions
back to the original, so this copy costs 269 MB rather than 1.3 GB.

Every fix is behind an environment gate, all **off by default**. An unset environment
reproduces upstream byte for byte, which is what makes the ablation meaningful — if the
baseline run does not land on 25/50, the copy is wrong and no other number from it counts.

| Gate | Fixes | What it turns on |
|---|---|---|
| `MGI_FIX_PROMPT` | 1, 2 | citations out of the answer field; abstention clause replaced |
| `MGI_FIX_BRIDGE` | 3, 4 | cross-encoder rerank on the bridge clause; clause-split hyphen bug |
| `MGI_FIX_ROUTER` | 5, 6 | aggregation route actually answers its questions; selective re-ranking |
| *(always on)* | 7 | portability — the code could not run off one specific macOS machine |
| *(always on)* | — | **environment correction**, see below — not a fix, no credit claimed |

---

## The environment correction — read this before trusting any new number

**`gemini-2.5-flash` changed underneath this benchmark, and their code cannot absorb it.**

On that model `maxOutputTokens` is a single budget covering **thinking *and* visible output**.
Measured live today against a real 10-passage P3 context (17,135 chars):

| | thinking | visible answer |
|---|---|---|
| upstream call (`maxOutputTokens=512`) | **489** | **11** |
| `thinkingBudget=0` | — | 32 |
| `maxOutputTokens=2048` | 489 | 11 |

489 of the 512-token budget goes to reasoning, leaving 11 tokens for the answer. That is the
mechanism behind the truncated answers sitting in their saved results — EQ03 stored as
`'$57,048 million\nCitations: [JPM/10'` at 18 completion tokens, cut mid-citation, nowhere
near what looks like a 512 ceiling. It bites hardest exactly where this benchmark is scored:
the prompt asks Tier-C questions for "the COMPLETE set of matching companies", and EQ25's gold
answer alone is ~24 tickers.

**Their recorded run did not have this.** Across all 150 saved records,
`totalTokenCount − (prompt + candidates)` is exactly **0**, and a live call confirms the API's
`totalTokenCount` *does* include thoughts (283 + 10 + 119 = 412). So their 25/50 was measured
with the full 512 tokens available to the answer.

Running their code verbatim today would therefore truncate answers they never had truncated —
**understating their baseline and inflating every fix measured against it.** So the correction
is applied to **every arm including the baseline**: a fix has to beat their code working
properly, not their code crippled by an API change.

`MGI_ENV_CORRECT=0` reproduces the literal upstream call. Useful for demonstrating the
truncation; not a fair baseline.

### Consequences for the numbers

1. **The published head-to-head is unaffected and stands.** Our arm uses
   `gemini-3.5-flash-lite`, which emits **0** thinking tokens (verified live, and our
   `llm_client` folds `thoughtsTokenCount` into completion tokens regardless). Their frozen run
   also emitted 0. So *ours 2,274 vs theirs 2,471 tok/q* was apples-to-apples.
2. **New runs are not comparable to the frozen 25/50 on tokens.** Today's runs will report
   thinking tokens that the frozen run did not have. Only within-today comparisons are valid,
   which is what the ablation is.
3. **A hypothesis I had to discard.** I first blamed upstream's `parts[0]`-only read for the
   truncation and probed for it — every response came back single-part with
   `finishReason=STOP`. That hypothesis was **wrong**; the truncation is entirely budget
   starvation. The `parts` handling is still fixed, but only because `parts` is a list in the
   API contract and a `thought` part would otherwise be concatenated in as answer text.

---

## Why these fixes, and what they are not

The head-to-head result was **ours-v3 32/50 · MG-reproduced 25/50** on MasterGraph's own
frozen 50 questions with one shared judge; excluding the 16 questions our tuning touched it
is 24/34 vs 19/34. A full audit of their code then found the gap is **not architectural**:

- their **router is better than ours** — 20/20 Tier C and 8/8 Tier A routed correctly
- their **schema, chunking and citation design are better** — we adopted all three into v3

So the interesting question is not who won, it is *what actually separates the two arms*. The
fixes below are the answer to that question. Every one of them is a prompt, a regex or a
retrieval call. None of them changes the graph, the schema or the extraction.

---

## Fix 1 — citations out of the answer field

`config.SYSTEM_PROMPT` tells the model to append a `Citations:` block to its answer. That
block is part of the `answer` string, so a ~26-token string gets graded against an ~11-token
gold reference.

**Re-measured on their own saved answers** (`wixqa-token-efficiency/verify_fix1_bertscore.py`
— local BERTScore, no API cost, retrieval held perfectly constant because the answers already
exist):

| arm | answers with a `Citations:` block | BERTScore as-is | stripped | delta |
|---|---|---|---|---|
| **MG P3 GraphRAG** | 37/50 | **0.0483** | **0.3229** | **+0.2745** |
| MG P2 Basic RAG | 38/50 | −0.0283 | 0.2955 | +0.3238 |
| MG P1 LLM-only | 38/50 | 0.0490 | 0.2593 | +0.2103 |

The examples show how mechanical this is — the answers are *already right*:

```
EQ01  gold      '$416,161M (~$416.2B)'
      as-is     '$416,161 million\nCitations: [AAPL/10-K_2025-10-31#item8-c11]'
      stripped  '$416,161 million'
```

Both pipelines already log `retrieved_ids` separately, which is the auditable citation trail —
the Round 3 requirement is still met, it just stops being graded as if it were the answer.

The prompt change alone is not enough: the model still emits the block on a minority of
questions and one leak costs more than the fix wins. `config.strip_citations()` enforces it.

## Fix 2 — the abstention clause

`"...if the answer is not supported by them, say so"` produces **13 abstentions out of 50**;
our arm abstains 0 times. Checked against their own logged context: **3 of the 13 had the
answer in context and were refused anyway**, 10 were correct refusals. On EQ18 the model
produced the *hard* half (an officer name from an 8-K) and then refused the *easy* half —
Chevron's auditor — which their graph returns correctly when queried directly.

Replaced with the commit-and-answer-every-part instruction from our `run_sp100.py`, so the
two arms differ on retrieval rather than on answer policy.

## Fix 3 — rerank the bridge clause

Their bridge route scores **0/3**, and **8 of their 10 genuine retrieval misses are Tier B**.

Their clause *extraction* is correct — verified identical to ours on 5 of 6 questions. The
weakness is what happens next:

| | clause retrieval |
|---|---|
| them | `_tg_vector_search(clause, k=6)` — plain cosine over `bge-small`, 384-dim, against 61,777 chunks |
| us | MMR (k=20, fetch_k=60) + cross-encoder rerank |

The cross-encoder is **already vendored in their repo** (`pipeline2_basic_rag.RERANKER_MODEL`,
disabled by their own P2 ablation), so this reuses it instead of adding a dependency. It is
compute-only — zero LLM tokens — so the P2/P3 token comparison stays intact.

## Fix 4 — the clause-split hyphen bug

```python
re.split(r"[—\-–]|,\s*(?:who|what|which|and)\b", question)[0]
```

The `\-` alternative matches the hyphen inside `8-K`, `GLP-1`, `T-Mobile` and every
hyphenated name, so the bridge clause truncates mid-phrase. EQ18's clause becomes
`"Chevron's departing Chief Legal Officer (per its 2026 8"`. Latent on the frozen 50 only
because EQ18 routes to `lookup`; it fires on any bridge question containing a hyphen. Em and
en dashes are the real separators and are kept.

## Fix 5 — the aggregation route does not answer four of its own questions

**My first diagnosis was wrong and the offline check caught it.** I assumed EQ11/19/20/23
were misroutes and wrote a rule to veto `aggregation` for singular-answer questions. Checking
the four against the index showed the route was right in three of four cases and the repair
belonged downstream. `build/router_diff.py` now reports **0 route changes** — their 20/20
Tier-C and 8/8 Tier-A routing is untouched, which is exactly what you want from a fix aimed
at four Tier-B questions.

What was actually wrong, all verified offline with `build/agg_check.py` (no TigerGraph, no
API calls):

| | bug | before | after |
|---|---|---|---|
| **5b** | **direction.** `"AMD names which companies as its competitors"` asks for AMD's out-edges. `_detect_aggregation_constraints` emits `("competitor_of","amd")`, the mirror question. | `{CL, NVDA}` — Colgate-Palmolive as an AMD competitor | `{INTEL, NVIDIA}` = gold |
| **5e** | **amount parsing.** The pattern only accepts the word `billion`, so EQ11's `"$25B"` parses as *no amount* and the constraint degrades to "any buyback event". | 96 of 100 companies | — |
| **5f** | **amount matching is a substring test.** `"25" in "125 million"` is true, so `$25B` matches `$125 million`, `125,854` and `$ 25,985`. Now compared numerically. | — | — |
| **5g** | **name resolution.** The first-word key keeps its punctuation, so `"Netflix, Inc."` registers as `netflix,` and a question saying `Netflix` resolves to `None`. This silently disabled 5a. | Netflix stayed in its own exclusion | resolves |
| **5a** | **negation.** `"Besides Netflix, which company…"` — the exclusion is ignored. | — | `NFLX` excluded |
| | *(5a/e/f/g combined on EQ11)* | **96 companies** | **4** (`ADBE, BAC, BMY, VZ`), gold `Adobe` |
| **5c** | **fiscal-year-end.** Only the auditor half of EQ20 is detected; `company_facts` carries an exact `fy_end` for all 100 companies. | 24 Deloitte companies | intersect with `{MSFT, PG}` |
| **5d** | **superlative.** EQ23's set is right but nothing resolves "which one had the *highest* revenue". `company_facts.revenue` cannot help — it holds `"$4.0 trillion"` for NVDA (market cap) and `"null"` for AMD — so each finalist's real Item 7/8 chunks are retrieved instead, the same source their own `compare` route trusts. | set only | set + figures |

### EQ20's answer key, independently reconfirmed

**EQ20 has two correct answers.** It asks for "the Deloitte-audited company with a fiscal year
ending June 30, 2025". Both **Microsoft** and **Procter & Gamble** are Deloitte-audited and
both carry `fy_end = "June 30, 2025"` in MasterGraph's own `company_facts.json`. The gold
answer is "Microsoft" alone.

This was already one of the five key defects the earlier answer-key audit found (see the
`score_round3.py` docstring); what is new is that it now falls out of *their* index rather
than ours, which is a stronger form of the same evidence. Fix 5c makes the retrieval correct;
nothing can make the question unambiguous.

## Fix 6 — selective re-ranking, on `lookup` only

Measured on `mastergraph-6stage/`, which applies relevance ranking to every route: Tier C went
**11 → 12** but Tier D went **4 → 2**. Net negative. Relevance ranking is order-destroying and
item-blind — EQ47 needs *both* Apple 8-Ks and asks what each one reports, so a ranker that
promotes the more "relevant" one and drops the other cannot be right however good its scores
are.

`lookup` is single-fact and order-free, so it is the one route where ranking can only help.
`temporal` (which owns EQ45/46/47) and `aggregation` (one evidence line per company) are
excluded **by construction, not by tuning**.

## Fix 7 — portability (always on, changes no result)

Two host dependencies stop the pipeline running anywhere but the machine it was written on.

**Hardcoded `.env` path** — `load_dotenv("/Users/kajal/Downloads/GRAPHRAG HACKATHON/.env")`
appears in six modules (`build/evaluate_round3.py`, `build/extract_kg.py`,
`build/precision_pass.py`, `build/round3_tg_ingest.py`, `build/smoke_test_tg.py`,
`pipeline3_round3/pipeline.py`) and `build/run_ablations.sh` cds to the same path. On any
other machine that call is a **silent no-op** and every credential resolves to `""`. Replaced
with `_portability.load_env()`, which honours `MG_ENV_FILE` and otherwise searches upward.

**No explicit encoding** — 0 of 32 file operations across their five main modules set
`encoding=`, so both writer and reader inherit the host locale. Self-consistent on one
machine, broken everywhere else. This is demonstrated, not theorised:

> `datasets/round3/corpus.jsonl`, **built on Windows by their own `build_round3_corpus.py`**,
> is cp1252. It raises `UnicodeDecodeError` at byte 307 when read as UTF-8 — which is what
> every Linux and macOS host does by default.

`_portability.ropen()` probes the whole file with an incremental decoder, caches the verdict,
and reads UTF-8 when the file is UTF-8 and cp1252 when it is not. On this machine it returns
**byte-identical** text to the bare `open()` the baseline used, so the ablation baseline is
unaffected. `wopen()` writes UTF-8 always, so new artifacts are portable.

**Console encoding** — nothing calls `sys.stdout.reconfigure`, so printing the `✓` in
`round3_tg_ingest.py` crashes on a cp1252 console. `init_console()` fixes it in-process
instead of requiring `PYTHONIOENCODING` to be set outside.

### Not done

Dead config (`MAX_DOCS_TO_INDEX` and friends) and the 8 print-only `except` blocks. Both real,
neither changes a number.

---

---

## RESULTS — 4 baseline runs vs 3 fully-fixed runs, one shared judge and key

Replicates, not single runs, because the noise floor turned out to be large (below).

| metric | baseline (n=4) | all fixes (n=3) | delta |
|---|---|---|---|
| **Tier B (16)** | **6/16 in all 4 runs** | **9/16 in all 3 runs** | **+3, zero variance** |
| **Tier C set-F1 (20)** | 0.768 (0.756–0.772) | **0.825 in all 3 runs** | **+0.057, no overlap** |
| judge, Tier B+D (22) | 8.5 (range 7–10) | 11.7 (range 11–12) | +3.2 |
| **BERTScore** | **−0.010** | **0.387** | decisive |
| Tier A (8) | 0.688 (5–6/8) | 0.750 (6/8 every run) | within noise |
| Tier D (6) | 0.417 (1–4/6) | 0.444 (2–3/6) | too noisy to call |
| blended ALL | 0.587 (0.569–0.629) | **0.683 (0.670–0.690)** | **+0.096, no overlap** |
| tokens/question | 2,936 | 3,297 | **+12.3% cost** |

Corpus-verified key tells the same story: 0.594 → 0.689.

Three results are beyond argument because their ranges do not overlap across seven runs:
**Tier B +3**, **Tier C set-F1 +0.057**, and **BERTScore from negative to 0.387**. Tier A and
Tier D moved less than the noise floor and are reported as ties.

### The noise floor, measured

Four baseline runs with **identical configuration** drift by:

| | spread across 4 identical runs |
|---|---|
| Tier A | 1 full question (5/8 ↔ 6/8) |
| judge B+D | 3 questions (7 ↔ 10 of 22) |
| blended ALL | 0.060 |

Two sources stack. **TigerGraph vector search is non-deterministic** — between two identical
runs, 9 of 50 questions retrieved a *different chunk set* and 18 more retrieved the same
chunks in a different order, which is enough to flip an answer because the model is sensitive
to passage order. On top of that the thinking model varies its output at temperature 0.

This invalidates `score_round3.py`'s own footnote that "Tier A and C carry NO judge variance —
a 2-point gap there is real". That was true when the judge was the only noise source; the
variance now enters *upstream* of the judge, in retrieval. **Anyone benchmarking on Savanna
vector search needs replicates — single-run comparisons at n=50 are not trustworthy.**

The earlier head-to-head is not affected: our arm retrieves from Chroma, which is
deterministic, and its 7-question margin is far outside a 1-question floor.

**The judge is also not deterministic**, contrary to what this project established earlier —
grading the *same* answers twice flipped one question on 2 of 7 arms.

### What each fix was worth (single runs, so read against the floor)

| arm | blended | Tier B | BERTScore |
|---|---|---|---|
| baseline | 0.629 | 6/16 | −0.014 |
| +prompt (1,2) | 0.618 | 7/16 | **0.302** |
| +bridge (3,4) | 0.630 | 7/16 | 0.333 |
| +router (5,6) | **0.690** | **9/16** | **0.411** |

Only the full stack clears the floor. `+prompt` *lowered* the blended score on its own — it
bought BERTScore and a Tier-B question but cost Tier D — and `+bridge` was flat. Fix 3 very
likely subsumes fix 2's gain, which is exactly the interaction an ablation exists to expose.
A single "we improved it" number would have hidden all of this.

### A regression I introduced, caught by checking

`SUBJECT_FIRST_RE` (fix 5b) was written with `re.I`, which cancels its own capitalisation
requirement — so it matched "**Among companies that** … name NVIDIA … competitor" and emitted
`("__competitors_of__", "among companies that")`. **EQ43 returned an empty answer in all three
fixed runs, a question the baseline, +prompt and +bridge arms all answered correctly as
"CSCO, QCOM".** Fixed by dropping the flag and additionally requiring the captured span to
resolve to a real company. The Tier C gain above (0.775 → 0.825) is almost entirely this
correction. Predicted improvements are not results; this one was only found by re-reading the
per-question output after the numbers looked good.

### Honest summary

The fixes are real and land where the audit said they would — Tier B, the two-hop tier — at a
12% token cost. They are worth **+3 to +4 questions**, not the +7 originally predicted. The
architectural conclusion is unchanged and matters more than the scoreboard: **the two
submissions are separated by prompt design and clause retrieval, not by graph architecture**,
and on architecture MasterGraph's router, schema, chunking and citation design were ahead of
ours throughout.

---

## Verification harnesses (no TigerGraph, no API cost)

```
python build/router_diff.py    # baseline vs fixed route for all 50 — must show 0 changes
python build/agg_check.py      # constraint detection + set resolution for the fix-5 cases
python build/run_ablation.py   # the four graded runs
```

`router_diff.py` and `agg_check.py` run in seconds and cost nothing. They are what caught the
wrong first diagnosis of fix 5 before an hour of API budget went into measuring it.
