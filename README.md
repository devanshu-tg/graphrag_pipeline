# GraphRAG Implementation — token-efficiency investigation

A workspace for one question: **on the same corpus, with the same model, how many tokens does
each retrieval approach need to answer the same question correctly — and does a knowledge graph
actually help?**

It holds an original build, the upstream implementation it replicates, two controlled variants of
that upstream, and the corpus downloader that feeds them. Each directory is a self-contained
project with its own README; this file explains what they are and how they relate.

> **We want GraphRAG to win. We are not going to assume it does.** Every arm is measured fairly
> and whatever comes out gets reported — including the results that do not flatter the graph.
> That principle is the reason the workspace has four sibling copies of the same benchmark rather
> than one tuned number.

---

## The six directories

| Directory | Role | What it is |
|---|---|---|
| [wixqa-token-efficiency/](wixqa-token-efficiency/) | **the build** | The original implementation and measurement harness. Governed by [SPEC.md](wixqa-token-efficiency/SPEC.md). |
| [mastergraph-round3/](mastergraph-round3/) | reference | Cloned upstream Round-3 submission — the implementation being replicated. |
| [mastergraph-fair/](mastergraph-fair/) | controlled variant | Upstream, with **one** change: the answer-policy clause in the system prompt. |
| [mastergraph-6stage/](mastergraph-6stage/) | controlled variant | Upstream, plus the two retrieval stages it lacks (rerank, per-side reserve). |
| [graphrag-inference-hackathon/](graphrag-inference-hackathon/) | reference | Cloned upstream Round-2 submission (BioASQ) — the origin of the "32% fewer tokens" claim. |
| [sp100_edgar_downloader/](sp100_edgar_downloader/) | data | Standalone SEC EDGAR downloader that builds the 400-filing S&P-100 corpus. |

The root is not a git repository, and the subprojects do not share an environment. Only
`mastergraph-round3` carries git history (remote:
`github.com/knightwalker96/context-optimization-using-graphrag`).

---

## wixqa-token-efficiency — the actual build

Owner: Devanshu Saxena · Reviewer: Sebastien Mancel · Target: Texas conference, 13 Aug 2026.

Read [SPEC.md](wixqa-token-efficiency/SPEC.md) first — it is the governing document, and it
records not just the design but every measurement mistake found along the way and the rule added
to prevent each one.

### Arms

Five arms, identical model, prompt, budget and packer; only retrieval differs.

| Arm | What it does |
|---|---|
| `p1` | LLM-only. Expected to score near zero — that is the point (see below). |
| `p2` | Basic RAG. Vector search + cross-encoder rerank. The baseline to beat. |
| `grep` | **The control that matters.** Same router as GraphRAG, constraints resolved by regex over the filings — no graph, no embeddings, no LLM. |
| `p3` | GraphRAG. Identical router; constraints resolved against TigerGraph Savanna. |
| `p3off` | GraphRAG with routing disabled — isolates routing from vector retrieval. |

**Why `p1` must fail.** The upstream Round-2 demo scored 90% with no retrieval at all, against
92% and 96% for its RAG arms. When the no-retrieval baseline nearly matches your pipelines, you
are not measuring retrieval. Both corpora here are chosen so that a general model cannot answer
from memory.

**Why `grep` exists.** If graph traversal cannot beat regex over the same documents, the honest
finding is *"you needed an index, not a graph."* No competing submission ran this arm.

### Two datasets

- **WixQA** (`data_wixqa/`) — 6,221 Wix Help Center articles. Real support content, but 87% of its
  questions are answered by a single article, so `build_multihop_eval.py` constructs a 2-hop bridge
  set with a shortcut filter (a chain is rejected if any single article mentions both endpoints).
- **SP100** (`data_sp100/`) — 400 SEC filings from [sp100_edgar_downloader/](sp100_edgar_downloader/),
  with the 50 frozen Round-3 questions stratified into Tier A (8 single-fact), B (16 two-hop
  bridge), C (20 ticker-set aggregation), D (6 cross-document/temporal).

### Latest scored results — SP100, n=50, judge `gemini-3.6-flash`

From `headline_v2.log`. Tokens are Gemini `usageMetadata`, never estimated.

| Arm | judge | BERTScore | tok/q | cost/q | Tier-C set F1 |
|---|---|---|---|---|---|
| MG-reproduced — Traditional RAG | 0.360 | −0.028 | 4,760 | $0.00150 | 0.593 |
| MG-reproduced — GraphRAG | 0.480 | 0.048 | 2,471 | $0.00081 | 0.813 |
| Ours v1 — GraphRAG | 0.460 | 0.349 | 3,463 | $0.00107 | 0.704 |
| Ours v2 — Traditional RAG | 0.340 | 0.296 | 3,495 | $0.00106 | 0.428 |
| Ours v2 — GraphRAG | 0.500 | 0.402 | 2,267 | $0.00070 | 0.790 |
| **Ours v2 — grep control** | **0.640** | **0.436** | 2,262 | $0.00070 | **0.947** |

GraphRAG beats its own RAG baseline in both implementations — ours by +0.160 judge and 35.1%
fewer tokens, theirs by +0.120 and 48.1%. **And the grep control beats both graphs**, at the same
token cost. On this benchmark, most of what looks like a graph win is a *routing* win: knowing
which constraint to resolve, not resolving it by traversal.

Read the judge column with care. It carries a measured ~8-point noise floor on this project's
data — two arms receiving byte-identical prompts once scored 8.4 apart. **Tier-C set F1 is the
only column with zero grading variance**, which is why 28 of the 50 questions (Tiers A and C) are
scored by exact/set match and never sent to a judge. Every score is also reported twice, against
the published answer key and against `answer_corpus` (what the filings actually say) — the
published key has at least five errors this project can cite chapter and verse for.

### Current state

`results/sp100/ours_v3.jsonl` holds 250 answers from the v3 run (section-aware chunking, 57,946
chunks, `gemini-embedding-2`, graph `sp100_v3`) — **run but not yet scored**. Next step:

```bash
cd wixqa-token-efficiency
python score_round3.py --answers results/sp100/ours_v3.jsonl
```

The `mastergraph-fair` rerun is blocked: its last attempt (`mg_fair.log`) failed on
`TG vector_search — System Memory in Critical state` from the Savanna instance.

---

## The three upstream copies, and why there are three

`mastergraph-round3` is the upstream Round-3 submission as published: 400 SEC filings, three
pipelines, reporting 48.7% strict pass vs 33.3% for RAG at 50.2% fewer tokens. `mastergraph-fair`
and `mastergraph-6stage` are byte-for-byte that code with exactly one intervention each, so any
difference in output is attributable to the intervention and nothing else.

**`mastergraph-fair` — one clause in `config.py`.** Upstream's system prompt says *"if the answer
is not supported by them, say so."* Its GraphRAG declines to answer on 13 of the frozen 50; ours
declines on 0 — and 10 of the 13 questions our arm "wins" are ones theirs *refused* rather than
got wrong. A head-to-head on raw scores was therefore comparing answer **policy**, not retrieval.
This copy swaps that clause for the commit-and-answer instruction our arm uses, changes nothing
else, and reuses their index and graph unchanged.

**`mastergraph-6stage` — two missing stages, off by default.** Upstream has no reranking in
Pipeline 3 at all: it truncates the deduped candidate list positionally. This copy adds stage 4
(cross-encoder rerank, placed after the cheap filter and before the context cut) and the second
clause of stage 5 (comparison questions reserve seats per side, round-robin, so one company's
chunks cannot take all ten slots). Both are gated behind `P3_RERANK=1`, so the file behaves
identically to upstream unless switched on.

---

## Findings worth reading before writing any of this yourself

All of these are recorded with reproductions in [SPEC.md §11](wixqa-token-efficiency/SPEC.md).

- **`pyTigerGraph.upsertVertices` takes a list of tuples, not a dict.** Passing a dict iterates
  its keys, so `"wix editor"` becomes `v[0]="w"`, `v[1]="i"` — every entity collapsing into one
  garbage vertex per distinct first character, with the real vertices created implicitly by the
  *edge* upserts carrying blank attributes. Reproduced: 5,107 vertices existed after sending
  5,071. It never raises, counts look plausible, and only reading an attribute **back** reveals
  it. Verification must sample attribute values, on a fresh connection — a load-time handle
  returns confidently wrong counts from stale cached schema.
- **A fixed passage count cannot demonstrate a token saving.** With passage count fixed, tokens
  can only differ if passage sizes differ — so upstream's 32% Round-2 token win was a per-passage
  character cap, not better retrieval. Budget must be a token cap, and the budget each arm
  actually received must be recorded and compared automatically.
- **A routed facts block must be charged against the budget, not added on top.** Otherwise
  GraphRAG spends the baseline's whole context *plus* the block and is arithmetically incapable
  of being cheaper.
- **Small-sample accuracy is the least reliable and most expensive number available.** The same
  code gave 60% vs 21% at n=15, 50% vs 40% at n=20, and 54.8% vs 54.3% at n=600. Gate on
  retrieval recall first — `eval_retrieval.py` runs it with no LLM calls.
- **Thinking models spend the output budget before emitting anything.** The judge burns ~240
  tokens reasoning, and `maxOutputTokens` covers reasoning *and* output — at a 200-token cap it
  returns `MAX_TOKENS`, an empty body, and HTTP 200. A truncated response must raise, never
  return.
- **Chunk boundary placement matters more than chunk size.** Recursive splitting cuts mid-sentence
  13% of the time on procedural articles; semantic (p95) cuts cleanly 99% of the time. Larger
  chunks were both larger *and* worse placed.
- **Hub thresholds must be calibrated per corpus.** Upstream's thresholds, tuned for 112,869
  biomedical entities, would have discarded the three most important Wix products outright.

---

## Getting started

Each project installs and runs on its own; see its README for keys and commands.

```bash
# Build the SEC corpus (stdlib only, no dependencies)
cd sp100_edgar_downloader && python download_sp100_edgar.py --email you@example.com

# The main build
cd wixqa-token-efficiency
pip install -r requirements.txt
cp .env.example .env          # GOOGLE_API_KEY + TIGERGRAPH_* — Gemini everywhere, no OpenAI
python eval_retrieval.py      # free gate: retrieval recall, no LLM calls
python run_sp100.py --arms p1 p2 grep p3 p3off --out results/sp100/ours.jsonl
python score_round3.py --answers results/sp100/ours.jsonl
python verify_run.py results/sp100/ours.jsonl    # asserts the §10 fairness checks
streamlit run app.py          # dashboard; reads the results file, computes nothing
```

`verify_run.py` is not optional. It enforces that the arms shared a model, prompt, reranker and
budget, that every token count is provider-reported, that the packed context never exceeded the
cap, and — specifically — that graph entity and document counts **vary** across questions, which
is how a dead traversal disguises itself as a working one.
