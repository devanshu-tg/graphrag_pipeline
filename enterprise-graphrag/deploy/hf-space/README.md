---
title: GraphRAG Pipeline Comparison
emoji: 🔗
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# GraphRAG vs Basic RAG vs LLM-only

Three retrieval strategies over the same 20k-document enterprise corpus, measured rather
than asserted. Ask a question and all three answer it live: no retrieval, vector top-k, and
a schema-driven graph traversal in TigerGraph.

## The measured result

50 questions — 20 single-document, 10 directory lookup, 20 multi-hop — same corpus, same
model (gemini-3.6-flash), same blind judge.

| | exact | LLM judge | tokens/query |
|---|---|---|---|
| LLM only | 0/50 | 0.8% | 273 |
| Basic RAG (reranked, 10 chunks) | 26/50 | 62.3% | 5,312 |
| **GraphRAG (3 chunks)** | **50/50** | **100%** | **2,733** |

**49% fewer tokens while reading 3 chunks against the baseline's 10.** The budgets are
deliberately unequal: 10 is where the reranked baseline scores best, so it is shown at its
strongest.

Multi-hop is where the difference lives. Basic RAG gets **0 of 20** exactly right there and
100% on directory lookups — same corpus, same model. The only variable is whether the
question needs a join.

## What is honest about this

- **43% of GraphRAG's 2,733 tokens are the planner call**, a fixed cost that does not shrink
  with the chunk budget. That is the floor on the token claim.
- **Not measured:** the baseline end-to-end at a 3-chunk budget. The retrieval curve predicts
  it gets worse (hit@k 82.5% → 75.0%), but predicted is not measured.
- **All 50 questions are relationship traversals** — the shape a graph is built for. On the
  SEC tab the same architecture *loses* single-fact lookup by 38 points.
- The company is fictional, which is why LLM-only scores 0.8% with no retrieval. That is the
  leak gate, and it is what makes the rest of the numbers mean anything.

## Rate limit

Live queries are capped per IP per hour. A query runs three pipelines and costs roughly
8,500 tokens. When you hit the limit the page still browses every recorded run, so nothing
stops working — you just stop spending someone else's quota.

Full evidence, per-tier breakdowns and the SEC comparison:
**https://graphrag-benchmark.netlify.app**
