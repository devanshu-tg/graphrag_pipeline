# CONTEXT — GraphRAG on SEC Filings

**Everything a fresh machine (or a fresh AI session) needs to pick this project up.**

Written 2026-08-08. Read this top to bottom before touching anything.

---

## 0. TL;DR for a new session

- **Owner:** Devanshu Saxena, DevRel at TigerGraph (`devanshu.saxena@tigergraph.com`)
- **Deadline:** conference in **Texas, 13 August 2026**
- **What this repo now is:** MasterGraph's Round-3 hackathon GraphRAG submission, the SEC
  filings dataset it runs on, and `_evidence/` — the frozen record of a long head-to-head
  evaluation whose *code* has been deleted but whose *results* are preserved here.
- **What was deleted (deliberately, by the owner, 2026-08-08):** our own competing GraphRAG
  implementation (`wixqa-token-efficiency/`, 3.3 GB), plus `mastergraph-6stage/`,
  `mastergraph-fair/` and `graphrag-inference-hackathon/`. **There is no git history — these
  are gone permanently.** Only `_evidence/` survives from that work.
- **Bottom line of the evaluation:** on the benchmark both teams tuned against, ours scored
  higher (0.776 vs 0.603). On **5 held-out questions neither team had seen, MasterGraph won
  4–3 and used a third of the tokens.** The tuned number was not predictive. Details in §7.

---

## 1. Repo layout (after the 2026-08-08 cleanup)

```
GraphRAG-Implementation/
├── README.md                    (original project readme)
├── mastergraph-round3/          THE SYSTEM — MasterGraph's submission, 1,314 MB
│   ├── config.py                central config + SYSTEM_PROMPT
│   ├── llm_client.py            Gemini / OpenAI / Ollama wrapper
│   ├── metrics.py               token counting, latency, cost
│   ├── chroma_store.py          Chroma vector store (used by P2 only)
│   ├── pipeline1_llm_only.py    P1 — no retrieval (control)
│   ├── pipeline2_basic_rag.py   P2 — Basic RAG over Chroma
│   ├── pipeline3_round3/
│   │   ├── pipeline.py          P3 — GraphRAG over TigerGraph  ← the main artifact
│   │   └── index/               extracted KG (edges_prov, company_facts, events, …) 60 MB
│   ├── build/                   corpus build, KG extraction, TG ingest, benchmark, eval
│   ├── datasets/round3/
│   │   ├── corpus.jsonl         61,777 pre-chunked passages, 113 MB  ⚠ cp1252, see §9
│   │   ├── documents.jsonl      400 whole filings, 86 MB
│   │   ├── eval_set.jsonl       50 held-out benchmark questions (EQ01–EQ50)
│   │   ├── dev_set.jsonl        24 dev questions (zero overlap with eval)
│   │   └── manifest_clean.json  ticker → company/sector map
│   ├── chroma_db_round3/        Chroma index, 928 MB
│   ├── Round 3 Dataset/         original problem statement + source data
│   ├── ARCHITECTURE.md          their own architecture write-up
│   └── .env                     ⚠ credentials — copied here 2026-08-08, see §2 and §11
├── sp100_edgar_downloader/      the 400 raw SEC filings (source of truth), 94 MB
└── _evidence/                   frozen results — this file lives here
```

### `_evidence/` inventory

| file | what it is |
|---|---|
| `CONTEXT.md` | this document |
| `MG_FIXES_FINDINGS.md` | the 9 bugs found in their pipeline, with measurements |
| `MG_SOURCE_FROZEN.sha256` | hashes of all 428 MasterGraph source files (still verifying) |
| `MG_FROZEN.sha256` | hashes of their frozen benchmark result files |
| `heldout_set.jsonl` | the 5 held-out questions (§7) |
| `heldout_truth.json` | auditor/sector/fy_end for all 100 companies, cross-verified |
| `heldout_theirs.jsonl` | MasterGraph's answers to the held-out 5 |
| `heldout_ours.jsonl` / `_k2` / `_k0` | our (deleted) system's answers at 3 token budgets |
| `cmp_mg_frozen.jsonl` | their original 50-question answers |
| `cmp_ours_v3.jsonl` | our original 50-question answers |
| `mgi_*.jsonl` | 15 files — the full fix-ablation (baseline ×4, +prompt, +bridge, +router ×3, +multi ×3) |

---

## 2. Credentials and environment

**⚠️ SECURITY — OUTSTANDING ACTION**

The Google API key, TigerGraph host URL and `TIGERGRAPH_SECRET` were **pasted into a chat
session** and are therefore compromised. **They must be rotated.** After rotating, update
`mastergraph-round3/.env`. This has been flagged repeatedly and is still not done.

`mastergraph-round3/.env` needs:

```
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=<rotate me>
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

TIGERGRAPH_HOST=https://<id>.i.tgcloud.io
TIGERGRAPH_GRAPH=finance
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=            # BLANK — the workspace uses Google SSO
TIGERGRAPH_SECRET=<rotate me>   # gsqlSecret; this is the ONLY auth that works

P3_REQUIRE_TG=1
CONTEXT_NUM_CHUNKS=10
CONTEXT_CHUNK_CHARS=2000
MAX_TOKENS=512
```

### ⚠️ The `.env` only works from one directory

`pipeline3_round3/pipeline.py` line 45 calls:

```python
load_dotenv("/Users/kajal/Downloads/GRAPHRAG HACKATHON/.env")
```

That is a **hardcoded path to the original author's Mac** and a silent no-op anywhere else —
every credential resolves to `""`. Six modules do this (`build/evaluate_round3.py`,
`build/extract_kg.py`, `build/precision_pass.py`, `build/round3_tg_ingest.py`,
`build/smoke_test_tg.py`, `pipeline3_round3/pipeline.py`) and `build/run_ablations.sh` cds to
the same path.

It works today **only** because `config.py` separately calls bare `load_dotenv()`, which
searches upward from the current directory. **So you must run everything with
`cwd = mastergraph-round3/`.** Run it from anywhere else and you get
`Invalid URL scheme. Supported schemes are http and https.` — which is what a blank host
looks like.

### Python environment

Global Python 3.12 on Windows (no venv was used). Install with:

```bash
cd mastergraph-round3
pip install -r requirements.txt
# these were missing at one point and are all required:
pip install sentence-transformers langchain-chroma chromadb bert-score evaluate tiktoken scikit-learn
```

Always export `PYTHONIOENCODING=utf-8` on Windows — see §9.

---

## 3. TigerGraph Savanna

- **Workspace size:** TG-1 — 8 CPU / 64 GiB. Resized up on 2026-08-08 from a smaller tier.
- **Graph in use:** `finance` — 73,646 entities, 61,777 chunks **with 384-dim vectors stored
  inside TigerGraph**, ~222K edges.
- A second graph `sp100_v3` (our deleted system's, ~340K objects) also exists on the
  instance. It is now orphaned — nothing left in this repo uses it. **Dropping it is safe and
  will free memory**, but see the warning below before you do.

### Hard-won operational lessons — read before touching Savanna

1. **Metadata endpoints lie about health.** When the instance is degraded or stopped,
   `echo`, `LS` and `getVertexTypes()` all keep returning success while **every real query
   fails**. A `/echo` healthcheck will show green through a total outage. Always gate on a
   query that returns rows.
2. **It ran out of memory at ~700K objects across two graphs** and never recovered — dropping
   a graph did **not** reclaim memory. `pyTigerGraph` 2.0.4 exposes no restart (only
   `clearGraphStore`, which is destructive, and `rebuildGraph`, which only rebuilds indices).
   The only fix was resizing the workspace, which reprovisions the compute.
3. **After a resize the workspace stays STOPPED** if auto-start is disabled. It returns
   `502 {"message": "Auto start is not enabled for this workspace"}` indefinitely — it is not
   "slow", it is off. Press **Start** in the tgcloud console.
4. **Creating a new graph wiped `finance`'s installed queries.** Every query 404'd until
   reinstalled with `python build/round3_tg_ingest.py --queries-only` (non-destructive).
   **If you drop `sp100_v3`, verify their queries afterwards.**
5. **Dropping a graph can fail on orphaned typed edges.** `Entity_sp100_v3` could not be
   recreated because `audited_by_sp100_v3` etc. still referenced it. Drop edges first,
   vertices second.
6. **Savanna vector search is NON-DETERMINISTIC.** See §8 — this invalidates single-run
   benchmarking.

### Readiness gate (the only trustworthy health check)

```python
conn.runInterpretedQuery(
  'INTERPRET QUERY () FOR GRAPH finance { S = {Chunk_finance.*}; T = SELECT x FROM S:x LIMIT 3; PRINT T.size(); }')
```

Healthy = returns rows. Anything else, including a clean `echo`, means do not run a benchmark.

---

## 4. The dataset

- **400 SEC filings** from **S&P 100 companies** — 10-K annual reports, 8-K current reports,
  DEF 14A proxy statements.
- **Future-dated / synthetic (filing dates in 2025–2026).** This is deliberate and important:
  the LLM cannot answer from memory, so the LLM-only control scores near zero. That is what
  makes the benchmark meaningful.
- Raw filings in `sp100_edgar_downloader/sp100_dataset/`.
- `corpus.jsonl` = 61,777 chunks, section-aware, with **citation-ready chunk IDs** like
  `AAPL/10-K_2025-10-31#item7-c2`. This ID design is genuinely good and worth copying.

### The 50-question benchmark (`eval_set.jsonl`)

| Tier | n | shape | scored how |
|---|---|---|---|
| **A** | 8 | single fact ("Apple's total net sales FY2025?") | exact match |
| **B** | 16 | two-hop bridge ("The company that appointed John Ternus as CEO — its net sales?") | LLM judge |
| **C** | 20 | ticker-set aggregation ("Which companies audited by KPMG?") | set precision/recall/F1 |
| **D** | 6 | cross-doc / temporal ("Put these 2026 buybacks in chronological order") | LLM judge |

`dev_set.jsonl` has 24 questions with **zero id overlap** with eval — both teams tuned their
routers against dev.

### ⚠️ The published answer key has at least 5 errors

Verified against the filings, with quotable evidence:

1. **Citigroup's auditor** is misattributed
2. **Intuit's declared dividend** is wrong
3. **EQ20 has two valid answers** — it asks for "the Deloitte-audited company with FY ending
   June 30, 2025". **Both Microsoft and Procter & Gamble** match, and both are in
   MasterGraph's own `company_facts.json`. Gold says Microsoft only.
4. **EQ44's Deloitte count**: their own dev set says 24, the eval key says 25, the corpus
   says **24**
5. one further mismatch recorded in the original audit

**Always report scores against both keys** — the published one (comparable to their reported
48.7%) and the corpus-verified one. Reporting only the corrected key looks like moving the
goalposts; reporting only the published one means knowingly scoring against errors.

---

## 5. MasterGraph's architecture (the system you now have)

Three pipelines over the same corpus, same generator, same token budget:

| | what it does | tokens/q (their frozen run) |
|---|---|---|
| **P1** LLM-only | no retrieval — proves the model isn't answering from memory | 286 |
| **P2** Basic RAG | Chroma cosine top-10 | **4,760** |
| **P3** GraphRAG | TigerGraph as **both graph AND vector DB** | **2,471** |

**P3 cuts tokens 48% versus their own P2** — that's their headline claim and it is legitimate.
(Caveat worth knowing: a big % reduction partly reflects how wasteful the baseline is. Our
deleted system went 3,466 → 2,274, a 34% cut, but landed at a *lower absolute* number.)

### P3's design — this is the interesting part

An **adaptive router** picks a retrieval strategy per question:

| route | when | how it retrieves |
|---|---|---|
| `aggregation` | "which/list/how many companies…" | **graph-first**: typed-edge GSQL queries (`agg_audited_by`, `agg_uses_supplier`, …), constraints **intersected** |
| `bridge` | "the company that/whose/which…" | vector-search the bridge clause → find the linking company → pull its curated facts |
| `compare` | "which had higher X: A or B" | ticker-scoped retrieval of each company's Item 7/8 financial chunks |
| `temporal` | chronological / "each 8-K" | vector, wider k |
| `lookup` | single fact | vector, tight k=6 |

**Their strongest ideas, all worth reusing:**

1. **Provenance-sentence substitution.** On aggregation, feed **one evidence sentence per
   graph edge** instead of a whole chunk. This is where the token win comes from — 434–682
   tokens per aggregation question versus 1,800–3,600 for a facts-plus-passages approach.
2. **Constraint intersection.** Every applicable constraint is detected (not first-match) and
   the resulting company sets are intersected, normalised to tickers first.
3. **Citation-ready chunk IDs** — `AAPL/10-K_2025-10-31#item8-c11` is directly quotable.
4. **Section-aware chunking** with form-aware headers (`Item N` for 10-K, `Proposal N` for
   DEF 14A).
5. **Their router is genuinely excellent** — measured **20/20 correct routing on Tier C** and
   **8/8 on Tier A**.

Read `mastergraph-round3/ARCHITECTURE.md` for their own account.

---

## 6. How to run it

**Always from `cd mastergraph-round3` (see §2).**

```bash
export PYTHONIOENCODING=utf-8          # required on Windows
export TIGERGRAPH_GRAPH=finance
export P3_REQUIRE_TG=1                 # fail loudly rather than silently skipping the graph
```

### Health check first

```bash
python build/smoke_test_tg.py
```

### Reinstall TigerGraph queries (non-destructive — do this if anything 404s)

```bash
python build/round3_tg_ingest.py --queries-only
```
Expect: 13 queries OK, then `agg_audited_by(KPMG LLP) -> 12 companies` and
`agg_uses_supplier(TSMC) -> 5 companies`.

### Full ingest flags

```bash
python build/round3_tg_ingest.py --fresh          # DESTRUCTIVE: drops and rebuilds the graph
python build/round3_tg_ingest.py --queries-only   # safe: reinstall queries
python build/round3_tg_ingest.py --vectors-only   # reload chunk vectors
python build/round3_tg_ingest.py --edges-only     # reload edges + mentions
```

### Run the benchmark

```bash
# held-out 50
python build/run_round3_benchmark.py --split test --runs 1 --pipelines 1,2,3 --tag myrun
# dev 24
python build/run_round3_benchmark.py --split dev  --runs 1 --pipelines 3
```
Writes `results/round3/bench_<split>_<tag>_run1.json`.

### Score

```bash
python build/evaluate_round3.py       # their own evaluator (needs an OpenAI key for the judge)
```

### Run a single question in Python

```python
import sys, importlib.util
sys.path.insert(0, '.')                      # cwd MUST be mastergraph-round3
spec = importlib.util.spec_from_file_location('p3', 'pipeline3_round3/pipeline.py')
p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)
p3._load(); p3._connect()
m = p3.run_single("Which Utilities-sector companies in the dataset are audited by Deloitte?")
print(m.route, m.answer, m.prompt_tokens, m.completion_tokens)
# -> aggregation  'DUK, NEE, SO'  472  89
```

### Ablation flags on P3

`P3_ROUTER_OFF=1` (force hybrid) · `P3_GRAPH_OFF=1` (vector only) ·
`P3_COMPRESS_OFF=1` (full chunks, no provenance substitution) · `P3_REQUIRE_TG=0` (allow
degraded runs — **never use this for a real measurement**).

---

## 7. THE HEADLINE RESULT — held-out test

The single most important experiment in this project.

**Why:** both teams hand-tuned their routers (hand-built `AUDITORS`, `SECTOR_LEX`, `RISK_LEX`
lexicons; their own comment reads *"Broadened from held-out+dev failures"*). So neither
team's score on the 50 measures generalisation.

**Method:** 5 brand-new questions, written from the schema, never shown to either system.
Gold answers derived mechanically from ground truth that **two independent extractions agree
on** (their `edges_prov.json` and our KG agreed on the auditor for **100/100 companies**).
Sector/auditor combinations deliberately avoided the four the eval 50 already uses.

### Result: **MasterGraph 4/5, ours 3/5**

| Q | shape | gold | theirs | ours |
|---|---|---|---|---|
| HQ1 | sector ∩ auditor | DUK, NEE, SO | ✅ | ✅ |
| HQ2 | ∩ + negation | ExxonMobil | ✅ `XOM` | ❌ **`CVX`** (the excluded company) |
| HQ3 | count over ∩ | 4 | ✅ `4` | ❌ **`32`** |
| HQ4 | 2-hop shared attribute | KPMG LLP | ❌ refused | ✅ `KPMG LLP` |
| HQ5 | sector ∩ auditor | AMZN, GM, MCD, TGT | ✅ | ✅ |

**Tokens: theirs 1,125/q vs ours 1,856/q** (ours at its tuned `P3_ROUTED_PASSAGES=2`; at the
default 10 it was 3,365 — a misconfiguration in the first run of this test).

### Why they won — diagnosed, not guessed

- **HQ3**: our KG had **no `in_sector` relation at all** (we extracted 8 relation types;
  sector wasn't one). We answered **32**, which is exactly the total number of PwC clients —
  the sector half of the constraint silently vanished.
- **HQ2**: our router didn't fire at all and fell back to vector search, returning the one
  company the question excludes.
- **Token gap**: entirely their provenance-sentence compression. On HQ4, where neither
  routes, the two are nearly identical (3,410 vs 3,686).

### Caveat on this test — stated honestly

**3 of the 5 questions leaned on `sector`, which only their schema models.** The test was
designed by me and that bias was mine. It tilted toward them. So 4–3 overstates the
*retrieval* gap — we won the one purely-multi-hop question, which they refused.

### The full scoreboard

| test | winner |
|---|---|
| Eval 50 (both tuned on it) | **Ours** 0.776 vs 0.603 |
| Eval 50, after fixing their 9 bugs | **Tie** 0.776 vs 0.737 (inside noise) |
| **Held-out 5 (neither tuned)** | **Theirs** 4/5 vs 3/5, and 40% fewer tokens |
| Tier B multi-hop (eval 50) | **Ours** 12/16 vs 5/16 |
| Tier C aggregation (after fixes) | **Theirs** 0.825 vs 0.791 |

**Honest verdict: their architecture is better; our retrieval was better; on the one untuned
test their architecture beat our retrieval.**

---

## 8. Measurement gotchas — these invalidate naive benchmarking

### Savanna vector search is non-deterministic

Two runs, **identical code and config**:

| | |
|---|---|
| identical retrieval | 23/50 questions |
| same chunks, **different order** | 21/50 |
| **genuinely different chunk set** | 9/50 |

Because the model is sensitive to passage order, this flips answers. Four identical baseline
runs drifted:

| metric | spread across 4 identical runs |
|---|---|
| Tier A | 1 full question (5/8 ↔ 6/8) |
| judge on Tier B+D | 3 questions (7 ↔ 10 of 22) |
| blended score | **0.060** |

**Any difference below ~0.06 blended, or ~1 Tier-A question, is noise. Use replicates.**

### The LLM judge is not deterministic either

Grading the **same answers** twice flipped one question on 2 of 7 arms (45.5% → 40.9%).

### `gemini-2.5-flash` changed under the benchmark

`maxOutputTokens` is a **single budget covering thinking AND visible output**. Measured on a
real 10-passage context: **thinking took 489 of the 512 budget, leaving 11 tokens for the
answer** — which truncates Tier-C list answers mid-sentence.

Their frozen run predates this: across all 150 saved records
`totalTokenCount − (prompt + candidates)` is exactly **0**. A live call confirms the API's
total *does* include thoughts (283 + 10 + 119 = 412).

**Consequence:** running their unmodified code today costs **2,936 tok/q instead of 2,471** —
a 19% cost rise with zero code change. And `candidatesTokenCount` reports only *visible*
output, so **thinking tokens are billed but invisible in their logs**. Their dashboard would
say "31 tokens" while they paid for 516.

**If you re-run anything, raise `MAX_TOKENS` to ~3072 or you are measuring truncation.**

---

## 9. Windows / encoding gotchas

- **`corpus.jsonl` is cp1252, not UTF-8.** It raises `UnicodeDecodeError` at byte 307 if read
  as UTF-8 — which is what every Linux and macOS host does by default. It was built on
  Windows by their own `build_round3_corpus.py`, which uses a bare `open()` and so inherits
  the host locale. **0 of 32 file operations across their five main modules set `encoding=`.**
  Read it with `encoding="cp1252"`, or detect: try UTF-8, fall back to cp1252.
- **No module calls `sys.stdout.reconfigure`**, so printing a `✓` crashes on a cp1252 console.
  Always `export PYTHONIOENCODING=utf-8`.
- **The six hardcoded `/Users/kajal/...` paths** — see §2.

---

## 10. The 9 bugs found in their pipeline

Full detail with measurements in **`MG_FIXES_FINDINGS.md`**. All were implemented in a copy
(`mastergraph-improved/`) which has since been **deleted** — the findings survive, the code
does not. Summary:

| # | bug | measured effect |
|---|---|---|
| 1 | `Citations:` block written **inside** the answer field, graded against an 11-token gold | BERTScore **0.023 → 0.323** with zero retrieval change (37/50 answers affected) |
| 2 | "say so if unsupported" clause | **13/50 refusals**; 3 of them had the answer in context |
| 3 | bridge clause searched with plain cosine top-6 | bridge route scored **0/3**; 8 of 10 genuine misses were Tier B |
| 4 | clause splitter `[—\-–]` splits on a bare hyphen | truncates at the `-` in `8-K`, `GLP-1`, `T-Mobile` |
| 5a | negation ignored — "Besides Netflix, …" | Netflix stayed in the answer set |
| 5b | competitor relation direction reversed | "AMD names which competitors?" → `{Colgate, NVDA}` instead of `{Intel, Nvidia}` |
| 5c | fiscal-year-end constraint not detected | returned 24 Deloitte companies instead of 2 |
| 5e/f | `$25B` unparsed (only "billion" matched); amount compared by **substring** (`"25" in "125 million"`) | one question matched **96 of 100 companies** |
| 5g | first-word company key keeps punctuation → `"netflix,"` | silently disabled the negation fix |
| 6 | re-ranking applied to every route | Tier C 11→12 but Tier D **4→2** — ranking is order-destroying and item-blind |
| 7 | portability (§2, §9) | pipeline cannot run off one specific Mac |
| 9 | `lookup`/`temporal` routes never consult the graph | EQ17/EQ18/EQ49 all ask "…and who is `<Company>`'s auditor?" — one `audited_by` hop away, never queried |

**Effect of fixing all of them:** 0.603 → **0.737** blended (n=3), Tier B 5/16 → 10.3/16,
BERTScore 0.023 → 0.387. Essentially level with our system.

**Key insight for the talk:** *none of these are architectural.* A prompt clause, a hyphen in
a regex, an `re.I` flag, a substring compare that should have been numeric, and a graph
lookup three routes never made.

---

## 11. What our (deleted) system did — for the record

Named `wixqa-token-efficiency/` (legacy name; it ran SP100 by the end). **Gone.** Its
distinguishing work, none of which exists in the surviving repo:

- **5 arms vs their 3**: LLM-only, Traditional RAG, **grep control**, GraphRAG, **graph-off ablation**
- **A grep control arm** — a deterministic keyword index built specifically to *falsify our
  own graph*. It did: grep scored **37/50 vs our GraphRAG's 32/50**. On the 34 questions our
  tuning never touched they **tied at 24/34**. The graph's real advantage was multi-hop
  (Tier B 12/16), not aggregate score.
- **The answer-key audit** (§4)
- **Deterministic scoring for 28 of 50** questions — no judge, no variance
- **Bias disclosure** — the clean-34 number excluding every tuned question (+5, not +7)
- **Closed-vocabulary canonicalisation** — repaired extraction 57/100 → **100/100 auditors**
  with zero new API calls
- **Context-gated auditor extraction** — "Deloitte" appears 15× in Bank of America's filing,
  all director bios; BAC's auditor is PwC. A naive extractor gets this wrong.
- **Per-relation typed edges** — found real data loss: TigerGraph keys edges by `(from,to)`,
  so `INTC→TSMC` collapsed `uses_foundry` and `names_competitor` into one edge

**Its two known weaknesses**, both confirmed by the held-out test:
1. **no `in_sector` relation** in the KG (router asked for it; graph couldn't answer)
2. **no provenance-sentence compression** — facts *plus* full passages, hence the token gap

---

## 12. Open items

1. **🔴 ROTATE THE CREDENTIALS** — Google API key, TigerGraph host, `TIGERGRAPH_SECRET`.
   Pasted in chat, therefore compromised. Update `mastergraph-round3/.env` after.
2. **Drop the orphaned `sp100_v3` graph** from Savanna to free memory — but reinstall
   `finance`'s queries afterwards and re-verify (§3, lesson 4).
3. **Paste Sebastien's architecture into §14 below.**
4. If our system is ever rebuilt: add `in_sector`, and adopt their provenance-sentence
   compression. Both are ~a day and both were the measured causes of losing the held-out test.

---

## 13. Candidate datasets for the next phase

The SEC/S&P 100 corpus in this repo covers the **fact-dense, entity-heavy, aggregation**
case. The open question is whether GraphRAG's token efficiency survives on **prose** and at
**deeper hop counts**. These are the candidates.

### The four filters — apply in this order

1. **The LLM-only arm must FAIL.** *This is the hard gate.* If a no-retrieval baseline scores
   well, the benchmark measures memorisation, not retrieval. This is exactly why the SEC
   corpus works: it is future-dated (2025–2026 filings), so P1 scores near zero. It is also
   why a previous hackathon's biomedical demo was meaningless — **their LLM-only baseline hit
   90%**. Any Wikipedia-sourced set must clear a contamination probe first.
2. **Genuinely multi-hop / cross-document.** Graph traversal only pays off when the answer
   spans documents.
3. **Textual prose, not fact/table-dense.** The complement of what this repo already has.
4. **Corpus sized for LLM-based KG extraction** — roughly 1–5M tokens is affordable. 5M+
   passages means subsetting, which builds the corpus around the questions and weakens any
   generalisation claim.

**Bonus:** gold document IDs per question let you measure *retrieval* directly instead of
inferring it from answer quality.

### The shortlist

| dataset | how to get it | size / shape | filter 1 (LLM-only fails?) | verdict |
|---|---|---|---|---|
| **PhantomWiki** | `pip install phantom-wiki` | generator; hop depth **1–20**, corpus 25 → 1M docs | ✅ **by construction** | **the contamination-proof option** |
| **MultiHop-RAG** | `yixuantt/MultiHopRAG` | 609 news articles, ~1.6M tokens, **2,556 queries**, each needing 2–4 docs | ⚠️ likely — news, but public | **best real-prose option** |
| **FanOutQA** | `pip install fanoutqa` | **1,034 questions at ~7 hops**, human-written decompositions | ❌ **Wikipedia — probe required** | highest hop count; gate it first |
| **Re-DocRED** | `tonytan48/Re-DocRED` | 4,053 docs, 25 MB, 96 relation types with **gold triples** | n/a — not QA | scores **KG extraction quality** directly (F1 vs gold relations) |

### Notes on each

**PhantomWiki** — the only option that satisfies filter 1 *absolutely*. It's a **generator**,
not a fixed set: run `pw-generate` with your own seed and the corpus has never existed
online, so the LLM-only arm is zero by construction.

> ⚠️ **Do NOT use the published `kilian-group/phantom-wiki-v1` on HuggingFace** — that one is
> online and therefore contaminated. Generate your own.

Uniquely offers a **hop-depth dial (up to 20)** and a **corpus-size dial**, so it can produce
accuracy-vs-hop-count and cost-vs-corpus-size curves no real dataset can. Its Prolog DB gives
exact gold reasoning paths. **Caveat: it's synthetic — report it alongside real numbers,
never instead of them.**

**MultiHop-RAG** — the best *real prose* option, and a natural complement to the SEC corpus
(news articles vs filings). All 2,556 queries need 2–4 documents. It also ships **null
queries**, which are a free hallucination test — an arm that answers them confidently is
fabricating.

**FanOutQA** — the deepest hop count on offer (~7) and human-written decompositions are
genuinely valuable for error analysis. **But it is Wikipedia-derived**, which is precisely the
class filter 1 exists to catch. Run the contamination probe *before* investing: take 30
questions, run the LLM-only arm with no retrieval, and if it scores materially above zero the
dataset is measuring memory. Worth doing — the payoff is a 7-hop benchmark — but do not skip
the probe.

**Re-DocRED** — not a QA set. It scores the **extraction** half of the pipeline directly,
against gold relation triples. That matters here because a chunk of this project's real work
was extraction quality (100/100 auditors after repair, the "Deloitte appears 15× in director
bios" false-positive trap). Re-DocRED would put a number on that instead of inferring it from
downstream answers.

### Ruled out, and why

| dataset | reason |
|---|---|
| **CRAG** | CC BY-NC — **non-commercial licence blocks customer demos** |
| **HotpotQA (BeIR)** | 5.2M passages — extraction cost prohibitive; Wikipedia-memorised |
| **NarrativeQA** | 3.43 GB; famous plots are memorised; questions are within-document |
| **WixQA** | only 26% multi-hop — useful as a stress test, not a showcase |

### The lesson from this project that applies to all of them

**Do not trust a score on a set you tuned against.** Both teams here hand-tuned routers with
lexicons built from the eval questions, and on 5 genuinely held-out questions the ranking
*reversed* (§7). Whatever dataset comes next, **hold out a slice from the very start and
never look at it** — and measure the noise floor with replicates before believing any gap
(§8).

---

## 14. Sebastien's architecture

> **← PASTE SEBASTIEN'S SHARED ARCHITECTURE HERE**
>
> Context for whoever reads it next: Sebastien is the stakeholder this work is being
> presented to. He previously shared a **6-stage retrieval poster**:
>
> `similarity search → expand → filter → re-rank → pack budget → compress`
>
> That 6-stage pipeline **was** implemented against MasterGraph's code (in the now-deleted
> `mastergraph-6stage/`) and measured **neutral-to-negative**: Tier C 11→12 but Tier D 4→2.
> The mechanism is that relevance ranking is **order-destroying and item-blind** — EQ47 needs
> *both* Apple 8-Ks and asks what each reports, so a ranker that promotes the more "relevant"
> one and drops the other cannot be right however good its scores are.
>
> The lesson that survived: **apply re-ranking selectively** (single-fact `lookup` routes
> only), never on temporal or set-valued questions.

---

## 15. What to actually say in the presentation

Not "we scored higher" — it won't survive a technical question. This is stronger and true:

> **Two teams built GraphRAG on the same 400 SEC filings. The gap between them was never the
> graph — it was nine non-architectural bugs: a prompt clause, a hyphen in a regex, a
> case-insensitivity flag, a substring compare that should have been numeric, and a graph
> lookup three routes never made. Fixing those took the trailing submission from 0.603 to
> level. And on five questions neither team had seen, the tuned scores reversed. What
> separates a working GraphRAG POC from a broken one is retrieval discipline and
> measurement — not graph design.**

Supporting points that will land with a technical audience:

- **"Does a graph beat plain search?"** is the first question any buyer asks. We built the
  keyword control arm to try to beat our own graph. On tuned questions it did. On untouched
  questions it tied. The graph's real advantage is multi-hop.
- **Benchmark numbers on a tuned set are not predictive.** We proved it on ourselves.
- **Savanna's vector search is non-deterministic**, and its metadata endpoints report healthy
  through a total outage. Both are live risks for a client POC and worth flagging as product
  feedback.
- **Thinking-model token accounting is broken by default** — `candidatesTokenCount` hides
  reasoning tokens that are billed. A customer's dashboard will understate cost by ~490
  tokens/question.

### Concede these before you're asked

MasterGraph's **router** (20/20 Tier C routing), **schema breadth**, **section-aware
chunking**, **citation-ready chunk IDs** and **provenance-sentence compression** are all
better than ours were. Saying so first is the difference between sounding rigorous and
sounding like a salesperson.
