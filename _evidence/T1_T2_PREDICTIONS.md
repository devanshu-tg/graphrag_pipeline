# Predictions — frozen before T1 and T2 were run

Written 2026-08-11, **before** either test executed. Recorded so no result below can be
retro-fitted to whatever happened. I expect some of these to be wrong; that is the point.

Context: the enterprise benchmark reports GraphRAG at 99.5–100% on 50 questions. An audit found
that (a) the gold answer IS the traversal output, (b) 50/50 questions contain a word matching
the relation they need, (c) the 50 questions are only 8 templates, and (d) zero content
questions have ever been tested despite 500 sitting unused in the repo.

---

## T1 — the 500 official content questions

Source: `datasets/enterprise/official_questions.jsonl`, the dataset's own published gold
standard. 500 questions, `gold_answer` on 500/500, 470/470 `expected_doc_ids` resolvable in our
corpus via the `uuid` field. Never run before.

**Predicted:**

1. **P2-RR and P3 land within noise of each other** on the `basic` (175) + `semantic` (125)
   types. The graph has no relation to follow for "what are the default size limits for file
   uploads", so P3 should fall back to vector search and become Basic RAG with an extra
   planner call.
2. **P3 may score LOWER than P2-RR.** Two mechanisms: the planner burns a call, and a
   mis-routed content question walks a graph path that returns a useless provenance sentence
   instead of the chunk that holds the answer.
3. **P1 well above its enterprise 0.8%** — these questions have generic technical answers
   (limits, metric names, acceptance criteria) that a model can sometimes guess — but still
   worst of the three.
4. **Retrieval hit@10 against `expected_doc_ids`: 0.50–0.75.**
5. **`info_not_found` (20 questions): all three arms hallucinate on most of them.** Our
   `config.SYSTEM_PROMPT` forbids saying "I cannot answer", so the correct answer is
   prohibited by construction. Predicted honest-abstention rate under 25%.

**What would falsify the audit's claim I5:** P3 beating P2-RR by more than ~10 points on
`basic` + `semantic`. That would mean the graph helps on content questions too, and the
"benchmark only tests what the graph does" criticism is weaker than stated.

---

## T2 — adversarial paraphrase of the same 50 questions

Only the question wording changes. `gold_members`, `answer`, `tier`, `shape` and
`question_id` are copied verbatim from `eval_mixed50.jsonl`, so the answer key is identical and
the two runs are directly comparable.

**Predicted:**

1. **P3 falls from 100% to 60–85%.** If the planner was relying on the question naming its own
   relation, removing that vocabulary should cost several shapes outright.
2. **`dept_in_channel` and `two_hop_org` degrade most** — three relations each, and the
   paraphrase removes every one of their trigger words.
3. **P2-RR barely moves (within ±8 points).** It is the control: vector retrieval never
   depended on relation vocabulary. **If P2-RR moves a lot, the paraphrases changed difficulty
   and T2 is void** — that check comes before reading P3's number.

**What would falsify the audit's claim I3:** P3 holding at or above 95%. That would mean the
router genuinely generalises over phrasings and is schema-driven rather than
vocabulary-matched — and I will say so plainly rather than defending the criticism.

---

## Ground rules

- Paraphrases are written in ONE pass and frozen before any run. If P3 scores badly they are
  **not** revised.
- Gold answers are never edited.
- `mastergraph-round3/` stays read-only; 428/428 hashes verified before starting.
- Existing artifacts (`eval_ent50.jsonl`, `eval_v2local.jsonl`, `eval_v2graph.jsonl`,
  `eval_sec_live.jsonl`) are not overwritten.
