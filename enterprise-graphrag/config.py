"""Central configuration (Round 3 — SEC filings, GraphRAG hackathon)."""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OPENAI_PRICING = {
    "gpt-4o-mini":     (0.00015,    0.00060),
    "gpt-4o":          (0.0025,     0.01),
    "gpt-4-turbo":     (0.01,       0.03),
    "gpt-4":           (0.03,       0.06),
    "gpt-3.5-turbo":   (0.0005,     0.0015),
}

# --- Gemini (generator for all three pipelines; extraction/judge configured separately) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# gemini-2.5-flash was two generations behind what the key can reach. Measured on the 20
# held-out questions, switching to 3.6-flash moved GraphRAG 0.717 -> 1.000 (20/20, all
# graph-routed) and CUT tokens 3,152 -> 2,008. The gain is in the PLANNER: the six failures
# were bad plans, not bad traversals. Basic RAG also improved (0.109 -> 0.171) and the leak
# gate got safer (0.079 -> 0.062), so the switch helps every arm and weakens no control.
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Gemini pricing per 1K tokens (input, output).
# WARNING: the gemini-3.x entries are ASSUMED EQUAL to their 2.5 counterparts. I do not have
# authoritative public rates for them, and metrics._cost() returns 0.0 for any model missing
# here — so leaving 3.6-flash out would have silently reported every cost as $0.00. Treat the
# cost column as an ESTIMATE until these are checked against current Google pricing.
GEMINI_PRICING = {
    "gemini-2.5-flash":      (0.00030, 0.00250),
    "gemini-2.5-flash-lite": (0.00010, 0.00040),
    "gemini-2.5-pro":        (0.00125, 0.01000),
    "gemini-3.6-flash":      (0.00030, 0.00250),   # assumed = 2.5-flash, UNVERIFIED
    "gemini-3.5-flash":      (0.00030, 0.00250),   # assumed
    "gemini-3-flash-preview": (0.00030, 0.00250),  # assumed
    "gemini-3.1-flash-lite": (0.00010, 0.00040),   # assumed
    "gemini-3.1-pro-preview": (0.00125, 0.01000),  # assumed = 2.5-pro
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

TOP_K = 10

# --- Fair-comparison context budget (Basic RAG vs GraphRAG) ---
# Basic RAG (P2) and GraphRAG (P3) use the SAME number of context items AND the
# SAME per-item character cap, so neither pipeline is fed more text than the other.
CONTEXT_NUM_CHUNKS  = int(os.getenv("CONTEXT_NUM_CHUNKS", "10"))
CONTEXT_CHUNK_CHARS = int(os.getenv("CONTEXT_CHUNK_CHARS", "2000"))

MAX_DOCS_TO_INDEX = int(os.getenv("MAX_DOCS_TO_INDEX", "10000"))
EVAL_SAMPLE_SIZE = 50

DATASET = "round3"
# Max corpus tokens to load (high default = use the full available corpus).
TARGET_CORPUS_TOKENS = int(os.getenv("TARGET_CORPUS_TOKENS", "100000000"))


def ds_paths() -> dict:
    """Standard artifact paths for the round3 dataset."""
    return {
        "chroma": "./chroma_db_round3/main", "collection": "round3_docs",
        "ground_truth": "./results/ground_truth_round3.json",
        "index_dir": "pipeline3_round3/index",
        "graph": os.getenv("TIGERGRAPH_GRAPH", "finance"),
        "benchmark": "./results/benchmark_round3.json",
    }


_ds = ds_paths()
CHROMA_PERSIST_DIR     = _ds["chroma"]
CHROMA_COLLECTION_NAME = _ds["collection"]

HF_TOKEN = os.getenv("HF_TOKEN", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai")

BENCHMARK_RESULTS_PATH = "./results/benchmark_results.json"
GROUND_TRUTH_PATH      = "./results/ground_truth.json"

TEMPERATURE = 0.0

# Shared across ALL THREE pipelines verbatim (fairness rule).
# Answers are financial facts (numbers with units), names, dates, or LISTS of companies.
# Citations must reference evidence actually supplied (the [chunk_id] labels shown in
# context). LLM-only has no context, so its citation block is naturally empty.
SYSTEM_PROMPT = (
    "You answer questions about a technology company's internal records — Slack threads, "
    "Jira and Linear tickets, Confluence pages, GitHub PRs, email threads, meeting notes "
    "and CRM records.\n"
    "Use the provided context passages and knowledge-graph facts. When a "
    "'Graph traversal' line is present it is AUTHORITATIVE — it was computed from the "
    "knowledge graph, so answer from it directly and completely.\n"
    # Measured on the 40-question run: without this, Basic RAG returned an EMPTY string or a
    # refusal on 21 of 40 questions. It was NOT token starvation — thinking used only 8% of an
    # 8,192 budget and finishReason was STOP, so the model was choosing silence. Adding this
    # moved mean set-F1 on those 21 from 0.000 to 0.190, and THREE went straight to 1.00: the
    # answer was in the context and it was declining to commit.
    # A baseline that stays mute understates plain retrieval and makes the comparison look
    # rigged, so this is applied to ALL arms verbatim under the fairness rule.
    "Give your best supported answer rather than declining. NEVER return an empty answer, "
    "and never reply 'None', 'N/A' or that you cannot answer. If the context does not fully "
    "settle it, name the MOST LIKELY item(s) the context supports and output only those. "
    "An empty reply is always worse than a best-effort one.\n"
    "Answer TERSELY — ONLY the requested item(s), no preamble, no explanation:\n"
    "- Set questions ('which channels', 'which people'): output the COMPLETE list, "
    "comma-separated, nothing else. Do not stop early.\n"
    "- Single-fact questions: the fact alone.\n"
    "- Names and identifiers exactly as they appear.\n"
    "Output the answer ONLY - no 'Citations:' line. Evidence is recorded automatically."
)

# ENT_PROMPT=v2 appends a constraint-filtering rule. Measured on FIN46/FIN49: the Graph
# traversal line listed every member of the channel and the model output all 200+, because the
# instruction above says "COMPLETE list, do not stop early" and declares the traversal
# AUTHORITATIVE. Gold was two names. The traversal answers ONE constraint; a two-constraint
# question needs the second applied afterwards, and only the model can do that from context.
# Default stays v1 so nothing already measured shifts underneath.
if os.getenv("ENT_PROMPT", "v1") != "v1":
    SYSTEM_PROMPT += (
        "\nIMPORTANT - multi-constraint questions:\n"
        "- If the question carries MORE THAN ONE constraint (e.g. 'people from the X department "
        "who take part in Y'), the Graph traversal line answers only ONE of them. Output ONLY "
        "the items satisfying EVERY constraint, checking each candidate against the context. "
        "A long traversal list is a list of CANDIDATES, not the answer. Leave out any candidate "
        "you cannot confirm against the other constraint.\n"
        "- A list longer than about ten items is almost always an unfiltered traversal. Filter "
        "it rather than repeating it.\n")

# gemini-2.5-flash draws THINKING and visible output from the same maxOutputTokens budget.
# Measured on the SEC run: ~489 of a 512 budget went to reasoning, leaving ~11 for the
# answer — and here it truncated the router's JSON plan at 7 tokens, which silently made
# every question fall back to vector search. Headroom, not a smaller model.


# Round 3 lists can be long (EQ44/EQ25 ~11-35 companies) — needs output headroom.
# gemini-2.5-flash draws THINKING and visible output from the SAME budget. Measured on the
# 40-question run: Basic RAG spent 2,763 and 2,945 tokens of a 3,072 budget on reasoning
# and emitted 0 and 26 visible tokens — 21 of 40 answers came back empty. It penalises the
# BASELINE hardest, because a 6,000-token retrieved context provokes the longest reasoning,
# while GraphRAG answers from a traversal line in 2-11 completion tokens. Leaving it low
# was therefore a thumb on the scale in our own favour.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
