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
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Gemini pricing per 1K tokens (input, output) — 2.5 Flash public rates.
GEMINI_PRICING = {
    "gemini-2.5-flash":      (0.00030, 0.00250),
    "gemini-2.5-flash-lite": (0.00010, 0.00040),
    "gemini-2.5-pro":        (0.00125, 0.01000),
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
    "You answer questions about SEC filings (10-K annual reports, 8-K current reports, "
    "DEF 14A proxy statements) for large public companies. Use ONLY the provided context "
    "passages and knowledge-graph facts; if the answer is not supported by them, say so. "
    "Answer TERSELY — give ONLY the requested fact(s), no restated question, no preamble, "
    "no explanation:\n"
    "- Financial figures: the exact number with unit (e.g. '$416,161 million'). No extra words.\n"
    "- List/aggregation questions: output the COMPLETE set of matching companies as a "
    "comma-separated list of tickers (e.g. 'AMD, AVGO, INTC, NVDA, QCOM'); do not stop early, "
    "do not add prose.\n"
    "- 'Which had higher X: A or B' questions: name the winner and give both figures.\n"
    "- Dates and names: give them exactly, nothing more.\n"
    "After the answer, on a new line write 'Citations:' followed by the passage IDs (the "
    "[bracketed labels]) you actually used, comma-separated. Cite only passages provided."
)

# Round 3 lists can be long (EQ44/EQ25 ~11-35 companies) — needs output headroom.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
