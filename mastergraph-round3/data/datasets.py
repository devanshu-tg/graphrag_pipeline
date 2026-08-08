"""Corpus + QA loader for the Round 3 SEC-filings dataset.

Returns (documents, qa_pairs) where:
  documents = list[langchain Document]  (page_content = passage, metadata title = doc_id)
  qa_pairs  = list[{"question","answer"}]
"""
from __future__ import annotations
import warnings
from pathlib import Path

from langchain_core.documents import Document

warnings.filterwarnings("ignore")


def _est_tokens(s: str) -> int:
    return len(s) // 4


def load_round3(target_tokens: int) -> tuple[list[Document], list[dict]]:
    """Round 3 SEC-filings corpus. corpus.jsonl is ALREADY chunked (section-aware,
    citation-ready IDs) with rich metadata, so we preserve each chunk as its own
    Document with full metadata and DO NOT re-split later (chroma build honours
    ROUND3_PRECHUNKED). Eval questions come from eval_set.jsonl (held-out) —
    dev_set.jsonl is loaded separately by the benchmark when tuning."""
    import json
    base = Path("datasets") / "round3"
    docs, toks = [], 0
    for line in open(base / "corpus.jsonl"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        text = r.get("text", "")
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata={
            "title": r["title"], "doc_id": r.get("doc_id", ""),
            "ticker": r.get("ticker", ""), "company": r.get("company", ""),
            "sector": r.get("sector", ""), "form": r.get("form", ""),
            "filing_date": r.get("filing_date", ""), "section": r.get("section", ""),
            "source": "round3",
        }))
        toks += _est_tokens(text)
        if target_tokens and toks >= target_tokens:
            break

    qa_pairs = []
    epath = base / "eval_set.jsonl"
    if epath.exists():
        for line in open(epath):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qa_pairs.append({
                "question": r["question"], "answer": r["answer"],
                "id": r.get("id", ""), "tier": r.get("tier", ""),
                "hop": r.get("hop", ""), "source_docs": r.get("source_docs", ""),
            })
    print(f"[round3] corpus={len(docs):,} chunks (~{toks:,} tokens), qa={len(qa_pairs):,}")
    return docs, qa_pairs


def load_corpus_qa(dataset: str, target_tokens: int) -> tuple[list[Document], list[dict]]:
    if dataset == "round3":
        return load_round3(target_tokens)
    raise ValueError(f"Unknown dataset {dataset!r} — this Round 3 submission only supports 'round3'")
