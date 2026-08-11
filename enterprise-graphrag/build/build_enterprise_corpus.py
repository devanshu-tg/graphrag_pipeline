"""Turn the 20k EnterpriseRAG-Bench subset into the corpus.jsonl the pipeline expects.

The subset was selected and extracted separately (see _evidence/CONTEXT.md); this only
reshapes it. Each source document becomes ONE chunk record with the same field names
MasterGraph's pipeline reads, so nothing downstream needs to know the domain changed:

    title    citation-ready chunk id, e.g.  slack/eng/1737412205#c0
    text     the document body
    doc_id   the document it came from
    source   slack | jira | confluence | github | gmail | ...   (replaces `ticker`)
    company  always the fictional company                        (replaces `company`)
    form     the source type                                     (replaces `form`)
    section  the sub-directory, e.g. eng, design                  (replaces `section`)

Documents longer than MAX_CHARS are split so no single chunk blows the context budget;
the split is on paragraph boundaries, and every piece keeps the parent doc_id so
provenance survives.

    python build/build_enterprise_corpus.py --input <corpus_20k.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen, wopen   # noqa: E402

init_console()

OUT_DIR = Path("datasets/enterprise")
COMPANY = "Redwood Inference"

# MUST stay at or under config.CONTEXT_CHUNK_CHARS (2000), or the prompt builder silently
# truncates every retrieved passage and the extra text is paid for but never used.
# A first pass at 6000 produced chunks averaging 3,991 chars — 84% of them would have been
# cut in half at prompt-build time, invisibly. SEC's chunks average 1,650 chars (max 1,800),
# which is the granularity this pipeline was tuned for, so match it.
MAX_CHARS = 1800


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) + 2 > limit:
            parts.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        parts.append(buf)
    # a single paragraph over the limit still has to be cut
    out = []
    for p in parts:
        while len(p) > limit:
            out.append(p[:limit])
            p = p[limit:]
        if p:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="corpus_20k.jsonl from the subset build")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_docs = n_chunks = n_chars = 0
    by_source: dict[str, int] = {}

    with wopen(OUT_DIR / "corpus.jsonl") as cf, wopen(OUT_DIR / "documents.jsonl") as df:
        for line in ropen(args.input):
            if not line.strip():
                continue
            d = json.loads(line)
            path, text = d["path"], d["content"]
            source = d["source"]
            section = path.split("/")[1] if path.count("/") >= 2 else source
            doc_id = path
            n_docs += 1
            by_source[source] = by_source.get(source, 0) + 1

            df.write(json.dumps({"doc_id": doc_id, "source": source, "form": source,
                                 "company": COMPANY, "section": section,
                                 "uuid": d.get("doc_id", ""), "text": text},
                                ensure_ascii=False) + "\n")

            for i, piece in enumerate(split_text(text)):
                cf.write(json.dumps({
                    "title": f"{doc_id}#c{i}",      # citation-ready id
                    "text": piece,
                    "doc_id": doc_id,
                    "ticker": source,               # pipeline reads `ticker` — reuse for source
                    "source": source,
                    "company": COMPANY,
                    "sector": source,               # pipeline reads `sector` — reuse for source
                    "form": source,
                    "section": section,
                    "filing_date": "",
                    "uuid": d.get("doc_id", ""),
                }, ensure_ascii=False) + "\n")
                n_chunks += 1
                n_chars += len(piece)

    # GUARD — a chunk larger than the context cap is paid for and never read.
    # The prompt builder slices every passage with text[:CONTEXT_CHUNK_CHARS], silently.
    # A first pass at MAX_CHARS=6000 put 84% of chunks over the cap: retrieval scored them
    # as hits, the model saw the first half, and nothing anywhere reported a problem.
    # Worse, it would have biased the comparison — P3's aggregation route feeds one ~240-char
    # evidence sentence per edge, so GraphRAG is immune exactly where Basic RAG bleeds.
    import config
    if MAX_CHARS > config.CONTEXT_CHUNK_CHARS:
        raise SystemExit(
            f"REFUSING TO BUILD: MAX_CHARS={MAX_CHARS} exceeds "
            f"CONTEXT_CHUNK_CHARS={config.CONTEXT_CHUNK_CHARS}. Every retrieved passage "
            f"would be truncated at prompt-build time, invisibly, and the P2/P3 comparison "
            f"would be biased toward P3."
        )
    over = sum(1 for line in ropen(OUT_DIR / "corpus.jsonl")
               if line.strip() and len(json.loads(line)["text"]) > config.CONTEXT_CHUNK_CHARS)
    if over:
        raise SystemExit(f"REFUSING: {over:,} written chunks exceed "
                         f"CONTEXT_CHUNK_CHARS={config.CONTEXT_CHUNK_CHARS}")

    print(f"documents : {n_docs:,}")
    print(f"chunks    : {n_chunks:,}   ({n_chunks/n_docs:.2f} per document)")
    print(f"content   : {n_chars/1048576:.1f} MB  (~{n_chars/4/1e6:.1f}M tokens)")
    print("\nby source:")
    for s, c in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {s:<16}{c:>8,}")
    print(f"\nwrote {OUT_DIR/'corpus.jsonl'} and {OUT_DIR/'documents.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
