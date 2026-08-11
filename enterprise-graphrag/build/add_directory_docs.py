"""Put the employee directory into the RETRIEVABLE corpus, so the comparison is fair.

THE PROBLEM THIS FIXES
----------------------
`reports_to` (160 edges) and `in_department` (167) were built from employee_directory.yaml,
which was never chunked into corpus.jsonl. So the org chart existed ONLY inside the graph.

That makes 11 of the 20 eval questions — every two_hop_org and every dept_in_channel —
impossible for Basic RAG by construction. Not hard: impossible. No amount of vector retrieval
can surface a fact that is absent from the index. Measured on the first run, P2 scored 0.000
and answered the literal string "None".

"GraphRAG beats Basic RAG" would then be an artefact of what we chose to index, and the first
question from an audience — "was the org chart in your vector store?" — would end the demo.
After this, both arms can reach exactly the same facts and the only difference left is whether
the system can JOIN them across hops. That is the claim worth making.

WHAT IS AND IS NOT WRITTEN
--------------------------
Written: exactly the fields the YAML holds — name, title, department, manager, email,
start date, bio; plus a per-department roster, which is the same information transposed.

NOT written: anything connecting a person to a channel, a document or a project. That is the
second hop, and it is what the questions ask for. Putting it here would hand the answer to
both arms and measure nothing. The directory states hop 1 only.

    python build/add_directory_docs.py            # then: --push to load into TigerGraph
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portability import init_console, ropen   # noqa: E402

init_console()

CORPUS = Path("datasets/enterprise/corpus.jsonl")
DOCS = Path("datasets/enterprise/documents.jsonl")
EMPLOYEES = Path("datasets/enterprise/employee_directory.yaml")
COMPANY = "Redwood Inference"
PREFIX = "directory/"


def person_doc(p: dict, dept: str, reports: list[str]) -> tuple[str, str]:
    """One page per employee, phrased the way an intranet profile would be.

    `reports` is listed because it is the exact inverse of the `manager` field already in the
    YAML — stating it one way and not the other would leave the fact retrievable only in the
    direction the graph happens to walk, which is the bias this file exists to remove.
    """
    name = p["name"]
    lines = [f"# {name}", "",
             f"**Title:** {p.get('title','')}",
             f"**Department:** {dept}",
             f"**Email:** {p.get('email','')}",
             f"**Start date:** {p.get('start_date','')}"]
    if p.get("manager"):
        lines += ["", f"{name} reports to {p['manager']}. "
                      f"{p['manager']} is {name}'s manager."]
    if reports:
        lines += ["", f"Direct reports of {name}: " + ", ".join(sorted(reports)) + ". "
                      f"The following people report to {name}: " + ", ".join(sorted(reports)) + "."]
    if p.get("bio"):
        lines += ["", "## Bio", p["bio"]]
    return f"{PREFIX}people/{name.replace(' ', '-').lower()}.md", "\n".join(lines)


ROSTER_PER_CHUNK = 12   # keeps every roster page under the 1800-char context cap


def dept_docs(dept: str, members: list[dict]) -> list[tuple[str, str]]:
    """Department rosters, split so no page is silently truncated.

    Engineering has 40 members and rendered to 2,600 characters — over the 1800 cap the prompt
    builder slices at. The tail of the roster would have been paid for at retrieval time and
    then cut before the model ever saw it, quietly removing people from the only page that
    lists them. Split into pages of 12, each self-describing.
    """
    slug = dept.replace(" ", "-").replace("/", "-").lower()
    ordered = sorted(members, key=lambda x: x["name"])
    pages = [ordered[i:i + ROSTER_PER_CHUNK] for i in range(0, len(ordered), ROSTER_PER_CHUNK)] or [[]]
    out = []
    for i, page in enumerate(pages):
        part = f" (part {i+1} of {len(pages)})" if len(pages) > 1 else ""
        lines = [f"# {dept} department{part}", "",
                 f"The {dept} department at {COMPANY} has {len(members)} members.", "",
                 "## Members"]
        for m in page:
            mgr = f" — reports to {m['manager']}" if m.get("manager") else ""
            lines.append(f"- {m['name']}, {m.get('title','')}{mgr}")
        lines += ["", f"All of the above are members of the {dept} department."]
        out.append((f"{PREFIX}departments/{slug}.md", "\n".join(lines)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="also upsert the new chunks into TigerGraph")
    args = ap.parse_args()

    import yaml
    depts = yaml.safe_load(ropen(EMPLOYEES))["departments"]

    reports_of: dict[str, list[str]] = {}
    for members in depts.values():
        for p in members:
            if p.get("manager"):
                reports_of.setdefault(p["manager"], []).append(p["name"])

    # (doc_id, chunk_index, text) — a department roster can span several pages
    new_docs: list[tuple[str, int, str]] = []
    for dept, members in depts.items():
        for p in members:
            did, txt = person_doc(p, dept, reports_of.get(p["name"], []))
            new_docs.append((did, 0, txt))
        for i, (did, txt) in enumerate(dept_docs(dept, members)):
            new_docs.append((did, i, txt))
    print(f"generated {len(new_docs)} directory documents "
          f"({sum(len(m) for m in depts.values())} people + {len(depts)} departments)")

    existing = {json.loads(l)["doc_id"] for l in ropen(CORPUS) if l.strip()}
    if any(d.startswith(PREFIX) for d in existing):
        print("directory documents already present in corpus.jsonl — nothing appended")
    else:
        n_chunks = 0
        with open(CORPUS, "a", encoding="utf-8") as cf, open(DOCS, "a", encoding="utf-8") as df:
            seen_docs: set[str] = set()
            for doc_id, ci, text in new_docs:
                # a multi-page roster is ONE document with several chunks, so documents.jsonl
                # gets one row while corpus.jsonl gets one row per page
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    df.write(json.dumps({"doc_id": doc_id, "source": "directory",
                                         "form": "directory", "company": COMPANY,
                                         "section": "directory", "uuid": doc_id,
                                         "text": text}, ensure_ascii=False) + "\n")
                cf.write(json.dumps({
                    "title": f"{doc_id}#c{ci}", "text": text, "doc_id": doc_id,
                    "ticker": "directory", "source": "directory", "company": COMPANY,
                    "sector": "directory", "form": "directory", "section": "directory",
                    "filing_date": "", "uuid": doc_id}, ensure_ascii=False) + "\n")
                n_chunks += 1
        print(f"appended {n_chunks} chunks to {CORPUS}")

    over = [f"{d}#c{i}" for d, i, t in new_docs if len(t) > 1800]
    print(f"chunks over the 1800-char context cap: {len(over)}"
          + (f"  <-- WOULD BE TRUNCATED: {over[:3]}" if over else "  (none)"))

    if args.push:
        push_to_tg([(f"{d}#c{i}", t) for d, i, t in new_docs])
    else:
        print("\nnot pushed. Re-run with --push to upsert these chunks into TigerGraph.")
    return 0


def push_to_tg(chunks: list[tuple[str, str]]) -> None:
    """Embed and upsert just these chunks — the other 73,425 are untouched.

    Uses the raw /restpp payload form, and embeds text[:1000] / stores text[:1500], because
    that is exactly what build/round3_tg_ingest.py did for every other chunk. A directory page
    embedded over its full length would sit in a slightly different part of the vector space
    from the corpus it competes with, which is a thumb on the scale in the retrieval
    comparison. pyTigerGraph's dict-form upsert rejects the vector outright here
    ('array value is only allowed for attribute value') — the same reason the main loader
    uses REST.
    """
    import os
    import time

    import pyTigerGraph as tg
    import requests
    from embeddings.embedder import get_embedder

    g = os.getenv("TIGERGRAPH_GRAPH", "enterprise")
    host = os.getenv("TIGERGRAPH_HOST", "")
    secret = os.getenv("TIGERGRAPH_SECRET") or None
    conn = tg.TigerGraphConnection(host=host, graphname=g, gsqlSecret=secret,
                                   username=os.getenv("TIGERGRAPH_USERNAME", "tigergraph"),
                                   password=os.getenv("TIGERGRAPH_PASSWORD", ""))
    headers = {"Authorization": f"Bearer {conn.getToken(secret)[0]}"}
    emb = get_embedder()

    total = 0
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        vecs = emb.embed_documents([t[:1000] for _, t in batch])
        verts = {cid: {"doc_id": {"value": cid.split("#")[0]},
                       "ticker": {"value": "directory"}, "company": {"value": COMPANY},
                       "sector": {"value": "directory"}, "form": {"value": "directory"},
                       "section": {"value": "directory"},
                       "text": {"value": txt[:1500]}, "emb": {"value": v}}
                 for (cid, txt), v in zip(batch, vecs)}
        payload = {"vertices": {f"Chunk_{g}": verts}}
        for attempt in range(4):
            try:
                r = requests.post(f"{host}/restpp/graph/{g}", json=payload,
                                  headers=headers, timeout=90)
                if r.status_code == 200 and not r.json().get("error", True):
                    total += len(verts)
                    break
                if attempt == 3:
                    print(f"    batch {i} failed: {str(r.text)[:140]}")
            except Exception as e:
                if attempt == 3:
                    print(f"    batch {i} exc: {str(e)[:100]}")
            time.sleep(2 ** attempt)
    print(f"upserted {total}/{len(chunks)} directory chunks into TigerGraph graph '{g}'")


if __name__ == "__main__":
    raise SystemExit(main())
