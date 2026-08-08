#!/usr/bin/env python3
"""
S&P 100 EDGAR corpus downloader.

For each company in sp100_companies.csv, downloads from SEC EDGAR:
  - latest 10-K (annual report)
  - latest DEF 14A (proxy statement: governance & executive pay)
  - N latest 8-Ks (event disclosures), default 2

Converts each filing's primary document to plain text and writes:
  sp100_dataset/<TICKER>/<FORM>_<FILING-DATE>.txt
  sp100_dataset/manifest.csv   (one row per document, with source URL)

Usage:
  python3 download_sp100_edgar.py --email you@example.com
  python3 download_sp100_edgar.py --email you@example.com --num-8k 4 --keep-html

Requirements: Python 3.8+, stdlib only. Safe to re-run: existing files are skipped.
SEC fair-access policy: max 10 requests/sec, User-Agent with contact required.
"""

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE_TICKERS = "https://www.sec.gov/files/company_tickers.json"
BASE_SUBS = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
BASE_DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

REQUEST_INTERVAL = 0.15  # seconds -> ~6.6 req/s, under SEC's 10/s limit
_last_request = [0.0]


def fetch(url: str, user_agent: str, retries: int = 3) -> bytes:
    wait = REQUEST_INTERVAL - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=60) as r:
                _last_request[0] = time.time()
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class TextExtractor(HTMLParser):
    """Minimal HTML -> text conversion using only the stdlib."""

    SKIP = {"script", "style", "head", "title"}
    BLOCK = {"p", "div", "tr", "li", "br", "table", "h1", "h2", "h3", "h4", "h5", "h6", "section"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = unicodedata.normalize("NFKC", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r" ?\n ?", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(content: bytes) -> str:
    try:
        html = content.decode("utf-8")
    except UnicodeDecodeError:
        html = content.decode("latin-1")
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def load_ticker_cik_map(user_agent: str) -> dict:
    data = json.loads(fetch(BASE_TICKERS, user_agent))
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def pick_filings(subs: dict, num_8k: int) -> list:
    """Return [(form, filing_date, accession, primary_doc), ...]."""
    recent = subs["filings"]["recent"]
    rows = list(zip(recent["form"], recent["filingDate"],
                    recent["accessionNumber"], recent["primaryDocument"]))
    picked, count_8k = [], 0
    have = set()
    for form, date, acc, doc in rows:  # rows are newest-first
        if not doc:
            continue
        if form == "10-K" and "10-K" not in have:
            picked.append((form, date, acc, doc)); have.add("10-K")
        elif form == "DEF 14A" and "DEF 14A" not in have:
            picked.append((form, date, acc, doc)); have.add("DEF 14A")
        elif form == "8-K" and count_8k < num_8k:
            picked.append((form, date, acc, doc)); count_8k += 1
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--email", required=True, help="Contact email for SEC User-Agent (required by SEC policy)")
    ap.add_argument("--companies", default="sp100_companies.csv")
    ap.add_argument("--out", default="sp100_dataset")
    ap.add_argument("--num-8k", type=int, default=2)
    ap.add_argument("--keep-html", action="store_true", help="Also save the original HTML")
    args = ap.parse_args()

    user_agent = f"SP100-GraphRAG-Hackathon-Dataset {args.email}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.companies, newline="") as f:
        companies = list(csv.DictReader(f))

    print(f"Resolving CIKs for {len(companies)} companies...")
    cik_map = load_ticker_cik_map(user_agent)

    manifest_path = out / "manifest.csv"
    manifest_rows, errors = [], []

    for i, comp in enumerate(companies, 1):
        ticker = comp["ticker"].upper()
        override = (comp.get("cik") or "").strip()  # optional CSV column; forces a CIK
        if override:                                 # use when a ticker resolves to the wrong entity
            cik = int(override)                      # (e.g. XOM -> empty holding co in SEC's ticker file)
        else:
            sec_ticker = ticker.replace(".", "-")  # SEC uses BRK-B
            cik = cik_map.get(sec_ticker) or cik_map.get(ticker)
        if cik is None:
            errors.append((ticker, "CIK not found"))
            print(f"[{i:3}/{len(companies)}] {ticker}: CIK not found — skipping")
            continue

        cdir = out / ticker
        cdir.mkdir(exist_ok=True)
        print(f"[{i:3}/{len(companies)}] {ticker} (CIK {cik})")

        try:
            subs = json.loads(fetch(BASE_SUBS.format(cik=cik), user_agent))
            filings = pick_filings(subs, args.num_8k)
        except Exception as e:
            errors.append((ticker, f"submissions fetch failed: {e}"))
            continue

        used_names = set()
        for form, date, acc, doc in filings:
            safe_form = form.replace(" ", "").replace("/", "-")
            stem = f"{safe_form}_{date}"
            if stem in used_names:  # same form filed twice on one day (e.g. two 8-Ks) -> disambiguate
                n = 2
                while f"{stem}_{n}" in used_names:
                    n += 1
                stem = f"{stem}_{n}"
            used_names.add(stem)
            txt_path = cdir / f"{stem}.txt"
            url = BASE_DOC.format(cik=cik, acc=acc.replace("-", ""), doc=doc)
            if txt_path.exists():
                manifest_rows.append([ticker, comp["name"], comp.get("sector", ""),
                                      form, date, acc, url, txt_path.as_posix(), "cached"])
                continue
            try:
                content = fetch(url, user_agent)
                if args.keep_html:
                    (cdir / f"{stem}.html").write_bytes(content)
                txt_path.write_text(html_to_text(content), encoding="utf-8")
                manifest_rows.append([ticker, comp["name"], comp.get("sector", ""),
                                      form, date, acc, url, txt_path.as_posix(), "ok"])
                print(f"      {form} {date}  ({txt_path.stat().st_size // 1024} KB text)")
            except Exception as e:
                errors.append((ticker, f"{form} {date}: {e}"))
                print(f"      {form} {date}: FAILED ({e})")

    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "name", "sector", "form", "filing_date",
                    "accession", "source_url", "local_path", "status"])
        w.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} documents in {out}/ — manifest: {manifest_path}")
    if errors:
        print(f"{len(errors)} errors:")
        for t, msg in errors:
            print(f"  {t}: {msg}")
        print("Re-run the same command to retry failures (existing files are skipped).")


if __name__ == "__main__":
    main()
