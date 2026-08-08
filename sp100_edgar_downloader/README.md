# S&P 100 EDGAR Corpus Downloader

Builds a heterogeneous document corpus for the GraphRAG hackathon: for each of the 100 S&P 100 companies (101 tickers; Alphabet's two share classes are deduplicated to one company), it downloads the latest **10-K** (financial reporting), latest **DEF 14A proxy statement** (governance and executive pay), and the **2 most recent 8-Ks** (event disclosures) from SEC EDGAR, converted to plain text ready for chunking and indexing.

## Usage

```bash
python3 download_sp100_edgar.py --email you@example.com
```

Python 3.8+, no dependencies (stdlib only). The email goes into the User-Agent header, which the SEC requires for programmatic access.

Options:

```bash
--num-8k 4        # more 8-Ks per company (default 2)
--keep-html       # also save original HTML alongside the text
--out my_folder   # output directory (default sp100_dataset/)
```

## Output

```
sp100_dataset/
  manifest.csv          # ticker, form, filing date, source URL, local path per document
  AAPL/
    10-K_2025-xx-xx.txt
    DEF14A_2025-xx-xx.txt
    8-K_2025-xx-xx.txt
    8-K_2025-xx-xx.txt
  ABBV/
  ...
```

Roughly **400 documents**. Expect ~30–45 minutes (the script stays under the SEC's 10 requests/second fair-access limit). Total size as plain text: roughly 300–600 MB. Safe to interrupt and re-run — already-downloaded files are skipped, so re-running also retries any failures.

## Notes

- Company list: S&P 100 constituents as of September 2025 (source: [Wikipedia S&P 100](https://en.wikipedia.org/wiki/S%26P_100)). Edit `sp100_companies.csv` to add/remove companies — any ticker EDGAR knows will work.
- Text conversion is table-preserving but lossy (no images/XBRL). Use `--keep-html` if you need faithful originals.
- Not included, because they aren't on EDGAR and don't have predictable URLs across 100 companies: sustainability reports, codes of conduct, earnings call transcripts. For a subset of anchor companies, add these manually — see the Microsoft example stack in the hackathon notes.
- Manifest `source_url` column gives the authoritative EDGAR link for every document, satisfying the "traceable to source" requirement in the Round 3 problem statement.
