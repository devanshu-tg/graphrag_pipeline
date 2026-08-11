"""Portability shim — the fixes proven on the SEC run, applied once here.

Upstream (mastergraph-round3) has host dependencies that stop it running anywhere but the
machine it was written on. None changes a retrieval result, so all are fixed before
anything else:

1. HARDCODED .env PATH — six modules load the .env from an absolute path under
   /Users/<author>/..., which is a silent no-op elsewhere, leaving every credential as "".
   The failure surfaces much later as `Invalid URL scheme. Supported schemes are http and
   https.` — which is what a blank host looks like.

2. NO EXPLICIT ENCODING — 0 of 32 file operations across their five main modules set
   `encoding=`, so writer and reader both inherit the host locale. Demonstrated, not
   theorised: MasterGraph's own corpus.jsonl, built on Windows by their own builder, is
   cp1252 and raises UnicodeDecodeError at byte 307 when read as UTF-8 — which is what every
   Linux and macOS host does by default.

3. CONSOLE — nothing calls sys.stdout.reconfigure, so printing a checkmark crashes a cp1252
   console. Fixed in-process rather than requiring PYTHONIOENCODING to be set outside.
"""
from __future__ import annotations

import codecs
import os
import sys
from pathlib import Path

LEGACY_ENCODING = "cp1252"
_ENC_CACHE: dict[str, str] = {}


def load_env() -> str:
    """Load credentials from MG_ENV_FILE, else the nearest .env searching upward.

    Values already in the process environment win (override=False), which is how benchmark
    runs supply them — so this never changes what a run sees, it only stops the no-.env case
    from silently resolving every key to "".
    """
    from dotenv import load_dotenv

    explicit = os.getenv("MG_ENV_FILE", "")
    if explicit and Path(explicit).is_file():
        load_dotenv(explicit, override=False)
        return explicit
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            env = d / ".env"
            if env.is_file():
                load_dotenv(env, override=False)
                return str(env)
    load_dotenv(override=False)
    return ""


def detect_encoding(path) -> str:
    """UTF-8 if the WHOLE file decodes as UTF-8, else cp1252 (which cannot fail).

    Probes the entire file with an incremental decoder — streaming, so it never holds the
    decoded text — because a file can be valid UTF-8 for megabytes and then not be.
    """
    key = str(path)
    if key in _ENC_CACHE:
        return _ENC_CACHE[key]
    dec = codecs.getincrementaldecoder("utf-8")()
    enc = "utf-8"
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                dec.decode(chunk)
            dec.decode(b"", final=True)
    except UnicodeDecodeError:
        enc = LEGACY_ENCODING
    _ENC_CACHE[key] = enc
    return enc


def ropen(path, **kw):
    """Read a text file with a deterministic encoding, never the host locale."""
    kw.pop("encoding", None)
    kw.pop("mode", None)
    return open(path, "r", encoding=detect_encoding(path), **kw)


def wopen(path, mode: str = "w", **kw):
    """Write UTF-8, always. New artifacts stay portable."""
    kw.pop("encoding", None)
    return open(path, mode, encoding="utf-8", **kw)


def init_console() -> None:
    """Make stdout/stderr tolerate non-ASCII on a legacy Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
