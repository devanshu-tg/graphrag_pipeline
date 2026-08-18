# Design snapshot — 2026-08-18, before the light-theme redesign

Taken before rebuilding the page to match the Round 3 hackathon eval layout.

## What's here

    index.html        the dark dashboard as shipped, 17px base, 3 tabs
    build_static.py   the static builder that patches it
    docs-built/       the exact bytes deployed to graphrag-benchmark.netlify.app
    GIT-COMMIT.txt    the commit these came from

## To revert

    cp _design_snapshot_2026-08-18/index.html      demo/index.html
    cp _design_snapshot_2026-08-18/build_static.py demo/build_static.py
    python demo/build_static.py

Or from git, since everything here is committed:

    git checkout <GIT-COMMIT.txt contents> -- enterprise-graphrag/demo/index.html

Git is the real safety net; this folder is here so a revert does not require
knowing which commit was good.

## What the snapshot is of

Dark theme (--bg #08090d), --fs 17px, three tabs (Textual/Factual + heldout),
tables reading from eval_pub50.jsonl and eval_sec_fair_prompt.jsonl, 100
questions replayable, optional live backend via --live-url.
