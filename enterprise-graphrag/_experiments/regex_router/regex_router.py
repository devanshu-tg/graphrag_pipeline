"""EXPERIMENT ONLY — a zero-token regex router for the enterprise graph.

Not imported by any production module. It is injected at runtime by the harnesses in this
folder, which monkeypatch `schema_router.plan`. Nothing under enterprise-graphrag/ is edited.

WHY THIS EXISTS
---------------
`schema_router.plan()` spends ~1,185 tokens per question — 43% of GraphRAG's measured 2,733 —
and the question itself is under 3% of that prompt. The plan it returns is a function of the
question's PHRASING, not of the entity named in it, so for a known phrasing the same plan can
be produced by a pattern for free.

FOUR ROUTES, mirroring the four the SEC tab reports (aggregation / bridge / lookup / default),
re-framed for what this dataset actually asks:

    lookup     one fact about one named thing, one hop      author, channel_of_doc, manager_of
    directory  org listing, one hop reversed               direct_reports, dept_members
    intersect  two seeds combined, then an optional tail   intersection, dept_in_channel
    chain      one seed, multi-hop with a tail             two_hop_org
    (default)  no pattern matched -> the LLM planner runs, exactly as today

WHAT IS DELIBERATELY ABSENT
---------------------------
Entity names. MasterGraph's SEC router carries 9 lexicons and 51 hardcoded strings ("tsmc",
"deloitte", "energy") because its questions ask "which companies use TSMC" — there the entity
IS the filter and has to be enumerated. Here the entity is a PARAMETER: every pattern below
captures it as a group. So the patterns reference relation names from schema_v2.json and
English verbs, and nothing that would die on a different corpus.

DIRECTIONS come from the schema's own example edges, not from guessing:
    authored         person -> doc        so doc -> person is "in"
    participated_in  person -> doc        person -> docs is "out", doc -> people is "in"
    in_channel       doc    -> channel    doc -> channel is "out", channel -> docs is "in"
    reports_to       person -> manager    manager-of is "out", direct-reports is "in"
    in_department    person -> dept       people-in-dept is "in"

Emits the SAME object `schema_router.plan()` returns, so pipeline._walk consumes it unchanged:
    {"legs": [{"seed": str, "path": [(rel, dir), ...]}], "combine": ..., "tail": [...],
     "intent": "set"|"lookup"}
Seed strings are passed through as the question spells them; pipeline._resolve_seed already
does the vertex matching, so no name normalisation belongs here.
"""
from __future__ import annotations

import re

# Relation names, checked against datasets/enterprise/schema_v2.json at import time by the
# harness. A typo here must fail loudly rather than silently route to nothing.
AUTHORED = "authored"
PARTICIPATED = "participated_in"
IN_CHANNEL = "in_channel"
REPORTS_TO = "reports_to"
IN_DEPARTMENT = "in_department"

RELATIONS = {AUTHORED, PARTICIPATED, IN_CHANNEL, REPORTS_TO, IN_DEPARTMENT}

# --------------------------------------------------------------------------------------------
# Patterns. ORDER MATTERS and it is the failure that bit MasterGraph: its route() tests a broad
# aggregation regex BEFORE bridge and misroutes 13 of 16 bridge questions, which
# demo/server_factual.py has to patch at runtime. So the specific patterns go first and every
# pattern is anchored with ^...$ — an unanchored "reports to" would also match "which channels
# do the people reporting to X take part in", stealing a chain question for the directory route.
# --------------------------------------------------------------------------------------------

# quoted document / message title
Q = r"['‘’\"]([^'‘’\"]+)['‘’\"]"
# a person or department name: anything up to the phrase that follows it
NAME = r"(.+?)"

PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # ---- lookup: one hop from one named thing --------------------------------------------
    ("lookup", "author", re.compile(
        rf"^who\s+(?:authored|wrote|created)\s+(?:the\s+)?(?:document|message|doc)\s*{Q}\s*\?$",
        re.I)),
    ("lookup", "channel_of_doc", re.compile(
        rf"^which\s+slack\s+channel\s+is\s+(?:the\s+)?(?:message|document|doc)\s*{Q}\s+in\s*\?$",
        re.I)),
    ("lookup", "manager_of", re.compile(
        rf"^who\s+is\s+{NAME}[''’]s\s+manager\s*\?$", re.I)),

    # ---- directory: one hop, reversed ------------------------------------------------------
    ("directory", "direct_reports", re.compile(
        rf"^who\s+reports\s+(?:directly\s+)?to\s+{NAME}\s*\?$", re.I)),
    ("directory", "dept_members", re.compile(
        rf"^which\s+people\s+are\s+in\s+the\s+{NAME}\s+department\s*\?$", re.I)),

    # ---- intersect: two seeds, combined ----------------------------------------------------
    # Must precede `chain`: both start "which slack channels", and this one is the narrower.
    ("intersect", "intersection", re.compile(
        rf"^which\s+slack\s+channels\s+contain\s+documents\s+involving\s+both\s+{NAME}"
        rf"\s+and\s+{NAME}\s*\?$", re.I)),
    ("intersect", "dept_in_channel", re.compile(
        rf"^which\s+people\s+from\s+the\s+{NAME}\s+department\s+take\s+part\s+in\s+the\s+"
        rf"#?([\w-]+)\s+slack\s+channel\s*\?$", re.I)),

    # ---- chain: one seed, multi-hop --------------------------------------------------------
    ("chain", "two_hop_org", re.compile(
        rf"^which\s+slack\s+channels\s+do\s+the\s+people\s+reporting\s+to\s+{NAME}"
        rf"\s+take\s+part\s+in\s*\?$", re.I)),
]


def _plan(legs, combine="union", tail=(), intent="lookup") -> dict:
    return {"legs": legs, "combine": combine, "tail": list(tail), "intent": intent}


def build(shape: str, m: re.Match) -> dict:
    """Turn a matched pattern into the plan object pipeline._walk expects."""
    g = m.groups()

    if shape == "author":
        # doc -> the person who authored it, so `authored` is walked backwards
        return _plan([{"seed": g[0], "path": [(AUTHORED, "in")]}], intent="lookup")

    if shape == "channel_of_doc":
        return _plan([{"seed": g[0], "path": [(IN_CHANNEL, "out")]}], intent="lookup")

    if shape == "manager_of":
        return _plan([{"seed": g[0], "path": [(REPORTS_TO, "out")]}], intent="lookup")

    if shape == "direct_reports":
        return _plan([{"seed": g[0], "path": [(REPORTS_TO, "in")]}], intent="set")

    if shape == "dept_members":
        return _plan([{"seed": g[0], "path": [(IN_DEPARTMENT, "in")]}], intent="set")

    if shape == "intersection":
        # The intersection is on shared DOCUMENTS, then the tail walks to channels. Doing it the
        # other way round — intersecting channel sets — also admits a channel where each person
        # has their own separate document, which pipeline._walk's docstring records as a
        # measured wrong answer on MH01.
        return _plan(
            [{"seed": g[0], "path": [(PARTICIPATED, "out")]},
             {"seed": g[1], "path": [(PARTICIPATED, "out")]}],
            combine="intersect", tail=[(IN_CHANNEL, "out")], intent="set")

    if shape == "dept_in_channel":
        # Each seed reaches PEOPLE by its own route: the department directly, the channel via
        # its documents. This is the shape a single shared path cannot express.
        return _plan(
            [{"seed": g[0], "path": [(IN_DEPARTMENT, "in")]},
             {"seed": g[1], "path": [(IN_CHANNEL, "in"), (PARTICIPATED, "in")]}],
            combine="intersect", intent="set")

    if shape == "two_hop_org":
        # reports of M -> their documents -> those documents' channels
        return _plan(
            [{"seed": g[0], "path": [(REPORTS_TO, "in")]}],
            tail=[(PARTICIPATED, "out"), (IN_CHANNEL, "out")], intent="set")

    raise KeyError(shape)


def route(question: str) -> tuple[str, str, dict] | None:
    """(route, shape, plan) for a recognised phrasing, else None -> caller uses the planner."""
    q = " ".join(question.strip().split())
    for route_name, shape, rx in PATTERNS:
        m = rx.match(q)
        if m:
            return route_name, shape, build(shape, m)
    return None
