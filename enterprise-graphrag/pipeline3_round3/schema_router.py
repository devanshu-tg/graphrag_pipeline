"""Schema-driven routing — replaces every hand-written lexicon and question pattern.

WHAT THIS REPLACES
------------------
MasterGraph's router is ~400 lines of regex and lexicons in pipeline3_round3/pipeline.py:

    AUDITORS     = {"kpmg": "kpmg llp", "ernst": "ernst & young llp", ...}
    SUPPLIER_LEX = {"tsmc": "tsmc", "taiwan semiconductor": "tsmc", ...}
    AGG_RE       = re.compile(r"...|use tsmc|name nvidia|...")

Those literally encode the entities in the eval questions. They route 20/20 Tier-C questions
correctly on the set they were written against — and they are the mechanism by which a
benchmark score stops predicting anything. On 5 held-out questions the ranking reversed.

They are also the single biggest porting cost: worthless the moment the domain changes.

HOW THIS WORKS INSTEAD
----------------------
    1. The GRAPH publishes its own relations (from schema.json, itself derived from data
       coverage — never from questions).
    2. ONE LLM call maps the question onto those relations. Structured output.
    3. Execute the constraints against the graph.
    4. UNION with vector search — always, never replacement, so the graph can only add.
    5. Nothing resolvable -> pure vector search, i.e. it degrades to Basic RAG.

Nothing here names an entity, a company, a person or a channel. Point it at a different
graph and it adapts with no code change — which is the property being tested.

COST: one extra call per question, ~200 tokens. Reported separately so the token comparison
stays honest.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

SCHEMA_PATH = Path(os.getenv("TG_SCHEMA", "datasets/enterprise/schema.json"))

# Attempts at a valid plan before giving up and letting P3 degrade to vector search. Default
# stays 2 so the published runs reproduce exactly; raise it for live demos, where one dropped
# plan is a question that visibly fails in front of an audience. The call is non-deterministic
# even at temperature 0 (see plan() below), so a retry is a genuinely fresh draw rather than a
# repeat of the same failure.
PLAN_RETRIES = max(1, int(os.getenv("P3_PLAN_RETRIES", "2")))

_schema_cache: dict | None = None
_plan_tokens = {"prompt": 0, "output": 0, "calls": 0}


def graph_relations() -> list[dict]:
    """The relations the graph actually holds, with real example values.

    Examples come from the schema derivation (which read the corpus), NOT from questions.
    They matter: without them the model has to guess what a `has_status` target looks like,
    and it will guess wrong.
    """
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    out = []
    for r in _schema_cache["relations"] + _schema_cache.get("long_tail", []):
        ex = r.get("examples") or []
        out.append({
            "relation": r["relation"],
            "curated": r in _schema_cache["relations"],
            "coverage": r.get("coverage", 0),
            "example_targets": [e.get("target", "") for e in ex if e.get("target")][:2],
            "example_sources": [e.get("source", "") for e in ex if e.get("source")][:1],
        })
    return out


PLAN_PROMPT = (
    "You translate a question into a PLAN over a knowledge graph.\n\n"
    "Relations available, with an example edge showing what each end looks like:\n"
    "{relations}\n\n"
    "Return STRICT JSON:\n"
    '  {{"legs": [{{"seed": "<entity named in the question>",\n'
    '               "path": [{{"relation": "<name from the list>", "direction": "out"|"in"}}]}}],\n'
    '   "combine": "intersect"|"union",\n'
    '   "tail": [{{"relation": "...", "direction": "out"|"in"}}],\n'
    '   "intent": "set"|"lookup"}}\n\n'
    "Execution model — plan against it exactly:\n"
    "  1. Each leg starts at its OWN seed and follows its OWN path. Different seeds often "
    "need completely different routes to arrive at the same kind of thing.\n"
    "  2. The legs' results are combined ('intersect' = reached by EVERY leg, 'union' = by "
    "any).\n"
    "  3. The tail is then walked from the combined set. Put a hop in the TAIL when it must "
    "happen AFTER combining, and in a LEG when it must happen BEFORE.\n\n"
    "'out' follows a relation from source to target; 'in' follows it backwards. Use 'in' when "
    "you need the entities pointing AT the seed (a manager's reports are found by following "
    "reports_to backwards).\n\n"
    "Worked examples (a DIFFERENT graph, shown only to explain the format):\n"
    '  "which cities do the employees of ACME live in?"\n'
    '     legs [{{"ACME", [works_at:in]}}], combine union, tail [lives_in:out]\n'
    '  "which cities are home to people working for both ACME and Globex?"\n'
    '     legs [{{"ACME", [works_at:in]}}, {{"Globex", [works_at:in]}}], combine intersect,\n'
    '     tail [lives_in:out]\n'
    '     (the shared PEOPLE are found first, then their cities -> lives_in is the tail)\n'
    '  "which staff in the Legal team attended the Q3 offsite?"\n'
    '     legs [{{"Legal team", [in_team:in]}}, {{"Q3 offsite", [attended:in]}}],\n'
    '     combine intersect, tail []\n'
    '     (each seed reaches PEOPLE by a different route, so each leg has its own path)\n\n'
    "Choosing BETWEEN relations that both look plausible:\n"
    "- Match the question's own verb. 'involving', 'mentions', 'takes part in', 'appears in' "
    "are BROAD participation. 'wrote', 'authored', 'created' are narrow authorship. They are "
    "not interchangeable: authorship is a special case of participation.\n"
    "- Each relation shows the share of documents it covers. When two relations could both "
    "answer the question, prefer the one with HIGHER coverage unless the question explicitly "
    "asks for the narrower one.\n"
    "- This matters most with 'intersect'. A narrow relation intersected across two different "
    "people is usually EMPTY — two people rarely both author the same document, though they "
    "routinely both take part in one. An empty result answers nothing, so when the question "
    "asks what two people have in common, choose the relation they can genuinely share.\n\n"
    "Rules:\n"
    "- Use ONLY relation names from the list above. Never invent one.\n"
    "- Seeds are entities NAMED IN THE QUESTION, spelled as the question spells them.\n"
    "- Every leg must end on the SAME kind of thing, so the legs can be combined.\n"
    "- After the tail, the result must be what the question asks for.\n"
    "- Use 'intersect' when the question says both/and/shared; 'union' for a single seed.\n"
    "- 1 to 3 hops per leg. If nothing can answer it, return an empty legs list.\n\n"
    "Question: {question}\n\nJSON:"
)


# ---------------------------------------------------------------------------------------------
# EXPERIMENTAL, default OFF (P3_PLAN_SLIM=1). The planner prompt is ~1,187 tokens on EVERY
# question and the question itself is under 3% of it - 774 tokens of instructions and worked
# examples plus 382 of relation listing. That fixed cost is 35% of GraphRAG's spend at a
# 3-chunk budget. This is a condensed prompt: one worked example instead of three, the rules
# compressed, and the relation listing without example edges (which carry long document paths).
# It is only worth adopting if accuracy holds - routing is what earns 48/50, so a cheaper prompt
# that routes worse is a bad trade.
PLAN_SLIM = os.getenv("P3_PLAN_SLIM", "0") != "0"

PLAN_PROMPT_SLIM = (
    "Translate the question into a PLAN over a knowledge graph.\n\n"
    "Relations (with the share of documents each covers):\n{relations}\n\n"
    "Return STRICT JSON:\n"
    '  {{"legs":[{{"seed":"<entity named in the question>",'
    '"path":[{{"relation":"<from the list>","direction":"out"|"in"}}]}}],\n'
    '   "combine":"intersect"|"union","tail":[...],"intent":"set"|"lookup"}}\n\n'
    "Model: each leg starts at its OWN seed and follows its OWN path; the legs' results are "
    "combined ('intersect' = reached by every leg, 'union' = by any); the tail is walked from "
    "the combined set. Put a hop in the TAIL if it happens AFTER combining, in a LEG if BEFORE. "
    "'out' follows source->target, 'in' follows it backwards.\n\n"
    'Example (different graph): "which cities are home to people working for both ACME and '
    'Globex?" -> legs [{{"ACME",[works_at:in]}},{{"Globex",[works_at:in]}}], combine intersect, '
    "tail [lives_in:out]. The shared PEOPLE are found first, then their cities.\n\n"
    "Rules:\n"
    "- Use ONLY relation names from the list. Never invent one.\n"
    "- Seeds are entities NAMED IN THE QUESTION, spelled as the question spells them.\n"
    "- Every leg must end on the SAME kind of thing.\n"
    "- Match the question's verb: 'involving'/'mentions'/'takes part in' are BROAD "
    "participation; 'wrote'/'authored' is narrow authorship. Prefer HIGHER coverage unless the "
    "question asks for the narrow one - a narrow relation intersected across two people is "
    "usually EMPTY.\n"
    "- 'intersect' when the question says both/and/shared; 'union' for a single seed.\n"
    "- 1 to 3 hops per leg. If nothing can answer it, return an empty legs list.\n\n"
    "Question: {question}\n\nJSON:"
)


def _short(v: str, n: int = 46) -> str:
    """Document ids are full paths; showing one whole makes the relation list unreadable."""
    v = str(v)
    return v if len(v) <= n else v[: n - 3] + "..."


# ---------------------------------------------------------------- pattern fast path
# A pattern that recognises the question answers the routing question for FREE: the plan is a
# function of the PHRASING, not of the entity named in it, so ~1,373 tokens of planner prompt
# buy nothing on a phrasing already known. Measured on the 50 enterprise questions: 2,700 ->
# 1,327 tokens at a 3-chunk budget, 49/50 and 99.0% judged either way, latency 9.5s -> 4.3s.
#
# This is NOT MasterGraph's lexicon router. That one hardcodes 51 entity strings ("tsmc",
# "deloitte") because its questions ask "which companies use TSMC" - there the entity IS the
# filter. Here every person, channel and department is a CAPTURE GROUP, so the patterns encode
# phrasing and relation names only, and the LLM planner still handles everything unmatched.
#
# P3_REGEX_ROUTER=0 restores planner-only behaviour, which is what the pre-2026-08-25 published
# numbers were measured under.
REGEX_ROUTER = os.getenv("P3_REGEX_ROUTER", "1") != "0"
_rr_mod = None
_rr_ok: bool | None = None


def _regex_router():
    """Load the sibling module by PATH, not by name: this file is imported both as
    `schema_router` (pipeline.py, with pipeline3_round3 on sys.path) and as
    `pipeline3_round3.schema_router`, and a bare `import regex_router` resolves in only one of
    those. Returns None if the module is missing."""
    global _rr_mod
    if _rr_mod is None:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "p3_regex_router", Path(__file__).with_name("regex_router.py"))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            _rr_mod = m
        except Exception as e:
            print(f"[router] pattern fast path unavailable: {e}")
            _rr_mod = False
    return _rr_mod or None


def _regex_applies() -> bool:
    """Only use the fast path if THIS graph publishes every relation the patterns name.
    Point the demo at a different schema and the patterns silently stop firing rather than
    routing to relations that do not exist - which is the property the schema router exists
    to protect."""
    global _rr_ok
    if _rr_ok is None:
        rr = _regex_router()
        if rr is None:
            _rr_ok = False
        else:
            known = {r["relation"] for r in graph_relations()}
            missing = rr.RELATIONS - known
            _rr_ok = not missing
            if missing:
                print(f"[router] pattern fast path OFF - schema lacks {sorted(missing)}")
    return _rr_ok


def plan(question: str, get_llm_response, attempt: int = 0) -> dict:
    """Map a question onto legs + combine + tail over the graph's relations.

    Returns {"legs": [{"seed": str, "path": [(rel, dir), ...]}, ...],
             "combine": "intersect"|"union", "tail": [(rel, dir), ...], "intent": ...}.

    Two earlier models failed for reasons worth keeping:
      - (relation, target) constraints INTERSECTED, copied from MasterGraph's aggregation
        model, could not express a CHAIN at all.
      - one shared path applied to every seed could not express two seeds that reach the same
        destination by DIFFERENT routes, which is what every 'which people from department D
        are in channel C' question needs. Measured: 0/5 on that shape, 2 of them empty plans.

    The plan is retried once on a parse or validity failure. The call is non-deterministic
    even at temperature 0 — measured on the regressed questions, the same prompt produced a
    valid plan twice and an empty one once — and a dropped plan costs the whole question,
    since P3 then falls back to plain vector search.
    """
    # `attempt` is the caller's re-plan counter. attempt>0 means the walk landed nothing on the
    # plan just returned, so re-serving the same pattern plan would fail identically - hand that
    # draw to the LLM planner instead.
    #
    # This used to infer the retry from a module-level "last question seen", which was WRONG:
    # asking the same question twice in a row is indistinguishable from a re-plan that way, so
    # the second ask silently paid the planner. Observed live - 1,650 tokens on the first call
    # and 3,023 on an identical second call. The caller knows which attempt it is on; guessing
    # from global state cannot.
    if REGEX_ROUTER and attempt == 0 and _regex_applies():
        hit = _regex_router().route(question)
        if hit:
            return hit[2]

    rels = graph_relations()
    # Coverage is shown because the planner cannot otherwise tell a broad relation from a
    # narrow one, and picking the narrow one under 'intersect' returns an empty set: measured on
    # five questions of the form "channels containing documents involving both X and Y", where
    # `authored` (33% coverage) intersected to nothing every time while `participated_in` (89%)
    # returned the gold answer exactly. Same failure shape, one wrong relation.
    if PLAN_SLIM:
        # no example edges: they are mostly long document paths and cost ~200 tokens
        listing = "\n".join(
            f"  {r['relation']} ({100 * r.get('coverage', 0):.0f}%)" for r in rels)
        prompt = PLAN_PROMPT_SLIM.format(relations=listing, question=question)
    else:
        listing = "\n".join(
            f"  {r['relation']}  [covers {100 * r.get('coverage', 0):.0f}% of documents]"
            + (f"   e.g. ({_short(r['example_sources'][0])}) -> ({_short(r['example_targets'][0])})"
               if r["example_sources"] and r["example_targets"] else "")
            for r in rels)
        prompt = PLAN_PROMPT.format(relations=listing, question=question)
    empty = {"legs": [], "combine": "union", "tail": [], "intent": "lookup"}
    known = {r["relation"] for r in rels}

    def hops(raw_path) -> list:
        out = []
        for h in (raw_path or [])[:3]:
            if not isinstance(h, dict):
                continue
            rel = str(h.get("relation", "")).strip()
            d = str(h.get("direction", "out")).strip().lower()
            # a hallucinated relation is dropped, not guessed at — the graph is the authority
            if rel in known:
                out.append((rel, "in" if d == "in" else "out"))
        return out

    for _attempt in range(PLAN_RETRIES):
        try:
            text, raw = get_llm_response(prompt, json_mode=True)
            u = raw.get("usage", {})
            _plan_tokens["prompt"] += u.get("prompt_tokens", 0)
            _plan_tokens["output"] += u.get("completion_tokens", 0)
            _plan_tokens["calls"] += 1
            data = json.loads(text)
        except Exception:
            continue

        legs = []
        for leg in (data.get("legs") or [])[:4]:
            if not isinstance(leg, dict):
                continue
            seed = str(leg.get("seed", "")).strip()
            path = hops(leg.get("path"))
            if seed and path:
                legs.append({"seed": seed, "path": path})
        if not legs:
            continue
        combine = "intersect" if str(data.get("combine", "")).lower() == "intersect" else "union"
        intent = data.get("intent") if data.get("intent") in ("set", "lookup") else "lookup"
        return {"legs": legs, "combine": combine, "tail": hops(data.get("tail")),
                "intent": intent}
    # SAY SO. An empty plan is not a neutral outcome: pipeline.py leaves route_name as
    # "vector", so P3 answers from plain vector search and presents as working while the graph
    # contributed nothing. Silent degradation is the one failure this project cares most about
    # catching, and it cost a full debugging pass to notice from the route field alone.
    print(f"[router] NO PLAN after {PLAN_RETRIES} attempts — P3 will fall back to vector "
          f"search for: {question[:80]!r}")
    return empty


def plan_token_usage() -> dict:
    return dict(_plan_tokens)


def reset_token_usage() -> None:
    for k in _plan_tokens:
        _plan_tokens[k] = 0
