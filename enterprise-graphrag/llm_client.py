"""Unified LLM client. Returns (answer_str, raw_response_dict).

Providers: "ollama" (local) or "openai" (Chat Completions API).
Only the *answering* LLM uses this; the judge in accuracy.py is independent.
"""

from __future__ import annotations
import os
import random
import time
from typing import Any

import config

_OPENAI_MAX_RETRIES = 5
_OPENAI_BACKOFF_BASE = 2.0


def get_llm_response(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    provider = config.LLM_PROVIDER
    if provider == "ollama":
        return _ollama_call(prompt)
    if provider == "openai":
        return _openai_call(prompt, json_mode=json_mode)
    if provider == "gemini":
        return _gemini_call(prompt, json_mode=json_mode)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


_GEMINI_MAX_RETRIES = 5
_GEMINI_BACKOFF_BASE = 2.0
# Upper bound for the truncation escalation below. 32k is generous for this workload — the
# largest visible answer in any published run is 344 tokens (a 35-ticker Tier-C list) — so
# hitting this ceiling means something is wrong, not that the question was merely big.
_GEMINI_MAX_OUTPUT_CEILING = int(os.getenv("GEMINI_MAX_OUTPUT_CEILING", "32768"))


def _gemini_call(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    """Gemini generateContent. Returns real token usage from usageMetadata so the
    benchmark reports ACTUAL inference tokens (PS requirement), not tiktoken estimates.

    A fixed temperature (config.TEMPERATURE, default 0) keeps generation deterministic.
    Same model + params are used by all three pipelines (fairness rule)."""
    import os
    import requests

    key = config.GOOGLE_API_KEY
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Add it to .env for LLM_PROVIDER=gemini.")
    model = config.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    gen: dict = {"temperature": config.TEMPERATURE, "maxOutputTokens": config.MAX_TOKENS}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}

    last_err: Exception | None = None
    for attempt in range(_GEMINI_MAX_RETRIES):
        try:
            r = requests.post(url, json=body, timeout=90)
            if r.status_code == 200:
                j = r.json()
                usage = j.get("usageMetadata", {})
                cand = j.get("candidates", [{}])
                content = ""
                if cand and cand[0].get("content", {}).get("parts"):
                    content = cand[0]["content"]["parts"][0].get("text", "").strip()
                # TRUNCATION IS NOT SUCCESS. finishReason=MAX_TOKENS means thinking plus output
                # exhausted the budget, so `content` is a fragment — and HTTP is still 200.
                # Returning it silently is what made the router's JSON plan arrive as
                # '{"legs": [{"seed": "Allison Gr' , fail json.loads, and drop P3 to plain
                # vector search with no error anywhere. Escalate the budget and retry instead
                # of guessing a fixed ceiling: a question whose answer is a 35-company list
                # needs more room than one answered in three words, and neither the caller nor
                # config.MAX_TOKENS can know which is which in advance.
                if cand and cand[0].get("finishReason") == "MAX_TOKENS":
                    cur = gen["maxOutputTokens"]
                    if cur < _GEMINI_MAX_OUTPUT_CEILING:
                        gen["maxOutputTokens"] = min(cur * 2, _GEMINI_MAX_OUTPUT_CEILING)
                        print(f"[llm] truncated at {cur} output tokens "
                              f"(thinking+output); retrying at {gen['maxOutputTokens']}")
                        last_err = RuntimeError(f"truncated at maxOutputTokens={cur}")
                        continue
                    raise RuntimeError(
                        f"Gemini truncated at the {_GEMINI_MAX_OUTPUT_CEILING}-token ceiling; "
                        "refusing to return a partial answer")
                usage_dict = {
                    "prompt_tokens":     usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens":      usage.get("totalTokenCount", 0),
                }
                return content, {"usage": usage_dict}
            if r.status_code in (429, 500, 503):
                last_err = RuntimeError(f"Gemini {r.status_code}: {r.text[:120]}")
                time.sleep(_GEMINI_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1))
                continue
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
            time.sleep(_GEMINI_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1))
    raise RuntimeError(f"Gemini call failed after {_GEMINI_MAX_RETRIES} retries: {last_err}")


def _ollama_call(prompt: str) -> tuple[str, Any]:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        temperature=config.TEMPERATURE,
        num_predict=config.MAX_TOKENS,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    # Ollama doesn't return token usage — metrics.py uses tiktoken estimation
    return response.content.strip(), {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _openai_call(prompt: str, json_mode: bool = False) -> tuple[str, Any]:
    from openai import (
        OpenAI,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
        AuthenticationError,
    )

    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=0, timeout=60.0)

    last_err: Exception | None = None
    for attempt in range(_OPENAI_MAX_RETRIES):
        try:
            kwargs: dict = {
                "model":       config.OPENAI_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": config.TEMPERATURE,
                "max_tokens":  64 if json_mode else config.MAX_TOKENS,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()
            usage = response.usage
            usage_dict = {
                "prompt_tokens":     getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens":      getattr(usage, "total_tokens", 0) if usage else 0,
            }
            return content, {"usage": usage_dict}

        except AuthenticationError as e:
            raise RuntimeError(
                f"OpenAI authentication failed ({e}). Check OPENAI_API_KEY in .env."
            ) from e

        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            last_err = e
            wait = _OPENAI_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            kind = type(e).__name__
            print(f"[llm_client] OpenAI {kind} — retrying in {wait:.1f}s ({attempt + 1}/{_OPENAI_MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError(f"OpenAI call failed after {_OPENAI_MAX_RETRIES} retries: {last_err}")
