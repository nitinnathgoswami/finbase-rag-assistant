"""Thin provider-agnostic wrapper: one function, `generate_json(system, user)`.

Supported providers (set LLM_PROVIDER): gemini | openai | anthropic | groq.
Temperature is 0 everywhere. Answers are cached on disk (data/llm_cache.json) so
identical calls are free, which saves quota when re-running the evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

from . import config

_CACHE_FILE = config.ROOT / "data" / "llm_cache.json"
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _gemini(system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return resp.text


def _openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def _anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=800,
        temperature=0,
        system=system + "\nRespond with a single JSON object and nothing else.",
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _groq(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


_PROVIDERS = {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic, "groq": _groq}


def _retry_after(msg: str) -> float | None:
    """Seconds suggested by messages like 'try again in 6m24.048s' / 'retry in 12h39m35s'."""
    m = re.search(r"(?:try again|retry) in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", msg, re.I)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s + 2


def generate_json(system: str, user: str, retries: int = 1) -> dict:
    provider = config.LLM_PROVIDER
    model = getattr(config, provider.upper() + "_MODEL")
    key = hashlib.sha256(f"{provider}|{model}|{system}|{user}".encode("utf-8")).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]

    max_wait = float(os.getenv("LLM_MAX_WAIT_S", "900"))  # longest wait we accept before giving up
    fn = _PROVIDERS[provider]
    last = None
    for attempt in range(6):
        try:
            out = _parse_json(fn(system, user))
            cache[key] = out
            _save_cache()
            return out
        except (json.JSONDecodeError, KeyError) as e:  # malformed output: retry once
            last = e
            if attempt >= retries:
                break
        except Exception as e:  # busy / rate-limited servers: wait and retry
            msg = str(e)
            if not any(c in msg for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "rate_limit")):
                raise
            last = e
            wait = _retry_after(msg)
            if wait is not None:
                if wait > max_wait:  # too long to wait: fail fast so the UI can show an error
                    raise
                print(f"[rate limit] waiting {int(wait)}s ...", flush=True)
                time.sleep(wait)
            else:
                time.sleep(min(60, 4 * 2 ** attempt))  # 4s, 8s, 16s, 32s, 60s
    raise RuntimeError(f"LLM call failed after retries: {last}")