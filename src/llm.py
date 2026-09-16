"""
Provider-agnostic LLM client supporting Anthropic, xAI Grok, and Groq.

Secrets are read from environment variables or a local .env file.
Env vars:
  LLM_PROVIDER        ``anthropic``, ``grok``, or ``groq`` (auto-detected if unset)
  GROK_API_KEY        required for xAI Grok (or XAI_API_KEY)
  GROQ_API_KEY        required for Groq
  ANTHROPIC_API_KEY   required for Anthropic
  GROK_MODEL          optional Grok model override (default: grok-2-latest)
  GROQ_MODEL          optional Groq model override (default: qwen/qwen3.8-27b)
  LLM_MODEL           optional Anthropic model override (default: claude-haiku-4-5-20251001)
"""

import json
import os
from pathlib import Path

# Automatically load .env if present in project root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.is_file():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _v = _v.strip().strip("'\"")
                if _k not in os.environ:
                    os.environ[_k] = _v

# Groq model: qwen/qwen3.8-27b and openai/gpt-oss-20b are available on this account.
# Llama models return 404 on current Groq tiers.
_raw_groq_model = os.environ.get("GROQ_MODEL", "")
if not _raw_groq_model or "llama" in _raw_groq_model.lower():
    DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
else:
    DEFAULT_GROQ_MODEL = _raw_groq_model

DEFAULT_ANTHROPIC_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_GROK_MODEL = os.environ.get("GROK_MODEL", os.environ.get("XAI_MODEL", "grok-2-latest"))

_anthropic_client = None
_groq_client = None


def get_provider() -> str:
    """Determine active provider from LLM_PROVIDER or auto-detect based on available keys."""
    # If the provided key starts with 'gsk_', it is a GroqCloud key (regardless of grok/groq label)
    raw_key = os.environ.get("GROK_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
    if raw_key.startswith("gsk_"):
        return "groq"

    env_provider = os.environ.get("LLM_PROVIDER")
    if env_provider:
        return env_provider.strip().lower()
    if os.environ.get("XAI_API_KEY") or (os.environ.get("GROK_API_KEY") and not raw_key.startswith("gsk_")):
        return "grok"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "anthropic"


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No LLM credentials configured. To use Grok/Groq, set export GROQ_API_KEY='gsk_...'. "
                "Or configure a .env file in the project root."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def complete(system: str, user: str, max_tokens: int = 512, model: str | None = None) -> str:
    provider = get_provider()

    if provider in ("grok", "xai"):
        return _grok_complete(system, user, max_tokens, model or DEFAULT_GROK_MODEL)

    if provider == "groq":
        target_model = model or DEFAULT_GROQ_MODEL
        if "llama" in target_model.lower():
            target_model = "qwen/qwen3.8-27b"
        return _groq_complete(system, user, max_tokens, target_model)

    if provider == "anthropic":
        client = _get_anthropic_client()
        resp = client.messages.create(
            model=model or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}; use 'grok', 'groq', or 'anthropic'.")


def _grok_complete(system: str, user: str, max_tokens: int, model: str) -> str:
    api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROK_API_KEY (or XAI_API_KEY) is not set. Export it in your terminal before running the pipeline."
        )

    import certifi
    import httpx

    try:
        response = httpx.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
            verify=certifi.where(),
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        detail = ""
        if hasattr(exc, "response") and exc.response is not None:
            detail = f": {exc.response.text}"
        raise RuntimeError(f"Grok API request failed: {exc}{detail}") from exc
    if not content:
        raise RuntimeError("Grok returned an empty completion.")
    return content


def _groq_complete(system: str, user: str, max_tokens: int, model: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Export it in your terminal before running the pipeline.")

    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=api_key)
    import time

    for attempt in range(4):
        try:
            response = _groq_client.chat.completions.create(
                model=model,
                max_completion_tokens=min(max_tokens, 800),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = response.choices[0].message.content
            if content:
                return content
        except Exception as exc:
            err_str = str(exc).lower()
            if ("rate" in err_str or "429" in err_str or "limit" in err_str) and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"Groq API request failed: {exc}") from exc

    raise RuntimeError("Groq returned an empty completion.")


def complete_json(system: str, user: str, max_tokens: int = 512, model: str | None = None) -> dict:
    """Ask for strict JSON and parse it. Retries once with a corrective nudge on parse failure."""
    text = complete(system + "\n\nRespond with ONLY a single valid JSON object, no prose.", user, max_tokens, model)
    try:
        return json.loads(_extract_json(text))
    except json.JSONDecodeError:
        retry_user = (
            f"{user}\n\nYour previous response was not valid JSON:\n{text}\n"
            "Return ONLY a single valid JSON object this time."
        )
        text2 = complete(system, retry_user, max_tokens, model)
        return json.loads(_extract_json(text2))


def _extract_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    return text[start : end + 1]