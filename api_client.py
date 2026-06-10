"""OpenAI-compatible API client wrapper.

Wraps the OpenAI Python SDK against a custom base URL. Adds simple retry logic,
timeout handling, and a /models loader for the model picklist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "https://langcc.maidstech.ai/v1"

# Hard upper limit on parallel in-flight requests. The Evaluator clamps the
# user's concurrency setting to this value.
MAX_CONCURRENCY = 64


@dataclass
class APIConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = ""
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 1500
    timeout: float = 60.0
    retries: int = 2
    concurrency: int = 8


def build_client(base_url: str, api_key: str) -> OpenAI:
    """Build an OpenAI client pointed at the configured base URL."""
    if not api_key:
        # The OpenAI SDK requires a non-empty string. Internal proxies may not require it.
        api_key = "EMPTY"
    return OpenAI(api_key=api_key, base_url=base_url)


def fetch_models(client: OpenAI) -> list[str]:
    """Return the list of model ids available from the OpenAI-compatible /models endpoint."""
    resp = client.models.list()
    ids: list[str] = []
    data = getattr(resp, "data", None)
    if data is None and isinstance(resp, dict):
        data = resp.get("data", [])
    for item in data or []:
        if hasattr(item, "id"):
            ids.append(item.id)
        elif isinstance(item, dict) and "id" in item:
            ids.append(item["id"])
    ids = sorted(set(ids))
    return ids


def chat_completion(
    client: OpenAI,
    config: APIConfig,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict]:
    """Run a chat completion with retry. Returns (raw_text, debug_info)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    attempts = max(1, int(config.retries) + 1)
    debug: dict[str, Any] = {"attempts": 0, "errors": []}

    for attempt in range(1, attempts + 1):
        debug["attempts"] = attempt
        try:
            kwargs: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "temperature": float(config.temperature),
                "top_p": float(config.top_p),
                "max_tokens": int(config.max_tokens),
                "timeout": float(config.timeout),
            }
            # Hint compatible endpoints to prefer JSON responses where supported.
            # Some proxies will ignore unknown params, so guard the call.
            try:
                response = client.chat.completions.create(
                    response_format={"type": "json_object"},
                    **kwargs,
                )
            except TypeError:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                # Some servers reject response_format; retry once without it before raising.
                response = client.chat.completions.create(**kwargs)

            content = ""
            if hasattr(response, "choices") and response.choices:
                first = response.choices[0]
                msg = getattr(first, "message", None) or {}
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "") or ""
            usage = getattr(response, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                debug["usage"] = usage.model_dump()
            elif isinstance(usage, dict):
                debug["usage"] = usage
            return content, debug
        except Exception as e:  # noqa: BLE001 — surface any provider error
            last_error = e
            debug["errors"].append({"attempt": attempt, "error": str(e)})
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
            continue

    raise RuntimeError(f"chat_completion failed after {attempts} attempts: {last_error}")
