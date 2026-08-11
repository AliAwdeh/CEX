"""OpenAI-compatible API client wrapper.

Wraps the OpenAI Python SDK against a custom base URL. Adds simple retry logic,
timeout handling, and a /models loader for the model picklist.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "https://langcc.maidstech.ai/v1"
DEFAULT_MODEL = "openai/gpt-5.4-mini"
MAX_CONCURRENCY = 200

@dataclass
class APIConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    service_tier: str | None = None
    # Cross-provider thinking control. Supported values are:
    # default, disabled, low, medium, high, maximum.
    thinking_effort: str = "default"
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 100000
    timeout: float = 300.0
    retries: int = 2
    concurrency: int = 60
    # When True, every chat_completion() call (system prompt, user prompt,
    # raw response, reasoning content, usage) is appended to a local JSONL
    # debug log. Off by default: prompts/responses can contain real customer
    # data, so this is an explicit per-run opt-in, not an always-on default.
    debug_log_calls: bool = False


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


def _looks_like_response_format_rejection(error: Exception) -> bool:
    text = str(error).lower()
    return "response_format" in text and any(
        token in text
        for token in (
            "unsupported",
            "not supported",
            "unknown",
            "unrecognized",
            "invalid",
            "extra",
        )
    )


def _thinking_request_kwargs(config: APIConfig) -> dict[str, Any]:
    """Translate the platform thinking setting to provider-compatible fields."""
    effort = str(config.thinking_effort or "default").strip().lower()
    if effort in {"", "default", "provider default"}:
        return {}

    is_deepseek = "deepseek" in f"{config.base_url} {config.model}".lower()
    if effort in {"disabled", "off", "none"}:
        if is_deepseek:
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {"reasoning_effort": "none"}

    effort = effort if effort in {"low", "medium", "high", "maximum"} else "medium"
    if is_deepseek:
        return {
            "reasoning_effort": "max" if effort == "maximum" else effort,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    return {"reasoning_effort": "xhigh" if effort == "maximum" else effort}


def _object_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _stringify_content(value: Any) -> str:
    """Extract visible assistant text from common Chat/Responses SDK shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            pass
    if isinstance(value, list):
        return "".join(_stringify_content(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            extracted = _stringify_content(value.get(key))
            if extracted:
                return extracted
        return ""
    return ""


def _response_choices(response: Any) -> list[Any]:
    choices = _object_get(response, "choices", None)
    return list(choices or [])


def _extract_response_text(response: Any) -> tuple[str, dict[str, Any]]:
    output_text = _stringify_content(_object_get(response, "output_text", None))
    choices = _response_choices(response)
    choice_debug: list[dict[str, Any]] = []

    texts: list[str] = []
    if output_text:
        texts.append(output_text)

    for index, choice in enumerate(choices):
        message = _object_get(choice, "message", None) or {}
        content_value = _object_get(message, "content", None)
        content = _stringify_content(content_value)
        choice_text = _stringify_content(_object_get(choice, "text", None))
        if content:
            texts.append(content)
        elif choice_text:
            texts.append(choice_text)

        refusal = _stringify_content(_object_get(message, "refusal", None))
        reasoning = _stringify_content(_object_get(message, "reasoning_content", None))
        choice_debug.append(
            {
                "index": _object_get(choice, "index", index),
                "finish_reason": _object_get(choice, "finish_reason", None),
                "message_role": _object_get(message, "role", None),
                "content_type": type(content_value).__name__,
                "content_chars": len(content),
                "choice_text_chars": len(choice_text),
                "refusal_chars": len(refusal),
                "reasoning_content_chars": len(reasoning),
                # The actual reasoning/chain-of-thought text, when the model
                # returns one (reasoning-effort models). Previously only the
                # character count was kept and the text itself was discarded.
                "reasoning_content": reasoning,
                "tool_call_count": len(_object_get(message, "tool_calls", None) or []),
            }
        )

    return "".join(texts), {
        "id": _object_get(response, "id", None),
        "model": _object_get(response, "model", None),
        "choice_count": len(choices),
        "output_text_chars": len(output_text),
        "choices": choice_debug,
    }


def _empty_response_error(response_debug: dict[str, Any], usage: Any = None) -> str:
    choices = response_debug.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    parts = ["Empty model response"]
    if finish_reason:
        parts.append(f"finish_reason={finish_reason}")
    if choices:
        reasoning_chars = choices[0].get("reasoning_content_chars")
        if reasoning_chars:
            parts.append(f"reasoning_content_chars={reasoning_chars}")
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning_tokens:
            parts.append(f"reasoning_tokens={reasoning_tokens}")
    return f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0]


_LOG_LOCK = threading.Lock()
_LOG_PATH = Path(__file__).resolve().parent / "logs" / "llm_calls.jsonl"


def _log_llm_call(
    *,
    config: APIConfig,
    context: str,
    system_prompt: str,
    user_prompt: str,
    debug: dict[str, Any],
    content: str | None,
    elapsed_seconds: float,
    success: bool,
    error: str | None = None,
) -> None:
    """Append one JSONL line describing a chat_completion() call.

    Best-effort only: a logging failure (disk full, permissions, etc.) must
    never break the actual LLM call it's describing, so all I/O errors here
    are swallowed.
    """
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context or "",
            "success": success,
            "model": config.model,
            "thinking_effort": str(config.thinking_effort or "default"),
            "temperature": config.temperature,
            "attempts": debug.get("attempts"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": content,
            "usage": debug.get("usage"),
            "reasoning": [
                choice.get("reasoning_content")
                for choice in (debug.get("response") or {}).get("choices") or []
                if choice.get("reasoning_content")
            ],
            "errors": debug.get("errors"),
            "error": error,
        }
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _LOG_LOCK:
            with _LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        pass


def chat_completion(
    client: OpenAI,
    config: APIConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    context: str = "",
) -> tuple[str, dict]:
    """Run a chat completion with retry. Returns (raw_text, debug_info).

    ``context`` is a short free-text label identifying what triggered this
    call (e.g. "ticket_segmentation:CONV123"). It has no effect on the
    request itself -- it's only used to make the optional debug log
    (config.debug_log_calls) readable.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    attempts = max(1, int(config.retries) + 1)
    debug: dict[str, Any] = {
        "attempts": 0,
        "errors": [],
        "responses": [],
        "thinking_effort": str(config.thinking_effort or "default"),
    }
    call_started = time.monotonic()

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
            if config.service_tier:
                kwargs["service_tier"] = str(config.service_tier)
            kwargs.update(_thinking_request_kwargs(config))
            # Hint compatible endpoints to prefer JSON responses where supported.
            # Some proxies will ignore unknown params, so guard the call.
            try:
                response = client.chat.completions.create(
                    response_format={"type": "json_object"},
                    **kwargs,
                )
            except TypeError:
                response = client.chat.completions.create(**kwargs)
            except Exception as response_format_error:
                # Some servers reject response_format; retry once without it before raising.
                if not _looks_like_response_format_rejection(response_format_error):
                    raise
                response = client.chat.completions.create(**kwargs)

            usage = _object_get(response, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                debug["usage"] = usage.model_dump()
            elif isinstance(usage, dict):
                debug["usage"] = usage
            response_service_tier = _object_get(response, "service_tier", None)
            if response_service_tier:
                debug["service_tier"] = response_service_tier
            content, response_debug = _extract_response_text(response)
            debug["response"] = response_debug
            debug["responses"].append({"attempt": attempt, **response_debug})
            if not str(content or "").strip():
                error = RuntimeError(_empty_response_error(response_debug, debug.get("usage")))
                last_error = error
                debug["errors"].append(
                    {
                        "attempt": attempt,
                        "error": str(error),
                        "retryable_empty_response": True,
                    }
                )
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                break
            if config.debug_log_calls:
                _log_llm_call(
                    config=config,
                    context=context,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    debug=debug,
                    content=content,
                    elapsed_seconds=time.monotonic() - call_started,
                    success=True,
                )
            return content, debug
        except Exception as e:  # noqa: BLE001 — surface any provider error
            last_error = e
            debug["errors"].append({"attempt": attempt, "error": str(e)})
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
            continue

    if config.debug_log_calls:
        _log_llm_call(
            config=config,
            context=context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            debug=debug,
            content=None,
            elapsed_seconds=time.monotonic() - call_started,
            success=False,
            error=str(last_error),
        )
    raise RuntimeError(f"chat_completion failed after {attempts} attempts: {last_error}")
