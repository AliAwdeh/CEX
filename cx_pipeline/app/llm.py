from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    api_key: str
    model: str
    service_tier: str | None = None
    thinking_effort: str = "default"
    temperature: float = 0.1
    top_p: float = 1.0
    timeout: float = 600.0
    retries: int = 2
    concurrency: int = 1
    debug_log_calls: bool = False


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key or "EMPTY", base_url=base_url)


def _object_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _stringify_content(value: Any) -> str:
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
            text = _stringify_content(value.get(key))
            if text:
                return text
    return str(value)


def _thinking_kwargs(config: APIConfig) -> dict[str, Any]:
    effort = str(config.thinking_effort or "default").strip().lower()
    if effort in {"", "default", "provider default"}:
        return {}
    if effort in {"disabled", "off", "none"}:
        return {"reasoning_effort": "none"}
    if effort == "maximum":
        effort = "xhigh"
    if effort not in {"low", "medium", "high", "xhigh"}:
        effort = "medium"
    return {"reasoning_effort": effort}


def chat_prompt(
    client: OpenAI,
    config: APIConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    context: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    attempts = max(1, int(config.retries) + 1)
    debug: dict[str, Any] = {"context": context, "attempts": 0, "errors": []}
    last_error: Exception | None = None
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        debug["attempts"] = attempt
        try:
            kwargs: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "temperature": float(config.temperature),
                "top_p": float(config.top_p),
                "timeout": float(config.timeout),
            }
            if config.service_tier:
                kwargs["service_tier"] = config.service_tier
            kwargs.update(_thinking_kwargs(config))
            try:
                response = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            except Exception as exc:
                if "response_format" not in str(exc).lower():
                    raise
                response = client.chat.completions.create(**kwargs)
            usage = _object_get(response, "usage", None)
            if usage is not None and hasattr(usage, "model_dump"):
                debug["usage"] = usage.model_dump()
            elif isinstance(usage, dict):
                debug["usage"] = usage
            choices = list(_object_get(response, "choices", []) or [])
            raw = ""
            for choice in choices:
                raw += _stringify_content(_object_get(_object_get(choice, "message", {}), "content", None))
            if not raw.strip():
                raise RuntimeError("Empty model response")
            parsed = extract_json_object(raw)
            debug["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return parsed, raw, debug
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            debug["errors"].append({"attempt": attempt, "error": str(exc)})
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 5))
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_error}")


def chat_json(
    client: OpenAI,
    config: APIConfig,
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    context: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    return chat_prompt(
        client,
        config,
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
        context=context,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        obj = json.loads(fence.group(1))
        if isinstance(obj, dict):
            return obj
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError("No JSON object found in model response")
