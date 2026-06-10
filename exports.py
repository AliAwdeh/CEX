"""Export helpers: conversation-level CSV, message-level CSV, and full JSON."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

from aggregation import flatten_conversation_row, flatten_message_row


def build_conversation_csv_bytes(conversation_results: list[dict]) -> bytes:
    """Build the conversation-level CSV (one row per conversation) as bytes."""
    rows = []
    for cr in conversation_results:
        rows.append(
            flatten_conversation_row(
                cr,
                cr.get("conversation_metadata", {}) or {},
                cr.get("computed_metadata", {}) or {},
            )
        )
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_message_csv_bytes(message_results: list[dict]) -> bytes:
    """Build the message-level CSV (one row per evaluated agent message) as bytes."""
    rows = [flatten_message_row(m) for m in message_results]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def build_full_json_bytes(
    run_config: dict,
    conversation_results: list[dict],
    message_level_results: list[dict],
    errors: list[dict],
) -> bytes:
    """Build the combined JSON export including raw responses, errors, and config."""
    payload = {
        "run_config": _json_safe(run_config),
        "conversation_results": _json_safe(conversation_results),
        "message_level_results": _json_safe(message_level_results),
        "errors": _json_safe(errors),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
