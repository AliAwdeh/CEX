"""Export helpers: journey-level CSV, message-level CSV, and full JSON."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

from aggregation import flatten_conversation_row, flatten_message_row


def build_conversation_csv_bytes(conversation_results: list[dict]) -> bytes:
    """Build the journey-level CSV (one row per customer journey) as bytes."""
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
    """Build the message-level CSV (one row per evaluated assistant message) as bytes."""
    rows = [flatten_message_row(m) for m in message_results]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_issue_analysis_csv_bytes(issue_analysis_results: list[dict]) -> bytes:
    """Build a flat CSV of Layer 3 patterns (one row per pattern) as bytes."""
    rows = []
    for ir in issue_analysis_results or []:
        issue_type = ir.get("issue_type", "")
        parsed = ir.get("parsed_json") or ir.get("evaluation_output") or {}
        if not isinstance(parsed, dict):
            parsed = {}
        summary = parsed.get("summary", "")
        confidence = parsed.get("confidence", "")
        patterns = parsed.get("patterns") or []
        if not patterns:
            rows.append(
                {
                    "variant": ir.get("variant", ""),
                    "journey_id": ir.get("journey_id", "") or "",
                    "issue_type": issue_type,
                    "journeys_analyzed": ir.get("journey_count", 0),
                    "parse_status": ir.get("parse_status", ""),
                    "pattern_description": "",
                    "trigger_source": "",
                    "trigger_type": "",
                    "trigger": "",
                    "customer_context": "",
                    "where_it_happens": "",
                    "root_cause_category": "",
                    "root_cause_explanation": "",
                    "recommended_solution": "",
                    "occurrence_count": 0,
                    "journey_ids": "",
                    "summary": summary,
                    "confidence": confidence,
                }
            )
            continue
        for pat in patterns:
            if not isinstance(pat, dict):
                continue
            journey_ids = pat.get("journey_ids") or []
            rows.append(
                {
                    "variant": ir.get("variant", ""),
                    "journey_id": ir.get("journey_id", "") or "",
                    "issue_type": issue_type,
                    "journeys_analyzed": ir.get("journey_count", 0),
                    "parse_status": ir.get("parse_status", ""),
                    "pattern_description": pat.get("pattern_description", ""),
                    "trigger_source": pat.get("trigger_source", ""),
                    "trigger_type": pat.get("trigger_type", ""),
                    "trigger": pat.get("trigger", ""),
                    "customer_context": pat.get("customer_context", ""),
                    "where_it_happens": pat.get("where_it_happens", ""),
                    "root_cause_category": pat.get("root_cause_category", ""),
                    "root_cause_explanation": pat.get("root_cause_explanation", ""),
                    "recommended_solution": pat.get("recommended_solution", ""),
                    "occurrence_count": pat.get("occurrence_count", len(journey_ids)),
                    "journey_ids": "; ".join(str(x) for x in journey_ids),
                    "summary": summary,
                    "confidence": confidence,
                }
            )
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
    issue_analysis_results: list[dict] | None = None,
) -> bytes:
    """Build the combined JSON export including raw responses, errors, and config."""
    payload = {
        "run_config": _json_safe(run_config),
        "conversation_results": _json_safe(conversation_results),
        "message_level_results": _json_safe(message_level_results),
        "issue_analysis_results": _json_safe(issue_analysis_results or []),
        "errors": _json_safe(errors),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
