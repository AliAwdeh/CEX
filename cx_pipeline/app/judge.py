from __future__ import annotations

import re
from typing import Any

from .llm import APIConfig, chat_prompt
from .prompts import (
    MESSAGE_LEVEL_PROMPT,
    TICKET_CX_PROMPT,
    TICKET_SEGMENTATION_PROMPT,
    build_message_level_payload,
    build_ticket_cx_payload,
    build_ticket_segmentation_payload,
)


STATUSES = {"resolved", "pending_unresolved", "totally_unresolved"}
CATEGORIES = {"issue", "request", "inquiry"}
ORIGINS = {"customer", "company"}


def _clean_snake(value: Any, default: str = "other") -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return re.sub(r"_+", "_", text) or default


def _int_list(values: Any, valid: set[int] | None = None) -> list[int]:
    out: list[int] = []
    for value in values or []:
        try:
            parsed = int(value)
        except Exception:
            continue
        if valid is not None and parsed not in valid:
            continue
        if parsed not in out:
            out.append(parsed)
    return sorted(out)


def normalize_ticket_segments(obj: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = {int(record["message_index"]) for record in records if record.get("message_index") is not None}
    normalized = []
    for idx, ticket in enumerate(obj.get("tickets") or [], start=1):
        if not isinstance(ticket, dict):
            continue
        included = _int_list(ticket.get("included_message_indexes"), valid)
        if not included:
            continue
        status = str(ticket.get("status") or "pending_unresolved").strip().lower()
        category = str(ticket.get("ticket_category") or "inquiry").strip().lower()
        origin = str(ticket.get("request_origin") or "customer").strip().lower()
        if status not in STATUSES:
            status = "pending_unresolved"
        if category not in CATEGORIES:
            category = "inquiry"
        if origin not in ORIGINS:
            origin = "customer"
        normalized.append(
            {
                "ticket_id": str(ticket.get("ticket_id") or f"ticket_{idx}"),
                "ticket_category": category,
                "request_origin": origin,
                "ticket_type": _clean_snake(ticket.get("ticket_type")),
                "customer_objective": str(ticket.get("customer_objective") or "").strip(),
                "start_message_index": min(included),
                "end_message_index": max(included),
                "included_message_indexes": included,
                "status": status,
                "should_append_future_conversations": bool(ticket.get("should_append_future_conversations")) or status == "pending_unresolved",
                "previous_ticket_id": str(ticket.get("previous_ticket_id") or "").strip(),
                "inquiries": ticket.get("inquiries") if isinstance(ticket.get("inquiries"), list) else [],
                "conversation_summaries": ticket.get("conversation_summaries") if isinstance(ticket.get("conversation_summaries"), list) else [],
                "segmentation_reason": str(ticket.get("segmentation_reason") or "").strip(),
            }
        )
    return normalized


def evaluate_ticket_segmentation_once(
    client,
    api: APIConfig,
    *,
    conversation_id: str,
    records: list[dict[str, Any]],
    conversation_metadata: dict[str, Any],
    truncate_chars: int | None,
    segmentation_context: dict[str, Any],
    normalization_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = build_ticket_segmentation_payload(
        conversation_id=conversation_id,
        records=records,
        conversation_metadata=conversation_metadata,
        truncate_chars=truncate_chars,
        segmentation_context=segmentation_context,
    )
    obj, raw, debug = chat_prompt(
        client,
        api,
        system_prompt=TICKET_SEGMENTATION_PROMPT.build_system(),
        user_prompt=TICKET_SEGMENTATION_PROMPT.build_user(payload),
        context=f"ticketing:{conversation_id}",
    )
    return normalize_ticket_segments(obj, normalization_records), {"raw_model_response": raw, "debug": debug}


def evaluate_message_level(
    client,
    api: APIConfig,
    *,
    conversation_id: str,
    target_record: dict[str, Any],
    history_records: list[dict[str, Any]],
    conversation_metadata: dict[str, Any],
    save_raw: bool,
    truncate_chars: int | None,
) -> dict[str, Any]:
    payload = build_message_level_payload(
        conversation_id=conversation_id,
        target_message=target_record,
        history=history_records,
        conversation_metadata=conversation_metadata,
        truncate_chars=truncate_chars,
    )
    record = {
        "conversation_id": conversation_id,
        "target_message_id": target_record.get("message_id"),
        "message_index": target_record.get("message_index"),
        "source_conversation_id": target_record.get("source_conversation_id"),
        "message_time": target_record.get("message_time"),
        "target_message_text": target_record.get("message_text"),
        "parse_status": "ok",
        "parsed_json": None,
        "raw_model_response": None,
        "debug": None,
        "error_message": None,
    }
    try:
        obj, raw, debug = chat_prompt(
            client,
            api,
            system_prompt=MESSAGE_LEVEL_PROMPT.build_system(),
            user_prompt=MESSAGE_LEVEL_PROMPT.build_user(payload),
            context=f"message:{conversation_id}:{target_record.get('message_index')}",
        )
        obj.setdefault("message_index", target_record.get("message_index"))
        record["parsed_json"] = obj
        record["raw_model_response"] = raw if save_raw else None
        record["debug"] = debug
    except Exception as exc:
        record["parse_status"] = "api_or_parse_error"
        record["error_message"] = str(exc)
    return record


def evaluate_ticket_cx(
    client,
    api: APIConfig,
    *,
    ticket_id: str,
    conversation_metadata: dict[str, Any],
    transcript: list[dict[str, Any]],
    message_level_evaluations: list[dict[str, Any]],
    computed_metadata: dict[str, Any],
    save_raw: bool,
    truncate_chars: int | None,
) -> dict[str, Any]:
    payload = build_ticket_cx_payload(
        conversation_id=ticket_id,
        conversation_metadata=conversation_metadata,
        full_transcript=transcript,
        message_level_evaluations=[
            row.get("parsed_json") or row.get("result")
            for row in message_level_evaluations
            if row.get("parsed_json") or row.get("result")
        ],
        computed_metadata=computed_metadata,
        truncate_chars=truncate_chars,
    )
    record = {"conversation_id": ticket_id, "parse_status": "ok", "parsed_json": None, "raw_model_response": None, "debug": None, "error_message": None}
    try:
        obj, raw, debug = chat_prompt(
            client,
            api,
            system_prompt=TICKET_CX_PROMPT.build_system(),
            user_prompt=TICKET_CX_PROMPT.build_user(payload),
            context=f"ticket_cx:{ticket_id}",
        )
        record["parsed_json"] = obj
        record["raw_model_response"] = raw if save_raw else None
        record["debug"] = debug
    except Exception as exc:
        record["parse_status"] = "api_or_parse_error"
        record["error_message"] = str(exc)
    return record
