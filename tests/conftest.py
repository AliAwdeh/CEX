"""Shared fixtures for the ticket-segmentation regression tests."""

from __future__ import annotations

from typing import Any


def make_record(
    index: int,
    role: str,
    text: str,
    *,
    source: str = "conv_1",
    raw_role: str | None = None,
    agent_full_name: str | None = None,
) -> dict[str, Any]:
    """Build a minimal message record in the shape evaluator.py's ticket
    pipeline expects (see data_loader.message_records_from_group)."""
    return {
        "message_id": f"{source}::{index}",
        "message_index": index,
        "appended_message_index": index,
        "source_conversation_id": source,
        "message_time": f"2026-01-01T00:{index:02d}:00",
        "sender_role": role,
        "raw_sender_role": raw_role or role,
        "message_text": text,
        "agent_full_name": agent_full_name,
    }


def make_ticket(ticket_id: str, indexes: list[int], **overrides: Any) -> dict[str, Any]:
    """Build a minimal already-normalized ticket dict (the shape
    _normalize_ticket_segments produces), for tests that exercise the
    post-normalization merge/split/carry-forward passes directly."""
    ticket: dict[str, Any] = {
        "ticket_id": ticket_id,
        "ticket_category": "request",
        "model_ticket_category": "request",
        "request_origin": "customer",
        "ticket_type": "other",
        "customer_objective": "Test objective",
        "start_message_index": min(indexes),
        "end_message_index": max(indexes),
        "included_message_indexes": list(indexes),
        "status": "resolved",
        "should_append_future_conversations": False,
        "previous_ticket_id": "",
        "inquiries": [],
        "segmentation_reason": "test fixture",
    }
    ticket.update(overrides)
    return ticket
