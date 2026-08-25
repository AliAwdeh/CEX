from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import strip_inline_rag_context


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "correct_prompt_files"


@dataclass(frozen=True)
class PromptTemplate:
    system_prompt: str
    output_schema: str
    user_prompt_template: str

    def build_system(self) -> str:
        if "{output_schema}" in self.system_prompt:
            return self.system_prompt.replace("{output_schema}", self.output_schema)
        return f"{self.system_prompt}\n\nRequired schema:\n{self.output_schema}"

    def build_user(self, payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if "{payload_json}" in self.user_prompt_template:
            return self.user_prompt_template.replace("{payload_json}", payload_json)
        return f"{self.user_prompt_template}\n\nInput:\n{payload_json}"


def _read_prompt(filename: str) -> str:
    path = PROMPT_ROOT / filename
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError(f"Prompt file is empty: {path}")
    return value


MESSAGE_LEVEL_PROMPT = PromptTemplate(
    system_prompt=_read_prompt("message prompt.txt"),
    output_schema=_read_prompt("Message scheme"),
    user_prompt_template=_read_prompt("message user input"),
)

TICKET_CX_PROMPT = PromptTemplate(
    system_prompt=_read_prompt("conversational prompt.txt"),
    output_schema=_read_prompt("conversational output scheme.txt"),
    user_prompt_template=_read_prompt("conversational user input.txt"),
)

TICKET_SEGMENTATION_PROMPT = PromptTemplate(
    system_prompt=_read_prompt("second ticket segmenation prompt.txt"),
    output_schema=_read_prompt("ticket segmentation scheme.txt"),
    user_prompt_template=_read_prompt("ticket segmentation user input.txt"),
)


_ALLOWED_METADATA_KEYS = {
    "customer_name",
    "customer_phone",
    "customer_journey_id",
    "journey_id",
    "source_conversation_ids",
    "source_conversation_count",
    "conversation_start_date",
    "conversation_end_date",
    "conversation_status",
    "conversation_agent_full_name",
    "conversation_agent_login_name",
    "initial_skill",
    "last_skill",
    "joined_skills",
    "total_visible_messages",
    "customer_message_count",
    "agent_message_count",
    "unknown_message_count",
    "evaluation_target_role",
    "ticket_db_id",
    "customer_db_id",
}


def sanitize_conversation_metadata_for_llm(conversation_metadata: dict | None) -> dict:
    if not isinstance(conversation_metadata, dict):
        return {}
    return {
        key: value
        for key, value in conversation_metadata.items()
        if key in _ALLOWED_METADATA_KEYS and value not in (None, "")
    }


def _trim(text: Any, truncate_chars: int | None) -> str:
    text = strip_inline_rag_context(text)
    if truncate_chars and len(text) > truncate_chars:
        return text[:truncate_chars] + "...[truncated]"
    return text


def _evaluator_role(message: dict) -> str:
    return "customer" if str(message.get("sender_role", "")).lower() == "customer" else "agent"


def _sender_entity(message: dict) -> str:
    raw_role = str(message.get("raw_sender_role", "") or "").strip().lower()
    if raw_role == "system":
        return "broadcast"
    if raw_role in {"bot", "assistant"}:
        return "bot"
    if raw_role == "agent":
        return "agent"
    role = str(message.get("sender_role", "") or "").strip().lower()
    if role in {"customer", "agent"}:
        return role
    return "unknown"


def build_message_level_payload(
    conversation_id: str,
    target_message: dict,
    history: list[dict],
    conversation_metadata: dict,
    truncate_chars: int | None = None,
) -> dict:
    target = {
        "message_id": target_message.get("message_id", ""),
        "message_index": target_message.get("message_index"),
        "sender_role": _evaluator_role(target_message),
        "raw_sender_role": target_message.get("raw_sender_role"),
        "sender_entity": _sender_entity(target_message),
        "message_time": str(target_message.get("message_time", "")),
        "message_text": _trim(target_message.get("message_text", ""), truncate_chars),
    }
    history_clean = [
        {
            "message_id": message.get("message_id", ""),
            "sender_role": _evaluator_role(message),
            "raw_sender_role": message.get("raw_sender_role"),
            "sender_entity": _sender_entity(message),
            "message_index": message.get("message_index"),
            "message_time": str(message.get("message_time", "")),
            "message_text": _trim(message.get("message_text", ""), truncate_chars),
        }
        for message in history
    ]
    return {
        "conversation_id": conversation_id,
        "target_message": target,
        "conversation_history_until_target": history_clean,
    }


def build_ticket_cx_payload(
    conversation_id: str,
    conversation_metadata: dict,
    full_transcript: list[dict],
    message_level_evaluations: list[dict],
    computed_metadata: dict,
    truncate_chars: int | None = None,
) -> dict:
    eval_by_idx: dict[Any, dict] = {}
    for evaluation in message_level_evaluations or []:
        if not isinstance(evaluation, dict):
            continue
        idx = evaluation.get("message_index")
        if idx is None:
            continue
        try:
            eval_by_idx[int(idx)] = evaluation
        except (TypeError, ValueError):
            eval_by_idx[idx] = evaluation

    transcript_clean = []
    for message in full_transcript:
        try:
            msg_idx = int(message.get("message_index", 0))
        except (TypeError, ValueError):
            msg_idx = message.get("message_index", 0)
        entry: dict[str, Any] = {
            "message_index": msg_idx,
            "appended_message_index": message.get("appended_message_index", msg_idx),
            "source_conversation_id": message.get("source_conversation_id"),
            "message_time": str(message.get("message_time", "")),
            "sender_role": message.get("sender_role", ""),
            "raw_sender_role": message.get("raw_sender_role"),
            "sender_entity": _sender_entity(message),
            "agent_full_name": message.get("agent_full_name"),
            "message_text": _trim(message.get("message_text", ""), truncate_chars),
        }
        if msg_idx in eval_by_idx:
            entry["message_level_evaluation"] = eval_by_idx[msg_idx]
        transcript_clean.append(entry)

    return {
        "conversation_id": conversation_id,
        "conversation_metadata": sanitize_conversation_metadata_for_llm(conversation_metadata),
        "full_transcript": transcript_clean,
        "message_level_evaluations": message_level_evaluations,
        "computed_metadata": computed_metadata,
        "message_level_summary": (computed_metadata or {}).get("message_level_summary", {}),
    }


def build_ticket_segmentation_payload(
    conversation_id: str,
    records: list[dict],
    conversation_metadata: dict,
    truncate_chars: int | None,
    segmentation_context: dict | None = None,
) -> dict:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for record in records:
        source_id = str(record.get("source_conversation_id") or "unknown").strip() or "unknown"
        if current is None or current["source_conversation_id"] != source_id:
            current = {"source_conversation_id": source_id, "messages": []}
            blocks.append(current)
        current["messages"].append(
            {
                "message_index": record.get("message_index"),
                "time": record.get("message_time"),
                "role": record.get("sender_role"),
                "text": _trim(record.get("message_text", ""), truncate_chars),
            }
        )
    payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "conversation_metadata": conversation_metadata,
        "input_format": "Messages are grouped under source_conversation_blocks. The source_conversation_id is the block header for the messages inside it.",
        "message_index_rule": "message_index is continuous across the full customer journey and does not reset inside source_conversation_blocks.",
        "source_conversation_blocks": blocks,
    }
    if segmentation_context:
        payload["segmentation_context"] = segmentation_context
    return payload

