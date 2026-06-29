"""Pre-run token and cost estimates for supported models."""

from __future__ import annotations

from typing import Any

import pandas as pd
import tiktoken

from aggregation import compute_metadata
from data_loader import (
    conversation_metadata_from_group,
    get_conversation_groups,
    message_records_from_group,
)
from evaluator import RunConfig
from prompts import (
    build_conversation_level_payload,
    build_message_level_payload,
)


GPT5_MINI_INPUT_PER_MILLION = 0.25
GPT5_MINI_OUTPUT_PER_MILLION = 2.00


def is_gpt5_mini(model: str | None) -> bool:
    """Return whether a provider model ID refers to GPT-5 mini."""
    normalized = str(model or "").strip().lower().replace("_", "-")
    return "gpt-5-mini" in normalized


def _encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _text_tokens(encoding: Any, text: Any) -> int:
    return len(encoding.encode(str(text or "")))


def _chat_input_tokens(
    encoding: Any,
    system_tokens: int,
    user_prompt: str,
) -> int:
    # Small allowance for the two message wrappers and assistant priming.
    return system_tokens + _text_tokens(encoding, user_prompt) + 10


def estimate_gpt5_mini_run_cost(df: pd.DataFrame, config: RunConfig) -> dict[str, int | float]:
    """Estimate tokens and USD cost for the run represented by ``config``.

    Loaded prompts and message histories are counted directly. Since future
    message-level JSON does not exist yet, its size is approximated from the
    active message output schema when estimating conversation-level inputs.
    Output size is likewise estimated from each active output schema.
    """
    encoding = _encoding(config.api.model)
    groups = get_conversation_groups(df)
    if config.selected_conversation_ids is not None:
        wanted = {str(value) for value in config.selected_conversation_ids}
        groups = [group for group in groups if str(group[0]) in wanted]
    elif config.max_conversations is not None:
        groups = groups[: config.max_conversations]

    target_role = str(config.message_target_role or "agent").strip().lower()
    if target_role not in {"agent", "customer"}:
        target_role = "agent"

    truncate_chars = config.max_chars_per_message if config.truncate_messages else None
    message_system = config.message_prompt.build_system()
    conversation_system = config.conversation_prompt.build_system()
    message_system_tokens = _text_tokens(encoding, message_system)
    conversation_system_tokens = _text_tokens(encoding, conversation_system)
    estimated_message_output_tokens = min(
        _text_tokens(encoding, config.message_prompt.output_schema),
        int(config.api.max_tokens),
    )
    estimated_conversation_output_tokens = min(
        _text_tokens(encoding, config.conversation_prompt.output_schema),
        int(config.api.max_tokens),
    )

    input_tokens = 0
    output_tokens = 0
    message_calls = 0
    conversation_calls = 0

    for conversation_id, group in groups:
        records = message_records_from_group(group, conversation_id)
        metadata = conversation_metadata_from_group(group)
        targets = [record for record in records if record.get("sender_role") == target_role]
        if config.max_agent_messages_per_conv is not None:
            targets = targets[: config.max_agent_messages_per_conv]

        for target in targets:
            history = [
                record
                for record in records
                if record.get("message_index") is not None
                and target.get("message_index") is not None
                and record["message_index"] <= target["message_index"]
                and (
                    config.include_unknown_in_history
                    or record.get("sender_role") != "unknown"
                )
            ]
            payload = build_message_level_payload(
                conversation_id=conversation_id,
                target_message=target,
                history=history,
                conversation_metadata=metadata,
                truncate_chars=truncate_chars,
            )
            input_tokens += _chat_input_tokens(
                encoding,
                message_system_tokens,
                config.message_prompt.build_user(payload),
            )
            output_tokens += estimated_message_output_tokens
            message_calls += 1

        full_transcript = (
            records
            if config.include_unknown_in_history
            else [record for record in records if record.get("sender_role") != "unknown"]
        )
        computed_metadata = compute_metadata([], records)
        computed_metadata["evaluation_target_role"] = target_role
        computed_metadata["target_messages_evaluated"] = len(targets)
        conversation_metadata = dict(metadata)
        conversation_metadata["evaluation_target_role"] = target_role
        conversation_payload = build_conversation_level_payload(
            conversation_id=conversation_id,
            conversation_metadata=conversation_metadata,
            full_transcript=full_transcript,
            message_level_evaluations=[],
            computed_metadata=computed_metadata,
            truncate_chars=truncate_chars,
        )
        input_tokens += _chat_input_tokens(
            encoding,
            conversation_system_tokens,
            config.conversation_prompt.build_user(conversation_payload),
        )
        # The real conversation payload carries each message evaluation twice:
        # inline beside its message and in message_level_evaluations.
        input_tokens += len(targets) * estimated_message_output_tokens * 2
        output_tokens += estimated_conversation_output_tokens
        conversation_calls += 1

    input_cost = input_tokens / 1_000_000 * GPT5_MINI_INPUT_PER_MILLION
    output_cost = output_tokens / 1_000_000 * GPT5_MINI_OUTPUT_PER_MILLION
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "input_cost": float(input_cost),
        "output_cost": float(output_cost),
        "total_cost": float(input_cost + output_cost),
        "message_calls": message_calls,
        "conversation_calls": conversation_calls,
    }
