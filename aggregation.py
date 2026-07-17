"""Compute conversation-level metadata from message-level evaluations, plus dashboard aggregations."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


FRUSTRATION_ORDER = ["none", "low", "medium", "high", "cancellation_risk"]
FRUSTRATION_RANK = {v: i for i, v in enumerate(FRUSTRATION_ORDER)}


def _is_human_agent_transcript_message(message: dict) -> bool:
    """Return whether a transcript message was sent by a human agent."""
    raw_role = str(message.get("raw_sender_role") or "").strip().lower()
    sender_role = str(message.get("sender_role") or "").strip().lower()
    return raw_role == "agent" or sender_role == "agent"


def _agent_presence_summary(
    transcript: list[dict],
    *,
    fallback_conversation_id: str = "",
) -> dict[str, Any]:
    """Summarize human-agent presence at journey and source-conversation level."""
    source_stats: dict[str, dict[str, Any]] = {}
    has_human_agent = False

    for message in transcript or []:
        if not isinstance(message, dict):
            continue
        source_id = str(
            message.get("source_conversation_id")
            or message.get("conversation_id")
            or fallback_conversation_id
            or ""
        ).strip()
        if not source_id:
            source_id = "__unknown__"

        stats = source_stats.setdefault(source_id, {"message_count": 0, "has_agent": False})
        stats["message_count"] += 1
        if _is_human_agent_transcript_message(message):
            stats["has_agent"] = True
            has_human_agent = True

    agentless_source_ids = [
        source_id
        for source_id, stats in source_stats.items()
        if int(stats.get("message_count") or 0) > 2 and not bool(stats.get("has_agent"))
    ]

    return {
        "has_human_agent_in_journey": has_human_agent,
        "has_agentless_source_conversation_over_2_messages": bool(agentless_source_ids),
        "agentless_source_conversation_ids_over_2_messages": " | ".join(agentless_source_ids),
    }


def humanize_label(value: Any) -> str:
    """Render enum/metric identifiers as readable labels."""
    text = str(value or "").strip()
    if not text:
        return ""
    special = {
        "many": "Many Issues",
        "zero_minimal": "Zero/Minimal issues",
        "good": "Good",
        "bad": "Bad",
    }
    normalized = text.lower().replace(" ", "_").replace("-", "_")
    if normalized in special:
        return special[normalized]
    text = text.replace("_", " ")
    text = " ".join(text.split())
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()



def _flatten_conversation_score(score: Any) -> dict[str, Any]:
    """Flatten model-produced conversation score into table-friendly columns."""
    if not isinstance(score, dict):
        return {}

    score_values = [
        score.get("resolution_score"),
        score.get("context_understanding_score"),
        score.get("customer_effort_score"),
        score.get("trust_frustration_risk_score", score.get("frustration_risk_score")),
        score.get("ai_judgment_score"),
        score.get("message_signal_score"),
        score.get("raw_total_score"),
        score.get("final_score"),
        score.get("final_score_100"),
    ]
    has_real_score = any(v not in (None, "", "none", "None") for v in score_values)
    if not has_real_score:
        return {}

    all_zero = True
    for value in score_values:
        if value in (None, "", "none", "None"):
            continue
        try:
            if float(value) != 0.0:
                all_zero = False
                break
        except (TypeError, ValueError):
            all_zero = False
            break
    is_new_score = any(
        key in score
        for key in ("ai_judgment_score", "message_signal_score", "final_score_100", "score_version")
    )
    if all_zero and not is_new_score and not str(score.get("score_explanation", "") or "").strip():
        return {}

    return {
        "score_resolution": score.get("resolution_score"),
        "score_context_understanding": score.get("context_understanding_score"),
        "score_customer_effort": score.get("customer_effort_score"),
        "score_frustration_risk": score.get(
            "trust_frustration_risk_score",
            score.get("frustration_risk_score"),
        ),
        "score_ai_judgment": score.get("ai_judgment_score"),
        "score_message_signals": score.get("message_signal_score"),
        "score_raw_total": score.get("raw_total_score"),
        "score_final": score.get("final_score"),
        "score_final_100": score.get("final_score_100"),
        "score_rating": score.get("score_rating"),
        "score_explanation": score.get("score_explanation"),
    }


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _sender_entity(record: dict | None) -> str:
    if not isinstance(record, dict):
        return "unknown"
    raw_role = _norm_text(record.get("raw_sender_role"))
    if raw_role == "system":
        return "broadcast"
    if raw_role in {"bot", "assistant"}:
        return "bot"
    if raw_role == "agent":
        return "agent"
    sender_role = _norm_text(record.get("sender_role"))
    if sender_role in {"customer", "agent"}:
        return sender_role
    return "unknown"


def _build_message_level_summary(message_evaluations: list[dict], message_records: list[dict]) -> dict:
    records_by_idx: dict[Any, dict] = {}
    for record in message_records or []:
        idx = record.get("message_index")
        if idx is not None:
            records_by_idx[idx] = record

    valid_evals = [e for e in message_evaluations if e.get("parse_status") == "ok" and e.get("parsed_json")]
    parsed = [e["parsed_json"] for e in valid_evals]

    effect_counter = Counter(p.get("message_level_effect", "neutral") for p in parsed)
    type_counter = Counter(p.get("issue_type", "none") for p in parsed)
    origin_counter = Counter(p.get("issue_origin", "none") for p in parsed)
    entity_counter: Counter[str] = Counter()
    issue_entity_counter: Counter[str] = Counter()
    major_entity_counter: Counter[str] = Counter()

    for evaluation in valid_evals:
        parsed_json = evaluation["parsed_json"]
        idx = parsed_json.get("message_index", evaluation.get("message_index"))
        record = records_by_idx.get(idx)
        entity = _sender_entity(record)
        entity_counter[entity] += 1
        if parsed_json.get("message_level_effect") in {"minor_issue", "major_issue"}:
            issue_entity_counter[entity] += 1
        if parsed_json.get("message_level_effect") == "major_issue":
            major_entity_counter[entity] += 1

    issue_count = int(effect_counter.get("minor_issue", 0) + effect_counter.get("major_issue", 0))
    major_issue_count = int(effect_counter.get("major_issue", 0))
    minor_issue_count = int(effect_counter.get("minor_issue", 0))
    recovered_issue_count = int(effect_counter.get("recovered_issue", 0))
    unresolved_issue_count = max(issue_count - recovered_issue_count, 0)
    evaluated_message_count = int(len(valid_evals))
    denominator = max(evaluated_message_count, 1)
    repetition_count = int(type_counter.get("repetition", 0))

    rates = {
        "major_issue_rate": round(major_issue_count / denominator, 4),
        "minor_issue_rate": round(minor_issue_count / denominator, 4),
        "unresolved_rate": round(unresolved_issue_count / denominator, 4),
        "repetition_rate": round(repetition_count / denominator, 4),
        "recovery_rate": round(recovered_issue_count / denominator, 4),
    }
    weights = {
        "major_issue_rate": 0.35,
        "minor_issue_rate": 0.20,
        "unresolved_rate": 0.25,
        "repetition_rate": 0.10,
        "recovery_rate": 0.10,
    }
    weighted_components = {
        "major_issue_rate": round(weights["major_issue_rate"] * rates["major_issue_rate"], 4),
        "minor_issue_rate": round(weights["minor_issue_rate"] * rates["minor_issue_rate"], 4),
        "unresolved_rate": round(weights["unresolved_rate"] * rates["unresolved_rate"], 4),
        "repetition_rate": round(weights["repetition_rate"] * rates["repetition_rate"], 4),
        "recovery_rate_credit": round(weights["recovery_rate"] * rates["recovery_rate"], 4),
    }
    burden = (
        weighted_components["major_issue_rate"]
        + weighted_components["minor_issue_rate"]
        + weighted_components["unresolved_rate"]
        + weighted_components["repetition_rate"]
        - weighted_components["recovery_rate_credit"]
    )
    burden = round(max(0.0, burden), 4)
    concrete_score = round(max(0.0, min(10.0, 10.0 * (1.0 - burden))), 1)

    summary_parts = [
        f"{issue_count} red flags",
        f"{major_issue_count} major",
        f"{minor_issue_count} minor",
        f"{recovered_issue_count} recovered",
        f"{unresolved_issue_count} still unresolved",
    ]
    top_issue_types = [
        issue_type
        for issue_type, count in type_counter.most_common()
        if issue_type not in {"none", ""} and count > 0
    ][:3]
    if top_issue_types:
        summary_parts.append(f"top issue types: {', '.join(top_issue_types)}")

    return {
        "evaluated_message_count": evaluated_message_count,
        "issue_count": issue_count,
        "major_issue_count": major_issue_count,
        "minor_issue_count": minor_issue_count,
        "recovered_issue_count": recovered_issue_count,
        "unresolved_issue_count": int(unresolved_issue_count),
        "effect_counts": dict(effect_counter),
        "issue_type_counts": dict(type_counter),
        "issue_origin_counts": dict(origin_counter),
        "evaluated_sender_entity_counts": dict(entity_counter),
        "issue_sender_entity_counts": dict(issue_entity_counter),
        "major_issue_sender_entity_counts": dict(major_entity_counter),
        "message_signal_score": concrete_score,
        "message_signal_trace": {
            "formula_type": "rate_based_v1",
            "rates": rates,
            "weights": weights,
            "weighted_components": weighted_components,
            "burden": burden,
            "final_score": concrete_score,
        },
        "summary_text": "; ".join(summary_parts),
    }


def _message_result_is_red(message_result: dict) -> bool:
    """Match the UI's red-severity rules for one message-level result."""
    if not message_result:
        return False
    if message_result.get("parse_status") != "ok":
        return True
    parsed = message_result.get("parsed_json") or {}
    effect = parsed.get("message_level_effect")
    frustration = parsed.get("frustration_level_after_message")
    change = parsed.get("frustration_change")
    issue_type = parsed.get("issue_type") or "none"
    issue_origin = parsed.get("issue_origin") or "none"
    has_issue = effect in {"minor_issue", "major_issue"} or issue_type != "none" or issue_origin != "none"
    if effect == "major_issue" or frustration in {"high", "cancellation_risk"}:
        return True
    if change == "created" and has_issue:
        return True
    return False


def _broadcast_only_red_issue_flag(
    message_evaluations: list[dict],
    message_records: list[dict],
) -> bool:
    """Return True when the journey's only red message is a system/broadcast row."""
    if not message_evaluations or not message_records:
        return False

    records_by_idx: dict[Any, dict] = {}
    for record in message_records:
        idx = record.get("message_index")
        if idx is not None and idx not in records_by_idx:
            records_by_idx[idx] = record

    red_results = [message for message in message_evaluations if _message_result_is_red(message)]
    if len(red_results) != 1:
        return False

    target = red_results[0]
    idx = target.get("message_index")
    if idx is None:
        parsed = target.get("parsed_json") or {}
        idx = parsed.get("message_index")
    record = records_by_idx.get(idx)
    if not record:
        return False

    return _norm_text(record.get("raw_sender_role")) == "system"


def _normalize_handled_status(cl: dict) -> str | None:
    handled = _norm_text(cl.get("handled_status"))
    if handled in {"handled", "unhandled"}:
        return handled
    old_class = str(cl.get("final_classification", "") or "").strip().lower()
    if old_class.startswith(("unhandled", "not handled")):
        return "unhandled"
    if old_class.startswith("handled"):
        return "handled"
    return None


def _normalize_customer_experience(cl: dict) -> str | None:
    experience = _norm_text(cl.get("customer_experience"))
    old_severity = _norm_text(cl.get("cx_issue_severity"))
    old_class = str(cl.get("final_classification", "") or "").strip().lower()
    legacy_bad = old_severity == "many" or any(
        marker in old_class for marker in ("many", "caused", "frustration")
    )
    if legacy_bad:
        return "bad"
    if experience in {"good", "bad"}:
        return experience
    if old_severity in {"zero_minimal", "minimal", "none"} or "minimal" in old_class:
        return "good"
    return None


def _normalize_origin(value: Any) -> str | None:
    origin = _norm_text(value)
    if origin in {"our_side", "customer_side", "shared", "none"}:
        return origin
    if origin in {"our", "agent", "company", "business"}:
        return "our_side"
    if origin == "customer":
        return "customer_side"
    return None


def _normalize_frustration_detected(cl: dict) -> bool:
    raw = cl.get("frustration_detected")
    if isinstance(raw, bool):
        return raw
    if _norm_text(raw) in {"true", "yes", "1", "frustrated"}:
        return True
    old_class = str(cl.get("final_classification", "") or "")
    return "frustration" in old_class.lower()


def _normalize_frustration_origin(cl: dict, main_issue: dict) -> str:
    origin = _normalize_origin(cl.get("frustration_origin"))
    if origin:
        return origin
    origin = _normalize_origin(cl.get("main_issue_origin") or main_issue.get("issue_origin"))
    if origin:
        return origin
    old_class = str(cl.get("final_classification", "") or "").lower()
    if _normalize_frustration_detected(cl) and "caused" in old_class:
        return "our_side"
    return "none"



def _max_frustration(levels: list[str]) -> str:
    rank = -1
    out = "none"
    for lv in levels:
        r = FRUSTRATION_RANK.get(lv, -1)
        if r > rank:
            rank = r
            out = lv if lv in FRUSTRATION_RANK else out
    return out


def compute_metadata(
    message_evaluations: list[dict],
    message_records: list[dict],
) -> dict:
    """Build the computed_metadata block expected by the conversation-level evaluator."""
    total = len(message_records)
    customer = sum(1 for m in message_records if m.get("sender_role") == "customer")
    agent = sum(1 for m in message_records if m.get("sender_role") == "agent")
    unknown = total - customer - agent

    valid_evals = [e for e in message_evaluations if e.get("parse_status") == "ok" and e.get("parsed_json")]
    parsed = [e["parsed_json"] for e in valid_evals]

    frustration_levels = [p.get("frustration_level_after_message", "none") for p in parsed]
    effects = [p.get("message_level_effect", "neutral") for p in parsed]
    issue_types = [p.get("issue_type", "none") for p in parsed]
    issue_origins = [p.get("issue_origin", "none") for p in parsed]

    issue_count = sum(1 for e in effects if e in ("minor_issue", "major_issue"))
    major = sum(1 for e in effects if e == "major_issue")
    minor = sum(1 for e in effects if e == "minor_issue")
    recovered = sum(1 for e in effects if e == "recovered_issue")

    type_counter = Counter(issue_types)
    origin_counter = Counter(issue_origins)
    message_summary = _build_message_level_summary(message_evaluations, message_records)

    first_frustration_idx: Any = None
    first_major_idx: Any = None
    for e in valid_evals:
        pj = e["parsed_json"]
        fl = pj.get("frustration_level_after_message", "none")
        idx = pj.get("message_index", e.get("message_index"))
        if first_frustration_idx is None and FRUSTRATION_RANK.get(fl, 0) >= FRUSTRATION_RANK["low"]:
            first_frustration_idx = idx
        if first_major_idx is None and pj.get("message_level_effect") == "major_issue":
            first_major_idx = idx
        if first_frustration_idx is not None and first_major_idx is not None:
            break

    cancellation = any(
        p.get("frustration_level_after_message") == "cancellation_risk" for p in parsed
    )
    broadcast_only_red_issue = _broadcast_only_red_issue_flag(message_evaluations, message_records)

    return {
        "total_messages": int(total),
        "customer_messages": int(customer),
        "agent_messages": int(agent),
        "unknown_messages": int(unknown),
        "agent_messages_evaluated": int(len(valid_evals)),
        "max_frustration_level": _max_frustration(frustration_levels),
        "issue_count": int(issue_count),
        "major_issue_count": int(major),
        "minor_issue_count": int(minor),
        "recovered_issue_count": int(recovered),
        "repetition_count": int(type_counter.get("repetition", 0)),
        "unclear_guidance_count": int(type_counter.get("unclear_guidance", 0)),
        "ignored_context_count": int(type_counter.get("ignored_context", 0)),
        "missing_next_step_count": int(type_counter.get("missing_next_step", 0)),
        "wrong_info_count": int(type_counter.get("wrong_info", 0)),
        "dead_end_count": int(type_counter.get("dead_end", 0)),
        "customer_side_issue_count": int(origin_counter.get("customer_side", 0)),
        "our_side_issue_count": int(origin_counter.get("our_side", 0)),
        "shared_issue_count": int(origin_counter.get("shared", 0)),
        "first_frustration_message_index": first_frustration_idx,
        "first_major_issue_message_index": first_major_idx,
        "cancellation_risk_detected": bool(cancellation),
        "broadcast_only_red_issue": bool(broadcast_only_red_issue),
        "message_level_summary": message_summary,
        "message_signal_score": message_summary.get("message_signal_score"),
    }


def flatten_conversation_row(
    conv_result: dict,
    conversation_metadata: dict,
    computed_metadata: dict,
) -> dict:
    """Flatten one conversation's results into a single CSV-friendly row."""
    cl = conv_result.get("parsed_json", {}) or {}
    main_issue = cl.get("main_issue", {}) or {}
    if not isinstance(main_issue, dict):
        main_issue = {}
    handled_status = _normalize_handled_status(cl)
    customer_experience = _normalize_customer_experience(cl)
    frustration_detected = _normalize_frustration_detected(cl)
    frustration_origin = _normalize_frustration_origin(cl, main_issue)
    main_issue_origin = (
        _normalize_origin(main_issue.get("issue_origin") or cl.get("main_issue_origin"))
        or "none"
    )
    main_issue_type = main_issue.get("issue_type", cl.get("main_issue_type"))
    main_issue_summary = main_issue.get("issue_summary", cl.get("main_issue_summary"))
    customer_impact = main_issue.get("customer_impact", cl.get("customer_impact"))
    culprits = cl.get("culprits") or []
    if not isinstance(culprits, list):
        culprits = []
    culprit_agent_names = cl.get("culprit_agent_names") or []
    if not isinstance(culprit_agent_names, list):
        culprit_agent_names = []

    def get_md(*keys: str) -> Any:
        for k in keys:
            if k in conversation_metadata and conversation_metadata[k] not in (None, ""):
                return conversation_metadata[k]
        return None

    transcript = conv_result.get("transcript") or []
    conversation_id = str(conv_result.get("conversation_id", "") or "")
    parent_journey_id = get_md("parent_journey_id") or (
        conversation_id.split("::ticket_", 1)[0]
        if "::ticket_" in conversation_id
        else conversation_id
    )
    ticket_id = get_md("ticket_id") or (
        f"ticket_{conversation_id.split('::ticket_', 1)[1]}"
        if "::ticket_" in conversation_id
        else None
    )
    ticket_label = str(ticket_id or "").strip()
    if ticket_label and ticket_label.lower().startswith("ticket_"):
        suffix = ticket_label.split("_", 1)[1]
        ticket_label = f"Ticket {suffix}"
    elif ticket_label:
        ticket_label = humanize_label(ticket_label)

    journey_starter = None
    if isinstance(transcript, list) and transcript and isinstance(transcript[0], dict):
        first_message = transcript[0]
        journey_starter = str(
            first_message.get("raw_sender_role")
            or first_message.get("sender_role")
            or "unknown"
        ).strip().lower()
        journey_starter = {
            "customer": "consumer",
            "assistant": "bot",
        }.get(journey_starter, journey_starter)

    row = {
        "conversation_id": conversation_id,
        "customer_journey_id": conversation_id,
        "parent_journey_id": parent_journey_id,
        "ticket_id": ticket_id,
        "ticket_label": ticket_label,
        "ticket_type": get_md("ticket_type"),
        "ticket_status": get_md("ticket_status"),
        "ticket_objective": get_md("ticket_objective"),
        "ticket_segmentation_reason": get_md("ticket_segmentation_reason"),
        "ticket_should_append_future": get_md("ticket_should_append_future"),
        "journey_starter": journey_starter,
        **_agent_presence_summary(
            transcript if isinstance(transcript, list) else [],
            fallback_conversation_id=conversation_id,
        ),
        "customer_name": get_md("customer_name"),
        "customer_phone": get_md("customer_phone"),
        "source_conversation_ids": get_md("source_conversation_ids", "conversation_ids"),
        "source_conversation_count": get_md("source_conversation_count"),
        "conversation_start_date": get_md("conversation_start_date"),
        "conversation_end_date": get_md("conversation_end_date"),
        "conversation_status": get_md("conversation_status"),
        "customer_objective_type": cl.get("customer_objective_type"),
        "customer_primary_objective": cl.get("customer_primary_objective"),
        "final_classification": cl.get("final_classification"),
        "cx_issue_severity": cl.get("cx_issue_severity"),
        "handled_status": handled_status,
        "customer_experience": customer_experience,
        "frustration_detected": frustration_detected,
        "frustration_origin": frustration_origin,
        "customer_started_frustrated": cl.get("customer_started_frustrated"),
        "customer_became_frustrated_during_chat": cl.get("customer_became_frustrated_during_chat"),
        "customer_ended_frustrated": cl.get("customer_ended_frustrated"),
        "frustration_timing": cl.get("frustration_timing"),
        "unhandled_resolution_subtype": cl.get("unhandled_resolution_subtype"),
        "final_customer_sentiment": cl.get("final_customer_sentiment"),
        "max_frustration_level": cl.get("max_frustration_level"),
        "main_issue_exists": main_issue.get("issue_exists"),
        "main_issue_origin": main_issue_origin,
        "main_issue_type": main_issue_type,
        "main_issue_summary": main_issue_summary,
        "customer_impact": customer_impact,
        "all_detected_issues": " | ".join(
            [
                f"{i.get('issue_type', '')}: {i.get('issue_summary', '')}".strip(": ")
                for i in (cl.get("all_detected_issues") or [])
                if isinstance(i, dict)
            ]
        ),
        "positive_signals": " | ".join(cl.get("positive_signals", []) or []),
        "negative_signals": " | ".join(cl.get("negative_signals", []) or []),
        "classification_reason": cl.get("classification_reason"),
        "management_summary": cl.get("management_summary"),
        "culprits": " | ".join(str(c) for c in culprits if c),
        "culprit_agent_names": " | ".join(str(name) for name in culprit_agent_names if name),
        "culprit_reason": cl.get("culprit_reason"),
        "recommended_actions": " | ".join(cl.get("recommended_actions", []) or []),
        "manual_review_required": cl.get("manual_review_required"),
        "manual_review_reason": cl.get("manual_review_reason"),
        "confidence": cl.get("confidence"),
        "parse_status": conv_result.get("parse_status"),
        "error_message": conv_result.get("error_message"),
    }
    # Append computed metadata fields directly.
    cm_fields = [
        "total_messages",
        "customer_messages",
        "unknown_messages",
        "issue_count",
        "major_issue_count",
        "minor_issue_count",
        "recovered_issue_count",
        "repetition_count",
        "unclear_guidance_count",
        "ignored_context_count",
        "missing_next_step_count",
        "wrong_info_count",
        "dead_end_count",
        "customer_side_issue_count",
        "our_side_issue_count",
        "shared_issue_count",
        "cancellation_risk_detected",
        "broadcast_only_red_issue",
        "message_signal_score",
    ]
    for f in cm_fields:
        row[f] = computed_metadata.get(f)
    # Older saved runs may predate this computed marker or contain a stale
    # value. Recalculate it from the persisted transcript and message results
    # whenever the platform rebuilds its conversation table.
    message_level_results = conv_result.get("message_level_results") or []
    if transcript and message_level_results:
        row["broadcast_only_red_issue"] = _broadcast_only_red_issue_flag(
            message_level_results,
            transcript,
        )
    row.update(_flatten_conversation_score(cl.get("conversation_score")))
    return row


def flatten_message_row(message_result: dict) -> dict:
    """Flatten one message-level evaluation into a CSV-friendly row."""
    pj = message_result.get("parsed_json") or {}
    return {
        "conversation_id": message_result.get("conversation_id", ""),
        "customer_journey_id": message_result.get("conversation_id", ""),
        "source_conversation_id": message_result.get("source_conversation_id"),
        "target_message_id": message_result.get("target_message_id", ""),
        "appended_message_index": message_result.get("message_index"),
        "message_index": message_result.get("message_index"),
        "message_time": message_result.get("message_time"),
        "target_message_text": message_result.get("target_message_text"),
        "message_level_effect": pj.get("message_level_effect"),
        "frustration_level_after_message": pj.get("frustration_level_after_message"),
        "frustration_change": pj.get("frustration_change"),
        "customer_effort_level": pj.get("customer_effort_level"),
        "clarity_level": pj.get("clarity_level"),
        "context_handling": pj.get("context_handling"),
        "issue_origin": pj.get("issue_origin"),
        "issue_type": pj.get("issue_type"),
        "contradiction": pj.get("contradiction"),
        "first_contradiction_message_id": pj.get("first_contradiction_message_id"),
        "contradiction_debug_message": pj.get("contradiction_debug_message"),
        "contradiction_agent_issue_suppressed": pj.get("contradiction_agent_issue_suppressed"),
        "contradiction_source_penalized": pj.get("contradiction_source_penalized"),
        "contradiction_transfer_skipped": pj.get("contradiction_transfer_skipped"),
        "generated_from_contradiction_transfer": pj.get("generated_from_contradiction_transfer"),
        "frustration_cause": pj.get("frustration_cause"),
        "evidence": pj.get("evidence"),
        "business_impact": pj.get("business_impact"),
        "recommended_fix": pj.get("recommended_fix"),
        "parse_status": message_result.get("parse_status"),
        "error_message": message_result.get("error_message"),
    }


def build_conversation_table(conversation_rows: list[dict]) -> pd.DataFrame:
    """Build a tidy DataFrame from flattened conversation rows."""
    if not conversation_rows:
        return pd.DataFrame()
    return pd.DataFrame(conversation_rows)


def build_message_table(message_rows: list[dict]) -> pd.DataFrame:
    """Build a tidy DataFrame from flattened message rows."""
    if not message_rows:
        return pd.DataFrame()
    return pd.DataFrame(message_rows)


def dashboard_aggregates(conv_df: pd.DataFrame) -> dict:
    """Compute the headline numbers and chart-ready breakdowns shown on the dashboard."""
    if conv_df.empty:
        return {
            "total": 0,
            "handled_pct": 0.0,
            "unhandled_pct": 0.0,
            "many_issues_pct": 0.0,
            "high_frustration_count": 0,
            "cancellation_risk_count": 0,
            "manual_review_count": 0,
            "classification_counts": {},
            "experience_counts": {},
            "unhandled_subtype_counts": {},
            "frustration_origin_counts": {},
            "issue_origin_counts": {},
            "issue_type_counts": {},
        }

    total = int(len(conv_df))

    def norm_series(col: str, default: str = "") -> pd.Series:
        if col not in conv_df.columns:
            return pd.Series([default] * len(conv_df), index=conv_df.index)
        return (
            conv_df[col]
            .fillna(default)
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

    def customer_experience_series() -> pd.Series:
        series = norm_series("customer_experience")
        series = series.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
        old_severity = norm_series("cx_issue_severity")
        final_class = norm_series("final_classification")
        legacy_bad = (old_severity == "many") | final_class.str.contains(
            "many|caused|frustration",
            regex=True,
        )
        return series.mask(legacy_bad, "bad")

    def safe_pct(col: str, value: Any) -> float:
        series = norm_series(col)
        if col == "customer_experience":
            series = customer_experience_series()
        return float((series == value).sum()) / total * 100.0 if total else 0.0

    handled_pct = safe_pct("handled_status", "handled")
    unhandled_pct = safe_pct("handled_status", "unhandled")
    many_issues_pct = safe_pct("customer_experience", "bad")

    high_frustration_count = 0
    if "max_frustration_level" in conv_df.columns:
        high_frustration_count = int(
            conv_df["max_frustration_level"].isin(["high", "cancellation_risk"]).sum()
        )

    cancellation_risk_count = 0
    if "cancellation_risk_detected" in conv_df.columns:
        cancellation_risk_count = int(
            conv_df["cancellation_risk_detected"]
            .map(lambda value: str(value if value is not None else False).strip().lower() in {"true", "1", "yes", "y"})
            .sum()
        )
    elif "max_frustration_level" in conv_df.columns:
        cancellation_risk_count = int((conv_df["max_frustration_level"] == "cancellation_risk").sum())

    manual_review_count = 0
    if "manual_review_required" in conv_df.columns:
        manual_review_count = int(
            conv_df["manual_review_required"]
            .map(lambda value: str(value if value is not None else False).strip().lower() in {"true", "1", "yes", "y"})
            .sum()
        )

    classification_counts = {}
    if "final_classification" in conv_df.columns:
        classification_counts = (
            conv_df["final_classification"].fillna("Unknown").value_counts().to_dict()
        )

    experience_counts = {}
    if "customer_experience" in conv_df.columns:
        experience_counts = customer_experience_series().replace({"": "unknown"}).value_counts().to_dict()

    unhandled_subtype_counts = {}
    if "unhandled_resolution_subtype" in conv_df.columns:
        subtype_series = conv_df["unhandled_resolution_subtype"].fillna("unknown")
        subtype_series = subtype_series[
            subtype_series.astype(str).str.strip().str.lower() != "not_applicable"
        ]
        unhandled_subtype_counts = subtype_series.value_counts().to_dict()

    issue_origin_counts = {}
    if "main_issue_origin" in conv_df.columns:
        issue_origin_counts = (
            conv_df["main_issue_origin"].fillna("none").value_counts().to_dict()
        )

    frustration_origin_counts = {}
    if "frustration_origin" in conv_df.columns:
        origin_series = norm_series("frustration_origin", "none")
        origin_series = origin_series.replace({"customer": "customer_side", "our": "our_side", "agent": "our_side"})
        frustration_origin_counts = origin_series.value_counts().to_dict()

    issue_type_counts = {}
    if "main_issue_type" in conv_df.columns:
        issue_type_counts = (
            conv_df["main_issue_type"].fillna("none").value_counts().to_dict()
        )

    return {
        "total": total,
        "handled_pct": handled_pct,
        "unhandled_pct": unhandled_pct,
        "many_issues_pct": many_issues_pct,
        "high_frustration_count": high_frustration_count,
        "cancellation_risk_count": cancellation_risk_count,
        "manual_review_count": manual_review_count,
        "classification_counts": classification_counts,
        "experience_counts": experience_counts,
        "unhandled_subtype_counts": unhandled_subtype_counts,
        "frustration_origin_counts": frustration_origin_counts,
        "issue_origin_counts": issue_origin_counts,
        "issue_type_counts": issue_type_counts,
    }


def top_frustration_causes(message_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return a DataFrame of the most common frustration causes from message-level evaluations."""
    if message_df.empty or "frustration_cause" not in message_df.columns:
        return pd.DataFrame(columns=["frustration_cause", "count"])
    series = message_df["frustration_cause"].fillna("none").astype(str).str.strip().str.lower()
    series = series[~series.isin(["none", "", "nan"])]
    if series.empty:
        return pd.DataFrame(columns=["frustration_cause", "count"])
    counts = series.value_counts().head(top_n).reset_index()
    counts.columns = ["frustration_cause", "count"]
    return counts
