"""Evaluation orchestration: message-level and conversation-level runs.

Includes robust JSON extraction and schema validation, plus a single entry point
``run_evaluation`` that drives the full pipeline with progress callbacks and
graceful per-conversation error handling.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from api_client import APIConfig, MAX_CONCURRENCY, chat_completion
from prompts import (
    DEFAULT_CONVERSATION_LEVEL_PROMPT,
    DEFAULT_MESSAGE_LEVEL_PROMPT,
    PromptTemplate,
    build_conversation_level_payload,
    build_message_level_payload,
)
from aggregation import compute_metadata
from data_loader import (
    conversation_metadata_from_group,
    get_conversation_groups,
    message_records_from_group,
)


# ---------- JSON robustness ----------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Best-effort extraction of a single JSON object from a model response."""
    if not text:
        raise ValueError("Empty model response")
    text = text.strip()

    # Plain JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Code fence
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Greedy first { ... } block
    m = _FIRST_OBJ_RE.search(text)
    if m:
        candidate = m.group(0).strip()
        # Try progressively trimming from the right end if there is trailing junk.
        for end in range(len(candidate), 0, -1):
            try:
                obj = json.loads(candidate[:end])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    raise ValueError("Could not extract a JSON object from the model response")


# ---------- Schema validators / normalizers ----------

_ML_ENUMS = {
    "message_level_effect": {"helped", "neutral", "minor_issue", "major_issue", "recovered_issue"},
    "frustration_level_after_message": {"none", "low", "medium", "high", "cancellation_risk"},
    "frustration_change": {"decreased", "unchanged", "increased", "created"},
    "customer_effort_level": {"low", "medium", "high"},
    "clarity_level": {"clear", "somewhat_clear", "unclear"},
    "context_handling": {"good", "partial", "poor", "not_applicable"},
    "issue_origin": {"our_side", "customer_side", "shared", "none"},
    "issue_type": {
        "none",
        "misunderstanding",
        "repetition",
        "delay",
        "unclear_guidance",
        "wrong_info",
        "ignored_context",
        "dead_end",
        "tool_or_system_failure",
        "poor_tone",
        "missing_next_step",
        "other",
    },
}

_ML_DEFAULTS = {
    "message_level_effect": "neutral",
    "frustration_level_after_message": "none",
    "frustration_change": "unchanged",
    "customer_effort_level": "low",
    "clarity_level": "clear",
    "context_handling": "not_applicable",
    "issue_origin": "none",
    "issue_type": "none",
    "frustration_cause": "none",
    "evidence": "",
    "business_impact": "",
    "recommended_fix": "",
}

_FORBIDDEN_ID_FIELDS = {
    "conversation_id",
    "thread_id",
    "run_id",
    "customer_id",
    "customer_phone",
    "phone_number",
    "target_message_id",
}


def validate_message_level_result(data: dict) -> dict:
    """Coerce a parsed message-level JSON object into the strict schema shape.

    Well-known fields are normalized to the dashboard's expected enums. Any
    additional fields produced by a custom schema are preserved so that
    downstream consumers (Debug tab, JSON export) can still see them.
    """
    if not isinstance(data, dict):
        raise ValueError("Message-level result is not a JSON object")

    out: dict[str, Any] = {}
    try:
        out["message_index"] = int(data.get("message_index") or 0)
    except (TypeError, ValueError):
        out["message_index"] = 0

    for field_name, allowed in _ML_ENUMS.items():
        val = str(data.get(field_name, "") or "").strip().lower().replace(" ", "_")
        if val not in allowed:
            val = _ML_DEFAULTS[field_name]
        out[field_name] = val

    for field_name in ("frustration_cause", "evidence", "business_impact", "recommended_fix"):
        out[field_name] = str(data.get(field_name) or _ML_DEFAULTS[field_name]) or _ML_DEFAULTS[field_name]

    # Preserve any fields the user's custom schema produced.
    for k, v in data.items():
        if k not in out and k not in _FORBIDDEN_ID_FIELDS:
            out[k] = v

    return out


_CL_CLASSIFICATION_RULES = {
    "Handled with Minimal Issues": ("handled", "zero_minimal", False, False),
    "Handled with Many Issues": ("handled", "many", False, False),
    "Handled with Minimal Issues and Frustration": ("handled", "zero_minimal", True, False),
    "Handled with Many Issues and Frustration": ("handled", "many", True, False),
    "Handled with Minimal Caused Issues and Frustration": ("handled", "zero_minimal", True, True),
    "Handled with Many Caused Issues and Frustration": ("handled", "many", True, True),
    "Not Handled with Minimal Issues": ("unhandled", "zero_minimal", False, False),
    "Not Handled with Many Issues": ("unhandled", "many", False, False),
    "Not Handled with Minimal Issues and Frustration": ("unhandled", "zero_minimal", True, False),
    "Not Handled with Many Issues and Frustration": ("unhandled", "many", True, False),
    "Not Handled with Minimal Caused Issues and Frustration": ("unhandled", "zero_minimal", True, True),
    "Not Handled with Many Caused Issues and Frustration": ("unhandled", "many", True, True),
}

_OLD_CL_CLASSIFICATION_RULES = {
    "Handled with Zero/Minimal Issues": ("handled", "zero_minimal", "not_applicable"),
    "Handled with Many Issues": ("handled", "many", "not_applicable"),
    "Unhandled with Zero/Minimal Issues - Totally Definitive Unresolved": (
        "unhandled",
        "zero_minimal",
        "totally_unresolved",
    ),
    "Unhandled with Zero/Minimal Issues - Pending Unresolved": (
        "unhandled",
        "zero_minimal",
        "pending_unresolved",
    ),
    "Unhandled with Many Issues - Totally Definitive Unresolved": (
        "unhandled",
        "many",
        "totally_unresolved",
    ),
    "Unhandled with Many Issues - Pending Unresolved": (
        "unhandled",
        "many",
        "pending_unresolved",
    ),
}

_UNHANDLED_SUBTYPES = {
    "not_applicable",
    "totally_unresolved",
    "pending_unresolved",
}

_MAIN_ISSUE_ORIGINS = {"our_side", "customer_side", "shared", "third_party", "unclear", "none"}
_TOP_LEVEL_MAIN_ISSUE_ORIGINS = {"our_side", "customer_side", "shared", "unclear", "none"}
_FRUSTRATION_TIMINGS = {"start", "during", "end", "multiple", "none"}


def _normalize_bool_flag(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n", "", "none", "null"}:
        return False
    return default


def _normalize_issue_origin(value: Any, *, allow_third_party: bool) -> str:
    origin = str(value or "").strip().lower().replace(" ", "_")
    if origin not in _MAIN_ISSUE_ORIGINS:
        return "none"
    if origin == "third_party" and not allow_third_party:
        return "unclear"
    return origin


def _normalize_unhandled_subtype(value: Any) -> str:
    subtype = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if subtype in {"n/a", "na", "none", "not_applicable"}:
        return "not_applicable"
    if subtype in {
        "totally_unresolved",
        "totally_definitive_unresolved",
        "definitive_unresolved",
        "definitive",
        "totally",
    }:
        return "totally_unresolved"
    if subtype in {"pending_unresolved", "pending"}:
        return "pending_unresolved"
    return ""


def _normalize_frustration_timing(value: Any) -> str:
    timing = str(value or "").strip().lower().replace(" ", "_")
    return timing if timing in _FRUSTRATION_TIMINGS else ""


def _infer_frustration_timing(started: bool, became: bool, ended: bool) -> str:
    active = [started, became, ended]
    if not any(active):
        return "none"
    if sum(1 for flag in active if flag) > 1:
        return "multiple"
    if started:
        return "start"
    if became:
        return "during"
    return "end"


def _classification_from_parts(
    handled_status: str,
    severity: str,
    frustration_detected: bool,
    main_issue_origin: str,
) -> str:
    if handled_status == "handled":
        if not frustration_detected:
            return "Handled with Many Issues" if severity == "many" else "Handled with Minimal Issues"
        if main_issue_origin == "our_side":
            return (
                "Handled with Many Caused Issues and Frustration"
                if severity == "many"
                else "Handled with Minimal Caused Issues and Frustration"
            )
        return (
            "Handled with Many Issues and Frustration"
            if severity == "many"
            else "Handled with Minimal Issues and Frustration"
        )
    if not frustration_detected:
        return "Not Handled with Many Issues" if severity == "many" else "Not Handled with Minimal Issues"
    if main_issue_origin == "our_side":
        return (
            "Not Handled with Many Caused Issues and Frustration"
            if severity == "many"
            else "Not Handled with Minimal Caused Issues and Frustration"
        )
    return (
        "Not Handled with Many Issues and Frustration"
        if severity == "many"
        else "Not Handled with Minimal Issues and Frustration"
    )


def _normalize_quantifiable_metrics(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []

    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        metrics = item.get("metrics") or {}
        if not category or not isinstance(metrics, dict):
            continue

        clean_metrics: dict[str, float | int] = {}
        for key, raw in metrics.items():
            metric_name = str(key or "").strip()
            if not metric_name:
                continue
            try:
                num = float(raw)
            except (TypeError, ValueError):
                num = 0.0
            clean_metrics[metric_name] = int(num) if num.is_integer() else num
        out.append({"category": category, "metrics": clean_metrics})
    return out


def validate_conversation_level_result(data: dict) -> dict:
    """Coerce a parsed conversation-level JSON object into the strict schema shape."""
    if not isinstance(data, dict):
        raise ValueError("Conversation-level result is not a JSON object")

    out: dict[str, Any] = {}

    objective_type = str(data.get("customer_objective_type", "") or "").strip()
    if objective_type not in {"Inquiry", "Issue"}:
        objective_type = "Inquiry"
    out["customer_objective_type"] = objective_type
    out["customer_primary_objective"] = str(data.get("customer_primary_objective", "") or "")

    classification = str(data.get("final_classification", "") or "").strip()
    handled_status = str(data.get("handled_status", "") or "").strip().lower()
    severity = str(data.get("cx_issue_severity", "") or "").strip().lower().replace(" ", "_")
    subtype = _normalize_unhandled_subtype(data.get("unhandled_resolution_subtype"))

    main = data.get("main_issue") or {}
    if not isinstance(main, dict):
        main = {}
    main_out = {
        "issue_exists": _normalize_bool_flag(main.get("issue_exists", False)),
        "issue_origin": _normalize_issue_origin(main.get("issue_origin", "none"), allow_third_party=True),
        "issue_type": str(main.get("issue_type", "none") or "none").strip().lower(),
        "issue_summary": str(main.get("issue_summary", "") or ""),
        "customer_impact": str(main.get("customer_impact", "") or ""),
    }

    main_issue_origin = _normalize_issue_origin(
        data.get("main_issue_origin", main_out["issue_origin"]),
        allow_third_party=False,
    )
    if main_issue_origin not in _TOP_LEVEL_MAIN_ISSUE_ORIGINS:
        main_issue_origin = "none"

    frustration_detected = _normalize_bool_flag(data.get("frustration_detected"), default=False)
    customer_started_frustrated = _normalize_bool_flag(data.get("customer_started_frustrated"), default=False)
    customer_became_frustrated_during_chat = _normalize_bool_flag(
        data.get("customer_became_frustrated_during_chat"),
        default=False,
    )
    customer_ended_frustrated = _normalize_bool_flag(data.get("customer_ended_frustrated"), default=False)
    frustration_timing = _normalize_frustration_timing(data.get("frustration_timing"))

    if classification in _CL_CLASSIFICATION_RULES:
        handled_status, severity, frustration_detected, caused_by_us = _CL_CLASSIFICATION_RULES[classification]
        if caused_by_us:
            main_issue_origin = "our_side"
        elif frustration_detected and main_issue_origin == "our_side":
            main_issue_origin = "shared"
    else:
        if classification in _OLD_CL_CLASSIFICATION_RULES:
            handled_status, severity, subtype = _OLD_CL_CLASSIFICATION_RULES[classification]
        else:
            # Backward-compatible recovery for older or partial outputs.
            if handled_status not in {"handled", "unhandled"}:
                handled_status = "handled" if classification.startswith("Handled") else "unhandled"
            if severity not in {"zero_minimal", "many"}:
                severity = "many" if "Many" in classification else "zero_minimal"
            if handled_status == "handled":
                subtype = "not_applicable"
            elif subtype not in {"totally_unresolved", "pending_unresolved"}:
                subtype = "pending_unresolved" if "pending" in classification.lower() else "totally_unresolved"
            if "Frustration" in classification and not frustration_detected:
                frustration_detected = True
            if "Caused Issues and Frustration" in classification:
                main_issue_origin = "our_side"

        if severity not in {"zero_minimal", "many"}:
            severity = "zero_minimal"
        if handled_status not in {"handled", "unhandled"}:
            handled_status = "handled"

    if handled_status == "handled":
        subtype = "not_applicable"
    elif subtype == "not_applicable" or subtype not in _UNHANDLED_SUBTYPES:
        subtype = "totally_unresolved"

    # Promote issue origin from main_issue when the top-level field is missing or unusable.
    if main_issue_origin == "none" and main_out["issue_origin"] in _TOP_LEVEL_MAIN_ISSUE_ORIGINS:
        main_issue_origin = main_out["issue_origin"]
    if main_issue_origin == "unclear" and main_out["issue_origin"] in {"our_side", "customer_side", "shared"}:
        main_issue_origin = main_out["issue_origin"]

    if not frustration_detected:
        customer_started_frustrated = False
        customer_became_frustrated_during_chat = False
        customer_ended_frustrated = False
        frustration_timing = "none"
    else:
        if frustration_timing:
            if frustration_timing == "start":
                customer_started_frustrated = True
            elif frustration_timing == "during":
                customer_became_frustrated_during_chat = True
            elif frustration_timing == "end":
                customer_ended_frustrated = True
            elif frustration_timing == "multiple":
                customer_started_frustrated = True
                customer_became_frustrated_during_chat = True
                customer_ended_frustrated = True
        elif not any(
            [
                customer_started_frustrated,
                customer_became_frustrated_during_chat,
                customer_ended_frustrated,
            ]
        ):
            customer_became_frustrated_during_chat = True
        frustration_timing = _infer_frustration_timing(
            customer_started_frustrated,
            customer_became_frustrated_during_chat,
            customer_ended_frustrated,
        )

    if severity not in {"zero_minimal", "many"}:
        severity = "zero_minimal"
    classification = _classification_from_parts(
        handled_status,
        severity,
        frustration_detected,
        main_issue_origin,
    )

    out["final_classification"] = classification
    out["handled_status"] = handled_status
    out["cx_issue_severity"] = severity
    out["frustration_detected"] = frustration_detected
    out["customer_started_frustrated"] = customer_started_frustrated
    out["customer_became_frustrated_during_chat"] = customer_became_frustrated_during_chat
    out["customer_ended_frustrated"] = customer_ended_frustrated
    out["frustration_timing"] = frustration_timing
    out["main_issue_origin"] = main_issue_origin
    out["unhandled_resolution_subtype"] = subtype

    sentiment = str(data.get("final_customer_sentiment", "") or "").strip().lower()
    if sentiment not in {"satisfied", "neutral", "frustrated", "confused", "dissatisfied", "unknown"}:
        sentiment = "unknown"
    out["final_customer_sentiment"] = sentiment

    max_fl = str(data.get("max_frustration_level", "") or "").strip().lower()
    if max_fl not in {"none", "low", "medium", "high", "cancellation_risk"}:
        max_fl = "none"
    out["max_frustration_level"] = max_fl

    if not main_out["issue_exists"]:
        main_out["issue_origin"] = "none"
        main_out["issue_type"] = "none"
        main_out["issue_summary"] = "none"
        main_out["customer_impact"] = "none"
    out["main_issue"] = main_out

    detected = data.get("all_detected_issues") or []
    if not isinstance(detected, list):
        detected = []
    out["all_detected_issues"] = [
        {
            "issue_origin": _normalize_issue_origin(d.get("issue_origin", ""), allow_third_party=True),
            "issue_type": str(d.get("issue_type", "") or ""),
            "issue_summary": str(d.get("issue_summary", "") or ""),
            "evidence": str(d.get("evidence", "") or ""),
            "impact": str(d.get("impact", "") or ""),
        }
        for d in detected
        if isinstance(d, dict)
    ]

    out["positive_signals"] = [str(x) for x in (data.get("positive_signals") or []) if x]
    out["negative_signals"] = [str(x) for x in (data.get("negative_signals") or []) if x]
    out["quantifiable_metrics"] = _normalize_quantifiable_metrics(data.get("quantifiable_metrics"))
    out["management_summary"] = str(data.get("management_summary", "") or "")
    out["recommended_actions"] = [str(x) for x in (data.get("recommended_actions") or []) if x]
    out["manual_review_required"] = _normalize_bool_flag(data.get("manual_review_required"), default=False)
    out["manual_review_reason"] = str(data.get("manual_review_reason", "") or "")
    if not out["manual_review_required"] and not out["manual_review_reason"].strip():
        out["manual_review_reason"] = "none"
    confidence = str(data.get("confidence", "") or "").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    out["confidence"] = confidence

    # Preserve any extra fields a custom schema may have introduced.
    for k, v in data.items():
        if k not in out and k not in _FORBIDDEN_ID_FIELDS:
            out[k] = v

    return out


# ---------- Run orchestration ----------


@dataclass
class RunConfig:
    api: APIConfig = field(default_factory=APIConfig)
    max_conversations: Optional[int] = None
    max_agent_messages_per_conv: Optional[int] = None
    truncate_messages: bool = False
    max_chars_per_message: int = 1500
    include_unknown_in_history: bool = True
    stop_on_error: bool = False
    save_raw_responses: bool = True
    # Which messages the judge evaluates as targets:
    #   "agent"    — judge the agent's response to a (possibly frustrated) customer message
    #   "customer" — judge the customer's state / frustration before the agent answers
    message_target_role: str = "agent"
    # Explicit set of conversation IDs to run on. When non-None, takes
    # precedence over ``max_conversations`` — used by the random sampler.
    selected_conversation_ids: Optional[list[str]] = None
    # Editable prompts (defaults to the in-memory defaults; the app loads
    # the active prompts from the DB before each run).
    message_prompt: PromptTemplate = field(default_factory=lambda: DEFAULT_MESSAGE_LEVEL_PROMPT)
    conversation_prompt: PromptTemplate = field(default_factory=lambda: DEFAULT_CONVERSATION_LEVEL_PROMPT)


@dataclass
class RunResults:
    conversation_results: list[dict] = field(default_factory=list)
    message_level_results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: Optional[float] = None


def _eval_message_level(
    client,
    api: APIConfig,
    conversation_id: str,
    target_record: dict,
    history_records: list[dict],
    conversation_metadata: dict,
    save_raw: bool,
    truncate_chars: Optional[int],
    prompt: PromptTemplate,
) -> dict:
    """Run one message-level evaluation. Always returns a record (success or failure)."""
    payload = build_message_level_payload(
        conversation_id=conversation_id,
        target_message=target_record,
        history=history_records,
        conversation_metadata=conversation_metadata,
        truncate_chars=truncate_chars,
    )
    system_prompt = prompt.build_system()
    user_prompt = prompt.build_user(payload)

    record: dict[str, Any] = {
        "thread_id": conversation_id,
        "conversation_id": conversation_id,
        "target_message_id": target_record.get("message_id", ""),
        "message_index": target_record.get("message_index"),
        "message_time": target_record.get("message_time", ""),
        "target_message_text": target_record.get("message_text", ""),
        "input_history": history_records if save_raw else None,
        "raw_model_response": None,
        "parsed_json": None,
        "evaluation_output": None,
        "parse_status": "ok",
        "error_message": None,
        "debug": None,
    }

    try:
        raw, debug = chat_completion(client, api, system_prompt, user_prompt)
        if save_raw:
            record["raw_model_response"] = raw
            record["debug"] = debug
        try:
            obj = extract_json_object(raw)
            validated = validate_message_level_result(obj)
            if not validated.get("message_index") and record["message_index"] is not None:
                try:
                    validated["message_index"] = int(record["message_index"])
                except (TypeError, ValueError):
                    pass
            record["parsed_json"] = validated
            record["evaluation_output"] = validated
        except Exception as je:
            record["parse_status"] = "failed"
            record["error_message"] = f"JSON parse failed: {je}"
    except Exception as e:
        record["parse_status"] = "api_error"
        record["error_message"] = f"API call failed: {e}"

    return record


def _eval_conversation_level(
    client,
    api: APIConfig,
    conversation_id: str,
    conversation_metadata: dict,
    full_transcript: list[dict],
    message_level_evaluations: list[dict],
    computed_metadata: dict,
    save_raw: bool,
    truncate_chars: Optional[int],
    prompt: PromptTemplate,
) -> dict:
    """Run one conversation-level evaluation. Always returns a record."""
    payload = build_conversation_level_payload(
        conversation_id=conversation_id,
        conversation_metadata=conversation_metadata,
        full_transcript=full_transcript,
        message_level_evaluations=[
            e["parsed_json"] for e in message_level_evaluations if e.get("parsed_json")
        ],
        computed_metadata=computed_metadata,
        truncate_chars=truncate_chars,
    )
    system_prompt = prompt.build_system()
    user_prompt = prompt.build_user(payload)

    record: dict[str, Any] = {
        "thread_id": conversation_id,
        "conversation_id": conversation_id,
        "run_id": None,
        "raw_model_response": None,
        "parsed_json": None,
        "evaluation_output": None,
        "parse_status": "ok",
        "error_message": None,
        "debug": None,
    }

    try:
        raw, debug = chat_completion(client, api, system_prompt, user_prompt)
        if save_raw:
            record["raw_model_response"] = raw
            record["debug"] = debug
        try:
            obj = extract_json_object(raw)
            validated = validate_conversation_level_result(obj)
            record["parsed_json"] = validated
            record["evaluation_output"] = validated
        except Exception as je:
            record["parse_status"] = "failed"
            record["error_message"] = f"JSON parse failed: {je}"
    except Exception as e:
        record["parse_status"] = "api_error"
        record["error_message"] = f"API call failed: {e}"

    return record


def run_evaluation(
    df: pd.DataFrame,
    client,
    config: RunConfig,
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
    on_message_result: Optional[Callable[[dict], None]] = None,
    on_conversation_result: Optional[Callable[[dict], None]] = None,
    on_error: Optional[Callable[[dict], None]] = None,
) -> RunResults:
    """Run the full message-level + conversation-level evaluation pipeline.

    All AI calls — both message-level and conversation-level, across all
    conversations — share ONE ``ThreadPoolExecutor`` whose worker count equals
    ``config.api.concurrency`` (clamped to ``MAX_CONCURRENCY``). The instant a
    worker is free it picks the next pending task from the queue, regardless of
    which conversation it belongs to. As soon as the *last* message-level call
    for a given conversation completes, that conversation's conversation-level
    call is submitted to the same pool — no cross-conversation barrier.

    Optional persistence callbacks (``on_message_result``,
    ``on_conversation_result``, ``on_error``) are invoked on the calling thread
    as each record finishes, so the app can write incrementally to its DB.

    Calls ``on_progress`` with a small status dict at each step. All callbacks
    fire on the calling thread; only the OpenAI API calls run in worker threads.
    """
    results = RunResults(started_at=time.time())
    truncate_chars = config.max_chars_per_message if config.truncate_messages else None

    workers = max(1, min(int(getattr(config.api, "concurrency", 1) or 1), MAX_CONCURRENCY))

    target_role = (config.message_target_role or "agent").strip().lower()
    if target_role not in ("agent", "customer"):
        target_role = "agent"

    groups = get_conversation_groups(df)
    # Selection precedence: explicit IDs (random sampler) > max_conversations slice.
    if config.selected_conversation_ids is not None:
        wanted = set(str(x) for x in config.selected_conversation_ids)
        groups = [g for g in groups if str(g[0]) in wanted]
    elif config.max_conversations is not None:
        groups = groups[: config.max_conversations]

    total_conversations = len(groups)
    if on_progress:
        on_progress(
            {
                "phase": "start",
                "total_conversations": total_conversations,
                "workers": workers,
            }
        )

    # ---- Pre-build per-conversation state on the main thread ----------------

    def visible_history_of(records: list[dict], up_to_index: Any) -> list[dict]:
        out = []
        for r in records:
            idx = r["message_index"]
            if idx is None:
                continue
            if idx > up_to_index:
                break
            role = r.get("sender_role", "unknown")
            if role == "unknown" and not config.include_unknown_in_history:
                continue
            out.append(r)
        return out

    conv_state: dict[str, dict[str, Any]] = {}
    conv_order: list[str] = []
    ml_tasks: list[tuple[str, dict, list[dict]]] = []  # (conversation_id, target_record, history)
    no_target_convs: list[str] = []

    for ci, (conversation_id, group) in enumerate(groups, start=1):
        records = message_records_from_group(group, conversation_id)
        conversation_metadata = conversation_metadata_from_group(group)
        targets = [r for r in records if r.get("sender_role") == target_role]
        if config.max_agent_messages_per_conv is not None:
            targets = targets[: config.max_agent_messages_per_conv]

        state = {
            "conversation_id": conversation_id,
            "conversation_index": ci,
            "records": records,
            "conversation_metadata": conversation_metadata,
            "targets": targets,
            "results_by_idx": {},          # message_index -> message-level record
            "ml_total": len(targets),
            "ml_done": 0,
            "cl_submitted": False,
            "cl_done": False,
        }
        conv_state[conversation_id] = state
        conv_order.append(conversation_id)

        if on_progress:
            on_progress(
                {
                    "phase": "conversation_start",
                    "conversation_index": ci,
                    "conversation_id": conversation_id,
                    "agent_messages": len(targets),
                    "target_messages": len(targets),
                    "target_role": target_role,
                    "total_conversations": total_conversations,
                    "workers": workers,
                }
            )

        if not targets:
            no_target_convs.append(conversation_id)
            continue

        for target in targets:
            history = visible_history_of(records, target["message_index"])
            ml_tasks.append((conversation_id, target, history))

    # ---- One shared pool drives everything ---------------------------------

    stop_signal = {"flag": False, "reason": None}

    def _submit_cl(ex: cf.ThreadPoolExecutor, conversation_id: str) -> cf.Future:
        """Build the conversation-level payload and submit it to the pool."""
        state = conv_state[conversation_id]
        message_results_ordered = [
            state["results_by_idx"][t["message_index"]]
            for t in state["targets"]
            if t["message_index"] in state["results_by_idx"]
        ]
        computed_md = compute_metadata(message_results_ordered, state["records"])
        computed_md["evaluation_target_role"] = target_role
        computed_md["target_messages_evaluated"] = sum(
            1 for m in message_results_ordered if m.get("parse_status") == "ok"
        )
        full_transcript = (
            state["records"] if config.include_unknown_in_history
            else [r for r in state["records"] if r.get("sender_role") != "unknown"]
        )
        conv_md_for_judge = dict(state["conversation_metadata"])
        conv_md_for_judge["evaluation_target_role"] = target_role

        state["message_results_ordered"] = message_results_ordered
        state["computed_metadata"] = computed_md
        state["full_transcript"] = full_transcript
        state["cl_submitted"] = True

        return ex.submit(
            _eval_conversation_level,
            client=client,
            api=config.api,
            conversation_id=conversation_id,
            conversation_metadata=conv_md_for_judge,
            full_transcript=full_transcript,
            message_level_evaluations=message_results_ordered,
            computed_metadata=computed_md,
            save_raw=config.save_raw_responses,
            truncate_chars=truncate_chars,
            prompt=config.conversation_prompt,
        )

    def _finalize_cl_record(conversation_id: str, cr: dict) -> dict:
        state = conv_state[conversation_id]
        cr["thread_id"] = conversation_id
        cr["conversation_metadata"] = state["conversation_metadata"]
        cr["computed_metadata"] = state["computed_metadata"]
        cr["transcript"] = state["records"]
        cr["message_level_results"] = state["message_results_ordered"]
        cr["evaluation_target_role"] = target_role
        if cr.get("parse_status") != "ok" and not cr.get("parsed_json"):
            # Inject a stub so the dashboard still has a row for this conversation.
            cr["parsed_json"] = {
                "customer_objective_type": "Inquiry",
                "customer_primary_objective": "",
                "final_classification": "Not Handled with Many Caused Issues and Frustration",
                "handled_status": "unhandled",
                "cx_issue_severity": "many",
                "frustration_detected": True,
                "customer_started_frustrated": False,
                "customer_became_frustrated_during_chat": True,
                "customer_ended_frustrated": False,
                "frustration_timing": "during",
                "main_issue_origin": "our_side",
                "unhandled_resolution_subtype": "totally_unresolved",
                "final_customer_sentiment": "unknown",
                "max_frustration_level": state["computed_metadata"].get("max_frustration_level", "none"),
                "main_issue": {
                    "issue_exists": True,
                    "issue_origin": "our_side",
                    "issue_type": "other",
                    "issue_summary": "Conversation-level evaluator failed to parse",
                    "customer_impact": "Unable to assess automatically",
                },
                "all_detected_issues": [],
                "positive_signals": [],
                "negative_signals": [],
                "quantifiable_metrics": [],
                "management_summary": "Automatic evaluation could not parse a result for this conversation. Manual review required.",
                "recommended_actions": ["Review this conversation manually."],
                "manual_review_required": True,
                "manual_review_reason": cr.get("error_message") or "Parse failure",
                "confidence": "low",
            }
        cr["evaluation_output"] = cr.get("parsed_json")
        return cr

    fut_info: dict[cf.Future, dict] = {}
    pending: set[cf.Future] = set()

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        # 1. Submit every message-level task across every conversation up front.
        for conversation_id, target, history in ml_tasks:
            fut = ex.submit(
                _eval_message_level,
                client=client,
                api=config.api,
                conversation_id=conversation_id,
                target_record=target,
                history_records=history,
                conversation_metadata=conv_state[conversation_id]["conversation_metadata"],
                save_raw=config.save_raw_responses,
                truncate_chars=truncate_chars,
                prompt=config.message_prompt,
            )
            pending.add(fut)
            fut_info[fut] = {"type": "ml", "conversation_id": conversation_id, "target": target}

        # 2. Conversations with no target messages can run their CL immediately.
        for conversation_id in no_target_convs:
            fut = _submit_cl(ex, conversation_id)
            pending.add(fut)
            fut_info[fut] = {"type": "cl", "conversation_id": conversation_id}

        # 3. Drain. As each ML finishes, check whether its conversation's CL is
        #    now ready to fire; if so submit it to the same pool. As each CL
        #    finishes, record the conversation result.
        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                info = fut_info.pop(fut)
                conversation_id = info["conversation_id"]
                state = conv_state[conversation_id]

                if info["type"] == "ml":
                    target = info["target"]
                    try:
                        mr = fut.result()
                    except Exception as e:  # noqa: BLE001
                        mr = {
                            "conversation_id": conversation_id,
                            "target_message_id": target.get("message_id", ""),
                            "message_index": target.get("message_index"),
                            "message_time": target.get("message_time", ""),
                            "target_message_text": target.get("message_text", ""),
                            "input_history": None,
                            "raw_model_response": None,
                            "parsed_json": None,
                            "parse_status": "api_error",
                            "error_message": f"Worker raised: {e}",
                            "debug": None,
                        }
                    state["results_by_idx"][target["message_index"]] = mr
                    state["ml_done"] += 1

                    if on_message_result:
                        try:
                            on_message_result(mr)
                        except Exception:
                            pass

                    if mr.get("parse_status") != "ok":
                        err = {
                            "level": "message",
                            "conversation_id": conversation_id,
                            "message_index": target.get("message_index"),
                            "error": mr.get("error_message"),
                        }
                        results.errors.append(err)
                        if on_error:
                            try:
                                on_error(err)
                            except Exception:
                                pass

                    if on_progress:
                        on_progress(
                            {
                                "phase": "message_done",
                                "conversation_index": state["conversation_index"],
                                "conversation_id": conversation_id,
                                "message_index": target.get("message_index"),
                                "message_in_conversation": state["ml_done"],
                                "total_in_conversation": state["ml_total"],
                                "status": mr.get("parse_status"),
                            }
                        )

                    if config.stop_on_error and mr.get("parse_status") == "api_error":
                        stop_signal["flag"] = True
                        stop_signal["reason"] = mr.get("error_message")
                    if cancel_requested and cancel_requested():
                        stop_signal["flag"] = True
                        stop_signal["reason"] = stop_signal["reason"] or "cancelled"

                    # Submit this conversation's CL now if its ML batch is complete.
                    if (
                        not stop_signal["flag"]
                        and not state["cl_submitted"]
                        and state["ml_done"] >= state["ml_total"]
                    ):
                        cl_fut = _submit_cl(ex, conversation_id)
                        pending.add(cl_fut)
                        fut_info[cl_fut] = {"type": "cl", "conversation_id": conversation_id}

                elif info["type"] == "cl":
                    try:
                        cr = fut.result()
                    except Exception as e:  # noqa: BLE001
                        cr = {
                            "conversation_id": conversation_id,
                            "raw_model_response": None,
                            "parsed_json": None,
                            "parse_status": "api_error",
                            "error_message": f"Worker raised: {e}",
                            "debug": None,
                        }
                    cr = _finalize_cl_record(conversation_id, cr)
                    state["cl_done"] = True

                    if cr.get("parse_status") != "ok":
                        err = {
                            "level": "conversation",
                            "conversation_id": conversation_id,
                            "error": cr.get("error_message"),
                        }
                        results.errors.append(err)
                        if on_error:
                            try:
                                on_error(err)
                            except Exception:
                                pass
                        if config.stop_on_error and cr.get("parse_status") == "api_error":
                            stop_signal["flag"] = True
                            stop_signal["reason"] = cr.get("error_message")

                    results.conversation_results.append(cr)
                    if on_conversation_result:
                        try:
                            on_conversation_result(cr)
                        except Exception:
                            pass

                    if on_progress:
                        on_progress(
                            {
                                "phase": "conversation_done",
                                "conversation_index": state["conversation_index"],
                                "conversation_id": conversation_id,
                                "total_conversations": total_conversations,
                                "status": cr.get("parse_status"),
                            }
                        )

            if stop_signal["flag"]:
                # Cancel anything that hasn't started yet and drop the rest.
                for f in list(pending):
                    if not f.done():
                        f.cancel()
                    pending.discard(f)
                if on_progress:
                    on_progress(
                        {
                            "phase": "stopped_on_error",
                            "error": stop_signal["reason"],
                        }
                    )
                break

    # Sort outputs by the original conversation order, then by message_index,
    # so the dashboard and exports stay deterministic regardless of completion
    # order in the streaming pool.
    order_by_cid = {cid: i for i, cid in enumerate(conv_order)}
    results.conversation_results.sort(
        key=lambda c: order_by_cid.get(c.get("conversation_id"), 0)
    )

    # Flatten per-conversation ordered message results into the global list.
    results.message_level_results = []
    for cid in conv_order:
        state = conv_state.get(cid, {})
        ordered = state.get("message_results_ordered")
        if ordered is None:
            ordered = [
                state.get("results_by_idx", {})[t["message_index"]]
                for t in state.get("targets", [])
                if t["message_index"] in state.get("results_by_idx", {})
            ]
        results.message_level_results.extend(ordered)

    results.finished_at = time.time()
    if on_progress:
        on_progress({"phase": "done", "total_conversations": total_conversations})
    return results
