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


def validate_message_level_result(data: dict) -> dict:
    """Coerce a parsed message-level JSON object into the strict schema shape.

    Well-known fields are normalized to the dashboard's expected enums. Any
    additional fields produced by a custom schema are preserved so that
    downstream consumers (Debug tab, JSON export) can still see them.
    """
    if not isinstance(data, dict):
        raise ValueError("Message-level result is not a JSON object")

    out: dict[str, Any] = {}
    out["conversation_id"] = str(data.get("conversation_id", ""))
    out["target_message_id"] = str(data.get("target_message_id", ""))
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
        if k not in out:
            out[k] = v

    return out


_CL_CLASSIFICATIONS = {
    "Handled with Zero/Minimal Issues",
    "Handled with Many Issues",
    "Unhandled with Zero/Minimal Issues",
    "Unhandled with Many Issues",
}


def validate_conversation_level_result(data: dict) -> dict:
    """Coerce a parsed conversation-level JSON object into the strict schema shape."""
    if not isinstance(data, dict):
        raise ValueError("Conversation-level result is not a JSON object")

    out: dict[str, Any] = {}
    out["conversation_id"] = str(data.get("conversation_id", ""))

    objective_type = str(data.get("customer_objective_type", "") or "").strip()
    if objective_type not in {"Inquiry", "Issue"}:
        objective_type = "Inquiry"
    out["customer_objective_type"] = objective_type
    out["customer_primary_objective"] = str(data.get("customer_primary_objective", "") or "")

    classification = str(data.get("final_classification", "") or "").strip()
    if classification not in _CL_CLASSIFICATIONS:
        classification = "Unhandled with Many Issues" if "many" in classification.lower() else "Handled with Zero/Minimal Issues"
        classification = classification if classification in _CL_CLASSIFICATIONS else "Handled with Zero/Minimal Issues"
    out["final_classification"] = classification

    handled_status = str(data.get("handled_status", "") or "").strip().lower()
    if handled_status not in {"handled", "unhandled"}:
        handled_status = "handled" if classification.startswith("Handled") else "unhandled"
    out["handled_status"] = handled_status

    severity = str(data.get("cx_issue_severity", "") or "").strip().lower().replace(" ", "_")
    if severity not in {"zero_minimal", "many"}:
        severity = "many" if "Many" in classification else "zero_minimal"
    out["cx_issue_severity"] = severity

    sentiment = str(data.get("final_customer_sentiment", "") or "").strip().lower()
    if sentiment not in {"satisfied", "neutral", "frustrated", "confused", "dissatisfied", "unknown"}:
        sentiment = "unknown"
    out["final_customer_sentiment"] = sentiment

    max_fl = str(data.get("max_frustration_level", "") or "").strip().lower()
    if max_fl not in {"none", "low", "medium", "high", "cancellation_risk"}:
        max_fl = "none"
    out["max_frustration_level"] = max_fl

    main = data.get("main_issue") or {}
    if not isinstance(main, dict):
        main = {}
    main_out = {
        "issue_exists": bool(main.get("issue_exists", False)),
        "issue_origin": str(main.get("issue_origin", "none") or "none").strip().lower(),
        "issue_type": str(main.get("issue_type", "none") or "none").strip().lower(),
        "issue_summary": str(main.get("issue_summary", "") or ""),
        "customer_impact": str(main.get("customer_impact", "") or ""),
    }
    if main_out["issue_origin"] not in {"our_side", "customer_side", "shared", "none"}:
        main_out["issue_origin"] = "none"
    out["main_issue"] = main_out

    detected = data.get("all_detected_issues") or []
    if not isinstance(detected, list):
        detected = []
    out["all_detected_issues"] = [
        {
            "issue_origin": str(d.get("issue_origin", "") or ""),
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
    out["management_summary"] = str(data.get("management_summary", "") or "")
    out["recommended_actions"] = [str(x) for x in (data.get("recommended_actions") or []) if x]
    out["manual_review_required"] = bool(data.get("manual_review_required", False))
    out["manual_review_reason"] = str(data.get("manual_review_reason", "") or "")
    confidence = str(data.get("confidence", "") or "").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    out["confidence"] = confidence

    # Preserve any extra fields a custom schema may have introduced.
    for k, v in data.items():
        if k not in out:
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
        "conversation_id": conversation_id,
        "target_message_id": target_record.get("message_id", ""),
        "message_index": target_record.get("message_index"),
        "message_time": target_record.get("message_time", ""),
        "target_message_text": target_record.get("message_text", ""),
        "input_history": history_records if save_raw else None,
        "raw_model_response": None,
        "parsed_json": None,
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
            # Backfill keys the model may have skipped.
            validated["conversation_id"] = validated.get("conversation_id") or conversation_id
            validated["target_message_id"] = validated.get("target_message_id") or record["target_message_id"]
            if not validated.get("message_index") and record["message_index"] is not None:
                try:
                    validated["message_index"] = int(record["message_index"])
                except (TypeError, ValueError):
                    pass
            record["parsed_json"] = validated
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
        "conversation_id": conversation_id,
        "raw_model_response": None,
        "parsed_json": None,
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
            validated["conversation_id"] = validated.get("conversation_id") or conversation_id
            record["parsed_json"] = validated
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

    Message-level calls within each conversation run concurrently via a
    ``ThreadPoolExecutor``. Concurrency is taken from ``config.api.concurrency``
    and is clamped to ``MAX_CONCURRENCY``. Conversation-level calls remain
    sequential so each one can incorporate its own message-level metadata.

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
    if config.max_conversations is not None:
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

    for ci, (conversation_id, group) in enumerate(groups, start=1):
        if cancel_requested and cancel_requested():
            break

        records = message_records_from_group(group, conversation_id)
        conversation_metadata = conversation_metadata_from_group(group)

        # History should respect the toggle for unknown messages.
        def visible_history(up_to_index: int) -> list[dict]:
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

        agent_targets = [r for r in records if r.get("sender_role") == target_role]
        if config.max_agent_messages_per_conv is not None:
            agent_targets = agent_targets[: config.max_agent_messages_per_conv]

        if on_progress:
            on_progress(
                {
                    "phase": "conversation_start",
                    "conversation_index": ci,
                    "conversation_id": conversation_id,
                    "agent_messages": len(agent_targets),
                    "target_messages": len(agent_targets),
                    "target_role": target_role,
                    "total_conversations": total_conversations,
                    "workers": workers,
                }
            )

        # Pre-compute history slices so worker threads never touch shared state.
        tasks = [(t, visible_history(t["message_index"])) for t in agent_targets]
        message_results_by_idx: dict[Any, dict] = {}
        stop_signal = False

        if tasks:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_target = {}
                for target, history in tasks:
                    fut = ex.submit(
                        _eval_message_level,
                        client=client,
                        api=config.api,
                        conversation_id=conversation_id,
                        target_record=target,
                        history_records=history,
                        conversation_metadata=conversation_metadata,
                        save_raw=config.save_raw_responses,
                        truncate_chars=truncate_chars,
                        prompt=config.message_prompt,
                    )
                    future_to_target[fut] = target

                completed = 0
                for fut in cf.as_completed(future_to_target):
                    target = future_to_target[fut]
                    try:
                        mr = fut.result()
                    except Exception as e:  # noqa: BLE001 — wrap worker crashes
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
                    message_results_by_idx[target["message_index"]] = mr
                    completed += 1

                    if on_message_result:
                        try:
                            on_message_result(mr)
                        except Exception:
                            # Persistence errors must not abort the run.
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
                                "conversation_index": ci,
                                "conversation_id": conversation_id,
                                "message_index": target.get("message_index"),
                                "message_in_conversation": completed,
                                "total_in_conversation": len(tasks),
                                "status": mr.get("parse_status"),
                            }
                        )

                    if config.stop_on_error and mr.get("parse_status") == "api_error":
                        stop_signal = True
                    if cancel_requested and cancel_requested():
                        stop_signal = True

                    if stop_signal:
                        # Cancel un-started futures so the executor exits quickly.
                        for f in future_to_target:
                            if not f.done():
                                f.cancel()
                        break

        # Restore original message order for downstream code and append the
        # whole conversation's results in order so the global list stays sorted
        # by conversation, then message_index.
        message_results = [
            message_results_by_idx[t["message_index"]]
            for t in agent_targets
            if t["message_index"] in message_results_by_idx
        ]
        results.message_level_results.extend(message_results)

        if stop_signal:
            if on_progress:
                on_progress(
                    {
                        "phase": "stopped_on_error",
                        "conversation_id": conversation_id,
                    }
                )
            results.finished_at = time.time()
            return results

        # Compute metadata + conversation-level call.
        computed_md = compute_metadata(message_results, records)
        computed_md["evaluation_target_role"] = target_role
        computed_md["target_messages_evaluated"] = sum(
            1 for m in message_results if m.get("parse_status") == "ok"
        )
        full_transcript = (
            records if config.include_unknown_in_history
            else [r for r in records if r.get("sender_role") != "unknown"]
        )
        # Include the target role in metadata sent to the conversation-level judge
        # so it understands what the inline message_level_evaluation entries judged.
        conv_md_for_judge = dict(conversation_metadata)
        conv_md_for_judge["evaluation_target_role"] = target_role
        cr = _eval_conversation_level(
            client=client,
            api=config.api,
            conversation_id=conversation_id,
            conversation_metadata=conv_md_for_judge,
            full_transcript=full_transcript,
            message_level_evaluations=message_results,
            computed_metadata=computed_md,
            save_raw=config.save_raw_responses,
            truncate_chars=truncate_chars,
            prompt=config.conversation_prompt,
        )
        cr["conversation_metadata"] = conversation_metadata
        cr["computed_metadata"] = computed_md
        cr["transcript"] = records
        cr["message_level_results"] = message_results
        cr["evaluation_target_role"] = target_role

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
            # If parse failed, force manual_review_required = True downstream by
            # injecting a minimal stub so the dashboard still has a row.
            if not cr.get("parsed_json"):
                cr["parsed_json"] = {
                    "conversation_id": conversation_id,
                    "customer_objective_type": "Inquiry",
                    "customer_primary_objective": "",
                    "final_classification": "Unhandled with Many Issues",
                    "handled_status": "unhandled",
                    "cx_issue_severity": "many",
                    "final_customer_sentiment": "unknown",
                    "max_frustration_level": computed_md.get("max_frustration_level", "none"),
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
                    "management_summary": "Automatic evaluation could not parse a result for this conversation. Manual review required.",
                    "recommended_actions": ["Review this conversation manually."],
                    "manual_review_required": True,
                    "manual_review_reason": cr.get("error_message") or "Parse failure",
                    "confidence": "low",
                }
            if config.stop_on_error and cr.get("parse_status") == "api_error":
                results.conversation_results.append(cr)
                if on_conversation_result:
                    try:
                        on_conversation_result(cr)
                    except Exception:
                        pass
                if on_progress:
                    on_progress(
                        {
                            "phase": "stopped_on_error",
                            "conversation_id": conversation_id,
                            "error": cr.get("error_message"),
                        }
                    )
                results.finished_at = time.time()
                return results

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
                    "conversation_index": ci,
                    "conversation_id": conversation_id,
                    "total_conversations": total_conversations,
                    "status": cr.get("parse_status"),
                }
            )

    results.finished_at = time.time()
    if on_progress:
        on_progress({"phase": "done", "total_conversations": total_conversations})
    return results
