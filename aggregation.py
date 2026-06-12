"""Compute conversation-level metadata from message-level evaluations, plus dashboard aggregations."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


FRUSTRATION_ORDER = ["none", "low", "medium", "high", "cancellation_risk"]
FRUSTRATION_RANK = {v: i for i, v in enumerate(FRUSTRATION_ORDER)}


def _flatten_quantifiable_metrics(metrics: Any) -> dict[str, Any]:
    """Flatten categorized quantifiable metrics into table-friendly columns."""
    if not isinstance(metrics, list):
        return {}

    out: dict[str, Any] = {}
    for category_obj in metrics:
        if not isinstance(category_obj, dict):
            continue
        category = str(category_obj.get("category") or "").strip()
        values = category_obj.get("metrics") or {}
        if not category or not isinstance(values, dict):
            continue

        category_key = (
            category.lower()
            .replace("&", "and")
            .replace("/", " ")
            .replace("-", " ")
        )
        category_key = "_".join(part for part in category_key.split() if part)
        for metric_name, raw in values.items():
            metric_key = str(metric_name or "").strip()
            if not metric_key:
                continue
            col = f"metric__{category_key}__{metric_key}"
            try:
                num = float(raw)
            except (TypeError, ValueError):
                num = 0.0
            out[col] = int(num) if num.is_integer() else num
    return out


def quantifiable_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return flattened quantifiable metric columns in a stable order."""
    return sorted([c for c in df.columns if c.startswith("metric__")])


def metric_display_name(column: str) -> str:
    """Convert a flattened metric column name into a readable label."""
    parts = column.split("__")
    if len(parts) < 3:
        return column
    return parts[-1].replace("_", " ").title()


def metric_category_display_name(column: str) -> str:
    parts = column.split("__")
    if len(parts) < 3:
        return "Metrics"
    return parts[1].replace("_", " ").replace(" And ", " & ").title()


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
    }


def flatten_conversation_row(
    conv_result: dict,
    conversation_metadata: dict,
    computed_metadata: dict,
) -> dict:
    """Flatten one conversation's results into a single CSV-friendly row."""
    cl = conv_result.get("parsed_json", {}) or {}
    main_issue = cl.get("main_issue", {}) or {}

    def get_md(*keys: str) -> Any:
        for k in keys:
            if k in conversation_metadata and conversation_metadata[k] not in (None, ""):
                return conversation_metadata[k]
        return None

    row = {
        "conversation_id": conv_result.get("conversation_id", ""),
        "customer_name": get_md("customer_name"),
        "customer_phone": get_md("customer_phone"),
        "conversation_start_date": get_md("conversation_start_date"),
        "conversation_end_date": get_md("conversation_end_date"),
        "conversation_status": get_md("conversation_status"),
        "initial_skill": get_md("initial_skill"),
        "last_skill": get_md("last_skill"),
        "joined_skills": get_md("joined_skills"),
        "conversation_agent_full_name": get_md("conversation_agent_full_name"),
        "conversation_agent_login_name": get_md("conversation_agent_login_name"),
        "customer_objective_type": cl.get("customer_objective_type"),
        "customer_primary_objective": cl.get("customer_primary_objective"),
        "final_classification": cl.get("final_classification"),
        "handled_status": cl.get("handled_status"),
        "cx_issue_severity": cl.get("cx_issue_severity"),
        "unhandled_resolution_subtype": cl.get("unhandled_resolution_subtype"),
        "final_customer_sentiment": cl.get("final_customer_sentiment"),
        "max_frustration_level": cl.get("max_frustration_level"),
        "main_issue_exists": main_issue.get("issue_exists"),
        "main_issue_origin": main_issue.get("issue_origin"),
        "main_issue_type": main_issue.get("issue_type"),
        "main_issue_summary": main_issue.get("issue_summary"),
        "customer_impact": main_issue.get("customer_impact"),
        "all_detected_issues": " | ".join(
            [
                f"{i.get('issue_type', '')}: {i.get('issue_summary', '')}".strip(": ")
                for i in (cl.get("all_detected_issues") or [])
                if isinstance(i, dict)
            ]
        ),
        "positive_signals": " | ".join(cl.get("positive_signals", []) or []),
        "negative_signals": " | ".join(cl.get("negative_signals", []) or []),
        "management_summary": cl.get("management_summary"),
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
        "agent_messages",
        "unknown_messages",
        "agent_messages_evaluated",
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
    ]
    for f in cm_fields:
        row[f] = computed_metadata.get(f)
    row.update(_flatten_quantifiable_metrics(cl.get("quantifiable_metrics")))
    return row


def flatten_message_row(message_result: dict) -> dict:
    """Flatten one message-level evaluation into a CSV-friendly row."""
    pj = message_result.get("parsed_json") or {}
    return {
        "conversation_id": message_result.get("conversation_id", ""),
        "target_message_id": message_result.get("target_message_id", ""),
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
            "unhandled_subtype_counts": {},
            "issue_origin_counts": {},
            "issue_type_counts": {},
            "metric_totals": pd.DataFrame(),
            "agent_breakdown": pd.DataFrame(),
            "skill_breakdown": pd.DataFrame(),
        }

    total = int(len(conv_df))

    def safe_pct(col: str, value: Any) -> float:
        if col not in conv_df.columns:
            return 0.0
        return float((conv_df[col] == value).sum()) / total * 100.0 if total else 0.0

    handled_pct = safe_pct("handled_status", "handled")
    unhandled_pct = safe_pct("handled_status", "unhandled")
    many_issues_pct = safe_pct("cx_issue_severity", "many")

    high_frustration_count = 0
    if "max_frustration_level" in conv_df.columns:
        high_frustration_count = int(
            conv_df["max_frustration_level"].isin(["high", "cancellation_risk"]).sum()
        )

    cancellation_risk_count = 0
    if "cancellation_risk_detected" in conv_df.columns:
        cancellation_risk_count = int(conv_df["cancellation_risk_detected"].fillna(False).astype(bool).sum())
    elif "max_frustration_level" in conv_df.columns:
        cancellation_risk_count = int((conv_df["max_frustration_level"] == "cancellation_risk").sum())

    manual_review_count = 0
    if "manual_review_required" in conv_df.columns:
        manual_review_count = int(conv_df["manual_review_required"].fillna(False).astype(bool).sum())

    classification_counts = {}
    if "final_classification" in conv_df.columns:
        classification_counts = (
            conv_df["final_classification"].fillna("Unknown").value_counts().to_dict()
        )

    unhandled_subtype_counts = {}
    if "unhandled_resolution_subtype" in conv_df.columns:
        unhandled_subtype_counts = (
            conv_df["unhandled_resolution_subtype"].fillna("unknown").value_counts().to_dict()
        )

    issue_origin_counts = {}
    if "main_issue_origin" in conv_df.columns:
        issue_origin_counts = (
            conv_df["main_issue_origin"].fillna("none").value_counts().to_dict()
        )

    issue_type_counts = {}
    if "main_issue_type" in conv_df.columns:
        issue_type_counts = (
            conv_df["main_issue_type"].fillna("none").value_counts().to_dict()
        )

    metric_totals = pd.DataFrame()
    metric_cols = quantifiable_metric_columns(conv_df)
    if metric_cols:
        rows = []
        for col in metric_cols:
            series = pd.to_numeric(conv_df[col], errors="coerce").fillna(0)
            rows.append(
                {
                    "Column": col,
                    "Category": metric_category_display_name(col),
                    "Metric": metric_display_name(col),
                    "Total": float(series.sum()),
                    "Average": float(series.mean()) if len(series) else 0.0,
                    "Conversations > 0": int((series > 0).sum()),
                }
            )
        metric_totals = pd.DataFrame(rows).sort_values(
            ["Total", "Conversations > 0", "Category", "Metric"],
            ascending=[False, False, True, True],
        )

    agent_breakdown = pd.DataFrame()
    if "conversation_agent_full_name" in conv_df.columns:
        try:
            agent_breakdown = (
                conv_df.groupby("conversation_agent_full_name", dropna=False)
                .agg(
                    conversations=("conversation_id", "count"),
                    handled=("handled_status", lambda s: int((s == "handled").sum())),
                    unhandled=("handled_status", lambda s: int((s == "unhandled").sum())),
                    many_issues=("cx_issue_severity", lambda s: int((s == "many").sum())),
                    manual_review=("manual_review_required", lambda s: int(s.fillna(False).astype(bool).sum())),
                )
                .reset_index()
                .rename(columns={"conversation_agent_full_name": "Agent"})
            )
        except Exception:
            agent_breakdown = pd.DataFrame()

    skill_breakdown = pd.DataFrame()
    skill_col = None
    for c in ("last_skill", "initial_skill"):
        if c in conv_df.columns:
            skill_col = c
            break
    if skill_col:
        try:
            skill_breakdown = (
                conv_df.groupby(skill_col, dropna=False)
                .agg(
                    conversations=("conversation_id", "count"),
                    handled=("handled_status", lambda s: int((s == "handled").sum())),
                    unhandled=("handled_status", lambda s: int((s == "unhandled").sum())),
                    many_issues=("cx_issue_severity", lambda s: int((s == "many").sum())),
                )
                .reset_index()
                .rename(columns={skill_col: "Skill"})
            )
        except Exception:
            skill_breakdown = pd.DataFrame()

    return {
        "total": total,
        "handled_pct": handled_pct,
        "unhandled_pct": unhandled_pct,
        "many_issues_pct": many_issues_pct,
        "high_frustration_count": high_frustration_count,
        "cancellation_risk_count": cancellation_risk_count,
        "manual_review_count": manual_review_count,
        "classification_counts": classification_counts,
        "unhandled_subtype_counts": unhandled_subtype_counts,
        "issue_origin_counts": issue_origin_counts,
        "issue_type_counts": issue_type_counts,
        "metric_totals": metric_totals,
        "agent_breakdown": agent_breakdown,
        "skill_breakdown": skill_breakdown,
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
