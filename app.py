"""Streamlit entry point for the AI-as-a-Judge CX Conversation Evaluator."""

from __future__ import annotations

import json
import html as html_lib
import importlib
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import ui_components as ui_components_module

from api_client import APIConfig, DEFAULT_BASE_URL, build_client, fetch_models
from data_loader import (
    JOURNEY_ID_COLUMN,
    METADATA_COLUMNS,
    REQUIRED_COLUMNS,
    conversation_metadata_from_group,
    estimate_call_counts,
    get_conversation_groups,
    load_csv,
    normalize_dataframe,
    summarize_dataframe,
    validate_csv,
)
from db import DEFAULT_DB_PATH, Database
from evaluator import RunConfig, RunResults, run_evaluation, validate_conversation_level_result
from prompts import (
    DEFAULT_CONVERSATION_LEVEL_PROMPT,
    DEFAULT_MESSAGE_LEVEL_PROMPT,
    PromptTemplate,
)
from aggregation import (
    build_conversation_table,
    build_message_table,
    dashboard_aggregates,
    flatten_conversation_row,
    flatten_message_row,
    humanize_label,
    top_frustration_causes,
)
from exports import (
    build_conversation_csv_bytes,
    build_full_json_bytes,
    build_message_csv_bytes,
)
from ui_components import (
    apply_conversation_filters,
    conversation_filters,
    metric_row,
    render_conversation_summary_card,
    render_conversation_transcript_with_evals,
    render_message_evaluation_panel,
    render_transcript,
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


st.set_page_config(
    layout="wide",
    page_title="CX Conversation Evaluator",
    page_icon="💬",
)


# --------- Session state defaults ---------


def _init_state() -> None:
    defaults = {
        "df_raw": None,
        "df_norm": None,
        "csv_summary": None,
        "csv_name": None,
        "run_name": "",
        "available_models": [],
        "models_loaded_at": None,
        "model_load_error": None,
        "api_base_url": DEFAULT_BASE_URL,
        "api_key": "",
        "selected_model": "",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_tokens": 100000,
        "timeout": 300.0,
        "retries": 2,
        "concurrency": 60,
        "run_all_conversations": False,
        "max_conversations": 50,
        "max_agent_messages_per_conv": 500,
        "truncate_messages": False,
        "max_chars_per_message": 1500,
        "include_unknown_in_history": True,
        "stop_on_error": False,
        "save_raw_responses": True,
        # Which side the message-level judge inspects per turn.
        "message_target_role": "agent",
        # When set, the run evaluates ONLY these IDs (random sampler).
        "selected_conversation_ids": None,
        "run_results": None,
        "run_in_progress": False,
        "progress_log": [],
        "cancel_flag": False,
        # DB integration
        "current_run_id": None,        # id of the run we're writing to (or loaded from)
        "loaded_run_label": None,
        "review_selected_conversation_id": None,
        "theme_mode": "Dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# --------- Database singleton ---------


@st.cache_resource(show_spinner=False)
def get_db(path: str = str(DEFAULT_DB_PATH)) -> Database:
    """Return a process-wide :class:`Database` instance (cached by Streamlit)."""
    return Database(path)


def _db_path(db: Database) -> str:
    return str(getattr(db, "path", DEFAULT_DB_PATH))


def _read_prompt_file(filename: str) -> str:
    root = Path(__file__).resolve().parent / "correct_prompt_files"
    for candidate in (filename, f"{filename}.txt"):
        path = root / candidate
        try:
            value = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if value.strip():
            return value
    return ""


def _refresh_default_prompts(db: Database) -> None:
    """Refresh DB default prompt rows directly from prompt files.

    Streamlit can keep imported modules cached across reruns. Reading the files
    here avoids stale prompt constants rewriting the DB back to an old schema.
    """
    prompt_files = {
        "message_level": ("message prompt", "Message scheme", "message user input"),
        "conversation_level": (
            "conversational prompt",
            "conversational output scheme",
            "conversational user input",
        ),
    }
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with sqlite3.connect(_db_path(db)) as con:
        for kind, filenames in prompt_files.items():
            system_prompt, output_schema, user_prompt = (_read_prompt_file(name) for name in filenames)
            if not (system_prompt and output_schema and user_prompt):
                continue
            default = con.execute(
                "SELECT id FROM prompt_templates WHERE kind=? AND is_default=1 LIMIT 1",
                (kind,),
            ).fetchone()
            if default:
                con.execute(
                    "UPDATE prompt_templates SET system_prompt=?, output_schema=?, "
                    "user_prompt_template=?, updated_at=? WHERE id=?",
                    (system_prompt, output_schema, user_prompt, now, int(default[0])),
                )
                default_id = int(default[0])
            else:
                con.execute("UPDATE prompt_templates SET is_active=0 WHERE kind=?", (kind,))
                cur = con.execute(
                    "INSERT INTO prompt_templates"
                    "(kind, name, system_prompt, output_schema, user_prompt_template, "
                    "is_default, is_active, created_at, updated_at)"
                    " VALUES(?, 'Default', ?, ?, ?, 1, 1, ?, ?)",
                    (kind, system_prompt, output_schema, user_prompt, now, now),
                )
                default_id = int(cur.lastrowid)

            if kind == "conversation_level":
                active = con.execute(
                    "SELECT id, output_schema FROM prompt_templates WHERE kind=? AND is_active=1 LIMIT 1",
                    (kind,),
                ).fetchone()
                active_schema = str(active[1] if active else "")
                stale_active = "customer_experience" not in active_schema and "cx_issue_severity" in active_schema
                if stale_active:
                    con.execute("UPDATE prompt_templates SET is_active=0 WHERE kind=?", (kind,))
                    con.execute(
                        "UPDATE prompt_templates SET is_active=1, updated_at=? WHERE id=?",
                        (now, default_id),
                    )


def _run_result_counts(db: Database, run_id: int) -> dict[str, int]:
    if hasattr(db, "get_run_result_counts"):
        return db.get_run_result_counts(run_id)
    with sqlite3.connect(_db_path(db)) as con:
        return {
            "conversation_results": int(con.execute(
                "SELECT COUNT(*) FROM conversation_results WHERE run_id=?",
                (int(run_id),),
            ).fetchone()[0]),
            "message_results": int(con.execute(
                "SELECT COUNT(*) FROM message_results WHERE run_id=?",
                (int(run_id),),
            ).fetchone()[0]),
            "run_errors": int(con.execute(
                "SELECT COUNT(*) FROM run_errors WHERE run_id=?",
                (int(run_id),),
            ).fetchone()[0]),
        }


def _fill_saved_run_counts(db: Database, df_runs: pd.DataFrame) -> pd.DataFrame:
    df_runs = df_runs.copy()
    for column in ("saved_conversations", "saved_message_results", "saved_errors"):
        if column not in df_runs.columns:
            df_runs[column] = pd.NA

    needs_counts = (
        df_runs[["saved_conversations", "saved_message_results", "saved_errors"]]
        .isna()
        .any(axis=1)
    )
    if not needs_counts.any():
        return df_runs

    for index, row in df_runs[needs_counts].iterrows():
        counts = _run_result_counts(db, int(row["id"]))
        df_runs.at[index, "saved_conversations"] = counts["conversation_results"]
        df_runs.at[index, "saved_message_results"] = counts["message_results"]
        df_runs.at[index, "saved_errors"] = counts["run_errors"]
    return df_runs


def _clear_run_results(db: Database, run_id: int) -> None:
    if hasattr(db, "clear_run_results"):
        db.clear_run_results(run_id)
        return
    with sqlite3.connect(_db_path(db)) as con:
        con.execute("DELETE FROM message_results WHERE run_id=?", (int(run_id),))
        con.execute("DELETE FROM conversation_results WHERE run_id=?", (int(run_id),))
        con.execute("DELETE FROM run_errors WHERE run_id=?", (int(run_id),))


def _load_active_prompts() -> tuple[PromptTemplate, PromptTemplate, int | None, int | None]:
    """Pull the currently active prompt templates (and their ids) from the DB."""
    db = get_db()
    ml_row = db.get_active_prompt("message_level")
    cl_row = db.get_active_prompt("conversation_level")
    ml_tpl = (
        PromptTemplate(
            system_prompt=ml_row["system_prompt"],
            output_schema=ml_row["output_schema"],
            user_prompt_template=ml_row["user_prompt_template"],
        )
        if ml_row
        else DEFAULT_MESSAGE_LEVEL_PROMPT
    )
    cl_tpl = (
        PromptTemplate(
            system_prompt=cl_row["system_prompt"],
            output_schema=cl_row["output_schema"],
            user_prompt_template=cl_row["user_prompt_template"],
        )
        if cl_row
        else DEFAULT_CONVERSATION_LEVEL_PROMPT
    )
    return ml_tpl, cl_tpl, (ml_row["id"] if ml_row else None), (cl_row["id"] if cl_row else None)


# --------- Helpers ---------


def _build_api_config() -> APIConfig:
    concurrency = max(1, int(st.session_state.concurrency))
    return APIConfig(
        base_url=st.session_state.api_base_url,
        api_key=st.session_state.api_key,
        model=st.session_state.selected_model,
        temperature=float(st.session_state.temperature),
        top_p=float(st.session_state.top_p),
        max_tokens=int(st.session_state.max_tokens),
        timeout=float(st.session_state.timeout),
        retries=int(st.session_state.retries),
        concurrency=concurrency,
    )


def _build_run_config() -> tuple[RunConfig, int | None, int | None]:
    """Build a RunConfig using the active prompts from the DB.

    Returns ``(config, message_prompt_id, conversation_prompt_id)`` so the run
    record can store the prompt versions used.
    """
    ml_tpl, cl_tpl, ml_id, cl_id = _load_active_prompts()
    max_conversations = (
        None
        if st.session_state.get("run_all_conversations")
        else int(st.session_state.max_conversations)
        if st.session_state.max_conversations
        else None
    )
    cfg = RunConfig(
        api=_build_api_config(),
        max_conversations=max_conversations,
        max_agent_messages_per_conv=(
            int(st.session_state.max_agent_messages_per_conv)
            if st.session_state.max_agent_messages_per_conv
            else None
        ),
        truncate_messages=bool(st.session_state.truncate_messages),
        max_chars_per_message=int(st.session_state.max_chars_per_message),
        include_unknown_in_history=bool(st.session_state.include_unknown_in_history),
        stop_on_error=bool(st.session_state.stop_on_error),
        save_raw_responses=bool(st.session_state.save_raw_responses),
        message_target_role=str(st.session_state.message_target_role or "agent"),
        selected_conversation_ids=(
            list(st.session_state.selected_conversation_ids)
            if st.session_state.selected_conversation_ids
            else None
        ),
        message_prompt=ml_tpl,
        conversation_prompt=cl_tpl,
    )
    return cfg, ml_id, cl_id


def _has_results() -> bool:
    return st.session_state.run_results is not None and bool(
        getattr(st.session_state.run_results, "conversation_results", [])
    )


def _normalize_conversation_result_for_display(cr: dict) -> dict:
    """Apply current conversation schema defaults to older saved result JSON."""
    parsed = cr.get("parsed_json") or cr.get("evaluation_output")
    if not isinstance(parsed, dict):
        return cr
    try:
        normalized = validate_conversation_level_result(parsed)
    except Exception:
        return cr
    cr["parsed_json"] = normalized
    cr["evaluation_output"] = normalized
    return cr


def _normalize_run_results_for_display(rr: RunResults) -> RunResults:
    rr.conversation_results = [
        _normalize_conversation_result_for_display(cr)
        for cr in rr.conversation_results
    ]
    return rr


def _conv_dataframe_from_results() -> pd.DataFrame:
    rr = st.session_state.run_results
    if not rr:
        return pd.DataFrame()
    rows = []
    for cr in rr.conversation_results:
        cr = _normalize_conversation_result_for_display(cr)
        rows.append(
            flatten_conversation_row(
                cr,
                cr.get("conversation_metadata", {}) or {},
                cr.get("computed_metadata", {}) or {},
            )
        )
    return _normalize_conversation_dataframe_markers(build_conversation_table(rows))


def _msg_dataframe_from_results() -> pd.DataFrame:
    rr = st.session_state.run_results
    if not rr:
        return pd.DataFrame()
    rows = [flatten_message_row(m) for m in rr.message_level_results]
    return build_message_table(rows)


def _normalize_conversation_dataframe_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and new marker columns before any UI aggregation."""
    if df.empty:
        return df
    out = df.copy()

    def norm_series(col: str, default: str = "") -> pd.Series:
        if col not in out.columns:
            return pd.Series([default] * len(out), index=out.index)
        return (
            out[col]
            .fillna(default)
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

    final_class = (
        out["final_classification"].fillna("").astype(str).str.strip().str.lower()
        if "final_classification" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )

    handled = norm_series("handled_status")
    handled = handled.where(handled.isin(["handled", "unhandled"]), None)
    handled = handled.mask(handled.isna() & final_class.str.startswith(("unhandled", "not handled")), "unhandled")
    handled = handled.mask(handled.isna() & final_class.str.startswith("handled"), "handled")
    out["handled_status"] = handled

    experience = norm_series("customer_experience")
    old_severity = norm_series("cx_issue_severity")
    legacy_bad_experience = (old_severity == "many") | final_class.str.contains(
        "many|caused|frustration",
        regex=True,
    )
    legacy_good_experience = old_severity.isin(["zero_minimal", "minimal"]) | final_class.str.contains("minimal")
    experience = experience.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
    experience = experience.mask(legacy_bad_experience, "bad")
    valid_experience = experience.isin(["good", "bad"])
    experience = experience.where(valid_experience, None)
    experience = experience.mask(experience.isna() & legacy_bad_experience, "bad")
    experience = experience.mask(experience.isna() & legacy_good_experience, "good")
    out["customer_experience"] = experience

    if "frustration_detected" in out.columns:
        out["frustration_detected"] = (
            out["frustration_detected"]
            .fillna(False)
            .map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "y", "frustrated"})
        )
    else:
        out["frustration_detected"] = final_class.str.contains("frustration")

    origin = norm_series("frustration_origin", "none")
    origin = origin.replace({"customer": "customer_side", "our": "our_side", "agent": "our_side"})
    main_origin = norm_series("main_issue_origin", "none")
    main_origin = main_origin.replace({"customer": "customer_side", "our": "our_side", "agent": "our_side"})
    valid_origin = origin.isin(["our_side", "customer_side", "shared", "none"])
    origin = origin.where(valid_origin, None)
    origin = origin.mask(origin.isna() & main_origin.isin(["our_side", "customer_side", "shared", "none"]), main_origin)
    origin = origin.mask(origin.isna() & out["frustration_detected"] & final_class.str.contains("caused"), "our_side")
    out["frustration_origin"] = origin.fillna("none")

    if "main_issue_origin" in out.columns:
        out["main_issue_origin"] = main_origin.where(
            main_origin.isin(["our_side", "customer_side", "shared", "none"]),
            out["frustration_origin"],
        )

    return out


def _ordered_selected_ids(all_ids: list[str], selected_ids: list[str] | None) -> list[str]:
    """Return selected journey IDs in the same order they appear in the CSV."""
    if not selected_ids:
        return []
    wanted = {str(x) for x in selected_ids}
    ordered = [str(x) for x in all_ids if str(x) in wanted]
    extra = [str(x) for x in selected_ids if str(x) not in set(ordered)]
    return ordered + extra


def _journey_selector_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Build one searchable row per customer journey for run scoping."""
    rows: list[dict[str, Any]] = []
    for journey_id, group in get_conversation_groups(df):
        md = conversation_metadata_from_group(group)
        customer_name = str(md.get("customer_name") or "").strip()
        customer_phone = str(md.get("customer_phone") or journey_id or "").strip()
        source_ids = str(md.get("source_conversation_ids") or "").strip()
        source_count = md.get("source_conversation_count") or 0
        message_count = int(md.get("total_visible_messages") or len(group))
        customer_messages = int(md.get("customer_message_count") or 0)
        agent_messages = int(md.get("agent_message_count") or 0)
        start_date = str(md.get("conversation_start_date") or "").strip()
        end_date = str(md.get("conversation_end_date") or "").strip()
        display_name = customer_name or "Unknown customer"
        label = (
            f"{customer_phone} • {display_name} • {source_count} source convs • "
            f"{message_count} msgs"
        )
        search_text = " ".join(
            [
                journey_id,
                customer_name,
                customer_phone,
                source_ids,
                start_date,
                end_date,
            ]
        ).lower()
        rows.append(
            {
                "journey_id": str(journey_id),
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "source_conversation_ids": source_ids,
                "source_conversation_count": source_count,
                "message_count": message_count,
                "customer_messages": customer_messages,
                "agent_messages": agent_messages,
                "conversation_start_date": start_date,
                "conversation_end_date": end_date,
                "label": label,
                "search_text": search_text,
            }
        )
    return pd.DataFrame(rows)


def _conversation_filters_with_keys(conv_df: pd.DataFrame, key_prefix: str) -> dict:
    try:
        return conversation_filters(conv_df, key_prefix=key_prefix)
    except TypeError:
        reloaded = importlib.reload(ui_components_module)
        return reloaded.conversation_filters(conv_df, key_prefix=key_prefix)


def _apply_conversation_filters_fresh(conv_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    reloaded = importlib.reload(ui_components_module)
    return reloaded.apply_conversation_filters(conv_df, filters)


def _render_conversation_summary_card_fresh(conv_result: dict) -> None:
    reloaded = importlib.reload(ui_components_module)
    reloaded.render_conversation_summary_card(conv_result)


def _humanize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].apply(humanize_label)
    return out


def _display_column_name(column: str) -> str:
    special = {
        "conversation_id": "ID",
        "customer_journey_id": "ID",
        "customer_name": "Customer name",
        "customer_phone": "Customer phone",
        "source_conversation_id": "Source conversation ID",
        "source_conversation_ids": "Source conversation IDs",
        "source_conversation_count": "Source conversations",
        "conversation_start_date": "Started",
        "conversation_end_date": "Ended",
        "conversation_status": "Conversation status",
        "customer_objective_type": "Customer goal type",
        "customer_primary_objective": "Customer goal",
        "final_classification": "Overall result",
        "handled_status": "Outcome",
        "cx_issue_severity": "Journey quality",
        "customer_experience": "Customer experience",
        "frustration_detected": "Customer frustration",
        "frustration_origin": "Frustration origin",
        "customer_started_frustrated": "Started frustrated",
        "customer_became_frustrated_during_chat": "Became frustrated during chat",
        "customer_ended_frustrated": "Ended frustrated",
        "frustration_timing": "When frustration appeared",
        "unhandled_resolution_subtype": "Unresolved status",
        "final_customer_sentiment": "Customer feeling at end",
        "max_frustration_level": "Highest frustration level",
        "score_resolution": "Resolution score",
        "score_context_understanding": "Context & Understanding score",
        "score_customer_effort": "Customer Effort score",
        "score_frustration_risk": "Frustration & Risk score",
        "score_raw_total": "Raw conversation score",
        "score_final": "Final conversation score",
        "score_rating": "Score rating",
        "score_explanation": "Score explanation",
        "main_issue_type": "Main problem type",
        "main_issue_origin": "Where the main problem came from",
        "main_issue_summary": "Main problem summary",
        "customer_impact": "Customer impact",
        "classification_reason": "Classification reason",
        "manual_review_required": "Needs human review",
        "manual_review_reason": "Reason for human review",
        "metric_value": "Metric value",
        "target_message_id": "Target message ID",
        "appended_message_index": "Appended message index",
        "message_index": "Appended message index",
        "message_time": "Message time",
        "target_message_text": "Assistant message",
    }
    return special.get(column, humanize_label(column))


def _prepare_display_table(df: pd.DataFrame, enum_columns: list[str] | None = None) -> pd.DataFrame:
    out = _humanize_columns(df, enum_columns or [])
    return out.rename(columns={col: _display_column_name(col) for col in out.columns})


def _theme_colors() -> dict[str, str]:
    dark = str(st.session_state.get("theme_mode") or "Light") == "Dark"
    return {
        "bg": "#0a0e27" if dark else "#ffffff",
        "panel": "#111827" if dark else "#f8fafc",
        "panel_2": "#1a202c" if dark else "#ffffff",
        "text": "#f0f4f8" if dark else "#0f172a",
        "muted": "#a0aec0" if dark else "#64748b",
        "border": "#2d3748" if dark else "#e5e7eb",
        "accent": "#3b82f6" if dark else "#2563eb",
        "accent_2": "#f59e0b" if dark else "#ef4444",
        "track": "#2d3748" if dark else "#e5e7eb",
        "grid": "#2d3748" if dark else "#e5e7eb",
    }


def _render_display_table(
    df: pd.DataFrame,
    *,
    enum_columns: list[str] | None = None,
    max_rows: int | None = None,
    height: int | None = None,
    empty_message: str = "No data.",
) -> None:
    """Render a theme-aware HTML table instead of Streamlit's iframe table."""
    if df is None or df.empty:
        if empty_message:
            st.caption(empty_message)
        return

    display_df = _prepare_display_table(df, enum_columns) if enum_columns is not None else df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)

    height_style = f' style="max-height: {height}px;"' if height else ""
    table_html = display_df.to_html(
        index=False,
        escape=True,
        border=0,
        classes="cx-data-table",
    )
    st.markdown(
        f'<div class="cx-table-wrap"{height_style}>{table_html}</div>',
        unsafe_allow_html=True,
    )



def _format_chart_value(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "0"
    value = float(value)
    if abs(value - round(value)) < 0.001:
        return f"{int(round(value))}{suffix}"
    return f"{value:.1f}{suffix}"


def _render_simple_bar_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    *,
    height: int = 360,
    max_value: float | None = None,
    value_suffix: str = "",
    empty_message: str = "No data.",
) -> None:
    if df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        st.caption(empty_message)
        return

    chart_df = df[[label_col, value_col]].copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce").fillna(0)
    chart_df = chart_df[chart_df[value_col] >= 0]
    if chart_df.empty:
        st.caption(empty_message)
        return

    colors = _theme_colors()
    max_seen = float(chart_df[value_col].max()) if not chart_df.empty else 0.0
    denominator = float(max_value) if max_value is not None else max_seen
    denominator = denominator if denominator > 0 else 1.0

    rows = []
    for _, row in chart_df.iterrows():
        label = html_lib.escape(str(row[label_col]))
        value = float(row[value_col])
        width = max(1.5, min(100.0, (value / denominator) * 100.0))
        value_text = html_lib.escape(_format_chart_value(value, value_suffix))
        rows.append(
            f"""
            <div class="cx-chart-row">
              <div class="cx-chart-label" title="{label}">{label}</div>
              <div class="cx-chart-track">
                <div class="cx-chart-bar" style="width: {width:.2f}%"></div>
              </div>
              <div class="cx-chart-value">{value_text}</div>
            </div>
            """
        )

    html_content = f"""
    <div class="cx-chart-wrap" style="max-height: {height}px;">
      {''.join(rows)}
    </div>
    <style>
    .cx-chart-wrap {{
      overflow: auto;
      background: {colors["panel_2"]};
      border: 1px solid {colors["border"]};
      border-radius: 8px;
      padding: 0.75rem;
      margin: 0.35rem 0 1rem;
    }}
    .cx-chart-row {{
      display: grid;
      grid-template-columns: minmax(160px, 32%) 1fr minmax(54px, auto);
      gap: 0.75rem;
      align-items: center;
      min-height: 34px;
    }}
    .cx-chart-row + .cx-chart-row {{
      margin-top: 0.55rem;
    }}
    .cx-chart-label {{
      color: {colors["text"]};
      font-weight: 600;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .cx-chart-track {{
      height: 14px;
      border-radius: 999px;
      background: {colors["track"]};
      overflow: hidden;
    }}
    .cx-chart-bar {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #3b82f6, #f59e0b);
      box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
    }}
    .cx-chart-value {{
      color: {colors["text"]};
      font-weight: 700;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    @media (max-width: 720px) {{
      .cx-chart-row {{
        grid-template-columns: 1fr minmax(48px, auto);
      }}
      .cx-chart-track {{
        grid-column: 1 / -1;
        grid-row: 2;
      }}
    }}
    </style>
    """
    components.html(html_content, height=height + 24, scrolling=False)


def _render_simple_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    height: int = 300,
    empty_message: str = "No data.",
) -> None:
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        st.caption(empty_message)
        return

    chart_df = df[[x_col, y_col]].copy()
    chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce").fillna(0)
    chart_df = chart_df.reset_index(drop=True)
    if chart_df.empty:
        st.caption(empty_message)
        return

    colors = _theme_colors()
    width = 900
    chart_h = max(180, height - 70)
    pad_x = 46
    pad_y = 28
    max_y = float(chart_df[y_col].max())
    min_y = float(chart_df[y_col].min())
    if max_y == min_y:
        max_y += 1.0
        min_y = 0.0
    span_x = max(len(chart_df) - 1, 1)

    points = []
    dots = []
    for i, row in chart_df.iterrows():
        x = pad_x + (i / span_x) * (width - pad_x * 2)
        y = pad_y + ((max_y - float(row[y_col])) / (max_y - min_y)) * (chart_h - pad_y * 2)
        points.append(f"{x:.2f},{y:.2f}")
        dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors["accent_2"]}" />')

    first_label = html_lib.escape(str(chart_df.iloc[0][x_col]))
    last_label = html_lib.escape(str(chart_df.iloc[-1][x_col]))
    max_label = html_lib.escape(_format_chart_value(max_y))
    min_label = html_lib.escape(_format_chart_value(min_y))
    path_points = " ".join(points)

    html_content = f"""
    <div class="cx-line-wrap" style="height: {height}px;">
      <svg class="cx-line-svg" viewBox="0 0 {width} {chart_h}" preserveAspectRatio="none">
        <line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{chart_h - pad_y}" stroke="{colors["grid"]}" />
        <line x1="{pad_x}" y1="{chart_h - pad_y}" x2="{width - pad_x}" y2="{chart_h - pad_y}" stroke="{colors["grid"]}" />
        <polyline points="{path_points}" fill="none" stroke="{colors["accent"]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        {''.join(dots)}
      </svg>
      <div class="cx-line-axis cx-line-axis-y-top">{max_label}</div>
      <div class="cx-line-axis cx-line-axis-y-bottom">{min_label}</div>
      <div class="cx-line-axis cx-line-axis-x-left">{first_label}</div>
      <div class="cx-line-axis cx-line-axis-x-right">{last_label}</div>
    </div>
    <style>
    .cx-line-wrap {{
      position: relative;
      background: {colors["panel_2"]};
      border: 1px solid {colors["border"]};
      border-radius: 8px;
      padding: 0.5rem;
      margin: 0.35rem 0 1rem;
    }}
    .cx-line-svg {{
      width: 100%;
      height: calc(100% - 1.8rem);
      display: block;
    }}
    .cx-line-axis {{
      position: absolute;
      color: {colors["muted"]};
      font-size: 0.82rem;
      font-weight: 600;
    }}
    .cx-line-axis-y-top {{
      top: 0.45rem;
      left: 0.65rem;
    }}
    .cx-line-axis-y-bottom {{
      bottom: 1.75rem;
      left: 0.65rem;
    }}
    .cx-line-axis-x-left {{
      left: 3rem;
      bottom: 0.45rem;
    }}
    .cx-line-axis-x-right {{
      right: 1rem;
      bottom: 0.45rem;
    }}
    </style>
    """
    components.html(html_content, height=height + 24, scrolling=False)


def _apply_theme() -> None:
    """Apply the selected app theme with CSS and Plotly template defaults."""
    mode = str(st.session_state.get("theme_mode") or "Light")
    dark = mode == "Dark"
    if HAS_PLOTLY:
        px.defaults.template = "plotly_dark" if dark else "plotly_white"
        px.defaults.color_continuous_scale = "Blues" if not dark else "Viridis"

    colors = {
        "bg": "#0a0e27" if dark else "#ffffff",
        "panel": "#111827" if dark else "#f8fafc",
        "panel_2": "#1a202c" if dark else "#ffffff",
        "text": "#f0f4f8" if dark else "#0f172a",
        "muted": "#a0aec0" if dark else "#64748b",
        "border": "#2d3748" if dark else "#e5e7eb",
        "input": "#0a0e27" if dark else "#ffffff",
        "input_text": "#f0f4f8" if dark else "#111827",
        "accent": "#3b82f6" if dark else "#ef4444",
        "button": "#2563eb" if dark else "#2563eb",
        "button_text": "#ffffff",
        "disabled": "#2d3748" if dark else "#e5e7eb",
        "disabled_text": "#a0aec0" if dark else "#94a3b8",
        "plot_bg": "#1a202c" if dark else "#ffffff",
        "grid": "#2d3748" if dark else "#e5e7eb",
    }
    color_scheme = "dark" if dark else "light"
    st.markdown(
        f"""
        <style>
        :root {{
          color-scheme: {color_scheme};
        }}
        .stApp {{
          background: {colors["bg"]};
          color: {colors["text"]};
        }}
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
          background: {colors["panel"]} !important;
          color: {colors["text"]} !important;
        }}
        [data-testid="stHeader"], [data-testid="stDecoration"] {{
          background: {colors["bg"]} !important;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp label, .stApp span, .stApp div {{
          color: {colors["text"]};
        }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
        small {{
          color: {colors["muted"]} !important;
        }}
        div[data-testid="stMetric"], div[data-testid="stExpander"],
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
          background-color: {colors["panel_2"]} !important;
          border-color: {colors["border"]} !important;
        }}
        .cx-table-wrap {{
          width: 100%;
          overflow: auto;
          border: 1px solid {colors["border"]};
          border-radius: 8px;
          background: {colors["panel_2"]};
          margin: 0.35rem 0 1rem;
        }}
        table.cx-data-table {{
          width: 100%;
          border-collapse: collapse;
          background: {colors["panel_2"]};
          color: {colors["text"]};
          font-size: 0.92rem;
          line-height: 1.35;
        }}
        table.cx-data-table thead th {{
          position: sticky;
          top: 0;
          z-index: 1;
          background: {colors["panel"]};
          color: {colors["muted"]};
          font-weight: 700;
          text-align: left;
          border-bottom: 1px solid {colors["border"]};
          padding: 0.65rem 0.75rem;
          white-space: nowrap;
        }}
        table.cx-data-table tbody td {{
          background: {colors["panel_2"]};
          color: {colors["text"]};
          border-bottom: 1px solid {colors["border"]};
          padding: 0.58rem 0.75rem;
          vertical-align: top;
        }}
        table.cx-data-table tbody tr:last-child td {{
          border-bottom: 0;
        }}
        table.cx-data-table tbody tr:hover td {{
          background: {colors["panel"]};
        }}
        table.cx-data-table td:nth-child(n+2):not(:last-child),
        table.cx-data-table th:nth-child(n+2):not(:last-child) {{
          text-align: right;
        }}
        input, textarea, select,
        div[data-baseweb="input"], div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"], div[data-baseweb="textarea"],
        div[data-baseweb="select"], div[data-baseweb="select"] > div {{
          color-scheme: {color_scheme};
          background-color: {colors["input"]} !important;
          color: {colors["input_text"]} !important;
          border-color: {colors["border"]} !important;
        }}
        input, textarea {{
          -webkit-text-fill-color: {colors["input_text"]} !important;
        }}
        input::placeholder, textarea::placeholder {{
          color: {colors["muted"]} !important;
          -webkit-text-fill-color: {colors["muted"]} !important;
        }}
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
        [data-testid="stRadio"] label, [data-testid="stRadio"] label *,
        div[role="radiogroup"] label, div[role="radiogroup"] label * {{
          color: {colors["text"]} !important;
          opacity: 1 !important;
        }}
        div[role="radiogroup"] [data-baseweb="radio"] {{
          color: {colors["text"]} !important;
        }}
        div[role="radiogroup"] [aria-checked="true"] div {{
          border-color: {colors["accent"]} !important;
        }}
        .stButton > button, button[kind="primary"], button[kind="secondary"] {{
          background-color: {colors["button"]} !important;
          color: {colors["button_text"]} !important;
          border-color: {colors["button"]} !important;
        }}
        .stButton > button:disabled, button:disabled {{
          background-color: {colors["disabled"]} !important;
          color: {colors["disabled_text"]} !important;
          border-color: {colors["border"]} !important;
          opacity: 1 !important;
        }}
        section[data-testid="stFileUploaderDropzone"] {{
          background-color: {colors["panel_2"]} !important;
          border: 1px solid {colors["border"]} !important;
        }}
        section[data-testid="stFileUploaderDropzone"] * {{
          color: {colors["text"]} !important;
        }}
        section[data-testid="stFileUploaderDropzone"] button {{
          background-color: {colors["input"]} !important;
          color: {colors["input_text"]} !important;
          border-color: {colors["border"]} !important;
        }}
        button[data-baseweb="tab"] p {{
          color: {colors["muted"]} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
          color: {colors["accent"]} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
          border-bottom-color: {colors["accent"]} !important;
        }}
        [data-testid="stAlert"] {{
          color: {colors["text"]} !important;
        }}
        [data-testid="stAlert"] * {{
          color: inherit !important;
        }}
        [data-testid="stSidebar"] hr {{
          border-color: {colors["border"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plotly_layout(fig, height: int | None = None, **layout):
    """Apply app theme colors to Plotly figures."""
    dark = str(st.session_state.get("theme_mode") or "Light") == "Dark"
    bg = "#0a0e27" if dark else "#ffffff"
    panel = "#1a202c" if dark else "#ffffff"
    text = "#f0f4f8" if dark else "#0f172a"
    grid = "#2d3748" if dark else "#e5e7eb"
    base = {
        "template": "plotly_dark" if dark else "plotly_white",
        "paper_bgcolor": bg,
        "plot_bgcolor": panel,
        "font": {"color": text},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": text}},
        "margin": dict(t=10, b=10),
    }
    if height is not None:
        base["height"] = height
    base.update(layout)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont={"color": text}, title_font={"color": text})
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont={"color": text}, title_font={"color": text})
    try:
        fig.update_traces(textfont_color=text, insidetextfont_color=text, outsidetextfont_color=text)
    except Exception:
        pass
    return fig


def _render_plotly(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "responsive": True})


def _segment_table(df: pd.DataFrame, column: str, label: str, *, base_count: int | None = None) -> pd.DataFrame:
    """Return count/share rows for a dashboard segment column."""
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[label, "Count", "Share", "_pct"])
    base = int(base_count if base_count is not None else len(df)) or 1
    counts = df[column].fillna("unknown").astype(str).value_counts(dropna=False)
    rows = []
    for value, count in counts.items():
        count = int(count)
        rows.append(
            {
                label: humanize_label(value),
                "Count": count,
                "Share": f"{(count / base * 100):.1f}%",
                "_pct": count / base * 100,
            }
        )
    return pd.DataFrame(rows)


def _render_segment_block(
    title: str,
    df: pd.DataFrame,
    label_col: str,
    *,
    color_col: str | None = None,
    chart: bool = True,
) -> None:
    """Render a compact dashboard block as chart plus table."""
    st.markdown(f"#### {title}")
    if df.empty:
        st.caption("No data.")
        return
    display_df = df[[label_col, "Count", "Share"]].copy()
    if chart and HAS_PLOTLY:
        chart_df = df.copy()
        fig = px.bar(
            chart_df,
            x="Count",
            y=label_col,
            color=color_col or label_col,
            orientation="h",
            text="Share",
            hover_data=["Count", "Share"],
        )
        _plotly_layout(fig, height=max(260, min(520, 68 + len(chart_df) * 38)), yaxis=dict(autorange="reversed"))
        _render_plotly(fig)
    _render_display_table(display_df, height=min(320, 74 + len(display_df) * 36))


def _comparison_matrix(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    *,
    row_label: str,
    col_label: str,
) -> pd.DataFrame:
    if df.empty or row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame()
    work = df[[row_col, col_col]].copy()
    work[row_col] = work[row_col].fillna("unknown").astype(str).apply(humanize_label)
    work[col_col] = work[col_col].fillna("unknown").astype(str).apply(humanize_label)
    matrix = pd.crosstab(work[row_col], work[col_col])
    matrix.index.name = row_label
    matrix.columns.name = col_label
    return matrix.reset_index()


# --------- Sidebar ---------


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Display")
        # Dark mode only
        st.session_state.theme_mode = "Dark"
        st.markdown("---")

        st.markdown("## API Settings")
        st.text_input(
            "Base URL",
            key="api_base_url",
            help="OpenAI-compatible base URL.",
        )
        st.text_input(
            "API Key",
            key="api_key",
            type="password",
        )

        if st.button("Load available models", use_container_width=True):
            try:
                with st.spinner("Loading models..."):
                    client = build_client(st.session_state.api_base_url, st.session_state.api_key)
                    models = fetch_models(client)
                st.session_state.available_models = models
                st.session_state.models_loaded_at = time.time()
                st.session_state.model_load_error = None
                if not models:
                    st.warning("No models returned from /models.")
                else:
                    st.success(f"Loaded {len(models)} models.")
            except Exception as e:
                st.session_state.available_models = []
                st.session_state.model_load_error = str(e)
                st.error(f"Could not load models: {e}")

        if st.session_state.model_load_error:
            st.caption(f"Last error: {st.session_state.model_load_error}")

        models = st.session_state.available_models or []
        if models:
            current = st.session_state.selected_model
            default_index = models.index(current) if current in models else 0
            st.selectbox("Model", models, index=default_index, key="selected_model")
        else:
            st.text_input(
                "Model",
                key="selected_model",
                help="Click 'Load available models' to populate this dropdown.",
            )

        st.markdown("---")
        st.markdown("### Generation parameters")
        st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.05, key="temperature")
        st.slider("Top P", min_value=0.0, max_value=1.0, step=0.05, key="top_p")
        st.number_input("Max tokens", min_value=128, step=64, key="max_tokens")
        st.number_input("Timeout (seconds)", min_value=5.0, step=5.0, key="timeout")
        st.number_input("Retry count", min_value=0, step=1, key="retries")
        st.session_state.concurrency = min(100, max(1, int(st.session_state.concurrency)))
        st.number_input(
            "Concurrency",
            min_value=1,
            max_value=100,
            step=1,
            key="concurrency",
            help=(
                "Number of message-level API calls dispatched in parallel. "
                "Lower this if the API returns 503s, rate limits, or timeouts."
            ),
        )

        st.markdown("---")
        st.markdown("### Evaluation safeguards")
        summary = st.session_state.get("csv_summary") or {}
        total_journeys = int(summary.get("journeys") or summary.get("conversations") or 0)
        if total_journeys:
            st.caption(f"Uploaded CSV has {total_journeys:,} customer journeys.")
            current_limit = int(st.session_state.max_conversations or 1)
            if st.session_state.get("run_all_conversations"):
                st.session_state.max_conversations = total_journeys
            elif current_limit > total_journeys:
                st.session_state.max_conversations = total_journeys
            elif current_limit < 1:
                st.session_state.max_conversations = min(50, total_journeys)
        else:
            st.session_state.run_all_conversations = False
        st.toggle(
            "Run all uploaded journeys",
            key="run_all_conversations",
            disabled=not total_journeys,
            help="When enabled, the run processes every customer journey in the uploaded CSV.",
        )
        if total_journeys and st.session_state.run_all_conversations:
            st.session_state.max_conversations = total_journeys
        st.number_input(
            "Customer journeys to process",
            min_value=1,
            max_value=total_journeys or None,
            step=1,
            key="max_conversations",
            disabled=bool(total_journeys and st.session_state.run_all_conversations),
            help="When 'Run all uploaded journeys' is off, this many journeys are processed from the CSV order.",
        )
        st.number_input(
            "Max target messages per journey",
            min_value=1,
            step=1,
            key="max_agent_messages_per_conv",
        )
        st.radio(
            "Evaluate which side?",
            options=["agent", "customer"],
            key="message_target_role",
            horizontal=True,
            format_func=lambda v: {
                "agent": "Assistant messages",
                "customer": "Customer messages",
            }.get(v, v),
            help=(
                "Assistant: judge each assistant reply — how it responded to a "
                "possibly-frustrated customer message.\n\n"
                "Customer: judge each customer message — capture the customer's "
                "state / frustration BEFORE the assistant answers."
            ),
        )
        st.toggle("Truncate message text", key="truncate_messages")
        if st.session_state.truncate_messages:
            st.number_input(
                "Max characters per message",
                min_value=200,
                step=100,
                key="max_chars_per_message",
            )
        st.toggle("Include unknown sender messages in history", key="include_unknown_in_history")
        st.toggle("Stop on API error", key="stop_on_error")
        st.toggle("Save raw model responses", key="save_raw_responses")

        st.markdown("---")
        st.caption(f"Database file: `{DEFAULT_DB_PATH}`")
        if st.session_state.current_run_id is not None:
            st.caption(f"Current run id: **#{st.session_state.current_run_id}**")


# --------- Tab: Upload & Settings ---------


def tab_upload() -> None:
    st.subheader("Upload Customer Journey CSV")
    st.caption(
        "Upload the Snowflake-exported CSV. One row per visible message in the appended customer journey. "
        "Tool calls and internal/system messages must already be removed."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            return

        previous_csv_name = st.session_state.csv_name
        if previous_csv_name and previous_csv_name != uploaded.name:
            st.session_state.selected_conversation_ids = None
            st.session_state.journey_selection_visible_labels = []

        st.session_state.csv_name = uploaded.name
        st.session_state.df_raw = df

        is_valid, missing, msg = validate_csv(df)
        if not is_valid:
            st.error(msg)
            with st.expander("Show CSV columns received"):
                st.write(list(df.columns))
            return

        df_norm = normalize_dataframe(df)
        st.session_state.df_norm = df_norm
        st.session_state.csv_summary = summarize_dataframe(df_norm)

    df_norm = st.session_state.df_norm
    if df_norm is None or df_norm.empty:
        st.info("Upload a CSV to continue.")
        return

    summary = st.session_state.csv_summary or {}
    st.markdown("### CSV Overview")
    metric_row(
        [
            ("Rows", f"{summary.get('rows', 0):,}", None),
            ("Customer journeys", f"{summary.get('journeys', summary.get('conversations', 0)):,}", None),
            ("Source conversations", f"{summary.get('source_conversations', 0):,}", None),
            ("Customer messages", f"{summary.get('customer_messages', 0):,}", None),
            ("Assistant messages", f"{summary.get('agent_messages', 0):,}", None),
            ("Unknown messages", f"{summary.get('unknown_messages', 0):,}", None),
        ]
    )
    if summary.get("date_min") and summary.get("date_max"):
        st.caption(f"Date range: {summary['date_min']} → {summary['date_max']}")

    st.markdown("### Required Columns")
    cols_present = list(df_norm.columns)
    req_status = []
    for c in REQUIRED_COLUMNS:
        req_status.append({"Column": c, "Present": "Yes" if c in cols_present else "Missing"})
    st.dataframe(pd.DataFrame(req_status), use_container_width=True, hide_index=True)

    st.markdown("### Useful Metadata Columns")
    md_status = []
    for c in METADATA_COLUMNS:
        md_status.append({"Column": c, "Present": "Yes" if c in cols_present else "—"})
    st.dataframe(pd.DataFrame(md_status), use_container_width=True, hide_index=True)

    st.markdown("### Preview")
    st.dataframe(df_norm.head(20), use_container_width=True)


# --------- Tab: Prompts ---------


def _render_prompt_editor(kind: str, label: str) -> None:
    """Reusable editor for one prompt template kind."""
    db = get_db()
    active = db.get_active_prompt(kind)
    versions = db.list_prompts(kind)

    st.markdown(f"### {label}")
    active_label = "—"
    if active:
        active_label = f"#{active['id']} • {active['name']} " + (
            "(default)" if active.get("is_default") else "(custom)"
        )
    st.caption(f"Active version: {active_label}")

    # Version picker
    if versions:
        version_labels = []
        version_ids = []
        for v in versions:
            marker = "★" if v.get("is_active") else " "
            tag = "default" if v.get("is_default") else "custom"
            version_labels.append(
                f"{marker} #{v['id']} • {v['name']} ({tag}) • {v['updated_at']}"
            )
            version_ids.append(v["id"])

        sel_idx = 0
        for i, v in enumerate(versions):
            if v.get("is_active"):
                sel_idx = i
                break
        chosen_label = st.selectbox(
            "Load a version into the editor",
            version_labels,
            index=sel_idx,
            key=f"version_pick_{kind}",
        )
        chosen_id = version_ids[version_labels.index(chosen_label)]
    else:
        chosen_id = None

    # Pull the chosen row for the editor.
    if chosen_id is None:
        editor_source = active or {}
    else:
        editor_source = db.get_prompt(chosen_id) or {}

    # State keys per kind for the editor textareas.
    sys_key = f"editor_system_{kind}"
    schema_key = f"editor_schema_{kind}"
    user_key = f"editor_user_{kind}"
    name_key = f"editor_name_{kind}"
    load_marker_key = f"loaded_prompt_id_{kind}"

    # If the user just changed the version dropdown, reload the editor contents.
    if st.session_state.get(load_marker_key) != chosen_id:
        st.session_state[sys_key] = editor_source.get("system_prompt", "")
        st.session_state[schema_key] = editor_source.get("output_schema", "")
        st.session_state[user_key] = editor_source.get("user_prompt_template", "")
        st.session_state[name_key] = ""
        st.session_state[load_marker_key] = chosen_id

    st.text_input("New version name", key=name_key, placeholder="e.g., Stricter tone v2")

    st.markdown("**System prompt**")
    st.caption(
        "Use `{output_schema}` where you want the schema block to appear. "
        "If the placeholder is missing, the schema is appended at the end."
    )
    st.text_area("system prompt body", key=sys_key, height=320, label_visibility="collapsed")

    st.markdown("**Output structure (JSON schema / example)**")
    st.caption("This is the JSON shape the LLM is told to return.")
    st.text_area("output schema", key=schema_key, height=260, label_visibility="collapsed")

    st.markdown("**User prompt template**")
    st.caption("Must contain `{payload_json}` — the per-call input is substituted there.")
    st.text_area("user prompt template", key=user_key, height=140, label_visibility="collapsed")

    btn_save, btn_activate, btn_reset, btn_delete = st.columns(4)
    with btn_save:
        if st.button("Save & Activate", key=f"save_{kind}", use_container_width=True, type="primary"):
            name = (st.session_state.get(name_key) or "").strip() or f"Custom {time.strftime('%Y-%m-%d %H:%M:%S')}"
            try:
                new_id = db.save_prompt(
                    kind=kind,
                    name=name,
                    system_prompt=st.session_state.get(sys_key, ""),
                    output_schema=st.session_state.get(schema_key, ""),
                    user_prompt_template=st.session_state.get(user_key, ""),
                    set_active=True,
                )
                st.session_state[load_marker_key] = new_id
                st.success(f"Saved as version #{new_id} and set active.")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")
    with btn_activate:
        if chosen_id is not None and st.button(
            "Set selected version active",
            key=f"activate_{kind}",
            use_container_width=True,
            disabled=(active and active.get("id") == chosen_id),
        ):
            try:
                db.set_active_prompt(chosen_id)
                st.success(f"Version #{chosen_id} is now active.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not activate: {e}")
    with btn_reset:
        if st.button("Reset to default", key=f"reset_{kind}", use_container_width=True):
            try:
                db.reset_to_default(kind)
                st.success("Default prompt is active again.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")
    with btn_delete:
        can_delete = (
            chosen_id is not None
            and editor_source
            and not editor_source.get("is_default")
        )
        if st.button(
            "Delete selected version",
            key=f"delete_{kind}",
            use_container_width=True,
            disabled=not can_delete,
        ):
            try:
                db.delete_prompt(chosen_id)
                st.success(f"Deleted version #{chosen_id}.")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    with st.expander("Preview combined system prompt"):
        try:
            tpl = PromptTemplate(
                system_prompt=st.session_state.get(sys_key, ""),
                output_schema=st.session_state.get(schema_key, ""),
                user_prompt_template=st.session_state.get(user_key, ""),
            )
            st.code(tpl.build_system(), language="markdown")
        except Exception as e:
            st.error(f"Could not build preview: {e}")


def tab_prompts() -> None:
    st.subheader("Prompts")
    st.caption(
        "Edit the prompts and output structures sent to the model. Changes are "
        "saved to the SQLite database. The active version of each kind is the "
        "one used on the next run."
    )

    sub_ml, sub_cl = st.tabs(["Message-Level Prompt", "Conversation-Level Prompt"])
    with sub_ml:
        _render_prompt_editor("message_level", "Message-Level Prompt")
    with sub_cl:
        _render_prompt_editor("conversation_level", "Conversation-Level Prompt")


# --------- Tab: Run Evaluation ---------


def tab_run() -> None:
    st.subheader("Run CX Evaluation")

    # --- Past runs (load from DB) ---
    db = get_db()
    with st.expander("Past runs (saved in the database)", expanded=False):
        if st.button("Refresh saved runs", key="refresh_saved_runs", use_container_width=True):
            st.rerun()
        runs = db.list_runs(limit=200)
        if not runs:
            st.caption("No saved runs yet.")
        else:
            df_runs = _fill_saved_run_counts(db, pd.DataFrame(runs))
            n_conversations = pd.to_numeric(
                df_runs["n_conversations"] if "n_conversations" in df_runs else pd.Series(0, index=df_runs.index),
                errors="coerce",
            ).fillna(0)
            saved_conversations = pd.to_numeric(
                df_runs["saved_conversations"] if "saved_conversations" in df_runs else pd.Series(0, index=df_runs.index),
                errors="coerce",
            ).fillna(0)
            df_runs["is_incomplete"] = (
                (n_conversations > 0)
                & (saved_conversations == 0)
            )
            df_runs["label"] = df_runs.apply(
                lambda r: (
                    f"#{r['id']} • {r.get('name') or 'Untitled run'} • "
                    f"{r.get('csv_name') or '—'} • {r['status']}"
                    f"{' • incomplete: no saved results' if r.get('is_incomplete') else ''} • "
                    f"{r['started_at']}"
                ),
                axis=1,
            )
            st.caption(f"Newest saved run in this database: #{int(df_runs.iloc[0]['id'])}")
            st.dataframe(
                df_runs[[
                    "id",
                    "csv_name",
                    "status",
                    "n_conversations",
                    "saved_conversations",
                    "n_message_calls",
                    "saved_message_results",
                    "started_at",
                ]].head(8),
                use_container_width=True,
                hide_index=True,
            )
            incomplete_count = int(df_runs["is_incomplete"].sum())
            show_incomplete = False
            if incomplete_count:
                show_incomplete = st.checkbox(
                    f"Show {incomplete_count} incomplete run(s) with no saved results",
                    value=False,
                )
            display_runs = df_runs if show_incomplete else df_runs[~df_runs["is_incomplete"]]
            if display_runs.empty:
                st.caption("No loadable saved runs. Enable incomplete runs above if you want to rename or delete them.")
                return

            sel = st.selectbox(
                "Select a saved run to load",
                display_runs["label"].tolist(),
                index=0,
                key=f"saved_run_select_{int(df_runs.iloc[0]['id'])}_{int(show_incomplete)}",
            )
            sel_id = int(display_runs.iloc[display_runs.index[display_runs["label"] == sel][0]]["id"])
            selected_run = display_runs[display_runs["id"] == sel_id].iloc[0].to_dict()
            rename_key = f"rename_run_{sel_id}"
            st.text_input(
                "Rename selected run",
                key=rename_key,
                value=selected_run.get("name") or "",
                placeholder="Untitled run",
            )
            col_load, col_rename, col_del = st.columns([1, 1, 1])
            with col_load:
                is_incomplete_run = bool(selected_run.get("is_incomplete"))
                if is_incomplete_run:
                    st.caption("This run has no saved result rows, so it cannot be loaded.")
                if st.button("Load this run", use_container_width=True, disabled=is_incomplete_run):
                    try:
                        loaded = db.load_run_results(sel_id)
                        if (
                            not loaded["conversation_results"]
                            and int(selected_run.get("n_conversations") or 0) > 0
                        ):
                            raise ValueError(
                                "This run has summary metadata but no saved result rows. "
                                "It cannot be reconstructed from the database."
                            )
                        rr = RunResults(
                            conversation_results=loaded["conversation_results"],
                            message_level_results=loaded["message_level_results"],
                            errors=loaded["errors"],
                            started_at=loaded["started_at"],
                            finished_at=loaded["finished_at"],
                        )
                        rr = _normalize_run_results_for_display(rr)
                        st.session_state.run_results = rr
                        st.session_state.current_run_id = sel_id
                        st.session_state.loaded_run_label = sel
                        st.success(f"Loaded run #{sel_id}.")
                    except Exception as e:
                        st.error(f"Could not load run: {e}")
            with col_rename:
                if st.button("Save name", use_container_width=True, type="secondary"):
                    try:
                        db.rename_run(sel_id, (st.session_state.get(rename_key) or "").strip())
                        st.success(f"Renamed run #{sel_id}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not rename run: {e}")
            with col_del:
                if st.button("Delete this run", use_container_width=True, type="secondary"):
                    try:
                        db.delete_run(sel_id)
                        if st.session_state.current_run_id == sel_id:
                            st.session_state.current_run_id = None
                            st.session_state.run_results = None
                        st.success(f"Deleted run #{sel_id}.")
                    except Exception as e:
                        st.error(f"Could not delete run: {e}")

    df = st.session_state.df_norm
    if df is None or df.empty:
        st.info("Upload a valid CSV in the Upload & Settings tab first.")
        return

    if not st.session_state.selected_model:
        st.warning(
            "Select a model from the sidebar before running. "
            "Click 'Load available models' to populate the list."
        )

    st.text_input(
        "Run name",
        key="run_name",
        placeholder="e.g., June renewal journeys - agent review",
        help="Saved with this run and shown in Past runs.",
    )

    target_role = str(st.session_state.message_target_role or "agent")

    # ---- Customer journey selection (sidebar scope, specific customers, random) ---
    selector_df = _journey_selector_rows(df)
    all_ids = selector_df["journey_id"].astype(str).tolist() if not selector_df.empty else []
    selected_ids = _ordered_selected_ids(all_ids, st.session_state.selected_conversation_ids)
    if selected_ids:
        st.session_state.selected_conversation_ids = selected_ids
    elif st.session_state.selected_conversation_ids:
        st.session_state.selected_conversation_ids = None

    st.markdown("### Customer journey selection")
    st.caption(
        "Leave selection empty to use the sidebar journey scope. "
        "Pin specific journeys to evaluate only those customers."
    )

    search = st.text_input(
        "Find customer journey",
        key="journey_selection_query",
        placeholder="Search by customer name, phone, journey ID, source conversation ID, or date",
    ).strip().lower()

    filtered_selector_df = selector_df
    if search and not selector_df.empty:
        filtered_selector_df = selector_df[
            selector_df["search_text"].fillna("").astype(str).str.contains(search, regex=False)
        ]

    max_visible_options = 250
    visible_selector_df = filtered_selector_df.head(max_visible_options).copy()
    visible_options = visible_selector_df["label"].tolist() if not visible_selector_df.empty else []
    label_to_id = (
        dict(zip(visible_selector_df["label"], visible_selector_df["journey_id"]))
        if not visible_selector_df.empty
        else {}
    )
    visible_key = "journey_selection_visible_labels"
    if visible_key in st.session_state:
        visible_option_set = set(visible_options)
        st.session_state[visible_key] = [
            label for label in st.session_state[visible_key] if label in visible_option_set
        ]

    st.caption(
        f"Showing {len(visible_selector_df):,} of {len(filtered_selector_df):,} matching journeys "
        f"({len(selector_df):,} total)."
    )
    picked_labels = st.multiselect(
        "Select customer journeys from the current search results",
        options=visible_options,
        key=visible_key,
        help="Pick one or more matching customer journeys, then add or replace the pinned run selection.",
    )
    picked_ids = [label_to_id[label] for label in picked_labels if label in label_to_id]

    pick_cols = st.columns([1, 1, 1, 1, 1])
    with pick_cols[0]:
        if st.button(
            "Add selected",
            use_container_width=True,
            disabled=not picked_ids,
            help="Add the selected visible journeys to the pinned run selection.",
        ):
            st.session_state.selected_conversation_ids = _ordered_selected_ids(all_ids, selected_ids + picked_ids)
            st.rerun()
    with pick_cols[1]:
        if st.button(
            "Replace with selected",
            use_container_width=True,
            disabled=not picked_ids,
            help="Run only the selected visible journeys.",
        ):
            st.session_state.selected_conversation_ids = _ordered_selected_ids(all_ids, picked_ids)
            st.rerun()
    with pick_cols[2]:
        if st.button(
            "Select all matches",
            use_container_width=True,
            disabled=filtered_selector_df.empty,
            help="Pin every journey matching the current search. If search is empty, this selects all journeys.",
        ):
            st.session_state.selected_conversation_ids = _ordered_selected_ids(
                all_ids,
                filtered_selector_df["journey_id"].astype(str).tolist(),
            )
            st.rerun()
    with pick_cols[3]:
        if st.button(
            "Random sample",
            use_container_width=True,
            help=(
                "Pick a random sample of IDs from the uploaded CSV. "
                "Sample size uses the sidebar journey count, or all journeys when Run all uploaded journeys is enabled."
            ),
            disabled=not all_ids,
        ):
            import random
            if st.session_state.get("run_all_conversations"):
                n = len(all_ids)
            else:
                n = max(1, int(st.session_state.max_conversations or 1))
            n = min(n, len(all_ids))
            st.session_state.selected_conversation_ids = _ordered_selected_ids(all_ids, random.sample(all_ids, n))
            st.rerun()
    with pick_cols[4]:
        if st.button(
            "Clear selection",
            use_container_width=True,
            disabled=not selected_ids,
        ):
            st.session_state.selected_conversation_ids = None
            st.rerun()

    if selected_ids:
        st.success(
            f"{len(selected_ids):,} customer journey/journeys pinned. "
            "The run will ignore the sidebar journey count and evaluate only this pinned selection."
        )
        selected_preview_df = selector_df[selector_df["journey_id"].astype(str).isin(set(selected_ids))].copy()
        selected_preview_df["order"] = selected_preview_df["journey_id"].astype(str).map(
            {journey_id: idx for idx, journey_id in enumerate(selected_ids)}
        )
        selected_preview_df = selected_preview_df.sort_values("order")
        preview_cols = [
            "customer_phone",
            "customer_name",
            "source_conversation_count",
            "message_count",
            "conversation_start_date",
            "conversation_end_date",
        ]
        with st.expander("Pinned customer journeys", expanded=False):
            st.dataframe(
                selected_preview_df[[c for c in preview_cols if c in selected_preview_df.columns]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        if st.session_state.get("run_all_conversations"):
            st.info("No pinned selection. The run will evaluate all customer journeys from the CSV.")
        else:
            st.info(
                "No pinned selection. The run will evaluate the first "
                f"{int(st.session_state.max_conversations):,} customer journeys from the CSV."
            )

    # Build the estimate. When a random selection is active, count over the
    # pinned IDs; otherwise apply the max_conversations slice.
    if selected_ids:
        df_for_estimate = df[df[JOURNEY_ID_COLUMN].astype(str).isin(set(map(str, selected_ids)))]
        estimate = estimate_call_counts(
            df_for_estimate,
            max_conversations=None,
            max_agent_messages_per_conv=int(st.session_state.max_agent_messages_per_conv),
            target_role=target_role,
        )
    else:
        max_conversations_for_estimate = (
            None
            if st.session_state.get("run_all_conversations")
            else int(st.session_state.max_conversations)
        )
        estimate = estimate_call_counts(
            df,
            max_conversations=max_conversations_for_estimate,
            max_agent_messages_per_conv=int(st.session_state.max_agent_messages_per_conv),
            target_role=target_role,
        )

    st.markdown("### Evaluation estimate")
    role_label = "assistant" if target_role == "agent" else "customer"
    st.caption(
        f"Message-level layer will evaluate **{role_label} messages** "
        + ("(judging the assistant's response to a possibly-frustrated customer message)."
           if target_role == "agent"
           else "(capturing the customer's state / frustration before the assistant answers).")
    )
    metric_row(
        [
            ("Customer journeys to evaluate", f"{estimate['conversations']:,}", None),
            (f"{role_label.capitalize()}-message AI calls", f"{estimate['message_level_calls']:,}", None),
            ("Journey-level AI calls", f"{estimate['conversation_level_calls']:,}", None),
            ("Total estimated AI calls", f"{estimate['total_calls']:,}", None),
        ]
    )

    large_job = estimate["total_calls"] > 200
    if large_job:
        scope_hint = (
            "Turn off Run all uploaded journeys, lower the journey count, or lower Max target messages per journey in the sidebar."
            if st.session_state.get("run_all_conversations") and not selected_ids
            else "Consider lowering the journey count or Max target messages per journey in the sidebar."
        )
        st.warning(
            f"This run will make ~{estimate['total_calls']:,} AI calls. "
            + scope_hint
        )

    run_col, cancel_col, _ = st.columns([1, 1, 4])
    with run_col:
        run_clicked = st.button(
            "Run CX Evaluation",
            type="primary",
            disabled=st.session_state.run_in_progress or not st.session_state.selected_model,
            use_container_width=True,
        )
    with cancel_col:
        if st.session_state.run_in_progress:
            if st.button("Cancel run", use_container_width=True):
                st.session_state.cancel_flag = True
                st.toast("Cancelling after current call finishes...")

    progress_box = st.empty()
    bar = st.progress(0, text="Idle")
    counter_box = st.empty()
    current_box = st.empty()
    log_box = st.empty()

    if run_clicked:
        st.session_state.run_in_progress = True
        st.session_state.cancel_flag = False
        st.session_state.progress_log = []

        config, ml_prompt_id, cl_prompt_id = _build_run_config()
        client = build_client(config.api.base_url, config.api.api_key)

        # Start a DB run record.
        db = get_db()
        run_config_serializable = {
            "api_base_url": config.api.base_url,
            "model": config.api.model,
            "temperature": config.api.temperature,
            "top_p": config.api.top_p,
            "max_tokens": config.api.max_tokens,
            "timeout": config.api.timeout,
            "retries": config.api.retries,
            "concurrency": config.api.concurrency,
            "run_all_conversations": bool(st.session_state.get("run_all_conversations")),
            "max_conversations": config.max_conversations,
            "max_target_messages_per_journey": config.max_agent_messages_per_conv,
            "truncate_messages": config.truncate_messages,
            "max_chars_per_message": config.max_chars_per_message,
            "include_unknown_in_history": config.include_unknown_in_history,
            "stop_on_error": config.stop_on_error,
            "save_raw_responses": config.save_raw_responses,
            "message_target_role": config.message_target_role,
            "selected_conversation_ids": config.selected_conversation_ids,
            "selected_conversation_count": len(config.selected_conversation_ids or []),
            "run_name": (st.session_state.run_name or "").strip(),
        }
        run_name = (st.session_state.run_name or "").strip() or None
        run_id = db.start_run(
            csv_name=st.session_state.csv_name,
            run_config=run_config_serializable,
            message_prompt_id=ml_prompt_id,
            conversation_prompt_id=cl_prompt_id,
            name=run_name,
        )
        st.session_state.current_run_id = run_id
        st.session_state.loaded_run_label = None

        total_conv = estimate["conversations"]
        total_msg = estimate["message_level_calls"] + estimate["conversation_level_calls"]
        progress_state = {"convs_done": 0, "calls_done": 0, "successes": 0, "failures": 0}

        def on_progress(evt: dict) -> None:
            phase = evt.get("phase")
            if phase == "conversation_start":
                current_box.info(
                    f"Journey {evt.get('conversation_index')}/{evt.get('total_conversations')} — "
                    f"Customer `{evt.get('conversation_id')}` — "
                    f"{evt.get('agent_messages', 0)} target messages"
                )
            elif phase == "message_done":
                progress_state["calls_done"] += 1
                if evt.get("status") == "ok":
                    progress_state["successes"] += 1
                else:
                    progress_state["failures"] += 1
            elif phase == "conversation_done":
                progress_state["convs_done"] += 1
                progress_state["calls_done"] += 1
                if evt.get("status") == "ok":
                    progress_state["successes"] += 1
                else:
                    progress_state["failures"] += 1

            if total_msg > 0:
                frac = min(progress_state["calls_done"] / max(total_msg, 1), 1.0)
            else:
                frac = 0.0
            bar.progress(
                frac,
                text=f"Journeys {progress_state['convs_done']}/{total_conv} • Calls {progress_state['calls_done']}/{total_msg}",
            )
            counter_box.markdown(
                f"**Successes:** {progress_state['successes']}  |  **Failures:** {progress_state['failures']}"
            )
            st.session_state.progress_log.append(evt)

        def cancel_requested() -> bool:
            return bool(st.session_state.cancel_flag)

        persistence_errors: list[str] = []

        def save_message(mr: dict) -> None:
            try:
                mr["run_id"] = run_id
                db.save_message_result(run_id, mr)
            except Exception as e:
                persistence_errors.append(f"message result: {e}")

        def save_conversation(cr: dict) -> None:
            try:
                cr["run_id"] = run_id
                db.save_conversation_result(run_id, cr)
            except Exception as e:
                persistence_errors.append(f"conversation result: {e}")

        def save_err(err: dict) -> None:
            try:
                db.save_error(run_id, err)
            except Exception as e:
                persistence_errors.append(f"run error: {e}")

        def persist_completed_results() -> None:
            if results is None:
                return
            counts = _run_result_counts(db, run_id)
            expected_convs = len(results.conversation_results)
            expected_msgs = len(results.message_level_results)
            expected_errors = len(results.errors)
            if (
                counts["conversation_results"] == expected_convs
                and counts["message_results"] == expected_msgs
                and counts["run_errors"] == expected_errors
            ):
                return
            _clear_run_results(db, run_id)
            for mr in results.message_level_results:
                mr["run_id"] = run_id
                db.save_message_result(run_id, mr)
            for cr in results.conversation_results:
                cr["run_id"] = run_id
                db.save_conversation_result(run_id, cr)
            for err in results.errors:
                db.save_error(run_id, err)

        results = None
        try:
            progress_box.info("Starting evaluation...")
            results = run_evaluation(
                df=df,
                client=client,
                config=config,
                on_progress=on_progress,
                cancel_requested=cancel_requested,
                on_message_result=save_message,
                on_conversation_result=save_conversation,
                on_error=save_err,
            )
            st.session_state.run_results = results
            persist_completed_results()
            progress_box.success(
                f"Evaluation finished. {len(results.conversation_results)} customer journeys processed, "
                f"{len(results.message_level_results)} message-level calls, "
                f"{len(results.errors)} errors. Saved as run #{run_id}."
            )
            if persistence_errors:
                st.warning(
                    "Some live DB saves failed during the run, but the completed results were saved again at the end. "
                    f"First error: {persistence_errors[0]}"
                )
        except Exception as e:
            progress_box.error(f"Evaluation failed: {e}")
        finally:
            # Finalize the run record regardless of outcome.
            try:
                status = "completed"
                if st.session_state.cancel_flag:
                    status = "cancelled"
                elif results is None:
                    status = "failed"
                n_convs = len(results.conversation_results) if results else 0
                n_msgs = len(results.message_level_results) if results else 0
                n_err = len(results.errors) if results else 0
                db.finish_run(run_id, status, n_convs, n_msgs, n_err)
            except Exception:
                pass
            st.session_state.run_in_progress = False
            st.session_state.cancel_flag = False

    if _has_results():
        rr = st.session_state.run_results
        st.markdown("### Last run")
        metric_row(
            [
                ("Customer journeys", f"{len(rr.conversation_results):,}", None),
                ("Message calls", f"{len(rr.message_level_results):,}", None),
                ("Errors", f"{len(rr.errors):,}", None),
                ("Duration (s)", f"{(rr.finished_at or 0) - (rr.started_at or 0):.1f}", None),
            ]
        )

        if rr.errors:
            with st.expander(f"View {len(rr.errors)} non-fatal errors"):
                st.dataframe(pd.DataFrame(rr.errors), use_container_width=True)


# --------- Tab: Dashboard ---------

# Color palette used across the dashboard (tuned for the dark theme).
_DASH_COLORS = {
    "panel_bg": "#11172a",
    "panel_top": "#0c1224",
    "panel_border": "#1f2a44",
    "text": "#f1f5f9",
    "muted": "#94a3b8",
    "dim": "#64748b",
    "track": "#1f2937",
    "handled": "#10b981",
    "unhandled": "#ef4444",
    "many": "#f97316",
    "minimal": "#22c55e",
    "frustrated": "#f59e0b",
    "calm": "#38bdf8",
    "our_side": "#fb923c",
    "customer": "#60a5fa",
    "shared": "#c084fc",
    "none": "#64748b",
    "unclear": "#475569",
    "review_yes": "#a78bfa",
    "review_no": "#334155",
    "heat_low": "#1e293b",
    "heat_mid": "#7c2d12",
    "heat_high": "#ef4444",
}


def _pct(part: float, whole: float) -> float:
    return float(part) / float(whole) * 100.0 if whole else 0.0


def _safe_col(df: pd.DataFrame, col: str, default: Any = "") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _norm_marker_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    series = _safe_col(df, col, default)
    return (
        series.fillna(default)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )


def _bool_marker_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return (
        df[col]
        .map(lambda value: str(value if value is not None else default).strip().lower() in {"true", "1", "yes", "y", "frustrated"})
    )


def _kpi_card_html(label: str, value: str, sub: str, segments: list[tuple[str, int, str]]) -> str:
    """Render a KPI card with a mini stacked bar and color-coded legend."""
    total = sum(max(int(c), 0) for _, c, _ in segments) or 1
    bar = ""
    legend = ""
    for name, count, color in segments:
        if count <= 0:
            continue
        share = max(int(count), 0) / total * 100
        bar += (
            f'<div style="flex:{share:.4f}; min-width:0; background:{color};"'
            f' title="{html_lib.escape(name)}: {count}"></div>'
        )
        legend += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:11px;line-height:1.4;">'
            f'<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:{color};"></span>'
            f'<span style="color:{_DASH_COLORS["muted"]};">{html_lib.escape(name)}</span>'
            f'<b style="color:{_DASH_COLORS["text"]};">{count:,}</b></span>'
        )
    return (
        f'<div style="border:1px solid {_DASH_COLORS["panel_border"]};'
        f'border-radius:14px;padding:14px 16px 12px;'
        f'background:linear-gradient(180deg,{_DASH_COLORS["panel_bg"]} 0%,{_DASH_COLORS["panel_top"]} 100%);">'
        f'<div style="font-size:0.7rem;letter-spacing:0.10em;text-transform:uppercase;'
        f'color:{_DASH_COLORS["muted"]};margin-bottom:6px;">{html_lib.escape(label)}</div>'
        f'<div style="font-size:1.7rem;font-weight:800;line-height:1.1;color:{_DASH_COLORS["text"]};">{html_lib.escape(value)}</div>'
        f'<div style="font-size:0.76rem;color:{_DASH_COLORS["muted"]};margin-top:3px;margin-bottom:11px;">{sub}</div>'
        f'<div style="display:flex;height:8px;border-radius:6px;overflow:hidden;background:{_DASH_COLORS["track"]};">{bar}</div>'
        f'<div style="font-size:0.7rem;margin-top:8px;">{legend}</div>'
        f'</div>'
    )


def _node_html(label: str, count: int, parent: int, total: int, depth: int, color: str) -> str:
    """Render one node in a cascading tree, showing share of parent and of total."""
    of_parent = _pct(count, parent)
    of_total = _pct(count, total)
    indent = depth * 16
    return (
        f'<div style="padding:5px 0 5px {indent + 12}px;border-left:2px solid {color};margin-left:{indent}px;">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;">'
        f'<div style="color:{_DASH_COLORS["text"]};font-size:0.86rem;"><b>{html_lib.escape(label)}</b>'
        f' <span style="color:{_DASH_COLORS["muted"]};font-weight:400;">· {count:,}</span></div>'
        f'<div style="color:{_DASH_COLORS["muted"]};font-size:0.74rem;white-space:nowrap;">'
        f'{of_parent:.1f}% of parent · <span style="color:{color};">{of_total:.1f}% of total</span></div>'
        f'</div>'
        f'<div style="margin-top:4px;height:5px;border-radius:3px;background:{_DASH_COLORS["track"]};overflow:hidden;">'
        f'<div style="width:{of_parent:.2f}%;height:100%;background:{color};"></div>'
        f'</div></div>'
    )


def _section_header(title: str, caption: str | None = None) -> None:
    st.markdown(
        f'<div style="margin-top:8px;margin-bottom:4px;">'
        f'<div style="font-size:1.15rem;font-weight:700;color:{_DASH_COLORS["text"]};">{html_lib.escape(title)}</div>'
        + (
            f'<div style="font-size:0.82rem;color:{_DASH_COLORS["muted"]};margin-top:2px;">{html_lib.escape(caption)}</div>'
            if caption else ""
        )
        + '</div>',
        unsafe_allow_html=True,
    )


def _render_kpi_strip(filtered: pd.DataFrame, msg_df: pd.DataFrame, agg: dict, total: int) -> None:
    handled_series = _norm_marker_series(filtered, "handled_status")
    experience_series = _norm_marker_series(filtered, "customer_experience")
    experience_series = experience_series.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
    handled = int((handled_series == "handled").sum())
    unhandled = int((handled_series == "unhandled").sum())
    bad = int((experience_series == "bad").sum())
    good = int((experience_series == "good").sum())
    unknown_experience = max(total - bad - good, 0)

    frustrated = int(_bool_marker_series(filtered, "frustration_detected").sum())
    calm = total - frustrated

    if "frustration_origin" in filtered.columns:
        origin_series = _norm_marker_series(filtered, "frustration_origin", "none")
        origin_series = origin_series.replace({"customer": "customer_side", "our": "our_side", "agent": "our_side"})
        oc = origin_series.value_counts().to_dict()
    else:
        oc = {}
    our_side = int(oc.get("our_side", 0))
    customer = int(oc.get("customer_side", 0))
    shared = int(oc.get("shared", 0))
    no_issue = int(oc.get("none", 0))
    unclear = max(total - our_side - customer - shared - no_issue, 0)

    review_flag = int(agg.get("manual_review_count", 0))
    high_frust = int(agg.get("high_frustration_count", 0))
    msg_count = int(len(msg_df)) if msg_df is not None else 0
    avg_score_text = ""
    if "score_final" in filtered.columns:
        score_series = pd.to_numeric(filtered["score_final"], errors="coerce").dropna()
        if not score_series.empty:
            avg_score_text = f" · Avg score {score_series.mean():.1f}"

    cards = [
        _kpi_card_html(
            "Total journeys",
            f"{total:,}",
            f"{msg_count:,} agent messages · {review_flag:,} flagged for review · {high_frust:,} high-frustration{avg_score_text}",
            [
                ("Flagged", review_flag, _DASH_COLORS["review_yes"]),
                ("Not flagged", max(total - review_flag, 0), _DASH_COLORS["review_no"]),
            ],
        ),
        _kpi_card_html(
            "Outcome",
            f"{_pct(handled, total):.1f}% handled",
            f"Handled {handled:,} · Not handled {unhandled:,}",
            [
                ("Handled", handled, _DASH_COLORS["handled"]),
                ("Not handled", unhandled, _DASH_COLORS["unhandled"]),
            ],
        ),
        _kpi_card_html(
            "Customer experience",
            f"{_pct(bad, total):.1f}% bad",
            f"Bad {bad:,} · Good {good:,}",
            [
                ("Bad", bad, _DASH_COLORS["many"]),
                ("Good", good, _DASH_COLORS["minimal"]),
                ("Unknown", unknown_experience, _DASH_COLORS["unclear"]),
            ],
        ),
        _kpi_card_html(
            "Frustration",
            f"{_pct(frustrated, total):.1f}% frustrated",
            f"Frustrated {frustrated:,} · Calm {calm:,}",
            [
                ("Frustrated", frustrated, _DASH_COLORS["frustrated"]),
                ("Calm", calm, _DASH_COLORS["calm"]),
            ],
        ),
        _kpi_card_html(
            "Frustration origin",
            f"{_pct(our_side, total):.1f}% our side",
            f"Our {our_side:,} · Customer {customer:,} · Shared {shared:,} · None {no_issue:,}",
            [
                ("Our side", our_side, _DASH_COLORS["our_side"]),
                ("Customer", customer, _DASH_COLORS["customer"]),
                ("Shared", shared, _DASH_COLORS["shared"]),
                ("None", no_issue, _DASH_COLORS["none"]),
                ("Unclear", unclear, _DASH_COLORS["unclear"]),
            ],
        ),
    ]
    cols = st.columns(5, gap="small")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)


def _render_outcome_sunburst(filtered: pd.DataFrame) -> None:
    if filtered.empty or not HAS_PLOTLY or "handled_status" not in filtered.columns:
        st.caption("Sunburst unavailable.")
        return
    work = filtered.copy()
    work["Outcome"] = _norm_marker_series(work, "handled_status", "unknown").map(
        {"handled": "Handled", "unhandled": "Not handled"}
    ).fillna("Unknown")
    experience_series = _norm_marker_series(work, "customer_experience", "unknown")
    experience_series = experience_series.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
    work["Experience"] = experience_series.map(
        {"bad": "Bad", "good": "Good"}
    ).fillna("Unknown")
    work["Frustration"] = _bool_marker_series(work, "frustration_detected").map(
        {True: "Frustrated", False: "Calm"}
    )

    grp = work.groupby(["Outcome", "Experience", "Frustration"]).size().reset_index(name="Count")
    grp = grp[grp["Count"] > 0]
    if grp.empty:
        st.caption("No data.")
        return
    fig = px.sunburst(
        grp,
        path=["Outcome", "Experience", "Frustration"],
        values="Count",
        color="Outcome",
        color_discrete_map={
            "Handled": _DASH_COLORS["handled"],
            "Not handled": _DASH_COLORS["unhandled"],
            "Unknown": _DASH_COLORS["none"],
            "Bad": _DASH_COLORS["many"],
            "Good": _DASH_COLORS["minimal"],
        },
        branchvalues="total",
    )
    fig.update_traces(textinfo="label+percent parent", insidetextorientation="radial")
    _plotly_layout(fig, height=440, margin=dict(t=8, b=8, l=8, r=8))
    _render_plotly(fig)


def _render_outcome_cascade(filtered: pd.DataFrame, total: int) -> None:
    if filtered.empty or total == 0:
        st.caption("No data.")
        return
    chunks: list[str] = [
        f'<div style="font-size:0.84rem;color:{_DASH_COLORS["muted"]};margin-bottom:8px;">'
        f'All journeys · <b style="color:{_DASH_COLORS["text"]};">{total:,}</b></div>'
    ]
    for outcome_val, outcome_label, outcome_color in (
        ("handled", "Handled", _DASH_COLORS["handled"]),
        ("unhandled", "Not handled", _DASH_COLORS["unhandled"]),
    ):
        outcome_df = filtered[_norm_marker_series(filtered, "handled_status") == outcome_val]
        outcome_count = int(len(outcome_df))
        if outcome_count == 0:
            continue
        chunks.append(_node_html(outcome_label, outcome_count, total, total, 0, outcome_color))
        for sev_val, sev_label, sev_color in (
            ("bad", "Bad experience", _DASH_COLORS["many"]),
            ("good", "Good experience", _DASH_COLORS["minimal"]),
        ):
            experience_series = _norm_marker_series(outcome_df, "customer_experience")
            experience_series = experience_series.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
            sev_df = outcome_df[experience_series == sev_val]
            sev_count = int(len(sev_df))
            if sev_count == 0:
                continue
            chunks.append(_node_html(sev_label, sev_count, outcome_count, total, 1, sev_color))
            fr_yes = int(_bool_marker_series(sev_df, "frustration_detected").sum())
            fr_no = sev_count - fr_yes
            if fr_yes:
                chunks.append(_node_html("Frustrated", fr_yes, sev_count, total, 2, _DASH_COLORS["frustrated"]))
            if fr_no:
                chunks.append(_node_html("Calm", fr_no, sev_count, total, 2, _DASH_COLORS["calm"]))
    st.markdown("".join(chunks), unsafe_allow_html=True)


def _render_issue_sunburst(filtered: pd.DataFrame) -> None:
    if not HAS_PLOTLY or filtered.empty or "main_issue_origin" not in filtered.columns:
        st.caption("Origin sunburst unavailable.")
        return
    work = filtered.copy()
    work["Origin"] = work["main_issue_origin"].fillna("none").astype(str).apply(humanize_label)
    work["Issue type"] = _safe_col(work, "main_issue_type", "none").fillna("none").astype(str).apply(humanize_label)
    experience_series = _norm_marker_series(work, "customer_experience", "unknown")
    experience_series = experience_series.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
    work["Experience"] = experience_series.map(
        {"bad": "Bad", "good": "Good"}
    ).fillna("Unknown")
    grp = work.groupby(["Origin", "Issue type", "Experience"]).size().reset_index(name="Count")
    grp = grp[grp["Count"] > 0]
    if grp.empty:
        st.caption("No data.")
        return
    fig = px.sunburst(
        grp,
        path=["Origin", "Issue type", "Experience"],
        values="Count",
        color="Origin",
        color_discrete_map={
            "Our Side": _DASH_COLORS["our_side"],
            "Customer side": _DASH_COLORS["customer"],
            "Shared": _DASH_COLORS["shared"],
            "None": _DASH_COLORS["none"],
            "Unclear": _DASH_COLORS["unclear"],
        },
        branchvalues="total",
    )
    fig.update_traces(textinfo="label+percent parent", insidetextorientation="radial")
    _plotly_layout(fig, height=440, margin=dict(t=8, b=8, l=8, r=8))
    _render_plotly(fig)


def _render_issue_cascade(filtered: pd.DataFrame, total: int) -> None:
    if filtered.empty or "main_issue_origin" not in filtered.columns:
        st.caption("No data.")
        return
    chunks: list[str] = [
        f'<div style="font-size:0.84rem;color:{_DASH_COLORS["muted"]};margin-bottom:8px;">'
        f'Issues across <b style="color:{_DASH_COLORS["text"]};">{total:,}</b> journeys</div>'
    ]
    origin_palette = {
        "our_side": _DASH_COLORS["our_side"],
        "customer_side": _DASH_COLORS["customer"],
        "shared": _DASH_COLORS["shared"],
        "none": _DASH_COLORS["none"],
        "unclear": _DASH_COLORS["unclear"],
    }
    origins = (
        filtered["main_issue_origin"].fillna("none").astype(str).value_counts()
    )
    for origin_val, origin_count in origins.items():
        color = origin_palette.get(origin_val, _DASH_COLORS["dim"])
        chunks.append(
            _node_html(humanize_label(origin_val), int(origin_count), total, total, 0, color)
        )
        sub_df = filtered[filtered["main_issue_origin"].fillna("none") == origin_val]
        type_counts = (
            _safe_col(sub_df, "main_issue_type", "none").fillna("none").astype(str).value_counts()
        )
        for type_val, type_count in type_counts.head(6).items():
            chunks.append(
                _node_html(
                    humanize_label(type_val),
                    int(type_count),
                    int(origin_count),
                    total,
                    1,
                    color,
                )
            )
    st.markdown("".join(chunks), unsafe_allow_html=True)


def _render_frustration_funnel(filtered: pd.DataFrame, total: int) -> None:
    if not HAS_PLOTLY or filtered.empty:
        st.caption("Funnel unavailable.")
        return
    frust_detected = int(_bool_marker_series(filtered, "frustration_detected").sum())
    if "frustration_timing" in filtered.columns:
        multi_or_during = int(
            filtered["frustration_timing"].fillna("").isin(["during", "multiple"]).sum()
        )
    else:
        multi_or_during = 0
    if "max_frustration_level" in filtered.columns:
        high_or_cancel = int(
            filtered["max_frustration_level"].fillna("").isin(["high", "cancellation_risk"]).sum()
        )
        cancel = int((filtered["max_frustration_level"].fillna("") == "cancellation_risk").sum())
    else:
        high_or_cancel = cancel = 0
    steps = [
        ("All journeys", total, _DASH_COLORS["calm"]),
        ("Frustration detected", frust_detected, _DASH_COLORS["frustrated"]),
        ("During or multi-timing", multi_or_during, "#fb7185"),
        ("High / cancellation level", high_or_cancel, _DASH_COLORS["unhandled"]),
        ("Cancellation risk", cancel, "#b91c1c"),
    ]
    fig = go.Figure(
        go.Funnel(
            y=[s[0] for s in steps],
            x=[s[1] for s in steps],
            textposition="inside",
            textinfo="value+percent initial",
            marker={"color": [s[2] for s in steps]},
            connector={"line": {"color": _DASH_COLORS["panel_border"], "width": 1}},
        )
    )
    _plotly_layout(fig, height=380, margin=dict(t=10, b=10, l=10, r=10))
    _render_plotly(fig)


def _render_frustration_cascade(filtered: pd.DataFrame, total: int) -> None:
    if filtered.empty or "frustration_detected" not in filtered.columns:
        st.caption("No data.")
        return
    chunks: list[str] = []
    frust_yes_df = filtered[_bool_marker_series(filtered, "frustration_detected")]
    frust_yes = int(len(frust_yes_df))
    frust_no = total - frust_yes
    chunks.append(
        _node_html("Frustration detected", frust_yes, total, total, 0, _DASH_COLORS["frustrated"])
    )
    if "frustration_timing" in frust_yes_df.columns:
        for timing in ["start", "during", "multiple"]:
            t_df = frust_yes_df[frust_yes_df["frustration_timing"].fillna("") == timing]
            t_count = int(len(t_df))
            if t_count == 0:
                continue
            chunks.append(
                _node_html(
                    humanize_label(timing), t_count, frust_yes, total, 1, _DASH_COLORS["frustrated"]
                )
            )
            if "max_frustration_level" in t_df.columns:
                level_counts = t_df["max_frustration_level"].fillna("none").astype(str).value_counts()
                level_palette = {
                    "low": "#fde68a",
                    "medium": "#fb923c",
                    "high": _DASH_COLORS["unhandled"],
                    "cancellation_risk": "#b91c1c",
                    "none": _DASH_COLORS["dim"],
                }
                for lvl, lvl_count in level_counts.items():
                    chunks.append(
                        _node_html(
                            humanize_label(lvl),
                            int(lvl_count),
                            t_count,
                            total,
                            2,
                            level_palette.get(lvl, _DASH_COLORS["dim"]),
                        )
                    )
    chunks.append(
        _node_html("No visible frustration", frust_no, total, total, 0, _DASH_COLORS["calm"])
    )
    st.markdown("".join(chunks), unsafe_allow_html=True)


def _render_timing_level_heatmap(filtered: pd.DataFrame) -> None:
    if not HAS_PLOTLY or filtered.empty:
        return
    if "frustration_timing" not in filtered.columns or "max_frustration_level" not in filtered.columns:
        return
    timing_order = ["none", "start", "during", "multiple"]
    level_order = ["none", "low", "medium", "high", "cancellation_risk"]
    work = filtered[["frustration_timing", "max_frustration_level"]].copy()
    work["frustration_timing"] = work["frustration_timing"].fillna("none").astype(str)
    work["max_frustration_level"] = work["max_frustration_level"].fillna("none").astype(str)
    mat = pd.crosstab(work["frustration_timing"], work["max_frustration_level"])
    rows = [r for r in timing_order if r in mat.index]
    cols = [c for c in level_order if c in mat.columns]
    if not rows or not cols:
        return
    mat = mat.reindex(index=rows, columns=cols, fill_value=0)
    fig = px.imshow(
        mat.values,
        x=[humanize_label(c) for c in mat.columns],
        y=[humanize_label(r) for r in mat.index],
        labels=dict(x="Max frustration level", y="Frustration timing", color="Journeys"),
        text_auto=True,
        color_continuous_scale=[_DASH_COLORS["heat_low"], _DASH_COLORS["heat_mid"], _DASH_COLORS["heat_high"]],
        aspect="auto",
    )
    _plotly_layout(fig, height=320, margin=dict(t=10, b=10, l=10, r=10))
    _render_plotly(fig)


def _render_overall_sankey(filtered: pd.DataFrame) -> None:
    if not HAS_PLOTLY or filtered.empty:
        st.caption("Sankey unavailable.")
        return
    work = filtered.copy()
    work["L1 Outcome"] = _norm_marker_series(work, "handled_status", "unknown").map(
        {"handled": "Handled", "unhandled": "Not handled"}
    ).fillna("Unknown")
    experience_series = _norm_marker_series(work, "customer_experience", "unknown")
    experience_series = experience_series.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
    work["L2 Experience"] = experience_series.map(
        {"bad": "Bad experience", "good": "Good experience"}
    ).fillna("Unknown")
    origin_series = _norm_marker_series(work, "frustration_origin", "none")
    origin_series = origin_series.replace({"customer": "customer_side", "our": "our_side", "agent": "our_side"})
    work["L3 Frustration Origin"] = origin_series.apply(humanize_label)
    work["L4 Frustration"] = _safe_col(work, "frustration_timing", "none").fillna("none").astype(str).apply(humanize_label)

    levels = ["L1 Outcome", "L2 Experience", "L3 Frustration Origin", "L4 Frustration"]
    label_to_id: dict[tuple[int, str], int] = {}
    labels: list[str] = []
    node_colors: list[str] = []

    color_map = {
        "Handled": _DASH_COLORS["handled"],
        "Not handled": _DASH_COLORS["unhandled"],
        "Bad experience": _DASH_COLORS["many"],
        "Good experience": _DASH_COLORS["minimal"],
        "Our Side": _DASH_COLORS["our_side"],
        "Customer side": _DASH_COLORS["customer"],
        "Shared": _DASH_COLORS["shared"],
        "None": _DASH_COLORS["none"],
        "Unclear": _DASH_COLORS["unclear"],
        "Start": "#fde68a",
        "During": "#fb923c",
        "Multiple": _DASH_COLORS["unhandled"],
    }
    for i, lev in enumerate(levels):
        for val in work[lev].dropna().unique().tolist():
            key = (i, val)
            if key not in label_to_id:
                label_to_id[key] = len(labels)
                labels.append(str(val))
                node_colors.append(color_map.get(str(val), _DASH_COLORS["dim"]))

    src: list[int] = []
    tgt: list[int] = []
    val: list[int] = []
    link_colors: list[str] = []
    for ai, a in enumerate(levels[:-1]):
        b = levels[ai + 1]
        pair_counts = work.groupby([a, b]).size()
        for (av, bv), count in pair_counts.items():
            src_id = label_to_id[(ai, av)]
            tgt_id = label_to_id[(ai + 1, bv)]
            src.append(src_id)
            tgt.append(tgt_id)
            val.append(int(count))
            base = node_colors[src_id]
            link_colors.append(_hex_to_rgba(base, 0.28))

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=18,
                thickness=16,
                color=node_colors,
                line=dict(color=_DASH_COLORS["panel_border"], width=0.5),
            ),
            link=dict(source=src, target=tgt, value=val, color=link_colors),
        )
    )
    _plotly_layout(fig, height=460, margin=dict(t=10, b=10, l=10, r=10))
    _render_plotly(fig)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(148,163,184,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------- Overview tab: marker family tree ---------


def _overview_tree_spec() -> list[dict]:
    """Describe the family tree using the Sami marker fields."""
    return [
        {
            "key": "handled",
            "title": "1. Handled",
            "short_name": "Handled",
            "tone": "good",
            "handled_status": "handled",
            "subtype": None,
            "experiences": [
                {"title": "Good experience", "tone": "good", "value": "good"},
                {"title": "Bad experience", "tone": "bad", "value": "bad"},
            ],
        },
        {
            "key": "pending",
            "title": "2.1 Not Handled — Pending Unresolved",
            "short_name": "Pending Unresolved",
            "tone": "warn",
            "handled_status": "unhandled",
            "subtype": "pending_unresolved",
            "experiences": [
                {"title": "Good experience", "tone": "good", "value": "good"},
                {"title": "Bad experience", "tone": "bad", "value": "bad"},
            ],
        },
        {
            "key": "totally",
            "title": "2.2 Not Handled — Totally Unresolved",
            "short_name": "Totally Unresolved",
            "tone": "bad",
            "handled_status": "unhandled",
            "subtype": "totally_unresolved",
            "experiences": [
                {"title": "Good experience", "tone": "good", "value": "good"},
                {"title": "Bad experience", "tone": "bad", "value": "bad"},
            ],
        },
    ]


def _overview_tone_color(tone: str) -> str:
    """Map a node tone to a dashboard color (looked up at call time)."""
    return {
        "good": _DASH_COLORS["handled"],
        "warn": _DASH_COLORS["many"],
        "bad": _DASH_COLORS["unhandled"],
    }.get(tone, _DASH_COLORS["none"])


def _overview_node_df(
    conv_df: pd.DataFrame,
    handled_status: str,
    subtype: str | None,
    customer_experience: str | None = None,
    frustration_origin: str | None = None,
) -> pd.DataFrame:
    """Slice the conversation table for one node of the family tree."""
    if conv_df.empty or "handled_status" not in conv_df.columns:
        return conv_df.iloc[0:0]

    mask = _norm_marker_series(conv_df, "handled_status") == handled_status

    if subtype is not None and "unhandled_resolution_subtype" in conv_df.columns:
        sub = _norm_marker_series(conv_df, "unhandled_resolution_subtype")
        mask &= sub == subtype

    if customer_experience is not None and "customer_experience" in conv_df.columns:
        exp = _norm_marker_series(conv_df, "customer_experience")
        exp = exp.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
        mask &= exp == customer_experience

    if frustration_origin is not None and "frustration_origin" in conv_df.columns:
        origin = _norm_marker_series(conv_df, "frustration_origin")
        origin = origin.replace({"customer": "customer_side", "our": "our_side", "agent": "our_side"})
        mask &= origin == frustration_origin

    return conv_df[mask]


def _overview_count_bar(label: str, count: int, total: int, color: str, depth: int = 0) -> str:
    """One row in the tree: label, count, % of total, and a progress bar."""
    share = _pct(count, total)
    indent = depth * 18
    return (
        f'<div style="padding:6px 0 6px {indent + 12}px;border-left:3px solid {color};'
        f'margin-left:{indent}px;margin-bottom:2px;">'
        f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;">'
        f'<div style="color:{_DASH_COLORS["text"]};font-size:0.9rem;">{html_lib.escape(label)}</div>'
        f'<div style="color:{_DASH_COLORS["muted"]};font-size:0.8rem;white-space:nowrap;">'
        f'<b style="color:{color};">{count:,}</b> · {share:.1f}%</div>'
        f'</div>'
        f'<div style="margin-top:4px;height:5px;border-radius:3px;background:{_DASH_COLORS["track"]};overflow:hidden;">'
        f'<div style="width:{share:.2f}%;height:100%;background:{color};"></div>'
        f'</div></div>'
    )


_OVERVIEW_JOURNEY_COLUMNS = {
    "conversation_id": "ID",
    "customer_name": "Customer",
    "handled_status": "Outcome",
    "customer_experience": "Experience",
    "unhandled_resolution_subtype": "Unresolved status",
    "frustration_origin": "Frustration origin",
    "main_issue_type": "Main issue",
    "main_issue_origin": "Origin",
    "max_frustration_level": "Max frustration",
    "final_customer_sentiment": "Final sentiment",
    "main_issue_summary": "Issue summary",
    "customer_impact": "Customer impact",
    "manual_review_required": "Needs review",
}


def _overview_journey_table(node_df: pd.DataFrame) -> pd.DataFrame:
    """Build the issue-focused journey list shown when a leaf is expanded."""
    cols = [c for c in _OVERVIEW_JOURNEY_COLUMNS if c in node_df.columns]
    view = node_df[cols].copy()
    for c in (
        "handled_status",
        "customer_experience",
        "unhandled_resolution_subtype",
        "frustration_origin",
        "main_issue_type",
        "main_issue_origin",
        "max_frustration_level",
        "final_customer_sentiment",
    ):
        if c in view.columns:
            view[c] = view[c].apply(humanize_label)
    view = view.rename(columns=_OVERVIEW_JOURNEY_COLUMNS)
    return view



def tab_overview() -> None:
    st.subheader("Overview")
    st.caption(
        "Management view of where journeys land across the handled / not-handled "
        "families, focused on surfacing problems and their measurable impact."
    )
    if not _has_results():
        st.info("Run an evaluation first.")
        return

    conv_df = _conv_dataframe_from_results()
    if conv_df.empty:
        st.info("No journeys to summarize.")
        return

    filtered = conv_df
    total = int(len(filtered))

    if total == 0:
        st.info("No journeys to summarize.")
        return

    tree = _overview_tree_spec()

    # --- Family summary cards ---
    st.markdown("---")
    family_cols = st.columns(len(tree), gap="medium")
    family_slices: dict[str, pd.DataFrame] = {}
    for col, family in zip(family_cols, tree):
        fdf = _overview_node_df(filtered, family["handled_status"], family["subtype"])
        family_slices[family["key"]] = fdf
        count = int(len(fdf))
        good_n = sum(
            len(_overview_node_df(filtered, family["handled_status"], family["subtype"], exp["value"]))
            for exp in family["experiences"] if exp["tone"] == "good"
        )
        bad_n = count - good_n
        with col:
            st.markdown(
                _kpi_card_html(
                    family["title"],
                    f"{count:,}",
                    f"{_pct(count, total):.1f}% of {total:,} journeys",
                    [("Bad experience", bad_n, _DASH_COLORS["unhandled"]),
                     ("Good experience", good_n, _DASH_COLORS["handled"])],
                ),
                unsafe_allow_html=True,
            )

    # --- Marker breakdown table per family ---
    st.markdown("---")
    _section_header(
        "Journey marker breakdown",
        "Each family split by customer experience and frustration origin. "
        "Bad experience shown first.",
    )

    family_tree_cols = st.columns(len(tree), gap="large")
    for col, family in zip(family_tree_cols, tree):
        fdf = family_slices[family["key"]]
        fcount = int(len(fdf))
        fcolor = _overview_tone_color(family["tone"])

        with col:
            # Family header
            st.markdown(
                f'<div style="padding:10px 14px;background:{fcolor}22;'
                f'border:2px solid {fcolor};border-radius:8px;margin-bottom:10px;">'
                f'<div style="font-size:0.95rem;font-weight:800;color:{fcolor};">'
                f'{html_lib.escape(family["title"])}</div>'
                f'<div style="font-size:0.82rem;color:{_DASH_COLORS["muted"]};margin-top:2px;">'
                f'{fcount:,} journeys &nbsp;·&nbsp; {_pct(fcount, total):.1f}% of total</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if fcount == 0:
                st.caption("No journeys in this family.")
                continue

            # Marker table: bad experience first, then good
            ordered_exps = sorted(family["experiences"], key=lambda e: 0 if e["tone"] == "bad" else 1)
            table_rows = []
            for exp in ordered_exps:
                edf = _overview_node_df(filtered, family["handled_status"], family["subtype"], exp["value"])
                if edf.empty:
                    continue
                origins = (
                    edf["frustration_origin"].fillna("none").astype(str).value_counts()
                    if "frustration_origin" in edf.columns
                    else pd.Series({"none": len(edf)})
                )
                for origin, lcount in origins.items():
                    table_rows.append({
                        "Experience": humanize_label(exp["value"]),
                        "Frustration origin": humanize_label(origin),
                        "Journeys": int(lcount),
                        f"% of {family['short_name']}": f"{_pct(lcount, fcount):.1f}%",
                        "% of total": f"{_pct(lcount, total):.1f}%",
                    })

            tdf = pd.DataFrame(table_rows)
            tdf = tdf[tdf["Journeys"] > 0]
            if not tdf.empty:
                st.dataframe(tdf, use_container_width=True, hide_index=True)


    # --- Issues table ---
    st.markdown("---")
    _section_header(
        "Detected issues across all journeys",
        "What went wrong — grouped by issue type and origin. "
        "Shows every journey where a main issue was identified.",
    )

    issue_cols_needed = ["main_issue_type", "main_issue_origin", "main_issue_summary", "customer_impact", "customer_experience"]
    available = [c for c in issue_cols_needed if c in filtered.columns]
    if not available:
        st.caption("No issue data available.")
        return

    issues_df = filtered[filtered["main_issue_type"].notna() & (filtered["main_issue_type"].astype(str).str.lower() != "none")].copy()
    if issues_df.empty:
        st.caption("No issues detected across evaluated journeys.")
        return

    # Filter controls
    if1, if2 = st.columns([1, 1])
    with if1:
        type_opts = sorted(issues_df["main_issue_type"].dropna().unique().tolist())
        sel_types = st.multiselect(
            "Issue type", [humanize_label(t) for t in type_opts], default=[], key="overview_issue_type",
        )
    with if2:
        if "main_issue_origin" in issues_df.columns:
            origin_opts = sorted(issues_df["main_issue_origin"].dropna().unique().tolist())
            sel_origins = st.multiselect(
                "Frustration origin", [humanize_label(o) for o in origin_opts], default=[], key="overview_issue_origin",
            )
        else:
            sel_origins = []

    view = issues_df.copy()
    if sel_types:
        view = view[view["main_issue_type"].apply(humanize_label).isin(sel_types)]
    if sel_origins and "main_issue_origin" in view.columns:
        view = view[view["main_issue_origin"].apply(humanize_label).isin(sel_origins)]

    if view.empty:
        st.caption("No issues match the selected filters.")
        return

    # Group by issue type + origin, count journeys per group
    group_cols = [c for c in ["main_issue_type", "main_issue_origin"] if c in view.columns]
    grouped = (
        view.groupby(group_cols)
        .size()
        .reset_index(name="Journeys")
        .sort_values("Journeys", ascending=False)
    )
    for c in group_cols:
        grouped[c] = grouped[c].apply(humanize_label)
    grouped = grouped.rename(columns={
        "main_issue_type": "Issue type",
        "main_issue_origin": "Origin",
    })

    st.caption(f"{len(view):,} journeys with detected issues — {len(grouped):,} distinct issue types")
    st.dataframe(grouped, use_container_width=True, hide_index=True)



def tab_dashboard() -> None:
    st.subheader("Management Dashboard")
    if not _has_results():
        st.info("Run an evaluation first.")
        return

    conv_df = _conv_dataframe_from_results()
    msg_df = _msg_dataframe_from_results()

    filters = _conversation_filters_with_keys(conv_df, "dashboard_filters")
    filtered = _apply_conversation_filters_fresh(conv_df, filters)
    agg = dashboard_aggregates(filtered)
    total = int(agg.get("total", 0))

    if total == 0:
        st.info("No journeys match the current filters.")
        return

    _render_kpi_strip(filtered, msg_df, agg, total)

    st.markdown("---")
    _section_header(
        "Outcome tree",
        "Distribution at each level: Outcome → Issue severity → Frustration. Each ring slice and each bar shows its share of its parent.",
    )
    out_cols = st.columns([1.25, 1], gap="medium")
    with out_cols[0]:
        _render_outcome_sunburst(filtered)
    with out_cols[1]:
        _render_outcome_cascade(filtered, total)

    st.markdown("---")
    _section_header(
        "Issue tree",
        "Where issues originate and what kind they are. Inner ring is the origin; outer rings are the most common issue types and severity inside each origin.",
    )
    issue_cols = st.columns([1.25, 1], gap="medium")
    with issue_cols[0]:
        _render_issue_sunburst(filtered)
    with issue_cols[1]:
        _render_issue_cascade(filtered, total)

    st.markdown("---")
    _section_header(
        "Frustration tree",
        "Funnel from all journeys down to cancellation risk. The breakdown beside it shows how frustration timing splits into max severity. The heatmap below crosses both.",
    )
    frust_cols = st.columns([1.25, 1], gap="medium")
    with frust_cols[0]:
        _render_frustration_funnel(filtered, total)
    with frust_cols[1]:
        _render_frustration_cascade(filtered, total)
    _render_timing_level_heatmap(filtered)

    st.markdown("---")
    _section_header(
        "End-to-end flow",
        "Trace every journey across four decision points: Outcome → Severity → Issue origin → Frustration timing. Hover any band to read the count.",
    )
    _render_overall_sankey(filtered)

    st.markdown("---")
    _section_header("Top issue types and frustration causes")
    cause_cols = st.columns(2, gap="medium")
    with cause_cols[0]:
        st.markdown(
            f'<div style="font-size:0.92rem;color:{_DASH_COLORS["text"]};font-weight:700;margin-bottom:4px;">'
            f'Main issue types (journey-level)</div>',
            unsafe_allow_html=True,
        )
        if agg["issue_type_counts"]:
            it_df = (
                pd.DataFrame([{"Issue type": k, "Count": v} for k, v in agg["issue_type_counts"].items()])
                .assign(**{"Issue type": lambda d: d["Issue type"].apply(humanize_label)})
                .query("Count > 0")
                .sort_values("Count", ascending=False)
                .head(12)
            )
            if HAS_PLOTLY:
                fig = px.bar(
                    it_df,
                    x="Count",
                    y="Issue type",
                    orientation="h",
                    text="Count",
                    color="Count",
                    color_continuous_scale=["#1e293b", _DASH_COLORS["many"], _DASH_COLORS["unhandled"]],
                )
                fig.update_traces(textposition="outside")
                _plotly_layout(
                    fig,
                    height=400,
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                )
                _render_plotly(fig)
            else:
                _render_simple_bar_chart(it_df, "Issue type", "Count", height=360)
        else:
            st.caption("No issue types recorded.")
    with cause_cols[1]:
        st.markdown(
            f'<div style="font-size:0.92rem;color:{_DASH_COLORS["text"]};font-weight:700;margin-bottom:4px;">'
            f'Frustration causes (message-level)</div>',
            unsafe_allow_html=True,
        )
        causes = top_frustration_causes(msg_df, top_n=15)
        if not causes.empty:
            causes = causes.copy()
            causes["frustration_cause"] = causes["frustration_cause"].apply(humanize_label)
            if HAS_PLOTLY:
                fig = px.bar(
                    causes,
                    x="count",
                    y="frustration_cause",
                    orientation="h",
                    text="count",
                    color="count",
                    color_continuous_scale=["#1e293b", _DASH_COLORS["frustrated"], _DASH_COLORS["unhandled"]],
                )
                fig.update_traces(textposition="outside")
                _plotly_layout(
                    fig,
                    height=400,
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                )
                _render_plotly(fig)
            else:
                _render_simple_bar_chart(causes, "frustration_cause", "count", height=360)
        else:
            st.caption("No frustration causes identified.")

    st.markdown("---")
    _section_header(
        "Activity over time",
        "Journeys per day across the filtered set.",
    )
    if "conversation_start_date" in filtered.columns and not filtered.empty:
        try:
            parsed = pd.to_datetime(filtered["conversation_start_date"], errors="coerce")
            ts = filtered.assign(_d=parsed.dt.date)
            daily = ts.groupby("_d").size().reset_index(name="count")
            if not daily.empty:
                if HAS_PLOTLY:
                    fig = px.area(
                        daily,
                        x="_d",
                        y="count",
                        labels={"_d": "Date", "count": "Customer journeys"},
                    )
                    fig.update_traces(
                        line=dict(color=_DASH_COLORS["customer"], width=2),
                        fillcolor=_hex_to_rgba(_DASH_COLORS["customer"], 0.18),
                        mode="lines+markers",
                        marker=dict(size=5, color=_DASH_COLORS["customer"]),
                    )
                    _plotly_layout(fig, height=300, margin=dict(t=10, b=10, l=10, r=10))
                    _render_plotly(fig)
                else:
                    _render_simple_line_chart(daily, "_d", "count", height=300)
            else:
                st.caption("No parseable dates.")
        except Exception:
            st.caption("Could not parse conversation_start_date.")
    else:
        st.caption("No start date column available.")

    st.caption(
        "Per-journey and per-message tables live in the Journey Review and Exports tabs."
    )


# --------- Tab: Customer Journey Review ---------


def tab_review() -> None:
    st.subheader("Customer Journey Review")
    if not _has_results():
        st.info("Run an evaluation first.")
        return

    rr = st.session_state.run_results
    conv_df = _conv_dataframe_from_results()
    if conv_df.empty:
        st.info("No customer journey results are available yet.")
        return

    st.caption(
        "Browse customer journeys by result, customer frustration, review priority, or the main customer problem."
    )

    review_filters = _conversation_filters_with_keys(conv_df, "review_filters")
    filtered_df = _apply_conversation_filters_fresh(conv_df, review_filters)

    search = st.text_input(
        "Search by ID, customer name, phone, source conversation ID, result, or problem summary",
        value="",
    ).strip()
    if search:
        search_text = search.lower()
        search_cols = [
            "conversation_id",
            "customer_name",
            "customer_phone",
            "source_conversation_ids",
            "handled_status",
            "customer_experience",
            "frustration_origin",
            "main_issue_summary",
        ]
        mask = pd.Series(False, index=filtered_df.index)
        for col in search_cols:
            if col in filtered_df.columns:
                mask = mask | filtered_df[col].fillna("").astype(str).str.lower().str.contains(search_text, regex=False)
        filtered_df = filtered_df[mask]

    if filtered_df.empty:
        st.warning("No customer journeys match the current filters.")
        return

    review_metrics = [
        ("Customer journeys shown", f"{len(filtered_df):,}", None),
        (
            "Handled",
            f"{int((_norm_marker_series(filtered_df, 'handled_status') == 'handled').sum()):,}",
            None,
        ),
        (
            "Need human review",
            f"{int(_bool_marker_series(filtered_df, 'manual_review_required').sum()):,}",
            None,
        ),
        (
            "High frustration",
            f"{int(filtered_df.get('max_frustration_level', pd.Series(dtype=str)).isin(['high', 'cancellation_risk']).sum()):,}",
            None,
        ),
    ]
    metric_row(review_metrics)

    options = []
    label_to_id = {}
    ordered_ids = []
    for row in filtered_df.to_dict(orient="records"):
        cid = str(row.get("conversation_id", "") or "")
        cust = row.get("customer_name") or "—"
        phone = row.get("customer_phone") or cid
        source_count = row.get("source_conversation_count") or "—"
        result = f"{humanize_label(row.get('handled_status')) or 'Unknown'} / {humanize_label(row.get('customer_experience')) or 'Unknown'}"
        label = f"{phone} • {cust} • {source_count} source convs • {result}"
        if label in label_to_id:
            label = f"{label} - {cid[:8]}"
        options.append(label)
        label_to_id[label] = cid
        ordered_ids.append(cid)

    current_id = str(st.session_state.get("review_selected_conversation_id") or "")
    if current_id not in ordered_ids:
        current_id = ordered_ids[0]
        st.session_state.review_selected_conversation_id = current_id
    current_index = ordered_ids.index(current_id)

    def set_review_index(index: int) -> None:
        if not ordered_ids:
            return
        index = index % len(ordered_ids)
        st.session_state.review_selected_conversation_id = ordered_ids[index]
        st.session_state.review_scroll_to_conversation_start = True

    def scroll_to_conversation_start_if_requested() -> None:
        if not st.session_state.pop("review_scroll_to_conversation_start", False):
            return
        components.html(
            """
            <script>
            const scrollToJourneyStart = () => {
              try {
                const parentWindow = window.parent;
                const parentDoc = parentWindow.document;
                const marker = parentDoc.getElementById("review-conversation-start");
                if (!marker) return;

                marker.scrollIntoView({ block: "start", behavior: "auto" });

                const top = marker.getBoundingClientRect().top + parentWindow.scrollY - 16;
                parentWindow.scrollTo({ top, behavior: "auto" });

                const scrollContainers = [
                  parentDoc.scrollingElement,
                  parentDoc.documentElement,
                  parentDoc.body,
                  parentDoc.querySelector("section.main"),
                  parentDoc.querySelector("[data-testid='stAppViewContainer']"),
                  parentDoc.querySelector("[data-testid='stMain']"),
                  parentDoc.querySelector("[data-testid='stMainBlockContainer']"),
                ].filter(Boolean);

                for (const container of scrollContainers) {
                  const rect = marker.getBoundingClientRect();
                  const containerRect = container.getBoundingClientRect
                    ? container.getBoundingClientRect()
                    : { top: 0 };
                  const nextTop = container.scrollTop + rect.top - containerRect.top - 16;
                  if (Number.isFinite(nextTop)) container.scrollTop = Math.max(nextTop, 0);
                }
              } catch (error) {
                window.parent.scrollTo(0, 0);
              }
            };
            requestAnimationFrame(scrollToJourneyStart);
            setTimeout(scrollToJourneyStart, 100);
            setTimeout(scrollToJourneyStart, 350);
            setTimeout(scrollToJourneyStart, 900);
            setTimeout(scrollToJourneyStart, 1600);
            </script>
            """,
            height=1,
        )

    def render_review_nav(position: str) -> None:
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        nav_cols = st.columns([2.7, 1.15, 1.15, 1.45, 2.7])
        with nav_cols[1]:
            if st.button(
                "Previous",
                key=f"review_prev_{position}",
                use_container_width=True,
                disabled=len(ordered_ids) <= 1,
            ):
                set_review_index(current_index - 1)
                st.rerun()
        with nav_cols[2]:
            if st.button(
                "Next",
                key=f"review_next_{position}",
                use_container_width=True,
                disabled=len(ordered_ids) <= 1,
            ):
                set_review_index(current_index + 1)
                st.rerun()
        with nav_cols[3]:
            st.markdown(
                f"<div style='height: 2.5rem; display: flex; align-items: center; "
                f"justify-content: center; color: #94a3b8;'>"
                f"Journey {current_index + 1:,} of {len(ordered_ids):,}</div>",
                unsafe_allow_html=True,
            )

    selection = st.selectbox(
        "Open a customer journey",
        options,
        index=current_index,
        key=f"review_journey_select_{current_id}",
    )
    target_id = label_to_id[selection]
    if target_id != current_id:
        st.session_state.review_scroll_to_conversation_start = True
    st.session_state.review_selected_conversation_id = target_id
    current_index = ordered_ids.index(target_id)

    render_review_nav("top")
    target_cr = next((c for c in rr.conversation_results if c.get("conversation_id") == target_id), None)
    if not target_cr:
        st.error("Customer journey not found.")
        return

    target_cr = _normalize_conversation_result_for_display(target_cr)
    _render_conversation_summary_card_fresh(target_cr)

    st.markdown("### Full Customer Journey")
    st.caption(
        "The full appended customer journey is shown below. Where available, assistant replies also include a short quality check underneath."
    )
    st.markdown("<div id='review-conversation-start'></div>", unsafe_allow_html=True)
    transcript = target_cr.get("transcript") or []
    msgs = target_cr.get("message_level_results") or []
    _, chat_col, _ = st.columns([0.15, 9.7, 0.15])
    with chat_col:
        render_conversation_transcript_with_evals(
            transcript=transcript,
            message_results=msgs,
        )

    render_review_nav("bottom")
    scroll_to_conversation_start_if_requested()


# --------- Tab: Exports ---------


def tab_exports() -> None:
    st.subheader("Exports")
    if not _has_results():
        st.info("Run an evaluation first to enable exports.")
        return

    rr = st.session_state.run_results
    run_config = {
        "api_base_url": st.session_state.api_base_url,
        "model": st.session_state.selected_model,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "max_tokens": st.session_state.max_tokens,
        "timeout": st.session_state.timeout,
        "retries": st.session_state.retries,
        "run_all_conversations": bool(st.session_state.get("run_all_conversations")),
        "max_conversations": (
            None
            if st.session_state.get("run_all_conversations")
            else st.session_state.max_conversations
        ),
        "max_target_messages_per_journey": st.session_state.max_agent_messages_per_conv,
        "truncate_messages": st.session_state.truncate_messages,
        "max_chars_per_message": st.session_state.max_chars_per_message,
        "include_unknown_in_history": st.session_state.include_unknown_in_history,
        "stop_on_error": st.session_state.stop_on_error,
        "save_raw_responses": st.session_state.save_raw_responses,
        "message_target_role": st.session_state.message_target_role,
        "started_at": rr.started_at,
        "finished_at": rr.finished_at,
    }

    conv_bytes = build_conversation_csv_bytes(rr.conversation_results)
    msg_bytes = build_message_csv_bytes(rr.message_level_results)
    json_bytes = build_full_json_bytes(
        run_config=run_config,
        conversation_results=rr.conversation_results,
        message_level_results=rr.message_level_results,
        errors=rr.errors,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Journey-Level CSV")
        st.caption("One row per customer journey, ready for spreadsheets.")
        st.download_button(
            "Download journey_results.csv",
            data=conv_bytes,
            file_name="cx_journey_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.markdown("#### Message-Level CSV")
        st.caption("One row per evaluated assistant message.")
        st.download_button(
            "Download message_results.csv",
            data=msg_bytes,
            file_name="cx_message_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.markdown("#### Full JSON Export")
        st.caption("Run config, all results, errors, and raw responses.")
        st.download_button(
            "Download full_results.json",
            data=json_bytes,
            file_name="cx_full_results.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Preview")
    tab_a, tab_b = st.tabs(["Journey-level preview", "Message-level preview"])
    with tab_a:
        conv_df = _conv_dataframe_from_results()
        st.dataframe(conv_df.head(50), use_container_width=True)
    with tab_b:
        msg_df = _msg_dataframe_from_results()
        st.dataframe(msg_df.head(100), use_container_width=True)


# --------- Tab: Debug ---------


def tab_debug() -> None:
    st.subheader("Debug")
    if not _has_results():
        st.info("Run an evaluation first.")
        return
    rr = st.session_state.run_results

    st.markdown("### Errors")
    if rr.errors:
        st.dataframe(pd.DataFrame(rr.errors), use_container_width=True)
    else:
        st.success("No errors recorded for this run.")

    st.markdown("### Failed message-level evaluations")
    failed_msgs = [m for m in rr.message_level_results if m.get("parse_status") != "ok"]
    if failed_msgs:
        st.write(f"{len(failed_msgs)} failed message-level evaluations.")
        for m in failed_msgs[:50]:
            source = m.get("source_conversation_id")
            source_part = f" source `{source}`" if source else ""
            label = f"`{m.get('conversation_id')}` #{m.get('message_index')}{source_part} — {m.get('parse_status')}"
            with st.expander(label):
                st.markdown("**Error message**")
                st.code(m.get("error_message") or "—")
                st.markdown("**Raw model response**")
                st.code(m.get("raw_model_response") or "—")
                st.markdown("**Debug info**")
                st.json(m.get("debug") or {}, expanded=False)
    else:
        st.caption("No failed message-level evaluations.")

    st.markdown("### Failed journey-level evaluations")
    failed_convs = [c for c in rr.conversation_results if c.get("parse_status") != "ok"]
    if failed_convs:
        st.write(f"{len(failed_convs)} failed journey-level evaluations.")
        for c in failed_convs[:50]:
            with st.expander(f"`{c.get('conversation_id')}` — {c.get('parse_status')}"):
                st.markdown("**Error message**")
                st.code(c.get("error_message") or "—")
                st.markdown("**Raw model response**")
                st.code(c.get("raw_model_response") or "—")
                st.markdown("**Debug info**")
                st.json(c.get("debug") or {}, expanded=False)
    else:
        st.caption("No failed journey-level evaluations.")

    st.markdown("### Inspect a specific record")
    st.caption("Pick any customer journey to view raw payloads, parsed JSON, and debug info.")
    ids = [c.get("conversation_id", "") for c in rr.conversation_results]
    if ids:
        sel = st.selectbox("ID", ids)
        target = next((c for c in rr.conversation_results if c.get("conversation_id") == sel), None)
        if target:
            with st.expander("Journey-level parsed JSON"):
                st.json(target.get("parsed_json") or {}, expanded=False)
            with st.expander("Journey-level raw model response"):
                st.code(target.get("raw_model_response") or "—")
            with st.expander("Computed metadata"):
                visible_cm = {
                    k: v for k, v in (target.get("computed_metadata") or {}).items()
                    if k not in {"agent_messages", "agent_messages_evaluated"}
                }
                st.json(visible_cm, expanded=False)
            with st.expander("Message-level records (parsed)"):
                st.json(
                    [
                        {
                            "message_index": m.get("message_index"),
                            "parse_status": m.get("parse_status"),
                            "parsed_json": m.get("parsed_json"),
                            "error_message": m.get("error_message"),
                        }
                        for m in target.get("message_level_results", [])
                    ],
                    expanded=False,
                )
            with st.expander("Message-level raw responses"):
                for m in target.get("message_level_results", []):
                    st.markdown(f"**#{m.get('message_index')}** — {m.get('parse_status')}")
                    st.code(m.get("raw_model_response") or "—")

    st.markdown("---")
    st.markdown("### Run config (sanitized)")
    cfg = {
        "api_base_url": st.session_state.api_base_url,
        "model": st.session_state.selected_model,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "max_tokens": st.session_state.max_tokens,
        "timeout": st.session_state.timeout,
        "retries": st.session_state.retries,
        "run_all_conversations": bool(st.session_state.get("run_all_conversations")),
        "max_conversations": (
            None
            if st.session_state.get("run_all_conversations")
            else st.session_state.max_conversations
        ),
        "max_target_messages_per_journey": st.session_state.max_agent_messages_per_conv,
        "truncate_messages": st.session_state.truncate_messages,
        "max_chars_per_message": st.session_state.max_chars_per_message,
        "include_unknown_in_history": st.session_state.include_unknown_in_history,
        "stop_on_error": st.session_state.stop_on_error,
        "save_raw_responses": st.session_state.save_raw_responses,
        "message_target_role": st.session_state.message_target_role,
    }
    st.json(cfg, expanded=False)


# --------- Main layout ---------


def main() -> None:
    _apply_theme()
    render_sidebar()

    st.title("CX Customer Journey Evaluator")
    st.caption(
        "AI-as-a-Judge evaluation of appended customer journeys across one or more source conversations. "
        "Built for management review — focused on outcomes, frustration, and root cause."
    )

    # Force DB initialization at app start so the seeded defaults exist before
    # any tab tries to read them.
    _refresh_default_prompts(get_db())

    tabs = st.tabs(
        [
            "Upload & Settings",
            "Prompts",
            "Run Evaluation",
            "Overview",
            "Dashboard",
            "Journey Review",
            "Exports",
            "Debug",
        ]
    )
    with tabs[0]:
        tab_upload()
    with tabs[1]:
        tab_prompts()
    with tabs[2]:
        tab_run()
    with tabs[3]:
        tab_overview()
    with tabs[4]:
        tab_dashboard()
    with tabs[5]:
        tab_review()
    with tabs[6]:
        tab_exports()
    with tabs[7]:
        tab_debug()


if __name__ == "__main__":
    main()

