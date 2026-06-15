"""Streamlit entry point for the AI-as-a-Judge CX Conversation Evaluator."""

from __future__ import annotations

import json
import html as html_lib
import importlib
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import ui_components as ui_components_module

from api_client import APIConfig, DEFAULT_BASE_URL, build_client, fetch_models
from data_loader import (
    METADATA_COLUMNS,
    REQUIRED_COLUMNS,
    estimate_call_counts,
    load_csv,
    normalize_dataframe,
    summarize_dataframe,
    validate_csv,
)
from db import DEFAULT_DB_PATH, Database
from evaluator import RunConfig, RunResults, run_evaluation
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
    get_metric_definition,
    humanize_label,
    metric_category_display_name,
    metric_display_name,
    quantifiable_metric_columns,
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
        "available_models": [],
        "models_loaded_at": None,
        "model_load_error": None,
        "api_base_url": DEFAULT_BASE_URL,
        "api_key": "",
        "selected_model": "",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_tokens": 50000,
        "timeout": 300.0,
        "retries": 2,
        # Always concurrent. Hard upper limit is 64 (api_client.MAX_CONCURRENCY).
        "concurrency": 50,
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
    return APIConfig(
        base_url=st.session_state.api_base_url,
        api_key=st.session_state.api_key,
        model=st.session_state.selected_model,
        temperature=float(st.session_state.temperature),
        top_p=float(st.session_state.top_p),
        max_tokens=int(st.session_state.max_tokens),
        timeout=float(st.session_state.timeout),
        retries=int(st.session_state.retries),
        concurrency=int(st.session_state.concurrency),
    )


def _build_run_config() -> tuple[RunConfig, int | None, int | None]:
    """Build a RunConfig using the active prompts from the DB.

    Returns ``(config, message_prompt_id, conversation_prompt_id)`` so the run
    record can store the prompt versions used.
    """
    ml_tpl, cl_tpl, ml_id, cl_id = _load_active_prompts()
    cfg = RunConfig(
        api=_build_api_config(),
        max_conversations=int(st.session_state.max_conversations) if st.session_state.max_conversations else None,
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


def _conv_dataframe_from_results() -> pd.DataFrame:
    rr = st.session_state.run_results
    if not rr:
        return pd.DataFrame()
    rows = []
    for cr in rr.conversation_results:
        rows.append(
            flatten_conversation_row(
                cr,
                cr.get("conversation_metadata", {}) or {},
                cr.get("computed_metadata", {}) or {},
            )
        )
    return build_conversation_table(rows)


def _msg_dataframe_from_results() -> pd.DataFrame:
    rr = st.session_state.run_results
    if not rr:
        return pd.DataFrame()
    rows = [flatten_message_row(m) for m in rr.message_level_results]
    return build_message_table(rows)


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
    if column.startswith("metric__"):
        return f"{metric_category_display_name(column)} / {metric_display_name(column)}"
    special = {
        "conversation_id": "ID",
        "customer_name": "Customer name",
        "customer_phone": "Customer phone",
        "conversation_start_date": "Started",
        "conversation_end_date": "Ended",
        "conversation_status": "Conversation status",
        "customer_objective_type": "Customer goal type",
        "customer_primary_objective": "Customer goal",
        "final_classification": "Overall result",
        "handled_status": "Outcome",
        "cx_issue_severity": "Journey quality",
        "frustration_detected": "Customer frustration",
        "customer_started_frustrated": "Started frustrated",
        "customer_became_frustrated_during_chat": "Became frustrated during chat",
        "customer_ended_frustrated": "Ended frustrated",
        "frustration_timing": "When frustration appeared",
        "unhandled_resolution_subtype": "Unresolved status",
        "final_customer_sentiment": "Customer feeling at end",
        "max_frustration_level": "Highest frustration level",
        "main_issue_type": "Main problem type",
        "main_issue_origin": "Where the main problem came from",
        "main_issue_summary": "Main problem summary",
        "customer_impact": "Customer impact",
        "manual_review_required": "Needs human review",
        "manual_review_reason": "Reason for human review",
        "metric_value": "Metric value",
        "target_message_id": "Target message ID",
        "message_index": "Message index",
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


def _render_metric_definition_table(metric_df: pd.DataFrame) -> None:
    """Render metrics with a clickable info control for each definition."""
    if metric_df is None or metric_df.empty:
        st.caption("No metrics.")
        return

    header_cols = st.columns([3.8, 0.7, 1.0, 1.4, 1.5])
    headers = ["Metric", "Info", "Total", "Average when flagged", "Conversations > 0"]
    for col, label in zip(header_cols, headers):
        with col:
            st.markdown(f"**{label}**")

    for _, row in metric_df.iterrows():
        metric_name = str(row.get("Metric", "") or "")
        metric_col = row.get("Column")
        definition = get_metric_definition(metric_col) if metric_col else ""
        row_cols = st.columns([3.8, 0.7, 1.0, 1.4, 1.5])
        with row_cols[0]:
            st.write(metric_name or "—")
        with row_cols[1]:
            if definition:
                with st.popover("i", use_container_width=True):
                    st.write(definition)
        with row_cols[2]:
            st.write(_format_chart_value(row.get("Total", 0)))
        with row_cols[3]:
            st.write(_format_chart_value(row.get("Average when flagged", 0)))
        with row_cols[4]:
            st.write(_format_chart_value(row.get("Conversations > 0", 0)))


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
        st.number_input("Max tokens", min_value=128, max_value=100000, step=64, key="max_tokens")
        st.number_input("Timeout (seconds)", min_value=5.0, max_value=600.0, step=5.0, key="timeout")
        st.number_input("Retry count", min_value=0, max_value=10, step=1, key="retries")
        st.number_input(
            "Concurrency",
            min_value=1,
            max_value=64,
            step=1,
            key="concurrency",
            help=(
                "Number of message-level API calls dispatched in parallel. "
                "Always concurrent — capped at 64. Raise if your endpoint allows "
                "high throughput; lower if you hit rate limits."
            ),
        )

        st.markdown("---")
        st.markdown("### Evaluation safeguards")
        st.number_input(
            "Max conversations to process",
            min_value=1,
            max_value=10000,
            step=1,
            key="max_conversations",
        )
        st.number_input(
            "Max target messages per conversation",
            min_value=1,
            max_value=2000,
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
                max_value=20000,
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
    st.subheader("Upload Conversation CSV")
    st.caption(
        "Upload the Snowflake-exported CSV. One row per visible message. "
        "Tool calls and internal/system messages must already be removed."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            return

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
            ("Conversations", f"{summary.get('conversations', 0):,}", None),
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
        runs = db.list_runs(limit=200)
        if not runs:
            st.caption("No saved runs yet.")
        else:
            df_runs = pd.DataFrame(runs)
            df_runs["label"] = df_runs.apply(
                lambda r: f"#{r['id']} • {r.get('csv_name') or '—'} • {r['status']} • {r['started_at']}",
                axis=1,
            )
            sel = st.selectbox("Select a saved run to load", df_runs["label"].tolist(), index=0)
            sel_id = int(df_runs.iloc[df_runs.index[df_runs["label"] == sel][0]]["id"])
            col_load, col_del = st.columns([1, 1])
            with col_load:
                if st.button("Load this run", use_container_width=True):
                    try:
                        loaded = db.load_run_results(sel_id)
                        rr = RunResults(
                            conversation_results=loaded["conversation_results"],
                            message_level_results=loaded["message_level_results"],
                            errors=loaded["errors"],
                            started_at=loaded["started_at"],
                            finished_at=loaded["finished_at"],
                        )
                        st.session_state.run_results = rr
                        st.session_state.current_run_id = sel_id
                        st.session_state.loaded_run_label = sel
                        st.success(f"Loaded run #{sel_id}.")
                    except Exception as e:
                        st.error(f"Could not load run: {e}")
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

    target_role = str(st.session_state.message_target_role or "agent")

    # ---- Conversation selection (first-N vs. random sample) -----------------
    all_ids = (
        df["CONVERSATION_ID"].astype(str).drop_duplicates().tolist()
        if "CONVERSATION_ID" in df.columns
        else []
    )
    selected_ids = st.session_state.selected_conversation_ids
    st.markdown("### Conversation selection")
    pick_cols = st.columns([1, 1, 1, 2])
    with pick_cols[0]:
        if st.button(
            "🎲 Random sample",
            use_container_width=True,
            help=(
                "Pick a random sample of IDs from the uploaded CSV. "
                "Sample size = 'Max conversations to process' from the sidebar."
            ),
            disabled=not all_ids,
        ):
            import random
            n = max(1, int(st.session_state.max_conversations or 1))
            n = min(n, len(all_ids))
            st.session_state.selected_conversation_ids = random.sample(all_ids, n)
            st.rerun()
    with pick_cols[1]:
        if st.button(
            "Clear selection",
            use_container_width=True,
            disabled=not selected_ids,
        ):
            st.session_state.selected_conversation_ids = None
            st.rerun()
    with pick_cols[2]:
        st.caption(
            f"**{len(selected_ids):,} pinned**"
            if selected_ids
            else "_No selection — runs use the first N from the CSV._"
        )
    with pick_cols[3]:
        if selected_ids:
            preview = ", ".join(str(x) for x in selected_ids[:6])
            if len(selected_ids) > 6:
                preview += f", … (+{len(selected_ids) - 6} more)"
            st.caption(f"Pinned IDs: `{preview}`")

    # Build the estimate. When a random selection is active, count over the
    # pinned IDs; otherwise apply the max_conversations slice.
    if selected_ids:
        df_for_estimate = df[df["CONVERSATION_ID"].astype(str).isin(set(map(str, selected_ids)))]
        estimate = estimate_call_counts(
            df_for_estimate,
            max_conversations=None,
            max_agent_messages_per_conv=int(st.session_state.max_agent_messages_per_conv),
            target_role=target_role,
        )
    else:
        estimate = estimate_call_counts(
            df,
            max_conversations=int(st.session_state.max_conversations),
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
            ("Conversations to evaluate", f"{estimate['conversations']:,}", None),
            (f"{role_label.capitalize()}-message AI calls", f"{estimate['message_level_calls']:,}", None),
            ("Conversation-level AI calls", f"{estimate['conversation_level_calls']:,}", None),
            ("Total estimated AI calls", f"{estimate['total_calls']:,}", None),
        ]
    )

    large_job = estimate["total_calls"] > 200
    if large_job:
        st.warning(
            f"This run will make ~{estimate['total_calls']:,} AI calls. "
            "Consider lowering Max conversations or Max target messages per conversation in the sidebar."
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
            "max_conversations": config.max_conversations,
            "max_target_messages_per_conversation": config.max_agent_messages_per_conv,
            "truncate_messages": config.truncate_messages,
            "max_chars_per_message": config.max_chars_per_message,
            "include_unknown_in_history": config.include_unknown_in_history,
            "stop_on_error": config.stop_on_error,
            "save_raw_responses": config.save_raw_responses,
            "message_target_role": config.message_target_role,
        }
        run_id = db.start_run(
            csv_name=st.session_state.csv_name,
            run_config=run_config_serializable,
            message_prompt_id=ml_prompt_id,
            conversation_prompt_id=cl_prompt_id,
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
                    f"Conversation {evt.get('conversation_index')}/{evt.get('total_conversations')} — "
                    f"ID `{evt.get('conversation_id')}` — "
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
                text=f"Conversations {progress_state['convs_done']}/{total_conv} • Calls {progress_state['calls_done']}/{total_msg}",
            )
            counter_box.markdown(
                f"**Successes:** {progress_state['successes']}  |  **Failures:** {progress_state['failures']}"
            )
            st.session_state.progress_log.append(evt)

        def cancel_requested() -> bool:
            return bool(st.session_state.cancel_flag)

        def save_message(mr: dict) -> None:
            try:
                mr["run_id"] = run_id
                db.save_message_result(run_id, mr)
            except Exception:
                pass

        def save_conversation(cr: dict) -> None:
            try:
                cr["run_id"] = run_id
                db.save_conversation_result(run_id, cr)
            except Exception:
                pass

        def save_err(err: dict) -> None:
            try:
                db.save_error(run_id, err)
            except Exception:
                pass

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
            progress_box.success(
                f"Evaluation finished. {len(results.conversation_results)} conversations processed, "
                f"{len(results.message_level_results)} message-level calls, "
                f"{len(results.errors)} errors. Saved as run #{run_id}."
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
                ("Conversations", f"{len(rr.conversation_results):,}", None),
                ("Message calls", f"{len(rr.message_level_results):,}", None),
                ("Errors", f"{len(rr.errors):,}", None),
                ("Duration (s)", f"{(rr.finished_at or 0) - (rr.started_at or 0):.1f}", None),
            ]
        )

        if rr.errors:
            with st.expander(f"View {len(rr.errors)} non-fatal errors"):
                st.dataframe(pd.DataFrame(rr.errors), use_container_width=True)


# --------- Tab: Dashboard ---------


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

    metric_row(
        [
            ("Total Conversations", f"{agg['total']:,}", None),
            ("Handled %", f"{agg['handled_pct']:.1f}%", None),
            ("Unhandled %", f"{agg['unhandled_pct']:.1f}%", None),
            ("Many Issues %", f"{agg['many_issues_pct']:.1f}%", None),
        ]
    )
    metric_row(
        [
            ("High Frustration", f"{agg['high_frustration_count']:,}", None),
            ("Cancellation Risk", f"{agg['cancellation_risk_count']:,}", None),
            ("Manual Review", f"{agg['manual_review_count']:,}", None),
            ("Errors", f"{len(st.session_state.run_results.errors):,}", None),
        ]
    )

    st.markdown("---")
    classification_order = [
        "Handled with Minimal Issues",
        "Handled with Many Issues",
        "Handled with Minimal Issues and Frustration",
        "Handled with Many Issues and Frustration",
        "Handled with Minimal Caused Issues and Frustration",
        "Handled with Many Caused Issues and Frustration",
        "Not Handled with Minimal Issues",
        "Not Handled with Many Issues",
        "Not Handled with Minimal Issues and Frustration",
        "Not Handled with Many Issues and Frustration",
        "Not Handled with Minimal Caused Issues and Frustration",
        "Not Handled with Many Caused Issues and Frustration",
    ]
    total_filtered = int(len(filtered))
    if "final_classification" in filtered.columns and total_filtered:
        class_counts = filtered["final_classification"].fillna("Unknown").value_counts()
    else:
        class_counts = pd.Series(dtype=int)

    classification_rows = []
    seen_classes = set()
    for classification in classification_order:
        count = int(class_counts.get(classification, 0))
        pct = (count / total_filtered * 100.0) if total_filtered else 0.0
        classification_rows.append(
            {
                "Classification": classification,
                "Count": count,
                "Percentage": pct,
                "Share": f"{pct:.1f}%",
                "Outcome": "Handled" if classification.startswith("Handled") else "Unhandled",
            }
        )
        seen_classes.add(classification)
    for classification, count in class_counts.items():
        if classification in seen_classes:
            continue
        pct = (int(count) / total_filtered * 100.0) if total_filtered else 0.0
        classification_rows.append(
            {
                "Classification": classification,
                "Count": int(count),
                "Percentage": pct,
                "Share": f"{pct:.1f}%",
                "Outcome": "Other",
            }
        )
    cls_df = pd.DataFrame(classification_rows)

    st.markdown("#### Final classification percentages")
    _render_display_table(
        cls_df[["Classification", "Count", "Share"]],
    )
    if HAS_PLOTLY and not cls_df.empty:
        fig = px.bar(
            cls_df,
            x="Percentage",
            y="Classification",
            color="Outcome",
            orientation="h",
            text="Share",
            hover_data=["Count"],
        )
        _plotly_layout(
            fig,
            height=430,
            xaxis_title="Percent of filtered conversations",
            yaxis_title="",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(autorange="reversed"),
        )
        _render_plotly(fig)
    elif not cls_df.empty:
        _render_simple_bar_chart(
            cls_df,
            "Classification",
            "Percentage",
            height=360,
            max_value=100,
            value_suffix="%",
        )

    st.markdown("---")
    c_breakdown_1, c_breakdown_2 = st.columns(2)
    with c_breakdown_1:
        st.markdown("#### Issue origin distribution")
        if agg["issue_origin_counts"]:
            io_df = pd.DataFrame(
                [{"Origin": humanize_label(k), "Count": v} for k, v in agg["issue_origin_counts"].items() if k.lower() != "none"]
            ).sort_values("Count", ascending=True)
            io_df = io_df[io_df["Count"] > 0]
            if io_df.empty:
                st.write("No issue origins identified.")
            elif HAS_PLOTLY:
                total_origins = max(int(io_df["Count"].sum()), 1)
                io_df["Share"] = io_df["Count"].apply(lambda c: f"{(c / total_origins * 100):.1f}%")
                io_df["Label"] = io_df.apply(lambda r: f"{int(r['Count'])} ({r['Share']})", axis=1)
                origin_colors = {
                    "Our side": "#ef4444",
                    "Shared": "#f59e0b",
                    "Customer side": "#2563eb",
                    "Third party": "#14b8a6",
                    "Unclear": "#64748b",
                }
                fig = px.bar(
                    io_df,
                    x="Count",
                    y="Origin",
                    orientation="h",
                    color="Origin",
                    color_discrete_map=origin_colors,
                    text="Label",
                    hover_data={"Share": True, "Count": True, "Origin": False, "Label": False},
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(showlegend=False)
                _plotly_layout(
                    fig,
                    height=320,
                    xaxis_title="Conversations",
                    yaxis_title="",
                    margin=dict(l=10, r=40, t=10, b=30),
                )
                _render_plotly(fig)
            else:
                _render_simple_bar_chart(io_df, "Origin", "Count", height=300)
        else:
            st.write("No data.")

    with c_breakdown_2:
        st.markdown("#### Unhandled subtype")
        if agg["unhandled_subtype_counts"]:
            subtype_df = pd.DataFrame(
                [{"Subtype": humanize_label(k), "Count": v} for k, v in agg["unhandled_subtype_counts"].items() if k.lower() != "not applicable"]
            )
            subtype_df = subtype_df[subtype_df["Count"] > 0]
            subtype_total = max(int(subtype_df["Count"].sum()), 1)
            subtype_df["Share"] = subtype_df["Count"].apply(lambda c: f"{(c / subtype_total * 100):.1f}%")
            if HAS_PLOTLY:
                fig = px.pie(subtype_df, names="Subtype", values="Count", hole=0.55)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                _plotly_layout(fig, height=340)
                _render_plotly(fig)
            else:
                _render_simple_bar_chart(subtype_df, "Subtype", "Count", height=300)
        else:
            st.write("No data.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top issue types")
        if agg["issue_type_counts"]:
            it_df = (
                pd.DataFrame([{"Issue type": k, "Count": v} for k, v in agg["issue_type_counts"].items()])
                .assign(**{"Issue type": lambda d: d["Issue type"].apply(humanize_label)})
                .query("Count > 0")
                .sort_values("Count", ascending=False)
                .head(12)
            )
            if HAS_PLOTLY:
                fig = px.scatter(
                    it_df,
                    x="Count",
                    y="Issue type",
                    size="Count",
                    color="Count",
                    text="Count",
                )
                fig.update_traces(textposition="middle right")
                _plotly_layout(fig, height=400, yaxis=dict(autorange="reversed"))
                _render_plotly(fig)
            else:
                _render_simple_bar_chart(it_df, "Issue type", "Count", height=360)
        else:
            st.write("No data.")

    with c2:
        st.markdown("#### Top frustration causes")
        causes = top_frustration_causes(msg_df, top_n=15)
        if not causes.empty:
            causes = causes.copy()
            causes["frustration_cause"] = causes["frustration_cause"].apply(humanize_label)
            if HAS_PLOTLY:
                fig = px.bar(causes, x="count", y="frustration_cause", orientation="h", text="count")
                _plotly_layout(fig, height=400, yaxis=dict(autorange="reversed"))
                _render_plotly(fig)
            else:
                _render_simple_bar_chart(causes, "frustration_cause", "count", height=360)
        else:
            st.write("No frustration causes identified.")

    st.markdown("---")
    st.markdown("### Quantifiable conversation metrics")
    st.caption(
        "Scroll down to the detailed category sections below if you want to see the definition of any metric."
    )
    metric_totals = agg.get("metric_totals", pd.DataFrame())
    if not metric_totals.empty:
        metric_view = metric_totals.copy()
        metric_view["Total"] = pd.to_numeric(metric_view["Total"], errors="coerce").fillna(0)
        metric_view["Average when flagged"] = pd.to_numeric(
            metric_view["Average"], errors="coerce"
        ).fillna(0)
        metric_view = metric_view.drop(columns=["Average"])
        metric_view["Conversations > 0"] = pd.to_numeric(
            metric_view["Conversations > 0"], errors="coerce"
        ).fillna(0).astype(int)

        f1, f2, f3, f4 = st.columns([1.3, 1.3, 1, 1])
        with f1:
            category_options = sorted(metric_view["Category"].dropna().unique().tolist())
            selected_categories = st.multiselect(
                "Metric categories",
                category_options,
                default=[],
                help="Leave empty to include all categories.",
            )
        with f2:
            metric_search = st.text_input("Search metrics", value="")
        with f3:
            only_nonzero = st.toggle("Only nonzero", value=True)
        with f4:
            min_total = st.number_input("Minimum total", min_value=0.0, value=0.0, step=1.0)

        sort_mode = st.selectbox(
            "Sort metrics by",
            [
                "Total descending",
                "Contributing conversations descending",
                "Category then metric",
                "Average when flagged descending",
            ],
            index=0,
        )

        if selected_categories:
            metric_view = metric_view[metric_view["Category"].isin(selected_categories)]
        if metric_search.strip():
            needle = metric_search.strip().lower()
            metric_view = metric_view[
                metric_view["Metric"].astype(str).str.lower().str.contains(needle, na=False)
                | metric_view["Category"].astype(str).str.lower().str.contains(needle, na=False)
            ]
        if only_nonzero:
            metric_view = metric_view[metric_view["Conversations > 0"] > 0]
        if min_total > 0:
            metric_view = metric_view[metric_view["Total"] >= float(min_total)]

        sort_map = {
            "Total descending": (["Total", "Conversations > 0", "Category", "Metric"], [False, False, True, True]),
            "Contributing conversations descending": (
                ["Conversations > 0", "Total", "Category", "Metric"],
                [False, False, True, True],
            ),
            "Category then metric": (["Category", "Metric"], [True, True]),
            "Average when flagged descending": (
                ["Average when flagged", "Total", "Category", "Metric"],
                [False, False, True, True],
            ),
        }
        sort_cols, sort_asc = sort_map[sort_mode]
        metric_view = metric_view.sort_values(sort_cols, ascending=sort_asc)

        if metric_view.empty:
            st.caption("No metrics match the selected filters.")
        else:
            top_metrics = metric_view[metric_view["Total"] > 0].head(15)
            if HAS_PLOTLY:
                fig = px.bar(
                    top_metrics,
                    x="Total",
                    y="Metric",
                    color="Category",
                    orientation="h",
                    text="Total",
                    hover_data=["Average when flagged", "Conversations > 0"],
                )
                _plotly_layout(fig, height=520, yaxis=dict(autorange="reversed"))
                _render_plotly(fig)
            else:
                _render_simple_bar_chart(top_metrics, "Metric", "Total", height=460)

            contributor_base_cols = [
                "conversation_id",
                "customer_name",
                "conversation_start_date",
                "final_classification",
                "handled_status",
                "cx_issue_severity",
                "unhandled_resolution_subtype",
                "main_issue_type",
                "main_issue_summary",
                "management_summary",
            ]
            metric_column_lookup = {
                (metric_category_display_name(c), metric_display_name(c)): c
                for c in quantifiable_metric_columns(filtered)
            }
            if "Column" not in metric_view.columns:
                metric_view["Column"] = metric_view.apply(
                    lambda r: metric_column_lookup.get((r.get("Category"), r.get("Metric"))),
                    axis=1,
                )
            else:
                metric_view["Column"] = metric_view.apply(
                    lambda r: (
                        r.get("Column")
                        if r.get("Column") in filtered.columns
                        else metric_column_lookup.get((r.get("Category"), r.get("Metric")))
                    ),
                    axis=1,
                )

            for category, category_metrics in metric_view.groupby("Category", sort=False):
                category_total = category_metrics["Total"].sum()
                category_cols = [
                    c for c in category_metrics.get("Column", pd.Series(dtype=str)).tolist()
                    if c in filtered.columns
                ]
                if category_cols:
                    category_values = filtered[category_cols].apply(
                        lambda s: pd.to_numeric(s, errors="coerce")
                    ).fillna(0)
                    category_conversations = int((category_values.sum(axis=1) > 0).sum())
                else:
                    category_conversations = 0
                with st.expander(
                    f"{category} - {len(category_metrics)} metrics, {category_conversations} contributing conversations",
                    expanded=len(metric_view["Category"].unique()) == 1,
                ):
                    category_table = category_metrics[
                        ["Metric", "Column", "Total", "Average when flagged", "Conversations > 0"]
                    ].copy()
                    _render_metric_definition_table(category_table)
                    st.caption(f"Category total across displayed metrics: {category_total:g}")

                    metric_options = category_metrics["Metric"].tolist()
                    selected_metric = st.selectbox(
                        "Metric contributors",
                        metric_options,
                        key=f"metric_contributors_{category}",
                    )
                    selected_row = category_metrics[category_metrics["Metric"] == selected_metric].iloc[0]
                    metric_col = selected_row.get("Column") or metric_column_lookup.get(
                        (category, selected_metric)
                    )
                    
                    # Show metric definition
                    if metric_col:
                        definition = get_metric_definition(metric_col)
                        if definition:
                            st.info(definition)
                    if metric_col in filtered.columns:
                        values = pd.to_numeric(filtered[metric_col], errors="coerce").fillna(0)
                        contributors = filtered.loc[values > 0].copy()
                        contributor_cols = [c for c in contributor_base_cols if c in contributors.columns]
                        contributor_table = contributors[contributor_cols].copy()
                        contributor_table.insert(1, "metric_value", values.loc[contributors.index].values)
                        contributor_table = contributor_table.sort_values(
                            ["metric_value", "conversation_id"],
                            ascending=[False, True],
                        )
                        contributor_table = _prepare_display_table(
                            contributor_table,
                            [
                                "handled_status",
                                "cx_issue_severity",
                                "unhandled_resolution_subtype",
                                "main_issue_type",
                            ],
                        )
                        st.caption(
                            f"{len(contributor_table):,} conversations contributed to "
                            f"{selected_metric}."
                        )
                        _render_display_table(contributor_table, height=360)
                    else:
                        st.caption("This metric column is not available in the filtered conversation table.")
    else:
        st.caption("No quantifiable metrics returned by the conversation-level evaluator yet.")

    st.markdown("#### Date / time summary")
    if "conversation_start_date" in filtered.columns and not filtered.empty:
        try:
            parsed = pd.to_datetime(filtered["conversation_start_date"], errors="coerce")
            ts = filtered.assign(_d=parsed.dt.date)
            daily = ts.groupby("_d").size().reset_index(name="count")
            if not daily.empty:
                if HAS_PLOTLY:
                    fig = px.line(daily, x="_d", y="count", markers=True)
                    _plotly_layout(fig, height=300, xaxis_title="Date", yaxis_title="Conversations")
                    _render_plotly(fig)
                else:
                    _render_simple_line_chart(daily, "_d", "count", height=300)
            else:
                st.caption("No parseable dates.")
        except Exception:
            st.caption("Could not parse conversation_start_date.")
    else:
        st.caption("No start date column available.")

    st.markdown("---")
    st.markdown("### Conversation results")
    display_cols = [
        "conversation_id",
        "customer_name",
        "customer_phone",
        "conversation_start_date",
        "final_classification",
        "handled_status",
        "cx_issue_severity",
        "frustration_detected",
        "customer_started_frustrated",
        "customer_became_frustrated_during_chat",
        "customer_ended_frustrated",
        "frustration_timing",
        "unhandled_resolution_subtype",
        "final_customer_sentiment",
        "max_frustration_level",
        "main_issue_type",
        "main_issue_origin",
        "main_issue_summary",
        "customer_impact",
        "all_detected_issues",
        "positive_signals",
        "negative_signals",
        "management_summary",
        "recommended_actions",
        "manual_review_required",
        "manual_review_reason",
        "confidence",
    ]
    display_cols.extend(quantifiable_metric_columns(filtered))
    existing_cols = [c for c in display_cols if c in filtered.columns]
    conversation_table = _prepare_display_table(
        filtered[existing_cols],
        [
            "handled_status",
            "cx_issue_severity",
            "frustration_detected",
            "frustration_timing",
            "unhandled_resolution_subtype",
            "final_customer_sentiment",
            "max_frustration_level",
            "main_issue_type",
            "main_issue_origin",
            "confidence",
        ],
    )
    _render_display_table(conversation_table, height=520)

    st.markdown("### Message-level results")
    message_display_cols = [
        "conversation_id",
        "target_message_id",
        "message_index",
        "message_time",
        "target_message_text",
        "message_level_effect",
        "frustration_level_after_message",
        "frustration_change",
        "customer_effort_level",
        "clarity_level",
        "context_handling",
        "issue_origin",
        "issue_type",
        "frustration_cause",
        "evidence",
        "business_impact",
        "recommended_fix",
        "parse_status",
        "error_message",
    ]
    message_existing_cols = [c for c in message_display_cols if c in msg_df.columns]
    if message_existing_cols:
        message_table = _prepare_display_table(
            msg_df[message_existing_cols],
            [
                "message_level_effect",
                "frustration_level_after_message",
                "frustration_change",
                "customer_effort_level",
                "clarity_level",
                "context_handling",
                "issue_origin",
                "issue_type",
                "parse_status",
            ],
        )
        _render_display_table(message_table, height=520)
    else:
        st.caption("No message-level results available.")


# --------- Tab: Conversation Review ---------


def tab_review() -> None:
    st.subheader("Conversation Review")
    if not _has_results():
        st.info("Run an evaluation first.")
        return

    rr = st.session_state.run_results
    conv_df = _conv_dataframe_from_results()
    if conv_df.empty:
        st.info("No conversation results are available yet.")
        return

    st.caption(
        "Browse conversations by result, customer frustration, review priority, or the main customer problem."
    )

    review_filters = _conversation_filters_with_keys(conv_df, "review_filters")
    filtered_df = _apply_conversation_filters_fresh(conv_df, review_filters)

    search = st.text_input(
        "Search by ID, customer name, result, or problem summary",
        value="",
    ).strip()
    if search:
        search_text = search.lower()
        search_cols = [
            "conversation_id",
            "customer_name",
            "final_classification",
            "main_issue_summary",
        ]
        mask = pd.Series(False, index=filtered_df.index)
        for col in search_cols:
            if col in filtered_df.columns:
                mask = mask | filtered_df[col].fillna("").astype(str).str.lower().str.contains(search_text, regex=False)
        filtered_df = filtered_df[mask]

    if filtered_df.empty:
        st.warning("No conversations match the current filters.")
        return

    metric_row(
        [
            ("Conversations shown", f"{len(filtered_df):,}", None),
            (
                "Handled",
                f"{int((filtered_df.get('handled_status') == 'handled').sum()):,}",
                None,
            ),
            (
                "Need human review",
                f"{int(filtered_df.get('manual_review_required', pd.Series(dtype=bool)).fillna(False).astype(bool).sum()):,}",
                None,
            ),
            (
                "High frustration",
                f"{int(filtered_df.get('max_frustration_level', pd.Series(dtype=str)).isin(['high', 'cancellation_risk']).sum()):,}",
                None,
            ),
        ]
    )

    options = []
    label_to_id = {}
    for row in filtered_df.to_dict(orient="records"):
        cid = row.get("conversation_id", "")
        cust = row.get("customer_name") or "—"
        result = row.get("final_classification") or "Unknown"
        label = f"{cid} • {cust} • {result}"
        options.append(label)
        label_to_id[label] = cid

    selection = st.selectbox("Open a conversation", options, index=0)
    target_id = label_to_id[selection]
    target_cr = next((c for c in rr.conversation_results if c.get("conversation_id") == target_id), None)
    if not target_cr:
        st.error("Conversation not found.")
        return

    _render_conversation_summary_card_fresh(target_cr)

    st.markdown("### Full Conversation")
    st.caption(
        "The full conversation is shown below. Where available, assistant replies also include a short quality check underneath."
    )
    transcript = target_cr.get("transcript") or []
    msgs = target_cr.get("message_level_results") or []
    render_conversation_transcript_with_evals(
        transcript=transcript,
        message_results=msgs,
    )


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
        "max_conversations": st.session_state.max_conversations,
        "max_target_messages_per_conversation": st.session_state.max_agent_messages_per_conv,
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
        st.markdown("#### Conversation-Level CSV")
        st.caption("One row per conversation, ready for spreadsheets.")
        st.download_button(
            "Download conversation_results.csv",
            data=conv_bytes,
            file_name="cx_conversation_results.csv",
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
    tab_a, tab_b = st.tabs(["Conversation-level preview", "Message-level preview"])
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
            label = f"`{m.get('conversation_id')}` #{m.get('message_index')} — {m.get('parse_status')}"
            with st.expander(label):
                st.markdown("**Error message**")
                st.code(m.get("error_message") or "—")
                st.markdown("**Raw model response**")
                st.code(m.get("raw_model_response") or "—")
                st.markdown("**Debug info**")
                st.json(m.get("debug") or {}, expanded=False)
    else:
        st.caption("No failed message-level evaluations.")

    st.markdown("### Failed conversation-level evaluations")
    failed_convs = [c for c in rr.conversation_results if c.get("parse_status") != "ok"]
    if failed_convs:
        st.write(f"{len(failed_convs)} failed conversation-level evaluations.")
        for c in failed_convs[:50]:
            with st.expander(f"`{c.get('conversation_id')}` — {c.get('parse_status')}"):
                st.markdown("**Error message**")
                st.code(c.get("error_message") or "—")
                st.markdown("**Raw model response**")
                st.code(c.get("raw_model_response") or "—")
                st.markdown("**Debug info**")
                st.json(c.get("debug") or {}, expanded=False)
    else:
        st.caption("No failed conversation-level evaluations.")

    st.markdown("### Inspect a specific record")
    st.caption("Pick any conversation to view raw payloads, parsed JSON, and debug info.")
    ids = [c.get("conversation_id", "") for c in rr.conversation_results]
    if ids:
        sel = st.selectbox("ID", ids)
        target = next((c for c in rr.conversation_results if c.get("conversation_id") == sel), None)
        if target:
            with st.expander("Conversation-level parsed JSON"):
                st.json(target.get("parsed_json") or {}, expanded=False)
            with st.expander("Conversation-level raw model response"):
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
        "max_conversations": st.session_state.max_conversations,
        "max_target_messages_per_conversation": st.session_state.max_agent_messages_per_conv,
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

    st.title("CX Conversation Evaluator")
    st.caption(
        "AI-as-a-Judge evaluation of customer/assistant conversations. "
        "Built for management review — focused on outcomes, frustration, and root cause."
    )

    # Force DB initialization at app start so the seeded defaults exist before
    # any tab tries to read them.
    get_db()

    tabs = st.tabs(
        [
            "Upload & Settings",
            "Prompts",
            "Run Evaluation",
            "Dashboard",
            "Conversation Review",
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
        tab_dashboard()
    with tabs[4]:
        tab_review()
    with tabs[5]:
        tab_exports()
    with tabs[6]:
        tab_debug()


if __name__ == "__main__":
    main()
