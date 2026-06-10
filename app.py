"""Streamlit entry point for the AI-as-a-Judge CX Conversation Evaluator."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

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
        # When set, the run evaluates ONLY these conversation IDs (random sampler).
        "selected_conversation_ids": None,
        "run_results": None,
        "run_in_progress": False,
        "progress_log": [],
        "cancel_flag": False,
        # DB integration
        "current_run_id": None,        # id of the run we're writing to (or loaded from)
        "loaded_run_label": None,
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


# --------- Sidebar ---------


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## API Settings")
        st.text_input(
            "Base URL",
            value=st.session_state.api_base_url,
            key="api_base_url",
            help="OpenAI-compatible base URL.",
        )
        st.text_input(
            "API Key",
            value=st.session_state.api_key,
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
                value=st.session_state.selected_model,
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
            "Max agent messages per conversation",
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
                "agent": "Agent messages",
                "customer": "Customer messages",
            }.get(v, v),
            help=(
                "Agent: judge each agent reply — how it responded to a "
                "possibly-frustrated customer message.\n\n"
                "Customer: judge each customer message — capture the customer's "
                "state / frustration BEFORE the agent answers."
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
            ("Agent messages", f"{summary.get('agent_messages', 0):,}", None),
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
                "Pick a random sample of conversation IDs from the uploaded CSV. "
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
    role_label = "agent" if target_role == "agent" else "customer"
    st.caption(
        f"Message-level layer will evaluate **{role_label} messages** "
        + ("(judging the agent's response to a possibly-frustrated customer message)."
           if role_label == "agent"
           else "(capturing the customer's state / frustration before the agent answers).")
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
            "Consider lowering Max conversations or Max agent messages per conversation in the sidebar."
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
            "max_agent_messages_per_conv": config.max_agent_messages_per_conv,
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
                    f"{evt.get('agent_messages', 0)} agent messages"
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
                db.save_message_result(run_id, mr)
            except Exception:
                pass

        def save_conversation(cr: dict) -> None:
            try:
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

    filters = conversation_filters(conv_df)
    filtered = apply_conversation_filters(conv_df, filters)
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
    left, right = st.columns(2)
    with left:
        st.markdown("#### Classification distribution")
        if agg["classification_counts"]:
            cls_df = pd.DataFrame(
                [{"Classification": k, "Count": v} for k, v in agg["classification_counts"].items()]
            )
            if HAS_PLOTLY:
                fig = px.bar(cls_df, x="Classification", y="Count", text="Count", color="Classification")
                fig.update_layout(showlegend=False, xaxis_tickangle=-15, height=380, margin=dict(t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(cls_df.set_index("Classification"))
        else:
            st.write("No data.")

        st.markdown("#### Handled vs Unhandled")
        if "handled_status" in filtered.columns and not filtered.empty:
            hs_df = filtered["handled_status"].fillna("unknown").value_counts().reset_index()
            hs_df.columns = ["handled_status", "count"]
            if HAS_PLOTLY:
                fig = px.pie(hs_df, names="handled_status", values="count", hole=0.45)
                fig.update_layout(height=340, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(hs_df.set_index("handled_status"))
        else:
            st.write("No data.")

    with right:
        st.markdown("#### Zero/Minimal vs Many Issues")
        if "cx_issue_severity" in filtered.columns and not filtered.empty:
            sev_df = filtered["cx_issue_severity"].fillna("unknown").value_counts().reset_index()
            sev_df.columns = ["cx_issue_severity", "count"]
            if HAS_PLOTLY:
                fig = px.pie(sev_df, names="cx_issue_severity", values="count", hole=0.45)
                fig.update_layout(height=340, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(sev_df.set_index("cx_issue_severity"))
        else:
            st.write("No data.")

        st.markdown("#### Issue origin distribution")
        if agg["issue_origin_counts"]:
            io_df = pd.DataFrame(
                [{"Origin": k, "Count": v} for k, v in agg["issue_origin_counts"].items()]
            )
            if HAS_PLOTLY:
                fig = px.bar(io_df, x="Origin", y="Count", text="Count", color="Origin")
                fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(io_df.set_index("Origin"))
        else:
            st.write("No data.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top issue types")
        if agg["issue_type_counts"]:
            it_df = (
                pd.DataFrame([{"Issue type": k, "Count": v} for k, v in agg["issue_type_counts"].items()])
                .sort_values("Count", ascending=False)
                .head(12)
            )
            if HAS_PLOTLY:
                fig = px.bar(it_df, x="Count", y="Issue type", orientation="h", text="Count")
                fig.update_layout(height=400, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(it_df.set_index("Issue type"))
        else:
            st.write("No data.")

    with c2:
        st.markdown("#### Top frustration causes")
        causes = top_frustration_causes(msg_df, top_n=15)
        if not causes.empty:
            if HAS_PLOTLY:
                fig = px.bar(causes, x="count", y="frustration_cause", orientation="h", text="count")
                fig.update_layout(height=400, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(causes.set_index("frustration_cause"))
        else:
            st.write("No frustration causes identified.")

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Agent-level summary")
        if not agg["agent_breakdown"].empty:
            st.dataframe(agg["agent_breakdown"], use_container_width=True, hide_index=True)
        else:
            st.caption("Agent names not available in the CSV.")

    with c4:
        st.markdown("#### Skill-level summary")
        if not agg["skill_breakdown"].empty:
            st.dataframe(agg["skill_breakdown"], use_container_width=True, hide_index=True)
        else:
            st.caption("Skill columns not available in the CSV.")

    st.markdown("#### Date / time summary")
    if "conversation_start_date" in filtered.columns and not filtered.empty:
        try:
            parsed = pd.to_datetime(filtered["conversation_start_date"], errors="coerce")
            ts = filtered.assign(_d=parsed.dt.date)
            daily = ts.groupby("_d").size().reset_index(name="count")
            if not daily.empty:
                if HAS_PLOTLY:
                    fig = px.line(daily, x="_d", y="count", markers=True)
                    fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Date", yaxis_title="Conversations")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(daily.set_index("_d"))
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
        "initial_skill",
        "last_skill",
        "final_classification",
        "handled_status",
        "cx_issue_severity",
        "final_customer_sentiment",
        "max_frustration_level",
        "main_issue_type",
        "main_issue_origin",
        "main_issue_summary",
        "management_summary",
        "manual_review_required",
        "confidence",
    ]
    existing_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[existing_cols], use_container_width=True, hide_index=True)


# --------- Tab: Conversation Review ---------


def tab_review() -> None:
    st.subheader("Conversation Review")
    if not _has_results():
        st.info("Run an evaluation first.")
        return

    rr = st.session_state.run_results
    options = []
    label_to_id = {}
    for cr in rr.conversation_results:
        cid = cr.get("conversation_id", "")
        md = cr.get("conversation_metadata", {}) or {}
        pj = cr.get("parsed_json", {}) or {}
        cust = md.get("customer_name") or "—"
        label = f"{cid} • {cust} • {pj.get('final_classification', 'Unknown')}"
        options.append(label)
        label_to_id[label] = cid

    search = st.text_input("Search by conversation id, customer name, or classification", value="")
    filtered_options = (
        [o for o in options if search.lower() in o.lower()] if search else options
    )
    if not filtered_options:
        st.warning("No conversations match your search.")
        return

    selection = st.selectbox("Select a conversation", filtered_options, index=0)
    target_id = label_to_id[selection]
    target_cr = next((c for c in rr.conversation_results if c.get("conversation_id") == target_id), None)
    if not target_cr:
        st.error("Conversation not found.")
        return

    render_conversation_summary_card(target_cr)

    st.markdown("### Transcript with inline evaluations")
    st.caption(
        "Each agent message includes its message-level evaluation directly below it. "
        "Customer and unknown messages are shown for context."
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
        "max_agent_messages_per_conv": st.session_state.max_agent_messages_per_conv,
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
        st.caption("One row per evaluated agent message.")
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
        sel = st.selectbox("Conversation ID", ids)
        target = next((c for c in rr.conversation_results if c.get("conversation_id") == sel), None)
        if target:
            with st.expander("Conversation-level parsed JSON"):
                st.json(target.get("parsed_json") or {}, expanded=False)
            with st.expander("Conversation-level raw model response"):
                st.code(target.get("raw_model_response") or "—")
            with st.expander("Computed metadata"):
                st.json(target.get("computed_metadata") or {}, expanded=False)
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
        "max_agent_messages_per_conv": st.session_state.max_agent_messages_per_conv,
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
    st.title("CX Conversation Evaluator")
    st.caption(
        "AI-as-a-Judge evaluation of customer/agent conversations. "
        "Built for management review — focused on outcomes, frustration, and root cause."
    )

    # Force DB initialization at app start so the seeded defaults exist before
    # any tab tries to read them.
    get_db()

    render_sidebar()

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
