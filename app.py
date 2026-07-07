"""Streamlit entry point for the AI-as-a-Judge CX Conversation Evaluator."""

from __future__ import annotations

import json
import html as html_lib
import hmac
import importlib
import os
import sqlite3
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import ui_components as ui_components_module

from api_client import APIConfig, DEFAULT_BASE_URL, MAX_CONCURRENCY, build_client, fetch_models
from cost_estimator import estimate_run_tokens_and_cost
from data_loader import (
    JOURNEY_ID_COLUMN,
    METADATA_COLUMNS,
    REQUIRED_COLUMNS,
    conversation_metadata_from_group,
    estimate_call_counts,
    get_conversation_groups,
    load_csv,
    normalize_dataframe,
    proportional_stratified_sample_ids,
    summarize_dataframe,
    validate_csv,
)
from db import DEFAULT_DB_PATH, Database
from evaluator import (
    RunConfig,
    RunResults,
    run_conversation_level_only,
    run_evaluation,
    validate_conversation_level_result,
)
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


REVIEW_DB_PATH = Path("cx_evaluator_review_runs_59_57_55.db")
DEFAULT_SELECTED_MODEL = "openai/gpt-5.4-mini"
DEFAULT_EVALUATION_SETTINGS = {
    "api_base_url": DEFAULT_BASE_URL,
    "selected_model": DEFAULT_SELECTED_MODEL,
    "conversation_selected_model": DEFAULT_SELECTED_MODEL,
    "message_thinking_effort": "default",
    "conversation_thinking_effort": "default",
    "use_flex_service_tier": False,
    "temperature": 0.1,
    "top_p": 1.0,
    "max_tokens": 100000,
    "timeout": 300.0,
    "retries": 2,
    "concurrency": 60,
    "message_target_role": "agent",
}
DEFAULT_RUN_SETTINGS = {
    "run_all_conversations": False,
    "max_conversations": 50,
    "max_agent_messages_per_conv": 500,
    "truncate_messages": False,
    "max_chars_per_message": 1500,
    "include_unknown_in_history": True,
    "stop_on_error": False,
    "save_raw_responses": True,
}
DEFAULT_CHOICES_VERSION = 6
DEFAULT_DB_SELECTION_VERSION = 2
ROLE_MASTER = "master"
ROLE_ACTIVE = "active"
ROLE_READ_ONLY = "read_only"
ROLE_ALIASES = {
    "admin": ROLE_MASTER,
    "master": ROLE_MASTER,
    "active": ROLE_ACTIVE,
    "operator": ROLE_ACTIVE,
    "analyst": ROLE_ACTIVE,
    "reviewer": ROLE_READ_ONLY,
    "read_only": ROLE_READ_ONLY,
    "readonly": ROLE_READ_ONLY,
    "viewer": ROLE_READ_ONLY,
}
PROMPT_EDITING_ENABLED = False


def _config_value(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value:
        return str(value)
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or default)


def _config_csv_set(name: str) -> set[str]:
    raw = _config_value(name)
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _sso_auth_enabled() -> bool:
    return _config_value("CEX_AUTH_MODE", "local").strip().lower() in {"sso", "google", "oidc"}


def _sso_provider_name() -> str | None:
    provider = _config_value("CEX_OIDC_PROVIDER", "").strip()
    return provider or None


def _authlib_available() -> bool:
    try:
        return importlib.util.find_spec("authlib") is not None
    except Exception:
        return False


def _sso_auth_section() -> Any:
    try:
        return st.secrets.get("auth", {}) or {}
    except Exception:
        return {}


def _sso_auth_config_errors() -> list[str]:
    errors: list[str] = []
    if not _authlib_available():
        errors.append("Install Authlib with `pip install Authlib>=1.3.2` or `pip install streamlit[auth]`.")

    auth_section = _sso_auth_section()
    if not auth_section:
        errors.append("Add an `[auth]` section to `.streamlit/secrets.toml`.")
        return errors

    for key in ("redirect_uri", "cookie_secret"):
        if not str(auth_section.get(key, "") or "").strip():
            errors.append(f"Set `[auth].{key}` in `.streamlit/secrets.toml`.")

    redirect_uri = str(auth_section.get("redirect_uri", "") or "").strip()
    if redirect_uri and not redirect_uri.endswith("/oauth2callback"):
        errors.append("Set `[auth].redirect_uri` to a URL ending in `/oauth2callback`.")

    provider = _sso_provider_name()
    provider_section = auth_section.get(provider, {}) if provider else auth_section
    provider_label = f"[auth.{provider}]" if provider else "[auth]"
    if provider and not provider_section:
        errors.append(f"Add a `{provider_label}` section to `.streamlit/secrets.toml`.")
        return errors

    for key in ("client_id", "client_secret", "server_metadata_url"):
        if not str(provider_section.get(key, "") or "").strip():
            errors.append(f"Set `{provider_label}.{key}` in `.streamlit/secrets.toml`.")
    return errors


def _role_for_sso_email(email: str) -> str:
    normalized_email = str(email or "").strip().lower()
    admin_emails = _config_csv_set("CEX_ADMIN_EMAILS") | _config_csv_set("CEX_MASTER_EMAILS")
    if normalized_email and normalized_email in admin_emails:
        return ROLE_MASTER
    return ROLE_READ_ONLY


def _sso_email_allowed(email: str) -> bool:
    normalized_email = str(email or "").strip().lower()
    allowed_emails = _config_csv_set("CEX_ALLOWED_EMAILS")
    allowed_domains = _config_csv_set("CEX_ALLOWED_DOMAINS")
    if not allowed_emails and not allowed_domains:
        return True
    if normalized_email in allowed_emails:
        return True
    if "@" not in normalized_email:
        return False
    domain = normalized_email.rsplit("@", 1)[-1]
    return domain in allowed_domains


def _normalize_role(value: Any) -> str:
    return ROLE_ALIASES.get(str(value or "").strip().lower(), ROLE_READ_ONLY)


def _current_role() -> str:
    role = _normalize_role(st.session_state.get("auth_role"))
    st.session_state.auth_role = role
    return role


def _is_master() -> bool:
    return _current_role() == ROLE_MASTER


def _can_run_evaluations() -> bool:
    return _current_role() in {ROLE_MASTER, ROLE_ACTIVE}


def _can_export_results() -> bool:
    return _can_run_evaluations()


def _can_manage_runs() -> bool:
    return _is_master()


def _is_read_only() -> bool:
    return _current_role() == ROLE_READ_ONLY


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
        "api_base_url": DEFAULT_EVALUATION_SETTINGS["api_base_url"],
        "api_key": "",
        "selected_model": DEFAULT_EVALUATION_SETTINGS["selected_model"],
        "conversation_selected_model": DEFAULT_EVALUATION_SETTINGS["conversation_selected_model"],
        "message_thinking_effort": DEFAULT_EVALUATION_SETTINGS["message_thinking_effort"],
        "conversation_thinking_effort": DEFAULT_EVALUATION_SETTINGS["conversation_thinking_effort"],
        "use_flex_service_tier": DEFAULT_EVALUATION_SETTINGS["use_flex_service_tier"],
        "temperature": DEFAULT_EVALUATION_SETTINGS["temperature"],
        "top_p": DEFAULT_EVALUATION_SETTINGS["top_p"],
        "max_tokens": DEFAULT_EVALUATION_SETTINGS["max_tokens"],
        "timeout": DEFAULT_EVALUATION_SETTINGS["timeout"],
        "retries": DEFAULT_EVALUATION_SETTINGS["retries"],
        "concurrency": DEFAULT_EVALUATION_SETTINGS["concurrency"],
        "run_all_conversations": DEFAULT_RUN_SETTINGS["run_all_conversations"],
        "max_conversations": DEFAULT_RUN_SETTINGS["max_conversations"],
        "max_agent_messages_per_conv": DEFAULT_RUN_SETTINGS["max_agent_messages_per_conv"],
        "truncate_messages": DEFAULT_RUN_SETTINGS["truncate_messages"],
        "max_chars_per_message": DEFAULT_RUN_SETTINGS["max_chars_per_message"],
        "include_unknown_in_history": DEFAULT_RUN_SETTINGS["include_unknown_in_history"],
        "stop_on_error": DEFAULT_RUN_SETTINGS["stop_on_error"],
        "save_raw_responses": DEFAULT_RUN_SETTINGS["save_raw_responses"],
        # Which side the message-level judge inspects per turn.
        "message_target_role": DEFAULT_EVALUATION_SETTINGS["message_target_role"],
        # When set, the run evaluates ONLY these IDs (random sampler).
        "selected_conversation_ids": None,
        "selection_import_feedback": None,
        "run_results": None,
        "run_in_progress": False,
        "progress_log": [],
        "cancel_flag": False,
        # DB integration
        "db_path": str(REVIEW_DB_PATH if REVIEW_DB_PATH.exists() else DEFAULT_DB_PATH),
        "current_run_id": None,        # id of the run we're writing to (or loaded from)
        "loaded_run_label": None,
        "review_selected_conversation_id": None,
        "theme_mode": "Dark",
        "auth_user": None,
        "auth_role": None,
        "auth_reviewer_key_id": None,
        "auth_db_path": None,
        "generated_reviewer_key": None,
        "_run_results_db_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.get("_default_db_selection_version") != DEFAULT_DB_SELECTION_VERSION:
        if REVIEW_DB_PATH.exists() and st.session_state.get("db_path") in {None, "", str(DEFAULT_DB_PATH)}:
            st.session_state.db_path = str(REVIEW_DB_PATH)
        st.session_state["_default_db_selection_version"] = DEFAULT_DB_SELECTION_VERSION
    if st.session_state.get("_default_choices_version") != DEFAULT_CHOICES_VERSION:
        for key, value in DEFAULT_EVALUATION_SETTINGS.items():
            st.session_state[key] = value
        for key, value in DEFAULT_RUN_SETTINGS.items():
            st.session_state[key] = value
        st.session_state["_default_choices_version"] = DEFAULT_CHOICES_VERSION
    if not st.session_state.get("selected_model"):
        st.session_state.selected_model = DEFAULT_SELECTED_MODEL
    if not st.session_state.get("conversation_selected_model"):
        st.session_state.conversation_selected_model = st.session_state.selected_model
    if not st.session_state.get("api_base_url"):
        st.session_state.api_base_url = DEFAULT_BASE_URL
    if st.session_state.get("_separate_thinking_effort_version") != 1:
        legacy_effort = str(st.session_state.get("thinking_effort") or "default")
        if legacy_effort in {"default", "disabled", "low", "medium", "high", "maximum"}:
            st.session_state.message_thinking_effort = legacy_effort
            st.session_state.conversation_thinking_effort = legacy_effort
        st.session_state["_separate_thinking_effort_version"] = 1


_init_state()


# --------- Database singleton ---------


@st.cache_resource(show_spinner=False)
def get_db(path: str = str(DEFAULT_DB_PATH)) -> Database:
    """Return a process-wide :class:`Database` instance (cached by Streamlit)."""
    return Database(path)


def _resolve_db_path(path: str | Path) -> Path:
    db_path = Path(path)
    return db_path if db_path.is_absolute() else Path.cwd() / db_path


def _active_db_path() -> str:
    return str(st.session_state.get("db_path") or DEFAULT_DB_PATH)


def get_active_db() -> Database:
    return get_db(_active_db_path())


def _clear_auth_state(*, clear_sensitive_data: bool = False) -> None:
    st.session_state.auth_user = None
    st.session_state.auth_role = None
    st.session_state.auth_reviewer_key_id = None
    st.session_state.auth_db_path = None
    st.session_state.generated_reviewer_key = None
    if clear_sensitive_data:
        st.session_state.df_raw = None
        st.session_state.df_norm = None
        st.session_state.csv_summary = None
        st.session_state.csv_name = None
        st.session_state.run_results = None
        st.session_state.current_run_id = None
        st.session_state.loaded_run_label = None
        st.session_state.review_selected_conversation_id = None
        st.session_state._run_results_db_path = None


def _logout_current_user() -> None:
    _clear_auth_state(clear_sensitive_data=True)
    if _sso_auth_enabled() and hasattr(st, "logout"):
        try:
            st.logout()
            return
        except Exception:
            pass
    st.rerun()


def _reset_default_choices() -> None:
    for key, value in DEFAULT_EVALUATION_SETTINGS.items():
        st.session_state[key] = value
    for key, value in DEFAULT_RUN_SETTINGS.items():
        st.session_state[key] = value
    st.session_state["_default_choices_version"] = DEFAULT_CHOICES_VERSION
    st.session_state.selected_conversation_ids = None
    st.session_state.selection_import_feedback = None


def _ensure_parameter_defaults_exist() -> None:
    for key, value in DEFAULT_EVALUATION_SETTINGS.items():
        if st.session_state.get(key) in (None, ""):
            st.session_state[key] = value
    for key, value in DEFAULT_RUN_SETTINGS.items():
        if st.session_state.get(key) is None:
            st.session_state[key] = value


def _local_secrets_path() -> Path:
    return Path(".streamlit") / "secrets.toml"


def _read_local_master_key() -> str:
    path = _local_secrets_path()
    if not path.exists():
        return ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or not stripped.startswith("CEX_MASTER_KEY"):
                continue
            _, raw_value = stripped.split("=", 1)
            raw_value = raw_value.strip()
            try:
                return str(json.loads(raw_value) or "")
            except Exception:
                return raw_value.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _write_local_master_key(master_key: str) -> None:
    path = _local_secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret_line = f"CEX_MASTER_KEY = {json.dumps(str(master_key))}"
    if not path.exists():
        path.write_text(
            "# Local Streamlit secrets. This file is ignored by git.\n"
            f"{secret_line}\n",
            encoding="utf-8",
        )
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("CEX_MASTER_KEY"):
            updated.append(secret_line)
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(secret_line)
    path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def _configured_master_key() -> str:
    env_value = os.environ.get("CEX_MASTER_KEY", "")
    if env_value:
        return env_value
    try:
        secret_value = str(st.secrets.get("CEX_MASTER_KEY", "") or "")
        if secret_value:
            return secret_value
    except Exception:
        pass
    return _read_local_master_key()


def _session_master_key() -> str:
    return str(st.session_state.get("session_master_key") or "")


def _verify_master_key_input(value: str) -> bool:
    configured = _configured_master_key()
    if configured:
        return hmac.compare_digest(str(value or ""), configured)
    session_master_key = _session_master_key()
    return bool(session_master_key) and hmac.compare_digest(str(value or ""), session_master_key)


def _render_auth_intro() -> None:
    st.markdown(
        """
        <div class="cx-auth-header">
          <div class="cx-brand cx-brand-large">
            <span class="cx-brand-mark">m</span>
            <span class="cx-brand-word">maids.cc</span>
          </div>
          <div class="cx-app-kicker">Secure access</div>
          <h1>CX Review Platform</h1>
          <p>Enter your access key once to open the workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workspace_header() -> None:
    db_name = html_lib.escape(Path(_active_db_path()).name)
    run_id = st.session_state.get("current_run_id")
    run_label = f"Run #{int(run_id)}" if run_id is not None else "No run loaded"
    run_label = html_lib.escape(run_label)
    st.markdown(
        f"""
        <div class="cx-app-header">
          <div class="cx-title-wrap">
            <div class="cx-brand">
              <span class="cx-brand-mark">m</span>
              <span class="cx-brand-word">maids.cc</span>
            </div>
            <div>
              <div class="cx-app-kicker">Review workspace</div>
              <h1>CX Journey Review</h1>
              <div class="cx-app-subtitle">Evaluate, analyze, and mark customer journeys from one place.</div>
            </div>
          </div>
          <div class="cx-header-meta">
            <span class="cx-pill">{db_name}</span>
            <span class="cx-pill">{run_label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _streamlit_user_field(name: str) -> str:
    user = getattr(st, "user", None)
    if user is None:
        return ""
    try:
        value = user.get(name, "")
    except Exception:
        value = getattr(user, name, "")
    return str(value or "")


def _streamlit_user_is_logged_in() -> bool:
    user = getattr(st, "user", None)
    if user is None:
        return False
    try:
        return bool(user.get("is_logged_in", False))
    except Exception:
        return bool(getattr(user, "is_logged_in", False))


def _render_sso_auth_gate() -> bool:
    if not hasattr(st, "login"):
        _render_auth_intro()
        st.error(
            "SSO mode is enabled, but this Streamlit version does not expose st.login. "
            "Upgrade Streamlit or set CEX_AUTH_MODE=local."
        )
        return False

    if _streamlit_user_is_logged_in():
        email = _streamlit_user_field("email")
        if not _sso_email_allowed(email):
            _render_auth_intro()
            st.error("This Google account is not allowed to access the CX Review Platform.")
            if st.button("Log out", use_container_width=True):
                _logout_current_user()
            return False
        name = _streamlit_user_field("name") or email or "SSO user"
        st.session_state.auth_user = name
        st.session_state.auth_role = _role_for_sso_email(email)
        st.session_state.auth_reviewer_key_id = None
        st.session_state.auth_db_path = _active_db_path()
        return True

    _render_auth_intro()
    _, auth_col, _ = st.columns([1, 1.1, 1])
    with auth_col:
        st.subheader("Google SSO")
        st.caption("New SSO users open in read-only mode unless their email is configured as an admin.")
        config_errors = _sso_auth_config_errors()
        if config_errors:
            st.error("Google SSO is enabled but not fully configured.")
            for error in config_errors:
                st.markdown(f"- {error}")
            return False
        if st.button("Sign in with Google", type="primary", use_container_width=True):
            try:
                provider = _sso_provider_name()
                if provider:
                    st.login(provider)
                else:
                    st.login()
            except Exception as exc:
                st.error(f"Could not start SSO login: {exc}")
    return False


def _render_auth_gate() -> bool:
    if _sso_auth_enabled():
        return _render_sso_auth_gate()

    active_db_path = _active_db_path()
    if st.session_state.get("auth_user"):
        return True

    db = get_active_db()
    has_configured_master = bool(_configured_master_key())
    has_session_master = bool(_session_master_key())

    _render_auth_intro()
    _, auth_col, _ = st.columns([1, 1.1, 1])

    if not has_configured_master and not has_session_master:
        with auth_col:
            st.subheader("Master key setup")
            with st.form("master_setup_form"):
                master_key = st.text_input("Master key", type="password")
                confirm_key = st.text_input("Confirm master key", type="password")
                submitted = st.form_submit_button("Continue", type="primary")
            if submitted:
                if len(master_key or "") < 12:
                    st.error("Use at least 12 characters for the master key.")
                elif master_key != confirm_key:
                    st.error("Master keys do not match.")
                else:
                    try:
                        _write_local_master_key(master_key)
                    except Exception as e:
                        st.error(f"Could not save the master key locally: {e}")
                        return False
                    st.session_state.session_master_key = master_key
                    st.session_state.auth_user = "Master admin"
                    st.session_state.auth_role = ROLE_MASTER
                    st.session_state.auth_reviewer_key_id = None
                    st.session_state.auth_db_path = active_db_path
                    st.session_state.run_results = None
                    st.session_state._run_results_db_path = None
                    st.rerun()
        return False

    with auth_col:
        with st.form("auth_login_form"):
            secret = st.text_input("Access key", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")

        if submitted:
            if _verify_master_key_input(secret):
                st.session_state.auth_user = "Master admin"
                st.session_state.auth_role = ROLE_MASTER
                st.session_state.auth_reviewer_key_id = None
                st.session_state.auth_db_path = active_db_path
                # Force a fresh "load the latest run by date" on every login,
                # rather than keeping whatever run_results a prior session on
                # this browser tab happened to leave behind.
                st.session_state.run_results = None
                st.session_state._run_results_db_path = None
                st.rerun()
            else:
                reviewer_db = db
                reviewer_db_path = active_db_path
                if REVIEW_DB_PATH.exists():
                    reviewer_db_path = str(REVIEW_DB_PATH)
                    reviewer_db = get_db(reviewer_db_path)
                reviewer = reviewer_db.verify_reviewer_key(secret)
                if reviewer:
                    st.session_state.db_path = reviewer_db_path
                    st.session_state.auth_user = reviewer["reviewer_name"]
                    st.session_state.auth_role = _normalize_role(reviewer.get("role"))
                    st.session_state.auth_reviewer_key_id = reviewer["id"]
                    st.session_state.auth_db_path = reviewer_db_path
                    st.session_state.run_results = None
                    st.session_state._run_results_db_path = None
                    st.rerun()
                else:
                    st.error("Invalid or revoked access key.")
    return False


def _render_read_only_sidebar() -> None:
    auth_name = st.session_state.get("auth_user") or "Read-only user"
    auth_role_label = humanize_label(_current_role())

    with st.sidebar:
        st.markdown(
            """
            <div class="cx-sidebar-brand">
              <span class="cx-brand-mark">m</span>
              <span class="cx-brand-word">maids.cc</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("## Access")
        st.markdown(
            f"""
            <div class="cx-sidebar-mini">
              <div>Signed in</div>
              <div>{html_lib.escape(str(auth_name))} - {html_lib.escape(str(auth_role_label))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Read-only access: load saved runs, review dashboards, inspect journeys, and use filters.")
        st.markdown("---")
        render_database_selector(in_sidebar=True)
        if st.button("Log out", use_container_width=True):
            _logout_current_user()


def _available_database_options() -> tuple[list[str], dict[str, str]]:
    options: list[str] = []
    labels: dict[str, str] = {}

    def add_option(path: str | Path, label: str) -> None:
        value = str(path)
        if value in labels:
            return
        options.append(value)
        labels[value] = label

    if REVIEW_DB_PATH.exists():
        add_option(REVIEW_DB_PATH, "Review DB - runs 5, 59, 57, 55")
    add_option(DEFAULT_DB_PATH, "Local working DB - cx_evaluator.db")

    known = {Path(str(DEFAULT_DB_PATH)).name, REVIEW_DB_PATH.name}
    for path in sorted(Path.cwd().glob("*.db")):
        if path.name in known:
            continue
        add_option(path.name, f"Database file - {path.name}")

    current = _active_db_path()
    if current not in labels:
        add_option(current, f"Selected DB - {Path(current).name}")

    return options, labels


def _on_database_source_changed() -> None:
    st.session_state.current_run_id = None
    st.session_state.loaded_run_label = None
    st.session_state.run_results = None
    st.session_state.review_selected_conversation_id = None
    st.session_state.selection_import_feedback = None
    st.session_state.progress_log = []
    st.session_state.generated_reviewer_key = None
    st.session_state.auth_db_path = _active_db_path()
    st.session_state._run_results_db_path = None
    if _is_read_only():
        st.session_state.auth_reviewer_key_id = None
    try:
        get_db.clear()
    except Exception:
        pass


def render_database_selector(*, in_sidebar: bool = False) -> None:
    options, labels = _available_database_options()
    if st.session_state.get("db_path") not in labels:
        st.session_state.db_path = options[0]

    if in_sidebar:
        st.markdown("## Data")
    else:
        st.markdown("### Data source")
    selected = st.selectbox(
        "Load runs from",
        options=options,
        key="db_path",
        format_func=lambda value: labels.get(value, value),
        on_change=_on_database_source_changed,
        help="Switch between your full local database and the small review database.",
    )

    resolved = _resolve_db_path(selected)
    if resolved.exists():
        size_mb = resolved.stat().st_size / (1024 * 1024)
        st.caption(f"Using `{selected}` ({size_mb:.2f} MB).")
    else:
        st.warning(f"`{selected}` does not exist yet. The app will create it when needed.")

    if Path(selected).name == REVIEW_DB_PATH.name:
        st.caption("Review DB: runs 5, 59, 57, and 55.")


def _prompt_status_label(row: sqlite3.Row | None) -> str:
    if not row:
        return "Not found"
    prompt_id = row["id"]
    prompt_name = str(row["name"] or "Untitled")
    prompt_type = "default" if row["is_default"] else "custom"
    return f"#{prompt_id} {prompt_name} ({prompt_type})"


def render_active_prompt_status() -> None:
    st.markdown("## Active prompts")
    try:
        db = get_active_db()
        message_prompt = db.get_active_prompt("message_level")
        conversation_prompt = db.get_active_prompt("conversation_level")
    except Exception as exc:
        st.warning(f"Could not read active prompts: {exc}")
        return

    st.caption(f"Database: `{Path(_active_db_path()).name}`")
    st.markdown(f"**Message:** `{_prompt_status_label(message_prompt)}`")
    st.markdown(f"**Conversation:** `{_prompt_status_label(conversation_prompt)}`")
    st.caption("Used for the next evaluation run.")


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

    missing_ids = [int(x) for x in df_runs.loc[needs_counts, "id"].tolist()]
    if hasattr(db, "get_run_result_counts_bulk"):
        counts_by_id = db.get_run_result_counts_bulk(missing_ids)
    else:
        counts_by_id = {rid: _run_result_counts(db, rid) for rid in missing_ids}

    default_counts = {"conversation_results": 0, "message_results": 0, "run_errors": 0}
    for index, row in df_runs[needs_counts].iterrows():
        counts = counts_by_id.get(int(row["id"]), default_counts)
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
    db = get_active_db()
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
        service_tier="flex" if st.session_state.use_flex_service_tier else None,
        thinking_effort=str(st.session_state.message_thinking_effort),
        temperature=float(st.session_state.temperature),
        top_p=float(st.session_state.top_p),
        max_tokens=int(st.session_state.max_tokens),
        timeout=float(st.session_state.timeout),
        retries=int(st.session_state.retries),
        concurrency=concurrency,
    )


def _build_conversation_api_config() -> APIConfig:
    api = _build_api_config()
    return APIConfig(
        base_url=api.base_url,
        api_key=api.api_key,
        model=str(
            st.session_state.get("conversation_selected_model")
            or st.session_state.selected_model
        ),
        service_tier=api.service_tier,
        thinking_effort=str(st.session_state.conversation_thinking_effort),
        temperature=api.temperature,
        top_p=api.top_p,
        max_tokens=api.max_tokens,
        timeout=api.timeout,
        retries=api.retries,
        concurrency=api.concurrency,
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
        conversation_api=_build_conversation_api_config(),
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


def _show_live_run_failure(
    failure_box,
    failures: list[dict],
    error: dict,
) -> None:
    """Show persisted evaluator failures immediately while a run is active."""
    failures.append(dict(error or {}))
    visible = failures[-5:]
    lines: list[str] = []
    for failure in visible:
        level = humanize_label(failure.get("level") or "evaluation")
        location = []
        if failure.get("conversation_id") not in (None, ""):
            location.append(f"Customer `{failure['conversation_id']}`")
        if failure.get("message_index") not in (None, ""):
            location.append(f"message #{failure['message_index']}")
        heading = f"**{level} failure**"
        if location:
            heading += " — " + ", ".join(location)
        message = (
            failure.get("error")
            or failure.get("error_message")
            or "The evaluation failed without an error message."
        )
        lines.append(f"{heading}\n\n{message}")

    prefix = f"**{len(failures):,} evaluation failure(s) detected.**"
    if len(failures) > len(visible):
        prefix += " Showing the latest 5."
    failure_box.error(prefix + "\n\n" + "\n\n---\n\n".join(lines))


def _show_live_message_rerun(
    rerun_box,
    rerun_events: list[dict],
    event: dict,
) -> None:
    """Show automatic message reruns and whether they recovered."""
    rerun_events.append(dict(event or {}))
    visible = rerun_events[-5:]
    lines: list[str] = []
    for rerun in visible:
        recovered = bool(rerun.get("recovered_after_rerun"))
        outcome = "Recovered successfully" if recovered else "Still failed"
        errors = [str(value) for value in (rerun.get("rerun_errors") or []) if value]
        detail = errors[-1] if errors else "The original response could not be parsed."
        lines.append(
            f"**{outcome}** — Customer `{rerun.get('conversation_id')}`, "
            f"message #{rerun.get('message_index')} — "
            f"{int(rerun.get('automatic_reruns') or 0)} automatic rerun(s)\n\n"
            f"Initial failure: {detail}"
        )
    prefix = f"**{len(rerun_events):,} message evaluation(s) automatically rerun.**"
    if len(rerun_events) > len(visible):
        prefix += " Showing the latest 5."
    rerun_box.warning(prefix + "\n\n" + "\n\n---\n\n".join(lines))


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


def _saved_run_label(row: dict) -> str:
    return (
        f"#{row.get('id')} • {row.get('name') or 'Untitled run'} • "
        f"{row.get('csv_name') or '—'} • {row.get('status') or 'unknown'} • "
        f"{row.get('started_at') or '—'}"
    )


def _saved_run_is_loadable(row: dict) -> bool:
    n_conversations = int(row.get("n_conversations") or 0)
    saved_conversations = int(row.get("saved_conversations") or 0)
    return not (n_conversations > 0 and saved_conversations == 0)


def _load_saved_run_into_session(db: Database, run_id: int, *, label: str | None = None) -> None:
    loaded = db.load_run_results(int(run_id))
    run = loaded.get("run") or db.get_run(int(run_id)) or {}
    if not loaded["conversation_results"] and int(run.get("n_conversations") or 0) > 0:
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
    st.session_state.run_results = _normalize_run_results_for_display(rr)
    st.session_state.current_run_id = int(run_id)
    st.session_state.loaded_run_label = label or _saved_run_label(run)
    st.session_state.review_selected_conversation_id = None
    st.session_state._run_results_db_path = _active_db_path()


def _auto_load_latest_run(db: Database) -> None:
    active_db_path = _active_db_path()
    if _has_results() and st.session_state.get("_run_results_db_path") == active_db_path:
        return
    try:
        runs = db.list_runs(limit=50)
    except Exception:
        return
    for row in runs:
        if not _saved_run_is_loadable(row):
            continue
        try:
            _load_saved_run_into_session(db, int(row["id"]), label=_saved_run_label(row))
            st.session_state.latest_run_autoloaded = True
            return
        except Exception:
            continue
    st.session_state.latest_run_autoloaded = False


def _render_last_run_summary() -> None:
    if not _has_results():
        return
    rr = st.session_state.run_results
    rerun_rows: list[dict] = []
    for result in rr.message_level_results:
        debug = result.get("debug") or {}
        automatic_reruns = int(
            result.get("automatic_reruns")
            or debug.get("automatic_reruns")
            or 0
        )
        if automatic_reruns <= 0:
            continue
        history = debug.get("evaluation_attempt_history") or []
        rerun_rows.append(
            {
                "customer": result.get("conversation_id") or result.get("thread_id"),
                "message_index": result.get("message_index"),
                "automatic_reruns": automatic_reruns,
                "outcome": (
                    "recovered"
                    if result.get("recovered_after_rerun")
                    or debug.get("recovered_after_rerun")
                    else "failed"
                ),
                "initial_failure": next(
                    (
                        attempt.get("error")
                        for attempt in history
                        if isinstance(attempt, dict) and attempt.get("error")
                    ),
                    "",
                ),
            }
        )
    total_reruns = sum(int(row["automatic_reruns"]) for row in rerun_rows)
    recovered = sum(1 for row in rerun_rows if row["outcome"] == "recovered")
    st.markdown("### Last run")
    metric_row(
        [
            ("Customer journeys", f"{len(rr.conversation_results):,}", None),
            ("Message calls", f"{len(rr.message_level_results):,}", None),
            ("Automatic reruns", f"{total_reruns:,}", None),
            ("Recovered messages", f"{recovered:,}", None),
            ("Errors", f"{len(rr.errors):,}", None),
            ("Duration (s)", f"{(rr.finished_at or 0) - (rr.started_at or 0):.1f}", None),
        ]
    )

    if rerun_rows:
        with st.expander(f"View {len(rerun_rows):,} automatically rerun messages"):
            st.dataframe(pd.DataFrame(rerun_rows), use_container_width=True)

    if rr.errors:
        with st.expander(f"View {len(rr.errors)} non-fatal errors"):
            st.dataframe(pd.DataFrame(rr.errors), use_container_width=True)


def _execute_conversation_only_run(
    df: pd.DataFrame | None,
    progress_box,
    bar,
    counter_box,
    current_box,
    selected_conversation_ids: list[str] | None = None,
) -> None:
    source_results = st.session_state.run_results
    source_message_results = list(getattr(source_results, "message_level_results", []) or [])
    source_conversation_results = list(getattr(source_results, "conversation_results", []) or [])
    if not source_message_results:
        progress_box.error("Conversation-only run needs loaded message-level results first.")
        return

    st.session_state.run_in_progress = True
    st.session_state.cancel_flag = False
    st.session_state.progress_log = []

    config, _, cl_prompt_id = _build_run_config()
    if selected_conversation_ids is not None:
        config.selected_conversation_ids = list(selected_conversation_ids)
        config.max_conversations = None
    client = build_client(config.api.base_url, config.api.api_key)
    source_run_id = st.session_state.current_run_id

    db = get_active_db()
    source_run = db.get_run(source_run_id) if source_run_id else None
    run_config_serializable = {
        "api_base_url": config.api.base_url,
        "model": config.api.model,
        "message_model": config.api.model,
        "conversation_model": config.conversation_api_config().model,
        "service_tier": config.api.service_tier,
        "message_thinking_effort": config.api.thinking_effort,
        "conversation_thinking_effort": config.conversation_api_config().thinking_effort,
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
        "conversation_level_only": True,
        "message_level_source_run_id": source_run_id,
        "reused_message_level_results": len(source_message_results),
        "used_loaded_run_without_csv": bool(df is None or df.empty),
    }
    run_name = (st.session_state.run_name or "").strip() or None
    run_id = db.start_run(
        csv_name=st.session_state.csv_name or (source_run or {}).get("csv_name"),
        run_config=run_config_serializable,
        message_prompt_id=None,
        conversation_prompt_id=cl_prompt_id,
        name=run_name,
    )
    st.session_state.current_run_id = run_id
    st.session_state.loaded_run_label = None

    selected_ids_set = set(map(str, config.selected_conversation_ids or []))
    if df is not None and not df.empty:
        if selected_ids_set:
            df_for_estimate = df[df[JOURNEY_ID_COLUMN].astype(str).isin(selected_ids_set)]
            total_conv = int(df_for_estimate[JOURNEY_ID_COLUMN].astype(str).nunique())
        else:
            total_conv = (
                int(df[JOURNEY_ID_COLUMN].astype(str).nunique())
                if config.max_conversations is None
                else min(int(config.max_conversations), int(df[JOURNEY_ID_COLUMN].astype(str).nunique()))
            )
    else:
        loaded_ids = [
            str(cr.get("conversation_id") or cr.get("thread_id") or "")
            for cr in source_conversation_results
            if str(cr.get("conversation_id") or cr.get("thread_id") or "").strip()
        ]
        if selected_ids_set:
            total_conv = sum(1 for conversation_id in loaded_ids if conversation_id in selected_ids_set)
        else:
            total_conv = len(loaded_ids)
            if config.max_conversations is not None:
                total_conv = min(total_conv, int(config.max_conversations))

    total_calls = total_conv
    progress_state = {"convs_done": 0, "calls_done": 0, "successes": 0, "failures": 0}
    live_failures: list[dict] = []
    failure_box = st.empty()

    def on_progress(evt: dict) -> None:
        nonlocal total_conv, total_calls
        phase = evt.get("phase")
        if phase == "start":
            total_conv = int(evt.get("total_conversations") or total_conv or 0)
            total_calls = total_conv
        elif phase == "conversation_start":
            current_box.info(
                f"Journey {evt.get('conversation_index')}/{evt.get('total_conversations')} - "
                f"Customer `{evt.get('conversation_id')}` - "
                f"{evt.get('target_messages', 0)} reused message evals"
            )
        elif phase == "conversation_done":
            progress_state["convs_done"] += 1
            progress_state["calls_done"] += 1
            if evt.get("status") == "ok":
                progress_state["successes"] += 1
            else:
                progress_state["failures"] += 1

        frac = min(progress_state["calls_done"] / max(total_calls, 1), 1.0) if total_calls > 0 else 0.0
        bar.progress(
            frac,
            text=f"Journeys {progress_state['convs_done']}/{total_conv} | Calls {progress_state['calls_done']}/{total_calls}",
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
        _show_live_run_failure(failure_box, live_failures, err)
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
        progress_box.info("Starting conversation-level-only evaluation...")
        results = run_conversation_level_only(
            df=df,
            existing_message_level_results=source_message_results,
            existing_conversation_results=source_conversation_results,
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
        completion_message = (
            f"Conversation-only evaluation finished. {len(results.conversation_results)} customer journeys processed, "
            f"{len(results.message_level_results)} reused message results, "
            f"{len(results.errors)} errors. Saved as run #{run_id}."
        )
        if results.errors:
            progress_box.warning(completion_message)
        else:
            progress_box.success(completion_message)
        if persistence_errors:
            st.warning(
                "Some live DB saves failed during the run, but the completed results were saved again at the end. "
                f"First error: {persistence_errors[0]}"
            )
    except Exception as e:
        progress_box.error(f"Conversation-only evaluation failed: {e}")
    finally:
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


def _conversation_rerun_filter_ids(
    conversation_results: list[dict],
    *,
    outcome: str = "All",
    experience: str = "All",
    unresolved_status: str = "All",
    result_status: str = "All",
) -> list[str]:
    """Return saved journey IDs matching conversation-level rerun filters."""
    matched: list[str] = []
    for raw_result in conversation_results or []:
        result = _normalize_conversation_result_for_display(dict(raw_result))
        parsed = result.get("parsed_json") or {}
        if outcome != "All" and str(parsed.get("handled_status") or "") != outcome:
            continue
        if experience != "All" and str(parsed.get("customer_experience") or "") != experience:
            continue
        if (
            unresolved_status != "All"
            and str(parsed.get("unhandled_resolution_subtype") or "") != unresolved_status
        ):
            continue
        parse_status = str(result.get("parse_status") or "")
        if result_status == "Successful only" and parse_status != "ok":
            continue
        if result_status == "Failed only" and parse_status == "ok":
            continue
        conversation_id = str(
            result.get("conversation_id") or result.get("thread_id") or ""
        ).strip()
        if conversation_id:
            matched.append(conversation_id)
    return matched


def _render_conversation_rerun_scope() -> list[str]:
    """Render filters for reusing message results from the loaded saved run."""
    source_results = st.session_state.get("run_results")
    conversation_results = list(
        getattr(source_results, "conversation_results", []) or []
    )
    if not conversation_results:
        return []

    st.markdown("### Conversation-level rerun from loaded run")
    st.caption(
        "Reuse the saved message-level analysis and run only the conversation layer. "
        "All filters default to All, which reruns every journey in the loaded run."
    )
    filter_columns = st.columns(4)
    with filter_columns[0]:
        outcome = st.selectbox(
            "Outcome",
            ["All", "handled", "unhandled"],
            key="conversation_rerun_outcome",
            format_func=humanize_label,
        )
    with filter_columns[1]:
        experience = st.selectbox(
            "Customer experience",
            ["All", "good", "bad"],
            key="conversation_rerun_experience",
            format_func=humanize_label,
        )
    with filter_columns[2]:
        unresolved_status = st.selectbox(
            "Unresolved status",
            ["All", "pending_unresolved", "totally_unresolved"],
            key="conversation_rerun_unresolved_status",
            format_func=humanize_label,
        )
    with filter_columns[3]:
        result_status = st.selectbox(
            "Current conversation result",
            ["All", "Successful only", "Failed only"],
            key="conversation_rerun_result_status",
        )

    matched = _conversation_rerun_filter_ids(
        conversation_results,
        outcome=outcome,
        experience=experience,
        unresolved_status=unresolved_status,
        result_status=result_status,
    )
    st.caption(
        f"{len(matched):,} of {len(conversation_results):,} saved journeys match this rerun scope. "
        f"Conversation model: `{st.session_state.get('conversation_selected_model') or st.session_state.selected_model}`."
    )
    return matched


def _execute_full_batch_into_run(
    *,
    df: pd.DataFrame,
    run_id: int,
    conversation_ids: list[str],
    mode: str,
    progress_box,
    bar,
    counter_box,
    current_box,
) -> None:
    """Evaluate a selected CSV batch and merge it into an existing saved run."""
    conversation_ids = list(
        dict.fromkeys(str(value) for value in conversation_ids if str(value))
    )
    if not conversation_ids:
        progress_box.warning("No journeys are available for this operation.")
        return

    db = get_active_db()
    config, message_prompt_id, conversation_prompt_id = _build_run_config()
    config.selected_conversation_ids = conversation_ids
    config.max_conversations = None
    client = build_client(config.api.base_url, config.api.api_key)

    st.session_state.run_in_progress = True
    st.session_state.cancel_flag = False
    st.session_state.progress_log = []

    db.mark_run_running(run_id)
    db.append_run_event(
        run_id,
        {
            "type": mode,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "journey_count": len(conversation_ids),
            "message_model": config.api.model,
            "conversation_model": config.conversation_api_config().model,
            "service_tier": config.api.service_tier,
            "message_thinking_effort": config.api.thinking_effort,
            "conversation_thinking_effort": config.conversation_api_config().thinking_effort,
            "message_prompt_id": message_prompt_id,
            "conversation_prompt_id": conversation_prompt_id,
        },
    )

    estimate = estimate_call_counts(
        df[df[JOURNEY_ID_COLUMN].astype(str).isin(set(conversation_ids))],
        max_conversations=None,
        max_agent_messages_per_conv=config.max_agent_messages_per_conv,
        target_role=config.message_target_role,
    )
    total_conversations = int(estimate["conversations"])
    total_calls = int(estimate["total_calls"])
    progress_state = {
        "conversations": 0,
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "reruns": 0,
        "recovered": 0,
    }
    live_failures: list[dict] = []
    live_reruns: list[dict] = []
    failure_box = st.empty()
    rerun_box = st.empty()

    def on_progress(event: dict) -> None:
        phase = event.get("phase")
        if phase == "conversation_start":
            current_box.info(
                f"Journey {event.get('conversation_index')}/{event.get('total_conversations')} — "
                f"Customer `{event.get('conversation_id')}`"
            )
        elif phase == "message_done":
            progress_state["calls"] += 1
            key = "successes" if event.get("status") == "ok" else "failures"
            progress_state[key] += 1
            automatic_reruns = int(event.get("automatic_reruns") or 0)
            if automatic_reruns:
                progress_state["reruns"] += automatic_reruns
                if event.get("recovered_after_rerun"):
                    progress_state["recovered"] += 1
                _show_live_message_rerun(rerun_box, live_reruns, event)
        elif phase == "conversation_done":
            progress_state["conversations"] += 1
            progress_state["calls"] += 1
            key = "successes" if event.get("status") == "ok" else "failures"
            progress_state[key] += 1
        fraction = min(progress_state["calls"] / max(total_calls, 1), 1.0)
        bar.progress(
            fraction,
            text=(
                f"Journeys {progress_state['conversations']}/{total_conversations} | "
                f"Calls {progress_state['calls']}/{total_calls}"
            ),
        )
        counter_box.markdown(
            f"**Successes:** {progress_state['successes']} | "
            f"**Failures:** {progress_state['failures']} | "
            f"**Automatic reruns:** {progress_state['reruns']} | "
            f"**Recovered:** {progress_state['recovered']}"
        )
        st.session_state.progress_log.append(event)

    persistence_errors: list[str] = []

    def save_message(result: dict) -> None:
        try:
            result["run_id"] = run_id
            db.replace_message_result(run_id, result)
        except Exception as exc:
            persistence_errors.append(f"message result: {exc}")

    def save_conversation(result: dict) -> None:
        try:
            result["run_id"] = run_id
            db.replace_conversation_result(run_id, result)
        except Exception as exc:
            persistence_errors.append(f"conversation result: {exc}")

    def save_error(error: dict) -> None:
        _show_live_run_failure(failure_box, live_failures, error)
        try:
            db.save_error(run_id, error)
        except Exception as exc:
            persistence_errors.append(f"run error: {exc}")

    results = None
    try:
        progress_box.info(
            "Retrying failed journeys..." if mode == "retry_failed" else "Appending the next journey batch..."
        )
        results = run_evaluation(
            df=df,
            client=client,
            config=config,
            on_progress=on_progress,
            cancel_requested=lambda: bool(st.session_state.cancel_flag),
            on_message_result=save_message,
            on_conversation_result=save_conversation,
            on_error=save_error,
        )
        db.replace_run_errors_for_journeys(
            run_id,
            conversation_ids,
            results.errors,
        )
        status = "completed" if not results.errors else "completed_with_errors"
        counts = db.finish_run_from_saved_results(run_id, status=status)
        _load_saved_run_into_session(db, run_id)
        completion_message = (
            f"Saved into run #{run_id}. This batch processed "
            f"{len(results.conversation_results):,} journeys. The run now contains "
            f"{counts['conversations']:,} journeys and {counts['errors']:,} recorded errors."
        )
        if results.errors:
            progress_box.warning(completion_message)
        else:
            progress_box.success(completion_message)
        if persistence_errors:
            st.warning(f"Some live saves failed. First error: {persistence_errors[0]}")
    except Exception as exc:
        # Successful rows already emitted by callbacks remain in the database.
        db.finish_run_from_saved_results(run_id, status="partial")
        progress_box.error(
            f"The batch stopped early: {exc}. Completed and failed rows emitted before "
            "the interruption remain saved and can be retried."
        )
    finally:
        st.session_state.run_in_progress = False
        st.session_state.cancel_flag = False


def _conv_dataframe_from_results() -> pd.DataFrame:
    rr = st.session_state.run_results
    if not rr:
        return pd.DataFrame()
    # Rebuilding this table re-validates and flattens every conversation result,
    # which is expensive for large runs. Streamlit reruns the whole script on
    # every widget interaction, so cache the result per run object/size and only
    # recompute when the underlying results actually change (new run loaded, or
    # more conversations appended during a live run).
    cache_key = len(rr.conversation_results)
    cached = st.session_state.get("_conv_df_cache")
    if cached is not None and cached[0] is rr and cached[1] == cache_key:
        return cached[2]
    rows = []
    for idx, cr in enumerate(rr.conversation_results):
        cr = _normalize_conversation_result_for_display(cr)
        row = flatten_conversation_row(
            cr,
            cr.get("conversation_metadata", {}) or {},
            cr.get("computed_metadata", {}) or {},
        )
        row["__run_order"] = idx
        rows.append(row)
    df = _normalize_conversation_dataframe_markers(build_conversation_table(rows))
    st.session_state["_conv_df_cache"] = (rr, cache_key, df)
    return df


def _msg_dataframe_from_results() -> pd.DataFrame:
    rr = st.session_state.run_results
    if not rr:
        return pd.DataFrame()
    cache_key = len(rr.message_level_results)
    cached = st.session_state.get("_msg_df_cache")
    if cached is not None and cached[0] is rr and cached[1] == cache_key:
        return cached[2]
    rows = [flatten_message_row(m) for m in rr.message_level_results]
    df = build_message_table(rows)
    st.session_state["_msg_df_cache"] = (rr, cache_key, df)
    return df


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


def _norm_export_marker(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _filter_conversation_results_for_export(
    conversation_results: list[dict],
    *,
    handled_status: str | None = None,
    customer_experience: str | None = None,
    unhandled_resolution_subtype: str | None = None,
) -> list[dict]:
    filtered: list[dict] = []
    for cr in conversation_results:
        normalized = _normalize_conversation_result_for_display(cr)
        parsed = normalized.get("parsed_json") or normalized.get("evaluation_output") or {}
        if not isinstance(parsed, dict):
            parsed = {}
        if handled_status and _norm_export_marker(parsed.get("handled_status")) != handled_status:
            continue
        if customer_experience and _norm_export_marker(parsed.get("customer_experience")) != customer_experience:
            continue
        if (
            unhandled_resolution_subtype
            and _norm_export_marker(parsed.get("unhandled_resolution_subtype")) != unhandled_resolution_subtype
        ):
            continue
        filtered.append(normalized)
    return filtered


def _ordered_selected_ids(all_ids: list[str], selected_ids: list[str] | None) -> list[str]:
    """Return selected journey IDs in the same order they appear in the CSV."""
    if not selected_ids:
        return []
    wanted = {str(x) for x in selected_ids}
    ordered = [str(x) for x in all_ids if str(x) in wanted]
    extra = [str(x) for x in selected_ids if str(x) not in set(ordered)]
    return ordered + extra


def _customer_ids_from_saved_run(db: Database, run_id: int) -> tuple[list[str], str]:
    """Return customer journey IDs represented by a saved run.

    Prefer the run's explicit pinned selection. If the run was not pinned,
    fall back to the saved conversation result IDs so a completed run can be
    reused as the next run's scope.
    """
    run = db.get_run(int(run_id)) or {}
    run_config = run.get("run_config") or {}
    selected_ids = [
        str(x)
        for x in (run_config.get("selected_conversation_ids") or [])
        if str(x).strip()
    ]
    if selected_ids:
        return selected_ids, "pinned selection"

    result_ids = [x for x in db.list_run_conversation_ids(int(run_id)) if x]
    return result_ids, "saved run results"


def _journey_selector_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Build one searchable row per customer journey for run scoping.

    Grouping/sorting the full CSV and extracting per-journey metadata is
    expensive and was previously redone on every rerun of the Run tab (i.e.
    every widget interaction) even though the uploaded CSV doesn't change in
    between. Cache by the dataframe's identity + row count so it only
    recomputes when a new CSV is uploaded.
    """
    cache_key = len(df)
    cached = st.session_state.get("_journey_selector_cache")
    if cached is not None and cached[0] is df and cached[1] == cache_key:
        return cached[2]
    rows: list[dict[str, Any]] = []
    for journey_id, group in get_conversation_groups(df):
        md = conversation_metadata_from_group(group)
        first_message = group.iloc[0] if not group.empty else pd.Series(dtype=object)
        raw_starter = first_message.get("RAW_SENDER_ROLE")
        if pd.isna(raw_starter) or not str(raw_starter).strip():
            raw_starter = first_message.get("SENDER_ROLE", "unknown")
        journey_starter = str(raw_starter or "unknown").strip().lower()
        journey_starter = {
            "customer": "consumer",
            "assistant": "bot",
        }.get(journey_starter, journey_starter)
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
                "journey_starter": journey_starter,
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
    result = pd.DataFrame(rows)
    st.session_state["_journey_selector_cache"] = (df, cache_key, result)
    return result


def _conversation_filters_with_keys(
    conv_df: pd.DataFrame,
    key_prefix: str,
    include_journey_starter: bool = False,
) -> dict:
    return ui_components_module.conversation_filters(
        conv_df,
        key_prefix=key_prefix,
        include_journey_starter=include_journey_starter,
    )


def _apply_conversation_filters_fresh(conv_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    return ui_components_module.apply_conversation_filters(conv_df, filters)


def _render_conversation_summary_card_fresh(
    conv_result: dict,
    show_details: bool = True,
) -> None:
    ui_components_module.render_conversation_summary_card(
        conv_result,
        show_details=show_details,
    )


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
        "score_ai_judgment": "AI judgment score",
        "score_message_signals": "Message signal score",
        "score_raw_total": "Raw conversation score",
        "score_final": "Final conversation score",
        "score_final_100": "Final conversation score (100 view)",
        "score_rating": "Score rating",
        "score_explanation": "Score explanation",
        "culprits": "Culprits",
        "culprit_reason": "Culprit reasoning",
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
        "bg": "#101113" if dark else "#f7f8f6",
        "app_bg": (
            "linear-gradient(135deg, #101113 0%, #12161a 52%, #151218 100%)"
            if dark
            else "linear-gradient(135deg, #f7f8f6 0%, #edfdf9 52%, #fff7ed 100%)"
        ),
        "panel": "#17191d" if dark else "#ffffff",
        "panel_2": "#202329" if dark else "#ffffff",
        "text": "#f5f2ea" if dark else "#16181d",
        "muted": "#a6aaa3" if dark else "#66716b",
        "border": "#333840" if dark else "#dce2de",
        "accent": "#14b8a6" if dark else "#0f766e",
        "accent_2": "#f59e0b" if dark else "#b45309",
        "track": "#2a2e35" if dark else "#e5ebe7",
        "grid": "#343941" if dark else "#dce2de",
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
        px.defaults.color_continuous_scale = "Teal" if not dark else "Viridis"

    colors = {
        "bg": "#101113" if dark else "#f7f8f6",
        "app_bg": (
            "linear-gradient(135deg, #101113 0%, #12161a 52%, #151218 100%)"
            if dark
            else "linear-gradient(135deg, #f7f8f6 0%, #edfdf9 52%, #fff7ed 100%)"
        ),
        "panel": "#17191d" if dark else "#ffffff",
        "panel_2": "#202329" if dark else "#ffffff",
        "panel_3": "#252932" if dark else "#f1f5f2",
        "text": "#f5f2ea" if dark else "#16181d",
        "muted": "#a6aaa3" if dark else "#66716b",
        "border": "#333840" if dark else "#dce2de",
        "border_soft": "#2a2e35" if dark else "#edf1ee",
        "input": "#121417" if dark else "#ffffff",
        "input_text": "#f5f2ea" if dark else "#16181d",
        "accent": "#14b8a6" if dark else "#0f766e",
        "accent_soft": "#0f766e33" if dark else "#ccfbf1",
        "accent_sky": "#38bdf8" if dark else "#0284c7",
        "accent_violet": "#a78bfa" if dark else "#7c3aed",
        "accent_rose": "#fb7185" if dark else "#e11d48",
        "accent_amber": "#f59e0b" if dark else "#d97706",
        "success": "#22c55e" if dark else "#16a34a",
        "header_bg": (
            "linear-gradient(135deg, #202329 0%, #15332f 48%, #2a2035 100%)"
            if dark
            else "linear-gradient(135deg, #ffffff 0%, #e7fffb 48%, #fff4de 100%)"
        ),
        "button": "#0d9488" if dark else "#0f766e",
        "button_hover": "#14b8a6" if dark else "#115e59",
        "button_text": "#ffffff",
        "disabled": "#2a2e35" if dark else "#e5ebe7",
        "disabled_text": "#828781" if dark else "#94a39c",
        "plot_bg": "#202329" if dark else "#ffffff",
        "grid": "#343941" if dark else "#dce2de",
        "shadow": "0 18px 46px rgba(0, 0, 0, 0.28)" if dark else "0 16px 38px rgba(21, 38, 32, 0.08)",
    }
    color_scheme = "dark" if dark else "light"
    st.markdown(
        f"""
        <style>
        :root {{
          color-scheme: {color_scheme};
        }}
        html, body, [class*="css"] {{
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          letter-spacing: 0;
        }}
        .stApp {{
          background: {colors["app_bg"]};
          color: {colors["text"]};
        }}
        .main .block-container,
        [data-testid="stMainBlockContainer"] {{
          max-width: 1480px;
          padding-top: 2.15rem;
          padding-bottom: 2.4rem;
        }}
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
          background: {colors["panel"]} !important;
          color: {colors["text"]} !important;
        }}
        [data-testid="stSidebarContent"] {{
          border-right: 1px solid {colors["border_soft"]};
          padding-top: 0.75rem;
        }}
        [data-testid="stHeader"], [data-testid="stDecoration"] {{
          background: {colors["bg"]} !important;
        }}
        [data-testid="stToolbar"] {{
          right: 0.8rem;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp label, .stApp span, .stApp div {{
          color: {colors["text"]};
        }}
        .stApp h1 {{
          font-size: clamp(1.65rem, 2.4vw, 2.15rem);
          line-height: 1.08;
          font-weight: 780;
          margin: 0;
          letter-spacing: 0;
        }}
        .stApp h2 {{
          font-size: 1.35rem;
          margin-top: 0.25rem;
          margin-bottom: 0.65rem;
        }}
        .stApp h3 {{
          font-size: 1.05rem;
          font-weight: 740;
          margin-top: 0.75rem;
          margin-bottom: 0.45rem;
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
        div[data-testid="stMetric"] {{
          border: 1px solid {colors["border"]};
          border-radius: 8px;
          padding: 0.78rem 0.9rem;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label * {{
          color: {colors["muted"]} !important;
          font-size: 0.76rem !important;
          font-weight: 690 !important;
          text-transform: uppercase;
          letter-spacing: 0.02em;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
          color: {colors["text"]} !important;
          font-size: 1.65rem !important;
          font-weight: 760 !important;
          line-height: 1.15;
        }}
        details[data-testid="stExpander"] {{
          border: 1px solid {colors["border"]} !important;
          border-radius: 8px !important;
          overflow: hidden;
          box-shadow: none;
        }}
        details[data-testid="stExpander"] summary {{
          background: {colors["panel_3"]};
          border-bottom: 1px solid {colors["border_soft"]};
        }}
        .cx-table-wrap {{
          width: 100%;
          overflow: auto;
          border: 1px solid {colors["border"]};
          border-radius: 8px;
          background: {colors["panel_2"]};
          margin: 0.35rem 0 1rem;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
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
          border-radius: 8px !important;
        }}
        input, textarea {{
          -webkit-text-fill-color: {colors["input_text"]} !important;
        }}
        textarea {{
          line-height: 1.45 !important;
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
          background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%) !important;
          color: {colors["button_text"]} !important;
          border-color: transparent !important;
          border-radius: 8px !important;
          min-height: 2.35rem;
          font-weight: 720 !important;
          box-shadow: 0 8px 18px rgba(20, 184, 166, 0.16) !important;
          transition: background-color 120ms ease, border-color 120ms ease, transform 120ms ease;
        }}
        .stButton > button:hover, button[kind="primary"]:hover, button[kind="secondary"]:hover {{
          background: linear-gradient(135deg, #14b8a6 0%, #8b5cf6 100%) !important;
          border-color: transparent !important;
          transform: translateY(-1px);
        }}
        .stButton > button:disabled, button:disabled {{
          background: {colors["disabled"]} !important;
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
        div[data-testid="stFileUploader"] {{
          background: {colors["panel_2"]};
          border: 1px solid {colors["border"]};
          border-radius: 8px;
          padding: 0.75rem;
        }}
        section[data-testid="stFileUploaderDropzone"] button {{
          background-color: {colors["input"]} !important;
          color: {colors["input_text"]} !important;
          border-color: {colors["border"]} !important;
        }}
        div[data-testid="stTabs"] {{
          margin-top: 0.35rem;
        }}
        div[data-testid="stTabs"] div[role="tablist"] {{
          gap: 0.3rem;
          border-bottom: 1px solid {colors["border"]};
          padding-bottom: 0;
          overflow-x: auto;
        }}
        button[data-baseweb="tab"] {{
          background: {colors["panel"]} !important;
          border: 1px solid transparent !important;
          border-radius: 8px 8px 0 0;
          padding: 0.55rem 0.78rem;
          margin-bottom: -1px;
        }}
        button[data-baseweb="tab"] p {{
          color: #d8d4c9 !important;
          font-weight: 720;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
          color: #ffffff !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
          background: linear-gradient(135deg, #123f3c 0%, #2b2443 100%) !important;
          border: 1px solid #22d3ee !important;
          box-shadow: inset 0 3px 0 {colors["accent_amber"]}, 0 8px 18px rgba(34, 211, 238, 0.12);
          border-bottom-color: #123f3c !important;
        }}
        [data-testid="stAlert"] {{
          color: {colors["text"]} !important;
          border-radius: 8px !important;
          border: 1px solid {colors["border"]} !important;
        }}
        [data-testid="stAlert"] * {{
          color: inherit !important;
        }}
        [data-testid="stSidebar"] hr {{
          border-color: {colors["border"]} !important;
        }}
        .cx-app-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 0.8rem;
          padding: 1.35rem 1rem 1rem;
          margin: 0.65rem 0 0.75rem;
          border: 1px solid {colors["border"]};
          border-left: 4px solid {colors["accent"]};
          border-radius: 8px;
          background: {colors["header_bg"]};
          box-shadow: {colors["shadow"]};
          overflow: visible;
        }}
        .cx-title-wrap {{
          display: flex;
          flex-direction: column;
          gap: 0.36rem;
          min-width: 0;
        }}
        .cx-brand {{
          display: inline-flex;
          align-items: center;
          gap: 0.42rem;
          line-height: 1.25;
          min-height: 2.25rem;
          overflow: visible;
        }}
        .cx-brand-mark {{
          display: none;
        }}
        .cx-brand-word {{
          display: inline-block;
          color: #6fa0ee !important;
          font-size: 1.04rem;
          font-weight: 860;
          letter-spacing: 0;
          line-height: 1.25;
          padding: 0.12rem 0;
          text-shadow: 0 10px 28px rgba(95, 143, 216, 0.22);
        }}
        .cx-brand-large .cx-brand-mark {{
          display: none;
        }}
        .cx-brand-large .cx-brand-word {{
          font-size: 1.42rem;
        }}
        .cx-app-kicker {{
          color: {colors["accent_amber"]};
          font-size: 0.76rem;
          font-weight: 780;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 0.16rem;
        }}
        .cx-app-subtitle {{
          color: {colors["muted"]};
          margin-top: 0.22rem;
          font-size: 0.94rem;
        }}
        .cx-header-meta {{
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 0.45rem;
          flex-wrap: wrap;
          min-width: 280px;
        }}
        .cx-pill {{
          display: inline-flex;
          align-items: center;
          min-height: 1.9rem;
          padding: 0.28rem 0.62rem;
          border-radius: 999px;
          border: 1px solid {colors["accent"]};
          background: linear-gradient(135deg, {colors["panel_3"]} 0%, {colors["accent_soft"]} 100%);
          color: {colors["text"]};
          font-size: 0.78rem;
          font-weight: 720;
          white-space: nowrap;
        }}
        .cx-pill strong {{
          color: {colors["accent"]};
          font-weight: 800;
        }}
        .cx-auth-header {{
          max-width: 620px;
          margin: 2.2rem auto 0.85rem;
          padding: 0.95rem 1.05rem;
          border: 1px solid {colors["border"]};
          border-left: 4px solid {colors["accent_amber"]};
          border-radius: 8px;
          background: {colors["header_bg"]};
          box-shadow: {colors["shadow"]};
        }}
        .cx-auth-header h1 {{
          margin: 0;
        }}
        .cx-auth-header p {{
          margin: 0.32rem 0 0;
          color: {colors["muted"]};
        }}
        .cx-review-status {{
          border: 1px solid {colors["border"]};
          border-radius: 8px;
          padding: 0.9rem 1rem;
          margin: 0.2rem 0 0.9rem;
          background: {colors["panel_2"]};
        }}
        .cx-review-status-title {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin-bottom: 0.55rem;
          font-weight: 760;
        }}
        .cx-review-list {{
          color: {colors["muted"]};
          font-size: 0.9rem;
          line-height: 1.45;
        }}
        .cx-sidebar-mini {{
          border: 1px solid {colors["border"]};
          border-left: 4px solid {colors["accent"]};
          border-radius: 8px;
          background: {colors["panel_2"]};
          padding: 0.7rem 0.8rem;
          margin: 0.45rem 0 0.75rem;
        }}
        .cx-sidebar-brand {{
          display: flex;
          align-items: center;
          gap: 0.46rem;
          padding: 0.72rem 0.75rem;
          margin: 0 0 0.75rem;
          border: 1px solid {colors["border"]};
          border-radius: 8px;
          background: {colors["header_bg"]};
        }}
        .cx-sidebar-mini div:first-child {{
          color: {colors["muted"]};
          font-size: 0.74rem;
          font-weight: 780;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }}
        .cx-sidebar-mini div:last-child {{
          margin-top: 0.18rem;
          color: {colors["text"]};
          font-weight: 760;
          word-break: break-word;
        }}
        @media (max-width: 780px) {{
          .main .block-container,
          [data-testid="stMainBlockContainer"] {{
            padding-left: 0.9rem;
            padding-right: 0.9rem;
            padding-top: 1.8rem;
          }}
          .cx-app-header {{
            align-items: flex-start;
            flex-direction: column;
          }}
          .cx-header-meta {{
            justify-content: flex-start;
            min-width: 0;
          }}
          .cx-auth-header {{
            margin-top: 1.2rem;
          }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plotly_layout(fig, height: int | None = None, **layout):
    """Apply app theme colors to Plotly figures."""
    dark = str(st.session_state.get("theme_mode") or "Light") == "Dark"
    bg = "#101113" if dark else "#f7f8f6"
    panel = "#202329" if dark else "#ffffff"
    text = "#f5f2ea" if dark else "#16181d"
    grid = "#343941" if dark else "#dce2de"
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
    _ensure_parameter_defaults_exist()
    st.session_state.theme_mode = "Dark"
    with st.sidebar:
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
            if current not in models and DEFAULT_SELECTED_MODEL in models:
                st.session_state.selected_model = DEFAULT_SELECTED_MODEL
                current = DEFAULT_SELECTED_MODEL
            default_index = models.index(current) if current in models else 0
            st.selectbox(
                "Message-level model",
                models,
                index=default_index,
                key="selected_model",
                help=f"Default: {DEFAULT_EVALUATION_SETTINGS['selected_model']}",
            )
            conversation_current = str(
                st.session_state.get("conversation_selected_model")
                or st.session_state.selected_model
            )
            if conversation_current not in models:
                conversation_current = (
                    st.session_state.selected_model
                    if st.session_state.selected_model in models
                    else models[0]
                )
                st.session_state.conversation_selected_model = conversation_current
            st.selectbox(
                "Conversation-level model",
                models,
                index=models.index(conversation_current),
                key="conversation_selected_model",
                help="Uses the same base URL and API key as the message-level model.",
            )
        else:
            st.text_input(
                "Message-level model",
                key="selected_model",
                help=(
                    "Default: "
                    f"{DEFAULT_EVALUATION_SETTINGS['selected_model']}. "
                    "Click 'Load available models' to populate this dropdown."
                ),
            )
            st.text_input(
                "Conversation-level model",
                key="conversation_selected_model",
                help="May differ from the message-level model; it reuses the same API key.",
            )

        st.markdown("---")
        st.markdown("### Generation parameters")
        st.button(
            "Reset parameters to defaults",
            use_container_width=True,
            on_click=_reset_default_choices,
        )
        thinking_options = ["default", "disabled", "low", "medium", "high", "maximum"]
        thinking_labels = {
            "default": "Provider default",
            "disabled": "Disabled",
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "maximum": "Maximum",
        }
        st.selectbox(
            "Message-level thinking effort",
            thinking_options,
            key="message_thinking_effort",
            format_func=thinking_labels.__getitem__,
            help=(
                "Reasoning used for every individual message evaluation. "
                "Lower values usually save the most time and output-token cost."
            ),
        )
        st.selectbox(
            "Conversation-level thinking effort",
            thinking_options,
            key="conversation_thinking_effort",
            format_func=thinking_labels.__getitem__,
            help=(
                "Reasoning used for the final whole-journey analysis. "
                "OpenAI receives reasoning_effort; DeepSeek also receives its "
                "compatible thinking toggle."
            ),
        )
        st.toggle(
            "Use Flex service tier",
            key="use_flex_service_tier",
            help=(
                "When enabled, sends service_tier='flex' with message- and "
                "conversation-level requests. When disabled, the service_tier "
                "field is omitted completely. Enable only for providers and "
                "models that support Flex processing."
            ),
        )
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            key="temperature",
            help=f"Default: {DEFAULT_EVALUATION_SETTINGS['temperature']}",
        )
        st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="top_p",
            help=f"Default: {DEFAULT_EVALUATION_SETTINGS['top_p']}",
        )
        st.number_input(
            "Max tokens",
            min_value=128,
            step=64,
            key="max_tokens",
            help=f"Default: {DEFAULT_EVALUATION_SETTINGS['max_tokens']:,}",
        )
        st.number_input(
            "Timeout (seconds)",
            min_value=5.0,
            step=5.0,
            key="timeout",
            help=f"Default: {DEFAULT_EVALUATION_SETTINGS['timeout']:.0f}",
        )
        st.number_input(
            "Retry count",
            min_value=0,
            step=1,
            key="retries",
            help=(
                f"Default: {DEFAULT_EVALUATION_SETTINGS['retries']}. "
                "Also controls automatic message reruns when a model returns "
                "empty or invalid JSON."
            ),
        )
        st.session_state.concurrency = min(
            MAX_CONCURRENCY,
            max(1, int(st.session_state.concurrency)),
        )
        st.number_input(
            "Concurrency",
            min_value=1,
            max_value=MAX_CONCURRENCY,
            step=1,
            key="concurrency",
            help=(
                f"Default: {DEFAULT_EVALUATION_SETTINGS['concurrency']}. "
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
            help=(
                f"Default: {'On' if DEFAULT_RUN_SETTINGS['run_all_conversations'] else 'Off'}. "
                "When enabled, the run processes every customer journey in the uploaded CSV."
            ),
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
            help=(
                f"Default: {DEFAULT_RUN_SETTINGS['max_conversations']}. "
                "When 'Run all uploaded journeys' is off, this many journeys are processed from the CSV order."
            ),
        )
        st.number_input(
            "Max target messages per journey",
            min_value=1,
            step=1,
            key="max_agent_messages_per_conv",
            help=f"Default: {DEFAULT_RUN_SETTINGS['max_agent_messages_per_conv']}",
        )
        st.radio(
            "Evaluate which side? (default: Assistant messages)",
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
        st.toggle(
            "Truncate message text",
            key="truncate_messages",
            help=f"Default: {'On' if DEFAULT_RUN_SETTINGS['truncate_messages'] else 'Off'}",
        )
        if st.session_state.truncate_messages:
            st.number_input(
                "Max characters per message",
                min_value=200,
                step=100,
                key="max_chars_per_message",
                help=f"Default: {DEFAULT_RUN_SETTINGS['max_chars_per_message']}",
            )
        st.toggle(
            "Include unknown sender messages in history",
            key="include_unknown_in_history",
            help=f"Default: {'On' if DEFAULT_RUN_SETTINGS['include_unknown_in_history'] else 'Off'}",
        )
        st.toggle(
            "Stop on API error",
            key="stop_on_error",
            help=f"Default: {'On' if DEFAULT_RUN_SETTINGS['stop_on_error'] else 'Off'}",
        )
        st.toggle(
            "Save raw model responses",
            key="save_raw_responses",
            help=f"Default: {'On' if DEFAULT_RUN_SETTINGS['save_raw_responses'] else 'Off'}",
        )

        st.markdown("---")
        if st.session_state.current_run_id is not None:
            st.caption(f"Current run id: **#{st.session_state.current_run_id}**")


def render_auth_sidebar() -> None:
    auth_name = st.session_state.get("auth_user") or "Unknown"
    auth_role = _current_role()
    auth_role_label = humanize_label(auth_role)

    with st.sidebar:
        st.markdown(
            """
            <div class="cx-sidebar-brand">
              <span class="cx-brand-mark">m</span>
              <span class="cx-brand-word">maids.cc</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("## Access")
        st.markdown(
            f"""
            <div class="cx-sidebar-mini">
              <div>Signed in</div>
              <div>{html_lib.escape(str(auth_name))} - {html_lib.escape(str(auth_role_label))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Log out", use_container_width=True):
            _logout_current_user()


def hide_sidebar() -> None:
    """Remove the sidebar and its expand/collapse controls for reviewer sessions."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tab_reviewer_admin() -> None:
    if not _is_master():
        st.warning("Only master admins can manage reviewer access.")
        return

    db = get_active_db()
    auth_name = st.session_state.get("auth_user") or "Master admin"

    st.subheader("Database Maintenance")
    db_path = Path(_active_db_path())
    if db_path.exists():
        size_before_mb = db_path.stat().st_size / (1024 * 1024)
        st.caption(
            f"`{db_path.name}` is currently {size_before_mb:.2f} MB. "
            "Deleting runs frees space inside the file but doesn't shrink it on disk "
            "until it's compacted."
        )
        if st.button("Compact database (reclaim space)", use_container_width=True):
            with st.spinner("Compacting database..."):
                try:
                    db.vacuum()
                except Exception as e:
                    st.error(f"Could not compact the database: {e}")
                else:
                    size_after_mb = db_path.stat().st_size / (1024 * 1024)
                    st.success(
                        f"Compacted `{db_path.name}`: {size_before_mb:.2f} MB -> {size_after_mb:.2f} MB."
                    )
    else:
        st.caption("Database file not found yet.")

    st.divider()

    st.subheader("Reviewer Access")
    st.caption("Generate reviewer keys and revoke access from the workspace.")

    create_col, keys_col = st.columns([1, 1.4])
    with create_col:
        st.markdown("### Add reviewer")
        with st.form("create_reviewer_key_form"):
            reviewer_name = st.text_input("Reviewer name")
            reviewer_role = st.selectbox(
                "Initial role",
                [ROLE_READ_ONLY, ROLE_ACTIVE],
                index=0,
                format_func=humanize_label,
                help="Read-only users can review saved runs. Active users can upload, run evaluations, and export.",
            )
            created = st.form_submit_button("Generate reviewer key", type="primary")
        if created:
            try:
                st.session_state.generated_reviewer_key = db.create_reviewer_key(
                    reviewer_name,
                    created_by=auth_name,
                    role=reviewer_role,
                )
            except Exception as exc:
                st.error(str(exc))

        generated = st.session_state.get("generated_reviewer_key")
        if generated:
            st.success(
                f"Key generated for {generated['reviewer_name']}. "
                "Copy it now; it will not be shown again."
            )
            st.code(generated["reviewer_key"], language="text")
            if st.button("Hide generated key", use_container_width=True):
                st.session_state.generated_reviewer_key = None
                st.rerun()

    with keys_col:
        st.markdown("### Existing reviewers")
        keys = db.list_reviewer_keys()
        if not keys:
            st.info("No reviewer keys yet.")
            return

        for key in keys:
            status = "Active" if key["is_active"] else "Revoked"
            current_role = _normalize_role(key.get("role"))
            cols = st.columns([2.2, 1.2, 1])
            with cols[0]:
                st.markdown(f"**{key['reviewer_name']}**")
                st.caption(f"{key['key_prefix']}... - {status}")
            with cols[1]:
                selected_role = st.selectbox(
                    "Role",
                    [ROLE_READ_ONLY, ROLE_ACTIVE],
                    index=0 if current_role == ROLE_READ_ONLY else 1,
                    key=f"reviewer_role_{key['id']}",
                    format_func=humanize_label,
                    disabled=not key["is_active"],
                    label_visibility="collapsed",
                )
                if key["is_active"] and selected_role != current_role:
                    db.update_reviewer_role(int(key["id"]), selected_role)
                    st.toast(f"Updated {key['reviewer_name']} to {humanize_label(selected_role)}.")
                    st.rerun()
            with cols[2]:
                if key["is_active"] and st.button("Revoke", key=f"revoke_reviewer_{key['id']}"):
                    db.revoke_reviewer_key(int(key["id"]))
                    st.rerun()


# --------- Tab: Upload & Settings ---------


def tab_upload() -> None:
    if not _can_run_evaluations():
        st.warning("Only active users and master admins can upload customer journey CSVs.")
        return

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
    db = get_active_db()
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


def _render_saved_runs_loader(key_prefix: str, *, expanded: bool = False) -> None:
    db = get_active_db()
    with st.expander("Past runs (saved in the database)", expanded=expanded):
        if st.button("Refresh saved runs", key=f"{key_prefix}_refresh_saved_runs", use_container_width=True):
            st.rerun()
        runs = db.list_runs(limit=200)
        if not runs:
            st.caption("No saved runs yet.")
            return

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
            lambda r: _saved_run_label(r.to_dict())
            + (" • incomplete: no saved results" if r.get("is_incomplete") else ""),
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
                key=f"{key_prefix}_show_incomplete_runs",
            )
        display_runs = df_runs if show_incomplete else df_runs[~df_runs["is_incomplete"]]
        if display_runs.empty:
            message = "No loadable saved runs."
            if _can_manage_runs():
                message += " Enable incomplete runs above if you want to rename or delete them."
            st.caption(message)
            return

        sel = st.selectbox(
            "Select a saved run to load",
            display_runs["label"].tolist(),
            index=0,
            key=f"{key_prefix}_saved_run_select_{int(df_runs.iloc[0]['id'])}_{int(show_incomplete)}",
        )
        sel_id = int(display_runs.iloc[display_runs.index[display_runs["label"] == sel][0]]["id"])
        selected_run = display_runs[display_runs["id"] == sel_id].iloc[0].to_dict()

        if _can_manage_runs():
            rename_key = f"{key_prefix}_rename_run_{sel_id}"
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
            if st.button("Load this run", key=f"{key_prefix}_load_run", use_container_width=True, disabled=is_incomplete_run):
                try:
                    _load_saved_run_into_session(db, sel_id, label=sel)
                    st.success(f"Loaded run #{sel_id}.")
                except Exception as e:
                    st.error(f"Could not load run: {e}")
        if _can_manage_runs():
            with col_rename:
                if st.button("Save name", key=f"{key_prefix}_save_run_name", use_container_width=True, type="secondary"):
                    try:
                        db.rename_run(sel_id, (st.session_state.get(rename_key) or "").strip())
                        st.success(f"Renamed run #{sel_id}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not rename run: {e}")
            with col_del:
                if st.button("Delete this run", key=f"{key_prefix}_delete_run", use_container_width=True, type="secondary"):
                    try:
                        db.delete_run(sel_id)
                        if st.session_state.current_run_id == sel_id:
                            st.session_state.current_run_id = None
                            st.session_state.run_results = None
                            st.session_state._run_results_db_path = None
                        st.success(f"Deleted run #{sel_id}.")
                    except Exception as e:
                        st.error(f"Could not delete run: {e}")


# --------- Tab: Run Evaluation ---------


def tab_run() -> None:
    if not _can_run_evaluations():
        st.warning("Only active users and master admins can run evaluations.")
        return

    st.subheader("Run CX Evaluation")

    db = get_active_db()
    _render_saved_runs_loader("run", expanded=False)

    df = st.session_state.df_norm
    conversation_only_available = bool(
        st.session_state.run_results
        and getattr(st.session_state.run_results, "message_level_results", [])
    )
    conversation_rerun_ids: list[str] = []
    if conversation_only_available:
        with st.expander(
            "Conversation-level rerun from the loaded run",
            expanded=False,
        ):
            conversation_rerun_ids = _render_conversation_rerun_scope()

    if df is None or df.empty:
        if not conversation_only_available:
            st.info("Upload a valid CSV in the Upload & Settings tab first.")
            return

        st.info("No CSV is loaded. Full CX evaluation is unavailable, but you can rerun conversation-level from the loaded run.")

        if not st.session_state.get("conversation_selected_model"):
            st.warning(
                "Select a conversation-level model from the sidebar before running. "
                "Click 'Load available models' to populate the list."
            )

        st.text_input(
            "Run name",
            key="run_name",
            placeholder="e.g., June renewal journeys - conversation-only rerun",
            help="Saved with this run and shown in Past runs.",
        )

        run_col, convo_only_col, cancel_col, _ = st.columns([1, 1, 1, 3])
        with run_col:
            st.button(
                "Run CX Evaluation",
                disabled=True,
                use_container_width=True,
                help="Upload a CSV first to run the full message-level + conversation-level pipeline.",
            )
        with convo_only_col:
            convo_only_clicked = st.button(
                f"Run Conversation Analysis ({len(conversation_rerun_ids):,})",
                disabled=(
                    st.session_state.run_in_progress
                    or not st.session_state.get("conversation_selected_model")
                    or not conversation_rerun_ids
                ),
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

        if convo_only_clicked:
            _execute_conversation_only_run(
                df=None,
                progress_box=progress_box,
                bar=bar,
                counter_box=counter_box,
                current_box=current_box,
                selected_conversation_ids=conversation_rerun_ids,
            )

        _render_last_run_summary()
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

    import_feedback = st.session_state.get("selection_import_feedback")
    if import_feedback:
        st.session_state.selection_import_feedback = None
        level, message = import_feedback
        if level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.success(message)

    with st.expander("Reuse customers from previous run", expanded=False):
        previous_runs = db.list_runs(limit=200)
        if not previous_runs:
            st.caption("No saved runs are available yet.")
        elif not all_ids:
            st.caption("Upload a CSV with customer journeys before importing a previous selection.")
        else:
            previous_runs_df = _fill_saved_run_counts(db, pd.DataFrame(previous_runs))
            previous_runs_df = previous_runs_df.copy()
            previous_runs_df["saved_conversations_num"] = pd.to_numeric(
                previous_runs_df.get("saved_conversations", pd.Series(0, index=previous_runs_df.index)),
                errors="coerce",
            ).fillna(0).astype(int)

            def reuse_label(row: pd.Series) -> str:
                run_id = int(row["id"])
                name = str(row.get("name") or "Untitled run").strip()
                csv_name = str(row.get("csv_name") or "unknown CSV").strip()
                status = str(row.get("status") or "unknown").strip()
                started_at = str(row.get("started_at") or "").strip()
                saved_count = int(row.get("saved_conversations_num") or 0)
                return f"#{run_id} - {name} - {csv_name} - {saved_count:,} saved - {status} - {started_at}"

            previous_runs_df["reuse_label"] = previous_runs_df.apply(reuse_label, axis=1)
            reuse_labels = previous_runs_df["reuse_label"].tolist()
            reuse_key = "selection_import_run_label"
            if reuse_key in st.session_state and st.session_state[reuse_key] not in reuse_labels:
                st.session_state[reuse_key] = reuse_labels[0] if reuse_labels else None

            selected_reuse_label = st.selectbox(
                "Previous run",
                options=reuse_labels,
                key=reuse_key,
                help="Use the previous run's pinned customers when available; otherwise use its saved result customers.",
            )
            label_to_run_id = dict(zip(previous_runs_df["reuse_label"], previous_runs_df["id"]))
            selected_reuse_run_id = int(label_to_run_id[selected_reuse_label])

            def import_previous_run_selection(replace: bool) -> None:
                previous_ids, source = _customer_ids_from_saved_run(db, selected_reuse_run_id)
                unique_previous_ids = list(dict.fromkeys(str(x) for x in previous_ids if str(x).strip()))
                current_id_set = {str(x) for x in all_ids}
                matched_ids = [journey_id for journey_id in unique_previous_ids if journey_id in current_id_set]
                missing_count = len(unique_previous_ids) - len(matched_ids)
                matched_ids = _ordered_selected_ids(all_ids, matched_ids)

                if not matched_ids:
                    st.session_state.selection_import_feedback = (
                        "warning",
                        (
                            f"Run #{selected_reuse_run_id} had no reusable customers in the current CSV "
                            f"from its {source}."
                        ),
                    )
                    st.rerun()

                if replace:
                    next_ids = matched_ids
                    action = "Replaced the pinned selection with"
                else:
                    next_ids = _ordered_selected_ids(all_ids, selected_ids + matched_ids)
                    action = "Added"

                st.session_state.selected_conversation_ids = next_ids
                missing_note = f" {missing_count:,} customer(s) were not found in the current CSV." if missing_count else ""
                st.session_state.selection_import_feedback = (
                    "success",
                    (
                        f"{action} {len(matched_ids):,} customer journey/journeys from run "
                        f"#{selected_reuse_run_id} ({source}).{missing_note}"
                    ),
                )
                st.rerun()

            import_cols = st.columns(2)
            with import_cols[0]:
                if st.button(
                    "Replace with previous run customers",
                    use_container_width=True,
                    help="Clear the current pinned selection and use customers from the chosen previous run.",
                ):
                    import_previous_run_selection(replace=True)
            with import_cols[1]:
                if st.button(
                    "Add previous run customers",
                    use_container_width=True,
                    help="Add customers from the chosen previous run to the current pinned selection.",
                ):
                    import_previous_run_selection(replace=False)

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
                "Pick a random sample while preserving the uploaded CSV's proportions of "
                "consumer-, system-, bot-, and agent-initiated journeys. "
                "Sample size uses the sidebar journey count, or all journeys when Run all uploaded journeys is enabled."
            ),
            disabled=not all_ids,
        ):
            if st.session_state.get("run_all_conversations"):
                n = len(all_ids)
            else:
                n = max(1, int(st.session_state.max_conversations or 1))
            n = min(n, len(all_ids))
            sampled_ids = proportional_stratified_sample_ids(selector_df, n)
            st.session_state.selected_conversation_ids = _ordered_selected_ids(
                all_ids,
                sampled_ids,
            )
            source_mix = selector_df["journey_starter"].value_counts()
            sample_mix = (
                selector_df[selector_df["journey_id"].astype(str).isin(sampled_ids)]
                ["journey_starter"]
                .value_counts()
            )
            mix_summary = ", ".join(
                (
                    f"{humanize_label(starter)} "
                    f"{int(sample_mix.get(starter, 0)):,}/{n:,} "
                    f"({int(sample_mix.get(starter, 0)) / n:.1%}; source {count / len(selector_df):.1%})"
                )
                for starter, count in source_mix.items()
            )
            st.session_state.selection_import_feedback = (
                "success",
                f"Selected a proportional random sample of {n:,} journeys: {mix_summary}.",
            )
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

    append_clicked = False
    retry_failed_clicked = False
    append_ids: list[str] = []
    retry_failed_ids: list[str] = []
    continuation_run_id = int(st.session_state.get("current_run_id") or 0)
    if continuation_run_id:
        continuation_run = db.get_run(continuation_run_id)
        uploaded_csv_name = str(st.session_state.get("csv_name") or "")
        saved_csv_name = str((continuation_run or {}).get("csv_name") or "")
        csv_matches = not (
            uploaded_csv_name
            and saved_csv_name
            and uploaded_csv_name != saved_csv_name
        )
        completed_ids = set(db.list_run_completed_conversation_ids(continuation_run_id))
        remaining_ids = [journey_id for journey_id in all_ids if journey_id not in completed_ids]
        failed_ids = set(db.list_run_failed_conversation_ids(continuation_run_id))
        retry_failed_ids = [journey_id for journey_id in all_ids if journey_id in failed_ids]

        with st.expander(
            f"Continue or repair loaded run #{continuation_run_id}",
            expanded=False,
        ):
            st.caption(
                f"CSV journeys: {len(all_ids):,} | Saved conversation results: "
                f"{len(completed_ids):,} | Remaining: {len(remaining_ids):,} | "
                f"Failed journeys: {len(retry_failed_ids):,}"
            )
            if not csv_matches:
                st.error(
                    f"The loaded run used `{saved_csv_name}`, but the uploaded file is "
                    f"`{uploaded_csv_name}`. Load the matching CSV before continuing it."
                )

            if remaining_ids:
                default_batch_size = min(
                    max(1, int(st.session_state.get("max_conversations") or 50)),
                    len(remaining_ids),
                )
                batch_key = f"continuation_batch_size_{continuation_run_id}"
                existing_batch_size = int(
                    st.session_state.get(batch_key) or default_batch_size
                )
                st.session_state[batch_key] = min(
                    max(existing_batch_size, 1),
                    len(remaining_ids),
                )
                batch_size = int(
                    st.number_input(
                        "Next batch size",
                        min_value=1,
                        max_value=len(remaining_ids),
                        key=batch_key,
                        help="Takes the next unprocessed journeys in CSV order.",
                    )
                )
                append_ids = remaining_ids[:batch_size]
                append_clicked = st.button(
                    f"Append next {len(append_ids):,} journeys to run #{continuation_run_id}",
                    key=f"append_run_batch_{continuation_run_id}",
                    type="primary",
                    disabled=st.session_state.run_in_progress or not csv_matches,
                    use_container_width=True,
                )
            else:
                st.success("This run already has a conversation result for every journey in the CSV.")

            if retry_failed_ids:
                st.caption(
                    "Retrying replaces results only for failed journeys. Successful journeys "
                    "already stored in the run remain untouched."
                )
                retry_failed_clicked = st.button(
                    f"Retry {len(retry_failed_ids):,} failed journeys in the same run",
                    key=f"retry_failed_run_{continuation_run_id}",
                    disabled=st.session_state.run_in_progress or not csv_matches,
                    use_container_width=True,
                )
            else:
                st.caption("No persisted failed journeys were found in this run.")

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

    cost_config, _, _ = _build_run_config()
    reasoning_db = get_active_db()
    message_reasoning = reasoning_db.estimate_reasoning_tokens_per_call(
        "message",
        model=cost_config.api.model,
        thinking_effort=cost_config.api.thinking_effort,
    )
    conversation_reasoning = reasoning_db.estimate_reasoning_tokens_per_call(
        "conversation",
        model=cost_config.conversation_api_config().model,
        thinking_effort=cost_config.conversation_api_config().thinking_effort,
    )
    estimated_message_reasoning_tokens = (
        int(message_reasoning["average_tokens"]) * int(estimate["message_level_calls"])
    )
    estimated_conversation_reasoning_tokens = (
        int(conversation_reasoning["average_tokens"])
        * int(estimate["conversation_level_calls"])
    )
    estimated_reasoning_tokens = (
        estimated_message_reasoning_tokens + estimated_conversation_reasoning_tokens
    )
    cost_estimate = estimate_run_tokens_and_cost(
        df,
        cost_config,
        reasoning_tokens=estimated_reasoning_tokens,
    )
    st.markdown("#### Estimated tokens and reference cost")
    metric_row(
        [
            ("Input tokens", f"{cost_estimate['input_tokens']:,}", None),
            ("Visible output tokens", f"{cost_estimate['visible_output_tokens']:,}", None),
            ("Message reasoning", f"~{estimated_message_reasoning_tokens:,}", None),
            ("Conversation reasoning", f"~{estimated_conversation_reasoning_tokens:,}", None),
            ("Total reasoning", f"~{cost_estimate['reasoning_tokens']:,}", None),
            ("Total billable tokens", f"~{cost_estimate['total_tokens']:,}", None),
        ]
    )
    metric_row(
        [
            ("Estimated input cost", f"${cost_estimate['input_cost']:,.4f}", None),
            ("Estimated output cost", f"${cost_estimate['output_cost']:,.4f}", None),
            ("Estimated total cost", f"${cost_estimate['total_cost']:,.4f}", None),
        ]
    )
    st.caption(
        "Reasoning estimate: "
        f"message calls average {int(message_reasoning['average_tokens']):,} tokens "
        f"from {int(message_reasoning['samples']):,} saved samples "
        f"({message_reasoning['basis']}); conversation calls average "
        f"{int(conversation_reasoning['average_tokens']):,} tokens from "
        f"{int(conversation_reasoning['samples']):,} saved samples "
        f"({conversation_reasoning['basis']})."
    )
    st.caption(
        "Reference pricing: $0.75 per 1M input tokens and $4.50 per 1M output tokens. "
        "Token counts use the loaded journeys, active prompts, output schemas, and recent saved reasoning usage. "
        "Your API proxy or selected model may charge different rates."
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

    run_col, convo_only_col, cancel_col, _ = st.columns([1, 1, 1, 3])
    with run_col:
        run_clicked = st.button(
            "Run CX Evaluation",
            type="primary",
            disabled=(
                st.session_state.run_in_progress
                or not st.session_state.selected_model
                or not st.session_state.get("conversation_selected_model")
            ),
            use_container_width=True,
        )
    with convo_only_col:
        convo_only_clicked = st.button(
            f"Run Conversation Analysis ({len(conversation_rerun_ids):,})",
            disabled=(
                st.session_state.run_in_progress
                or not st.session_state.get("conversation_selected_model")
                or not conversation_only_available
                or not conversation_rerun_ids
            ),
            use_container_width=True,
        )
    with cancel_col:
        if st.session_state.run_in_progress:
            if st.button("Cancel run", use_container_width=True):
                st.session_state.cancel_flag = True
                st.toast("Cancelling after current call finishes...")
    if not conversation_only_available:
        st.caption("Load or run message-level results first to enable conversation-only reruns.")

    progress_box = st.empty()
    bar = st.progress(0, text="Idle")
    counter_box = st.empty()
    current_box = st.empty()
    log_box = st.empty()
    rerun_box = st.empty()

    if run_clicked:
        st.session_state.run_in_progress = True
        st.session_state.cancel_flag = False
        st.session_state.progress_log = []

        config, ml_prompt_id, cl_prompt_id = _build_run_config()
        client = build_client(config.api.base_url, config.api.api_key)

        # Start a DB run record.
        run_config_serializable = {
            "api_base_url": config.api.base_url,
            "model": config.api.model,
            "message_model": config.api.model,
            "conversation_model": config.conversation_api_config().model,
            "service_tier": config.api.service_tier,
            "message_thinking_effort": config.api.thinking_effort,
            "conversation_thinking_effort": config.conversation_api_config().thinking_effort,
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
        st.session_state._run_results_db_path = _active_db_path()

        total_conv = estimate["conversations"]
        total_msg = estimate["message_level_calls"] + estimate["conversation_level_calls"]
        progress_state = {
            "convs_done": 0,
            "calls_done": 0,
            "successes": 0,
            "failures": 0,
            "reruns": 0,
            "recovered": 0,
        }
        live_failures: list[dict] = []
        live_reruns: list[dict] = []

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
                automatic_reruns = int(evt.get("automatic_reruns") or 0)
                if automatic_reruns:
                    progress_state["reruns"] += automatic_reruns
                    if evt.get("recovered_after_rerun"):
                        progress_state["recovered"] += 1
                    _show_live_message_rerun(rerun_box, live_reruns, evt)
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
                f"**Successes:** {progress_state['successes']} | "
                f"**Failures:** {progress_state['failures']} | "
                f"**Automatic reruns:** {progress_state['reruns']} | "
                f"**Recovered:** {progress_state['recovered']}"
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
            _show_live_run_failure(log_box, live_failures, err)
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
            st.session_state._run_results_db_path = _active_db_path()
            persist_completed_results()
            completion_message = (
                f"Evaluation finished. {len(results.conversation_results)} customer journeys processed, "
                f"{len(results.message_level_results)} message-level calls, "
                f"{len(results.errors)} errors. Saved as run #{run_id}."
            )
            if results.errors:
                progress_box.warning(completion_message)
            else:
                progress_box.success(completion_message)
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

    if convo_only_clicked:
        _execute_conversation_only_run(
            df=df,
            progress_box=progress_box,
            bar=bar,
            counter_box=counter_box,
            current_box=current_box,
            selected_conversation_ids=conversation_rerun_ids,
        )
    elif append_clicked:
        _execute_full_batch_into_run(
            df=df,
            run_id=continuation_run_id,
            conversation_ids=append_ids,
            mode="append_batch",
            progress_box=progress_box,
            bar=bar,
            counter_box=counter_box,
            current_box=current_box,
        )
    elif retry_failed_clicked:
        _execute_full_batch_into_run(
            df=df,
            run_id=continuation_run_id,
            conversation_ids=retry_failed_ids,
            mode="retry_failed",
            progress_box=progress_box,
            bar=bar,
            counter_box=counter_box,
            current_box=current_box,
        )

    _render_last_run_summary()


# --------- Tab: Dashboard ---------

# Color palette used across the dashboard (tuned for the dark theme).
_DASH_COLORS = {
    "panel_bg": "#202329",
    "panel_top": "#17191d",
    "panel_border": "#333840",
    "text": "#f5f2ea",
    "muted": "#a6aaa3",
    "dim": "#777d76",
    "track": "#2a2e35",
    "handled": "#10b981",
    "unhandled": "#ef4444",
    "many": "#f97316",
    "minimal": "#22c55e",
    "frustrated": "#f59e0b",
    "calm": "#14b8a6",
    "our_side": "#fb923c",
    "customer": "#2dd4bf",
    "shared": "#c084fc",
    "none": "#777d76",
    "unclear": "#525861",
    "review_yes": "#a78bfa",
    "review_no": "#333840",
    "heat_low": "#252932",
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


def _stats_summary_rows(conv_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build the management stats table rows from normalized conversation markers."""
    total = int(len(conv_df))
    handled = _norm_marker_series(conv_df, "handled_status")
    experience = _norm_marker_series(conv_df, "customer_experience")
    experience = experience.replace({"many": "bad", "zero_minimal": "good", "minimal": "good"})
    subtype = _norm_marker_series(conv_df, "unhandled_resolution_subtype", "not_applicable")

    handled_mask = handled == "handled"
    unhandled_mask = handled == "unhandled"
    good_mask = experience == "good"
    bad_mask = experience == "bad"
    pending_mask = subtype == "pending_unresolved"
    totally_mask = subtype == "totally_unresolved"

    pending_good = int((unhandled_mask & pending_mask & good_mask).sum())
    totally_good = int((unhandled_mask & totally_mask & good_mask).sum())

    def row(metric: str, count: int, kind: str, depth: int = 0) -> dict[str, Any]:
        return {
            "Metric": metric,
            "Count": int(count),
            "Percentage": f"{_pct(count, total):.1f}%",
            "_kind": kind,
            "_depth": depth,
        }

    rows = [
        row("Total Journeys", total, "total"),
        row("Handled", int(handled_mask.sum()), "section"),
        row("Handled / Good", int((handled_mask & good_mask).sum()), "child", 1),
        row("Handled / Bad", int((handled_mask & bad_mask).sum()), "child", 1),
        row("Not Handled", int(unhandled_mask.sum()), "section"),
        row("Pending Unresolved / Bad", int((unhandled_mask & pending_mask & bad_mask).sum()), "child", 1),
        row("Totally Unresolved / Bad", int((unhandled_mask & totally_mask & bad_mask).sum()), "child", 1),
    ]

    rows.extend(
        [
            row("Pending Unresolved / Good", pending_good, "child", 1),
            row("Totally Unresolved / Good", totally_good, "child", 1),
        ]
    )

    captured_unhandled_bad = int((unhandled_mask & (pending_mask | totally_mask) & bad_mask).sum())
    other_unhandled_bad = int((unhandled_mask & bad_mask).sum()) - captured_unhandled_bad
    if other_unhandled_bad > 0:
        rows.append(row("Other Unhandled / Bad", other_unhandled_bad, "child", 1))

    captured_unhandled_good = pending_good + totally_good
    other_unhandled_good = int((unhandled_mask & good_mask).sum()) - captured_unhandled_good
    if other_unhandled_good > 0:
        rows.append(row("Other Unhandled / Good", other_unhandled_good, "child", 1))

    rows.extend(
        [
            row("Overall Customer Experience - Good", int(good_mask.sum()), "section"),
            row("Overall Customer Experience - Bad", int(bad_mask.sum()), "section"),
        ]
    )
    return rows


def _render_stats_summary_table(rows: list[dict[str, Any]]) -> None:
    """Render the Stats tab table with the app dashboard palette."""
    header_bg = "#24272d"
    total_bg = "#202329"
    section_bg = "#1b2024"
    child_bg = "#17191d"
    border = _DASH_COLORS["panel_border"]
    text = _DASH_COLORS["text"]
    muted = _DASH_COLORS["muted"]

    def row_colors(metric: str, kind: str) -> tuple[str, str]:
        marker = metric.lower()
        if kind == "total":
            return total_bg, _DASH_COLORS["calm"]
        if "not handled" in marker or "unhandled" in marker or "totally" in marker:
            return "#24191b", _DASH_COLORS["unhandled"]
        if "bad" in marker:
            return "#261d18", _DASH_COLORS["many"]
        if "good" in marker:
            return "#17231b", _DASH_COLORS["minimal"]
        if "handled" in marker:
            return "#16221d", _DASH_COLORS["handled"]
        return section_bg if kind == "section" else child_bg, _DASH_COLORS["dim"]

    body = []
    for row in rows:
        kind = row["_kind"]
        bg, accent = row_colors(str(row["Metric"]), kind)
        weight = "600" if kind in {"total", "section"} else "400"
        metric = html_lib.escape(str(row["Metric"]))
        metric_text = str(row["Metric"])
        align = "left" if row["_depth"] == 0 or len(metric_text) > 42 else "center"
        value_color = text if kind in {"total", "section"} else muted
        metric_pad = 10 + int(row["_depth"]) * 18
        body.append(
            "<tr>"
            f'<td style="background:{bg};border:1px solid {border};border-left:4px solid {accent};padding:8px 10px 8px {metric_pad}px;'
            f'text-align:{align};font-weight:{weight};color:{text};">{metric}</td>'
            f'<td style="background:{bg};border:1px solid {border};padding:8px 10px;'
            f'text-align:left;font-weight:{weight};color:{value_color};">{int(row["Count"]):,}</td>'
            f'<td style="background:{bg};border:1px solid {border};padding:8px 10px;'
            f'text-align:left;font-weight:{weight};color:{accent if kind in {"total", "section"} else value_color};">{html_lib.escape(str(row["Percentage"]))}</td>'
            "</tr>"
        )

    st.markdown(
        f"""
        <style>
        .stats-summary-table {{
            width: min(820px, 100%);
            border-collapse: separate;
            border-spacing: 0;
            color: {text};
            font-size: 1rem;
            line-height: 1.25;
            border: 1px solid {border};
            border-radius: 10px;
            overflow: hidden;
            background: {total_bg};
        }}
        .stats-summary-table th {{
            background: {header_bg};
            border: 1px solid {border};
            padding: 10px;
            text-align: left;
            font-weight: 800;
            color: {text};
        }}
        .stats-summary-table td:nth-child(1) {{
            width: 65%;
        }}
        .stats-summary-table td:nth-child(2),
        .stats-summary-table td:nth-child(3) {{
            width: 17.5%;
            white-space: nowrap;
        }}
        </style>
        <table class="stats-summary-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {''.join(body)}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def _message_flag_stats(conversation_results: list[dict]) -> pd.DataFrame:
    """Count evaluated message flag colors by sender origin."""
    sender_types = ("Agent", "Bot", "Broadcast")
    flag_levels = ("Red", "Yellow", "Green")
    counts = {
        sender_type: {flag_level: 0 for flag_level in flag_levels}
        for sender_type in sender_types
    }

    for conversation_result in conversation_results:
        transcript = conversation_result.get("transcript") or []
        eval_by_idx = _message_results_by_index(
            conversation_result.get("message_level_results")
        )
        for message in transcript:
            message_index = message.get("message_index")
            if message_index is None:
                continue
            sender_type = _flagged_sender_type(message)
            flag_level = _message_flag_level(eval_by_idx.get(str(message_index)))
            if sender_type in counts and flag_level in flag_levels:
                counts[sender_type][flag_level] += 1

    rows = []
    for sender_type in sender_types:
        red = counts[sender_type]["Red"]
        yellow = counts[sender_type]["Yellow"]
        green = counts[sender_type]["Green"]
        flagged = red + yellow
        total_evaluated = flagged + green
        rows.append(
            {
                "Sender origin": sender_type,
                "Red": red,
                "Yellow": yellow,
                "Green": green,
                "Flagged (red + yellow)": flagged,
                "Total evaluated": total_evaluated,
            }
        )

    all_flagged = sum(row["Flagged (red + yellow)"] for row in rows)
    all_evaluated = sum(row["Total evaluated"] for row in rows)
    for row in rows:
        sender_evaluated = int(row["Total evaluated"])
        sender_flagged = int(row["Flagged (red + yellow)"])
        row["Flagged rate"] = (
            f"{_pct(sender_flagged, sender_evaluated):.1f}%"
        )
        row["Share of all flagged"] = (
            f"{_pct(sender_flagged, all_flagged):.1f}%"
        )

    total_row = {
        "Sender origin": "Total",
        "Red": sum(row["Red"] for row in rows),
        "Yellow": sum(row["Yellow"] for row in rows),
        "Green": sum(row["Green"] for row in rows),
        "Flagged (red + yellow)": all_flagged,
        "Total evaluated": all_evaluated,
        "Flagged rate": f"{_pct(all_flagged, all_evaluated):.1f}%",
        "Share of all flagged": f"{_pct(all_flagged, all_flagged):.1f}%",
    }
    return pd.DataFrame([*rows, total_row])


def _render_message_flag_stats_table(flag_stats_df: pd.DataFrame) -> None:
    """Render message flag stats with the same visual system as the summary table."""
    header_bg = "#24272d"
    total_bg = "#202329"
    row_bg = "#17191d"
    border = _DASH_COLORS["panel_border"]
    text = _DASH_COLORS["text"]
    muted = _DASH_COLORS["muted"]
    accent_by_sender = {
        "Agent": _DASH_COLORS["handled"],
        "Bot": _DASH_COLORS["calm"],
        "Broadcast": _DASH_COLORS["many"],
        "Total": _DASH_COLORS["calm"],
    }
    value_colors = {
        "Red": _DASH_COLORS["unhandled"],
        "Yellow": _DASH_COLORS["many"],
        "Green": _DASH_COLORS["minimal"],
    }

    body = []
    for row in flag_stats_df.to_dict(orient="records"):
        sender = str(row["Sender origin"])
        is_total = sender == "Total"
        bg = total_bg if is_total else row_bg
        weight = "700" if is_total else "500"
        accent = accent_by_sender.get(sender, _DASH_COLORS["dim"])

        cells = [
            (
                sender,
                text,
                f"border-left:4px solid {accent};",
            )
        ]
        for column in flag_stats_df.columns[1:]:
            value = row[column]
            if column in {"Red", "Yellow", "Green"}:
                color = value_colors[column]
            elif column in {"Flagged rate", "Share of all flagged"}:
                color = accent
            else:
                color = muted if not is_total else text
            display_value = f"{int(value):,}" if isinstance(value, (int, float)) else str(value)
            cells.append((display_value, color, ""))

        body.append(
            "<tr>"
            + "".join(
                f'<td style="background:{bg};border:1px solid {border};{extra}'
                f'padding:8px 10px;text-align:left;font-weight:{weight};'
                f'color:{color};white-space:nowrap;">{html_lib.escape(value)}</td>'
                for value, color, extra in cells
            )
            + "</tr>"
        )

    headers = "".join(
        f"<th>{html_lib.escape(str(column))}</th>"
        for column in flag_stats_df.columns
    )
    st.markdown(
        f"""
        <style>
        .message-flag-stats-table {{
            width: min(1180px, 100%);
            border-collapse: separate;
            border-spacing: 0;
            color: {text};
            font-size: 1rem;
            line-height: 1.25;
            border: 1px solid {border};
            border-radius: 10px;
            overflow: hidden;
            background: {total_bg};
        }}
        .message-flag-stats-table th {{
            background: {header_bg};
            border: 1px solid {border};
            padding: 10px;
            text-align: left;
            font-weight: 800;
            color: {text};
            white-space: nowrap;
        }}
        .message-flag-stats-table td:first-child {{
            min-width: 145px;
        }}
        </style>
        <table class="message-flag-stats-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{''.join(body)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def tab_stats() -> None:
    st.subheader("Stats")
    st.caption(
        "A simple count of journey outcomes, customer experience, and message quality "
        "for the currently loaded review."
    )
    with st.expander("How to read the Stats page", expanded=False):
        st.markdown(textwrap.dedent(
            """
            - **Count** is the number of customer journeys in that result.
            - **Percentage** shows how much of the full review that result represents.
            - **Handled** means the customer's request was completed. **Pending unresolved**
              means more work or follow-up was still needed. **Totally unresolved** means the
              request was not solved.
            - The message table separates replies from a human **Agent**, an automated **Bot**,
              and a system **Broadcast**.
            - **Red** messages need attention, **Yellow** messages may need improvement, and
              **Green** messages were assessed as acceptable.
            - **Flagged rate** is the share of that sender's evaluated messages that were red or
              yellow. **Share of all flagged** shows how much that sender contributed to all red
              and yellow messages.

            Use these totals to spot patterns, then open **Journey Review** to read the actual
            conversation and check its journey analysis before making a decision.
            """
        ))
    conv_df = _conv_dataframe_from_results()
    if conv_df.empty:
        st.info("No journeys to summarize.")
        return

    rows = _stats_summary_rows(conv_df)
    _render_stats_summary_table(rows)

    flag_stats_df = _message_flag_stats(
        st.session_state.run_results.conversation_results
    )
    st.markdown("### Message flags by sender origin")
    st.caption(
        "Evaluated message counts using the same red, yellow, and green rules as Journey Review."
    )
    _render_message_flag_stats_table(flag_stats_df)

    stats_df = pd.DataFrame(
        [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in rows
        ]
    )
    st.download_button(
        "Download stats CSV",
        data=stats_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cx_stats_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download message flag stats CSV",
        data=flag_stats_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cx_message_flag_stats.csv",
        mime="text/csv",
    )


def tab_overview() -> None:
    st.subheader("Overview")
    st.caption(
        "A quick picture of how customer journeys ended, where frustration came from, "
        "and which problems appeared most often."
    )
    with st.expander("How to read the Overview page", expanded=False):
        st.markdown(textwrap.dedent(
            """
            Start with the three summary cards:

            - **Handled** means the customer's request was completed.
            - **Pending unresolved** means the request still needed follow-up or another action.
            - **Totally unresolved** means the request was not solved.

            Each card also shows whether the overall customer experience was good or bad.
            The **Journey marker breakdown** explains where frustration came from inside each
            outcome group. The **Detected issues** section shows the most common problem types
            and whether they came from our side, the customer side, or another source.

            This page is for finding patterns. To confirm what happened in a specific case,
            open **Journey Review**, turn on **Show journey analysis and review metrics**, and
            read the full conversation.
            """
        ))
    _render_saved_runs_loader("overview", expanded=not _has_results())
    if not _has_results():
        st.info("Load a saved run above to start reviewing.")
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


def _ensure_journey_review_comment_column(db: Database) -> None:
    try:
        with db._lock:
            columns = {
                row["name"]
                for row in db._conn.execute("PRAGMA table_info(journey_reviews)").fetchall()
            }
            if "review_comment" not in columns:
                db._conn.execute("ALTER TABLE journey_reviews ADD COLUMN review_comment TEXT")
    except Exception:
        pass


def _list_journey_reviews_with_comments(db: Database, run_id: int, conversation_id: str) -> list[dict]:
    _ensure_journey_review_comment_column(db)
    try:
        rows = db._fetchall(
            "SELECT id, reviewer_name, reviewed_at, review_comment FROM journey_review_history "
            "WHERE run_id=? AND conversation_id=? ORDER BY reviewed_at ASC, id ASC",
            (int(run_id), str(conversation_id)),
        )
        reviews = [dict(row) for row in rows]
        if reviews:
            return reviews
    except Exception:
        pass

    try:
        reviews = db.list_journey_reviews(int(run_id), conversation_id)
        for row in reviews:
            row.setdefault("review_comment", None)
        return reviews
    except Exception:
        rows = db._fetchall(
            "SELECT id, reviewer_name, reviewed_at, review_comment FROM journey_reviews "
            "WHERE run_id=? AND conversation_id=? ORDER BY reviewed_at ASC",
            (int(run_id), str(conversation_id)),
        )
        return [dict(row) for row in rows]


def _record_journey_review_with_comment(
    db: Database,
    run_id: int,
    conversation_id: str,
    reviewer_name: str,
    reviewer_key_id: int | None,
    review_comment: str | None,
) -> None:
    _ensure_journey_review_comment_column(db)
    try:
        db.record_journey_review(
            int(run_id),
            conversation_id,
            reviewer_name,
            reviewer_key_id,
            review_comment,
        )
        return
    except TypeError:
        pass

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with db._lock:
        db._conn.execute(
            "CREATE TABLE IF NOT EXISTS journey_review_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "run_id INTEGER NOT NULL, "
            "conversation_id TEXT NOT NULL, "
            "reviewer_key_id INTEGER, "
            "reviewer_name TEXT NOT NULL, "
            "reviewed_at TEXT NOT NULL, "
            "review_comment TEXT)"
        )
        db._conn.execute(
            "INSERT INTO journey_review_history"
            "(run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                str(conversation_id),
                int(reviewer_key_id) if reviewer_key_id is not None else None,
                str(reviewer_name),
                now,
                str(review_comment or "").strip() or None,
            ),
        )
        db._conn.execute(
            "INSERT INTO journey_reviews(run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, conversation_id, reviewer_name) DO UPDATE SET "
            "reviewer_key_id=excluded.reviewer_key_id, reviewed_at=excluded.reviewed_at, "
            "review_comment=excluded.review_comment",
            (
                int(run_id),
                str(conversation_id),
                int(reviewer_key_id) if reviewer_key_id is not None else None,
                str(reviewer_name),
                now,
                str(review_comment or "").strip() or None,
            ),
        )


def _render_journey_review_tracking(run_id: int | None, conversation_id: str) -> None:
    if not run_id:
        st.caption("Review tracking is available after this journey is attached to a saved run.")
        return

    db = get_active_db()
    reviews = _list_journey_reviews_with_comments(db, int(run_id), conversation_id)
    reviewer_name = st.session_state.get("auth_user") or "Unknown"
    reviewer_key_id = st.session_state.get("auth_reviewer_key_id")

    def format_review_time(value: Any) -> str:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return str(value or "")
        return parsed.strftime("%Y-%m-%d %H:%M:%S.%f UTC")

    if reviews:
        reviewer_history: dict[str, list[dict]] = {}
        for row in reviews:
            reviewer_history.setdefault(str(row.get("reviewer_name") or "Unknown"), []).append(row)
        reviewer_count = len(reviewer_history)
    else:
        reviewer_history = {}
        reviewer_count = 0

    st.markdown(
        f"""
        <div class="cx-review-status">
          <div class="cx-review-status-title">
            <span>Review status</span>
            <span class="cx-pill">{reviewer_count:,} reviewer{"s" if reviewer_count != 1 else ""}</span>
            <span class="cx-pill">{len(reviews):,} review{"s" if len(reviews) != 1 else ""}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if reviewer_history:
        for reviewer, history in reviewer_history.items():
            latest = history[-1]
            latest_time = format_review_time(latest.get("reviewed_at"))
            with st.expander(f"{reviewer} - {len(history):,} review{'s' if len(history) != 1 else ''} - latest {latest_time}"):
                newest_first = list(reversed(history))
                for history_index, row in enumerate(newest_first, start=1):
                    comment = str(row.get("review_comment") or "").strip()
                    st.markdown(f"**History {history_index}**")
                    st.caption(format_review_time(row.get("reviewed_at")))
                    st.write(comment or "No comment added.")
                    if history_index < len(newest_first):
                        st.markdown("---")
    else:
        st.caption("No reviewer has marked this journey as reviewed yet.")

    current_review = next(
        (row for row in reversed(reviews) if str(row.get("reviewer_name") or "") == str(reviewer_name)),
        None,
    )
    with st.form(f"journey_review_comment_{run_id}_{conversation_id}"):
        comment = st.text_area(
            "Your review comment",
            value=str((current_review or {}).get("review_comment") or ""),
            key=f"journey_review_comment_text_{run_id}_{conversation_id}_{reviewer_name}",
            height=90,
            placeholder="Add what you checked, agreed with, or want another reviewer to revisit.",
        )
        saved = st.form_submit_button("Mark as reviewed", type="primary")
    if saved:
        _record_journey_review_with_comment(
            db,
            int(run_id),
            conversation_id,
            reviewer_name,
            int(reviewer_key_id) if reviewer_key_id is not None else None,
            comment,
        )
        st.rerun()


def _message_flag_level(message_result: dict | None) -> str | None:
    """Return the transcript flag color for an evaluated message."""
    if not message_result:
        return None
    if message_result.get("parse_status") != "ok":
        return "Red"
    parsed = message_result.get("parsed_json") or {}
    effect = parsed.get("message_level_effect")
    frustration = parsed.get("frustration_level_after_message")
    change = parsed.get("frustration_change")
    issue_type = parsed.get("issue_type") or "none"
    issue_origin = parsed.get("issue_origin") or "none"
    has_issue = effect in {"minor_issue", "major_issue"} or issue_type != "none" or issue_origin != "none"
    if effect == "major_issue" or frustration in {"high", "cancellation_risk"}:
        return "Red"
    if change == "created" and has_issue:
        return "Red"
    if effect == "minor_issue" or (has_issue and frustration in {"medium", "low"}) or change == "increased":
        return "Yellow"
    return "Green"


def _flagged_message_level(message_result: dict | None) -> str | None:
    """Return only issue flags for the flagged-message detail table."""
    level = _message_flag_level(message_result)
    return level if level in {"Red", "Yellow"} else None


def _flagged_sender_type(message: dict) -> str | None:
    raw_role = str(message.get("raw_sender_role") or "").strip().lower()
    sender_role = str(message.get("sender_role") or "").strip().lower()
    if raw_role == "system":
        return "Broadcast"
    if raw_role == "bot":
        return "Bot"
    if raw_role == "agent" or sender_role == "agent":
        return "Agent"
    return None


def _message_results_by_index(message_results: list[dict] | None) -> dict[str, dict]:
    """Index message evaluations while tolerating numeric/string index differences."""
    return {
        str(result.get("message_index")): result
        for result in (message_results or [])
        if result.get("message_index") is not None
    }


def _sender_flag_levels_for_journey(conversation_result: dict) -> dict[str, set[str]]:
    """Collect the message flag colors present for each supported sender type."""
    levels = {kind: set() for kind in ("Agent", "Bot", "Broadcast")}
    transcript = conversation_result.get("transcript") or []
    eval_by_idx = _message_results_by_index(conversation_result.get("message_level_results"))
    for message in transcript:
        message_index = message.get("message_index")
        if message_index is None:
            continue
        result = eval_by_idx.get(str(message_index))
        sender_type = _flagged_sender_type(message)
        flag_level = _message_flag_level(result)
        if sender_type and flag_level:
            levels[sender_type].add(flag_level)
    return levels


def _filter_conversations_by_sender_flags(
    conv_df: pd.DataFrame,
    conversation_results: list[dict],
    filters: dict[str, str | None],
) -> pd.DataFrame:
    """Keep journeys containing each requested sender/severity combination."""
    active_filters = {
        sender_type: flag_level
        for sender_type, flag_level in filters.items()
        if flag_level
    }
    if not active_filters or conv_df.empty or "conversation_id" not in conv_df.columns:
        return conv_df

    matching_ids: set[str] = set()
    for conversation_result in conversation_results:
        levels = _sender_flag_levels_for_journey(conversation_result)
        if all(
            flag_level in levels.get(sender_type, set())
            for sender_type, flag_level in active_filters.items()
        ):
            matching_ids.add(str(conversation_result.get("conversation_id") or ""))

    return conv_df[conv_df["conversation_id"].astype(str).isin(matching_ids)]


def _render_sender_flag_filters() -> dict[str, str | None]:
    """Render independent Agent/Bot/Broadcast message flag filters."""
    st.markdown("**Message flag filters**")
    st.caption(
        "Show journeys containing the selected flag for each sender type. "
        "When several filters are selected, a journey must match all of them."
    )
    options = ["All (no filter)", "Red", "Yellow", "Green"]
    filters: dict[str, str | None] = {}
    columns = st.columns(3)
    for column, sender_type in zip(columns, ("Agent", "Bot", "Broadcast")):
        with column:
            selected = st.selectbox(
                f"{sender_type} messages",
                options,
                index=0,
                key=f"review_{sender_type.lower()}_flag_filter",
            )
        filters[sender_type] = None if selected == options[0] else selected
    return filters


def _flagged_messages_by_sender(transcript: list[dict], message_results: list[dict]) -> list[dict[str, Any]]:
    eval_by_idx = _message_results_by_index(message_results)
    rows: list[dict[str, Any]] = []
    for message in transcript or []:
        msg_index = message.get("message_index")
        result = eval_by_idx.get(str(msg_index))
        flag_level = _flagged_message_level(result)
        sender_type = _flagged_sender_type(message)
        if not flag_level or not sender_type:
            continue
        parsed = result.get("parsed_json") or {}
        rows.append(
            {
                "Type": sender_type,
                "Flag": flag_level,
                "Message #": msg_index,
                "Source conversation": message.get("source_conversation_id") or result.get("source_conversation_id") or "",
                "Effect": humanize_label(parsed.get("message_level_effect")),
                "Issue type": humanize_label(parsed.get("issue_type")),
                "Frustration": humanize_label(parsed.get("frustration_level_after_message")),
                "Message": message.get("message_text") or result.get("target_message_text") or "",
            }
        )
    return rows


def _render_flagged_message_checker(transcript: list[dict], message_results: list[dict], conversation_id: str) -> None:
    flagged_rows = _flagged_messages_by_sender(transcript, message_results)
    counts = {kind: sum(1 for row in flagged_rows if row["Type"] == kind) for kind in ("Agent", "Bot", "Broadcast")}
    total = len(flagged_rows)

    st.markdown("### Flagged message check")
    st.caption("Check whether flagged evaluated messages came from an agent, bot, or broadcast/system message.")

    state_key = f"review_flagged_sender_{conversation_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = "All"
    button_cols = st.columns(4)
    choices = [
        ("All", total),
        ("Agent", counts["Agent"]),
        ("Bot", counts["Bot"]),
        ("Broadcast", counts["Broadcast"]),
    ]
    for col, (choice, count) in zip(button_cols, choices):
        with col:
            button_type = "primary" if st.session_state[state_key] == choice else "secondary"
            if st.button(f"{choice} ({count})", key=f"review_flagged_{choice}_{conversation_id}", type=button_type, use_container_width=True):
                st.session_state[state_key] = choice

    selected = st.session_state[state_key]
    shown = flagged_rows if selected == "All" else [row for row in flagged_rows if row["Type"] == selected]
    severity_counts = {
        "Red": sum(1 for row in shown if row.get("Flag") == "Red"),
        "Yellow": sum(1 for row in shown if row.get("Flag") == "Yellow"),
    }
    if not flagged_rows:
        st.success("No evaluated agent, bot, or broadcast messages were flagged for this journey.")
        return
    if not shown:
        st.info(f"No {selected.lower()} messages were flagged for this journey.")
        return

    metric_row(
        [
            ("Flagged red", f"{severity_counts['Red']:,}", None),
            ("Flagged yellow", f"{severity_counts['Yellow']:,}", None),
        ]
    )
    st.dataframe(pd.DataFrame(shown), use_container_width=True, hide_index=True, height=min(360, 88 + len(shown) * 54))


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
    st.info(
        "For a complete review, turn on **Show journey analysis and review metrics**, "
        "check the flagged messages, read the full conversation, and add your review comment "
        "before marking the journey as reviewed."
    )
    with st.expander("How to use Journey Review and its filters", expanded=False):
        st.markdown(textwrap.dedent(
            """
            1. Use the main filters to narrow the list by outcome, customer experience,
               unresolved status, frustration, problem type, who started the journey, review
               requirement, or date.
            2. Leaving a filter empty means it does not limit the results. Choosing several
               values inside one filter shows journeys matching any of those values. Filters
               used in different boxes work together, so a journey must match all of them.
            3. **Broadcast-only issue journeys** are hidden by default. Select
               **Only broadcast-only issue journeys** to show exclusively the journeys where
               the only red issue came from a system broadcast.
            4. The Agent, Bot, and Broadcast message filters look for journeys containing at
               least one evaluated message with the selected color:
               **Red** needs attention, **Yellow** may need improvement, and **Green** was
               assessed as acceptable. **All (no filter)** places no restriction on that sender.
               If you select colors for several sender types, the journey must match every
               selected sender filter.
            5. Use search to find a customer or conversation directly. Use **Worst score first**
               to begin with the journeys that may need the most attention.
            6. Turn on **Show journey analysis and review metrics**. Check the journey outcome,
               experience, score, frustration, main issue, recommended actions, and flagged
               message summary.
            7. Read the full conversation. Use the information button beside an evaluated
               message to understand why it was marked red, yellow, or green.
            8. Add a clear review comment and select **Mark as reviewed** when your check is
               complete.
            """
        ))

    show_review_details = st.toggle(
        "Show journey analysis and review metrics",
        value=False,
        key="review_show_supporting_details",
        help="Filters and conversation transcripts remain visible. Turn this on to show the KPI strip and selected journey analysis.",
    )

    review_filters = _conversation_filters_with_keys(
        conv_df,
        "review_filters",
        include_journey_starter=True,
    )
    filtered_df = _apply_conversation_filters_fresh(conv_df, review_filters)
    sender_flag_filters = _render_sender_flag_filters()
    filtered_df = _filter_conversations_by_sender_flags(
        filtered_df,
        rr.conversation_results,
        sender_flag_filters,
    )

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

    order_choice = st.radio(
        "Journey order",
        ["Current order", "Worst score first", "Best score first"],
        horizontal=True,
        key="review_score_order",
        help="Use the final conversation score to browse from worst to best, or best to worst.",
    )
    previous_order_choice = st.session_state.get("review_last_applied_order_choice")
    if "__run_order" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("__run_order", kind="stable")
    if order_choice != "Current order":
        if "score_final" not in filtered_df.columns:
            st.caption("No final conversation score is available for this run.")
        else:
            filtered_df = (
                filtered_df.assign(
                    _score_sort=pd.to_numeric(filtered_df["score_final"], errors="coerce")
                )
                .sort_values(
                    "_score_sort",
                    ascending=(order_choice == "Worst score first"),
                    na_position="last",
                    kind="stable",
                )
                .drop(columns=["_score_sort"])
            )

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
    if show_review_details:
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
        score = pd.to_numeric(pd.Series([row.get("score_final")]), errors="coerce").iloc[0]
        score_label = (
            f"Score {score:.1f}" if pd.notna(score) and float(score) <= 10 else
            (f"Score {score:.0f}" if pd.notna(score) else "No score")
        )
        label = f"{phone} • {cust} • {source_count} source convs • {result}"
        label = f"{score_label} - {label}"
        if label in label_to_id:
            label = f"{label} - {cid[:8]}"
        options.append(label)
        label_to_id[label] = cid
        ordered_ids.append(cid)

    current_id = str(st.session_state.get("review_selected_conversation_id") or "")
    if previous_order_choice is not None and previous_order_choice != order_choice and ordered_ids:
        current_id = ordered_ids[0]
        st.session_state.review_selected_conversation_id = current_id
        st.session_state.review_scroll_to_conversation_start = True
    st.session_state.review_last_applied_order_choice = order_choice
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
    run_id_for_review = st.session_state.get("current_run_id") or target_cr.get("run_id")
    _render_journey_review_tracking(run_id_for_review, target_id)
    _render_conversation_summary_card_fresh(
        target_cr,
        show_details=show_review_details,
    )

    st.markdown("### Full Customer Journey")
    st.caption(
        "The full appended customer journey is shown below. Where available, assistant replies also include a short quality check underneath."
    )
    st.markdown("<div id='review-conversation-start'></div>", unsafe_allow_html=True)
    transcript = target_cr.get("transcript") or []
    msgs = target_cr.get("message_level_results") or []
    if show_review_details:
        _render_flagged_message_checker(transcript, msgs, target_id)
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
    if not _can_export_results():
        st.warning("Only active users and master admins can export results.")
        return

    st.subheader("Exports")
    if not _has_results():
        st.info("Run an evaluation first to enable exports.")
        return

    rr = st.session_state.run_results
    run_config = {
        "api_base_url": st.session_state.api_base_url,
        "model": st.session_state.selected_model,
        "message_model": st.session_state.selected_model,
        "conversation_model": st.session_state.get("conversation_selected_model"),
        "service_tier": "flex" if st.session_state.use_flex_service_tier else None,
        "message_thinking_effort": st.session_state.message_thinking_effort,
        "conversation_thinking_effort": st.session_state.conversation_thinking_effort,
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

    conversation_results = [
        _normalize_conversation_result_for_display(cr)
        for cr in rr.conversation_results
    ]
    conv_bytes = build_conversation_csv_bytes(conversation_results)
    msg_bytes = build_message_csv_bytes(rr.message_level_results)
    json_bytes = build_full_json_bytes(
        run_config=run_config,
        conversation_results=conversation_results,
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
    st.markdown("### Journey Category CSVs")
    st.caption("Each file uses the same journey-level columns, filtered by outcome and customer experience.")

    category_specs = [
        (
            "Handled",
            "cx_journeys_handled.csv",
            {"handled_status": "handled"},
        ),
        (
            "Handled / Good",
            "cx_journeys_handled_good.csv",
            {"handled_status": "handled", "customer_experience": "good"},
        ),
        (
            "Handled / Bad",
            "cx_journeys_handled_bad.csv",
            {"handled_status": "handled", "customer_experience": "bad"},
        ),
        (
            "Not Handled",
            "cx_journeys_not_handled.csv",
            {"handled_status": "unhandled"},
        ),
        (
            "Pending Unresolved / Bad",
            "cx_journeys_pending_unresolved_bad.csv",
            {
                "handled_status": "unhandled",
                "customer_experience": "bad",
                "unhandled_resolution_subtype": "pending_unresolved",
            },
        ),
        (
            "Totally Unresolved / Bad",
            "cx_journeys_totally_unresolved_bad.csv",
            {
                "handled_status": "unhandled",
                "customer_experience": "bad",
                "unhandled_resolution_subtype": "totally_unresolved",
            },
        ),
        (
            "Not Handled / Good",
            "cx_journeys_not_handled_good.csv",
            {"handled_status": "unhandled", "customer_experience": "good"},
        ),
        (
            "Overall Good",
            "cx_journeys_overall_good.csv",
            {"customer_experience": "good"},
        ),
        (
            "Overall Bad",
            "cx_journeys_overall_bad.csv",
            {"customer_experience": "bad"},
        ),
    ]

    for row_start in range(0, len(category_specs), 3):
        cols = st.columns(3)
        for col, (label, file_name, filters) in zip(cols, category_specs[row_start:row_start + 3]):
            subset = _filter_conversation_results_for_export(conversation_results, **filters)
            with col:
                st.download_button(
                    f"{label} ({len(subset):,})",
                    data=build_conversation_csv_bytes(subset),
                    file_name=file_name,
                    mime="text/csv",
                    disabled=not subset,
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
    if not _is_master():
        st.warning("Only master admins can open Debug.")
        return

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
        "message_model": st.session_state.selected_model,
        "conversation_model": st.session_state.get("conversation_selected_model"),
        "service_tier": "flex" if st.session_state.use_flex_service_tier else None,
        "message_thinking_effort": st.session_state.message_thinking_effort,
        "conversation_thinking_effort": st.session_state.conversation_thinking_effort,
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


def _workspace_tab_specs(auth_role: str) -> list[tuple[str, Any]]:
    """Return the tabs available to the signed-in role."""
    role = _normalize_role(auth_role)
    review_tabs = [
        ("Overview", tab_overview),
        ("Stats", tab_stats),
        ("Dashboard", tab_dashboard),
        ("Journey Review", tab_review),
    ]
    if role == ROLE_READ_ONLY:
        return review_tabs
    if role == ROLE_ACTIVE:
        return [
            ("Upload & Settings", tab_upload),
            ("Run Evaluation", tab_run),
            *review_tabs,
            ("Exports", tab_exports),
        ]
    admin_tabs = [
        ("Reviewer Admin", tab_reviewer_admin),
        ("Upload & Settings", tab_upload),
        ("Run Evaluation", tab_run),
        *review_tabs,
        ("Exports", tab_exports),
        ("Debug", tab_debug),
    ]
    if PROMPT_EDITING_ENABLED:
        admin_tabs.insert(2, ("Prompts", tab_prompts))
    return admin_tabs


def main() -> None:
    _apply_theme()
    if not _render_auth_gate():
        return

    auth_role = _current_role()
    is_master = auth_role == ROLE_MASTER
    is_active = auth_role == ROLE_ACTIVE

    # Force DB initialization at app start so the seeded defaults exist before
    # the sidebar status or any tab tries to read them.
    db = get_active_db()
    if is_master:
        _refresh_default_prompts(db)
    _auto_load_latest_run(db)

    if is_master:
        render_auth_sidebar()
        with st.sidebar:
            render_database_selector(in_sidebar=True)
            st.markdown("---")
            render_active_prompt_status()
        render_sidebar()
    elif is_active:
        render_auth_sidebar()
        with st.sidebar:
            render_database_selector(in_sidebar=True)
            st.markdown("---")
            render_active_prompt_status()
        render_sidebar()
    else:
        _render_read_only_sidebar()

    _render_workspace_header()

    _ = (
        "AI-as-a-Judge evaluation of appended customer journeys across one or more source conversations. "
        "Built for management review — focused on outcomes, frustration, and root cause."
    )

    tab_specs = _workspace_tab_specs(auth_role)
    tabs = st.tabs([label for label, _render in tab_specs])
    for tab, (_label, render_tab) in zip(tabs, tab_specs):
        with tab:
            render_tab()


if __name__ == "__main__":
    main()
