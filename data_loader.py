"""CSV loading, validation, and conversation preparation."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "CONVERSATION_ID",
    "MESSAGE_INDEX",
    "MESSAGE_TIME",
    "SENDER_ROLE",
    "MESSAGE_TEXT",
]


METADATA_COLUMNS = [
    "CONVERSATION_START_DATE",
    "CONVERSATION_END_DATE",
    "CONVERSATION_STATUS",
    "INITIAL_SKILL",
    "LAST_SKILL",
    "JOINED_SKILLS",
    "CONVERSATION_AGENT_FULL_NAME",
    "CONVERSATION_AGENT_LOGIN_NAME",
    "CUSTOMER_NAME",
    "CUSTOMER_PHONE",
    "TOTAL_VISIBLE_MESSAGES",
    "CUSTOMER_MESSAGE_COUNT",
    "AGENT_MESSAGE_COUNT",
]


def load_csv(file_obj: Any) -> pd.DataFrame:
    """Load CSV file into a DataFrame.

    Accepts a file-like object (Streamlit upload), a path, or bytes.
    """
    if isinstance(file_obj, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file_obj))
    return pd.read_csv(file_obj)


def validate_csv(df: pd.DataFrame) -> tuple[bool, list[str], str]:
    """Check the DataFrame has the columns required to run evaluation.

    Returns (is_valid, missing_columns, message).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = (
            "This CSV is missing required columns needed for evaluation:\n- "
            + "\n- ".join(missing)
            + "\n\nPlease export the CSV using the expected Snowflake query structure."
        )
        return False, missing, msg
    return True, [], "CSV passes validation."


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and clean key columns; do not drop required columns."""
    df = df.copy()

    # Coerce MESSAGE_INDEX numeric and sort-safe.
    if "MESSAGE_INDEX" in df.columns:
        df["MESSAGE_INDEX"] = pd.to_numeric(df["MESSAGE_INDEX"], errors="coerce")

    # Stringify MESSAGE_TEXT to avoid NaN type issues downstream.
    if "MESSAGE_TEXT" in df.columns:
        df["MESSAGE_TEXT"] = df["MESSAGE_TEXT"].fillna("").astype(str)

    # Lowercase SENDER_ROLE for predictable comparisons.
    if "SENDER_ROLE" in df.columns:
        df["SENDER_ROLE"] = df["SENDER_ROLE"].fillna("unknown").astype(str).str.strip().str.lower()

    # Stringify CONVERSATION_ID to handle mixed int/str ids.
    if "CONVERSATION_ID" in df.columns:
        df["CONVERSATION_ID"] = df["CONVERSATION_ID"].astype(str)

    return df


def generate_message_id(conversation_id: str, message_index: Any) -> str:
    """Generate a stable message id from conversation id and message index."""
    try:
        idx = int(message_index)
    except (TypeError, ValueError):
        idx = message_index
    return f"{conversation_id}-{idx}"


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Produce a small summary used on the Upload page."""
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "conversations": 0,
        "customer_messages": 0,
        "agent_messages": 0,
        "unknown_messages": 0,
        "date_min": None,
        "date_max": None,
    }
    if "CONVERSATION_ID" in df.columns:
        summary["conversations"] = int(df["CONVERSATION_ID"].nunique())
    if "SENDER_ROLE" in df.columns:
        role_series = df["SENDER_ROLE"].astype(str).str.lower()
        summary["customer_messages"] = int((role_series == "customer").sum())
        summary["agent_messages"] = int((role_series == "agent").sum())
        summary["unknown_messages"] = int(
            ((role_series != "customer") & (role_series != "agent")).sum()
        )

    for date_col in ("CONVERSATION_START_DATE", "MESSAGE_TIME"):
        if date_col in df.columns:
            try:
                parsed = pd.to_datetime(df[date_col], errors="coerce", utc=False)
                non_null = parsed.dropna()
                if len(non_null) > 0:
                    summary["date_min"] = str(non_null.min())
                    summary["date_max"] = str(non_null.max())
                    break
            except Exception:
                continue

    return summary


def get_conversation_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return list of (conversation_id, sorted_dataframe) tuples."""
    if "CONVERSATION_ID" not in df.columns:
        return []
    out = []
    for conv_id, group in df.groupby("CONVERSATION_ID", sort=False):
        sorted_group = group.sort_values("MESSAGE_INDEX", kind="stable").reset_index(drop=True)
        out.append((str(conv_id), sorted_group))
    return out


def conversation_metadata_from_group(group: pd.DataFrame) -> dict:
    """Extract conversation-level metadata from the first row of a conversation group."""
    if group.empty:
        return {}
    first = group.iloc[0]
    md: dict[str, Any] = {}
    for col in METADATA_COLUMNS:
        if col in group.columns:
            val = first.get(col)
            if pd.isna(val):
                md[col.lower()] = None
            else:
                md[col.lower()] = str(val) if not isinstance(val, (int, float, bool)) else val
    return md


def message_records_from_group(group: pd.DataFrame, conversation_id: str) -> list[dict]:
    """Return list of message dicts for a conversation group, in order."""
    records: list[dict] = []
    for _, row in group.iterrows():
        msg_index = row.get("MESSAGE_INDEX")
        records.append(
            {
                "message_id": generate_message_id(conversation_id, msg_index),
                "message_index": int(msg_index) if pd.notna(msg_index) else None,
                "message_time": str(row.get("MESSAGE_TIME", "")) if pd.notna(row.get("MESSAGE_TIME")) else "",
                "sender_role": str(row.get("SENDER_ROLE", "unknown")),
                "raw_sender_role": (
                    str(row.get("RAW_SENDER_ROLE"))
                    if "RAW_SENDER_ROLE" in group.columns and pd.notna(row.get("RAW_SENDER_ROLE"))
                    else None
                ),
                "message_text": str(row.get("MESSAGE_TEXT", "") or ""),
                "agent_full_name": (
                    str(row.get("MESSAGE_AGENT_FULL_NAME"))
                    if "MESSAGE_AGENT_FULL_NAME" in group.columns and pd.notna(row.get("MESSAGE_AGENT_FULL_NAME"))
                    else None
                ),
            }
        )
    return records


def estimate_call_counts(
    df: pd.DataFrame,
    max_conversations: int | None = None,
    max_agent_messages_per_conv: int | None = None,
    target_role: str = "agent",
) -> dict:
    """Compute the planned call counts for an evaluation run.

    ``target_role`` selects which messages will be judged at the message level:
    ``"agent"`` (default) or ``"customer"``.
    """
    role = (target_role or "agent").strip().lower()
    if role not in ("agent", "customer"):
        role = "agent"

    groups = get_conversation_groups(df)
    if max_conversations is not None:
        groups = groups[:max_conversations]

    conv_count = len(groups)
    message_calls = 0
    for _, g in groups:
        target_rows = g[g["SENDER_ROLE"] == role]
        n = len(target_rows)
        if max_agent_messages_per_conv is not None:
            n = min(n, max_agent_messages_per_conv)
        message_calls += n

    return {
        "conversations": conv_count,
        "message_level_calls": int(message_calls),
        "conversation_level_calls": int(conv_count),
        "total_calls": int(message_calls + conv_count),
        "target_role": role,
    }
