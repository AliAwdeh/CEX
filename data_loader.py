"""CSV loading, validation, and customer journey preparation."""

from __future__ import annotations

import io
import random
from typing import Any

import pandas as pd


JOURNEY_ID_COLUMN = "CUSTOMER_PHONE"
MESSAGE_ORDER_COLUMN = "APPENDED_MESSAGE_INDEX"
LEGACY_MESSAGE_ORDER_COLUMN = "MESSAGE_INDEX"

REQUIRED_COLUMNS = [
    JOURNEY_ID_COLUMN,
    MESSAGE_ORDER_COLUMN,
    "MESSAGE_TIME",
    "SENDER_ROLE",
    "MESSAGE_TEXT",
]

ID_COLUMNS = [JOURNEY_ID_COLUMN]


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
    "CONVERSATION_IDS",
    "SOURCE_CONVERSATION_COUNT",
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
            + "\n\nThe current data model expects one row per visible message in a "
            "customer journey: CUSTOMER_PHONE is the journey key and "
            "APPENDED_MESSAGE_INDEX is the message order within that journey."
        )
        return False, missing, msg
    return True, [], "CSV passes validation."


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and clean key columns; do not drop required columns."""
    df = df.copy()

    # Coerce the appended journey order numeric and mirror it into MESSAGE_INDEX
    # for the older downstream evaluator/export code paths.
    if MESSAGE_ORDER_COLUMN in df.columns:
        df[MESSAGE_ORDER_COLUMN] = pd.to_numeric(df[MESSAGE_ORDER_COLUMN], errors="coerce")
        df[LEGACY_MESSAGE_ORDER_COLUMN] = df[MESSAGE_ORDER_COLUMN]
    elif LEGACY_MESSAGE_ORDER_COLUMN in df.columns:
        df[LEGACY_MESSAGE_ORDER_COLUMN] = pd.to_numeric(df[LEGACY_MESSAGE_ORDER_COLUMN], errors="coerce")

    # Stringify MESSAGE_TEXT to avoid NaN type issues downstream.
    if "MESSAGE_TEXT" in df.columns:
        df["MESSAGE_TEXT"] = df["MESSAGE_TEXT"].fillna("").astype(str)

    # Lowercase SENDER_ROLE for predictable comparisons.
    if "SENDER_ROLE" in df.columns:
        df["SENDER_ROLE"] = df["SENDER_ROLE"].fillna("unknown").astype(str).str.strip().str.lower()

    # CUSTOMER_PHONE is now the stable parent key for the full appended journey.
    # Keep THREAD_ID/JOURNEY_ID aliases so older internal names keep working.
    if JOURNEY_ID_COLUMN in df.columns:
        df[JOURNEY_ID_COLUMN] = df[JOURNEY_ID_COLUMN].fillna("").astype(str)
        df["JOURNEY_ID"] = df[JOURNEY_ID_COLUMN]
        df["THREAD_ID"] = df[JOURNEY_ID_COLUMN]
    if "CONVERSATION_ID" in df.columns:
        df["CONVERSATION_ID"] = df["CONVERSATION_ID"].fillna("").astype(str)

    return df


def generate_message_id(conversation_id: str, message_index: Any) -> str:
    """Generate a stable message id from ID and message index."""
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
    id_col = JOURNEY_ID_COLUMN if JOURNEY_ID_COLUMN in df.columns else "THREAD_ID"
    if id_col in df.columns:
        summary["conversations"] = int(df[id_col].nunique())
        summary["journeys"] = int(df[id_col].nunique())
    if "CONVERSATION_ID" in df.columns:
        summary["source_conversations"] = int(df["CONVERSATION_ID"].nunique())
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
    """Return list of (journey_id, sorted_dataframe) tuples."""
    id_col = JOURNEY_ID_COLUMN if JOURNEY_ID_COLUMN in df.columns else "THREAD_ID"
    if id_col not in df.columns:
        return []
    out = []
    sort_cols = [col for col in (MESSAGE_ORDER_COLUMN, LEGACY_MESSAGE_ORDER_COLUMN, "MESSAGE_TIME") if col in df.columns]
    for conv_id, group in df.groupby(id_col, sort=False):
        if sort_cols:
            sorted_group = group.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        else:
            sorted_group = group.reset_index(drop=True)
        out.append((str(conv_id), sorted_group))
    return out


def proportional_stratified_sample_ids(
    rows: pd.DataFrame,
    sample_size: int,
    id_column: str = "journey_id",
    stratum_column: str = "journey_starter",
    rng: random.Random | None = None,
) -> list[str]:
    """Randomly sample IDs while preserving each stratum's source proportion.

    Whole-journey allocations use the largest-remainder method, so percentages
    are exact when possible and otherwise as close as the sample size permits.
    """
    if rows.empty or id_column not in rows.columns:
        return []

    work = rows.copy()
    work[id_column] = work[id_column].astype(str)
    if stratum_column not in work.columns:
        work[stratum_column] = "unknown"
    work[stratum_column] = (
        work[stratum_column]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "unknown")
    )

    total = len(work)
    target = min(max(int(sample_size), 0), total)
    if target == 0:
        return []

    randomizer = rng or random.Random()
    groups = {
        stratum: group[id_column].tolist()
        for stratum, group in work.groupby(stratum_column, sort=True)
    }

    exact_allocations = {
        stratum: target * len(ids) / total
        for stratum, ids in groups.items()
    }
    allocations = {
        stratum: int(exact_allocations[stratum])
        for stratum in groups
    }
    remaining = target - sum(allocations.values())
    remainder_order = sorted(
        groups,
        key=lambda stratum: (
            -(exact_allocations[stratum] - allocations[stratum]),
            -len(groups[stratum]),
            stratum,
        ),
    )
    for stratum in remainder_order[:remaining]:
        allocations[stratum] += 1

    selected: list[str] = []
    for stratum, ids in groups.items():
        count = allocations[stratum]
        if count:
            selected.extend(randomizer.sample(ids, count))
    randomizer.shuffle(selected)
    return selected


def conversation_metadata_from_group(group: pd.DataFrame) -> dict:
    """Extract journey-level metadata from a grouped customer timeline."""
    if group.empty:
        return {}
    first = group.iloc[0]
    last = group.iloc[-1]
    md: dict[str, Any] = {}

    def clean(value: Any) -> Any:
        if pd.isna(value):
            return None
        return str(value) if not isinstance(value, (int, float, bool)) else value

    def unique_join(column: str) -> str | None:
        if column not in group.columns:
            return None
        values: list[str] = []
        for val in group[column].dropna().astype(str):
            for part in val.split(","):
                item = part.strip()
                if item and item not in values:
                    values.append(item)
        return ", ".join(values) if values else None

    for col in METADATA_COLUMNS:
        if col not in group.columns:
            continue
        key = col.lower()
        if col == "CONVERSATION_START_DATE":
            md[key] = clean(first.get(col))
        elif col == "CONVERSATION_END_DATE":
            md[key] = clean(last.get(col))
        elif col in {"JOINED_SKILLS", "CONVERSATION_IDS"}:
            md[key] = unique_join(col)
        elif col in {"TOTAL_VISIBLE_MESSAGES", "CUSTOMER_MESSAGE_COUNT", "AGENT_MESSAGE_COUNT"}:
            md[key] = clean(first.get(col))
        else:
            md[key] = clean(first.get(col))

    journey_id = clean(first.get(JOURNEY_ID_COLUMN)) if JOURNEY_ID_COLUMN in group.columns else None
    source_ids = unique_join("CONVERSATION_IDS") or unique_join("CONVERSATION_ID")
    md["customer_journey_id"] = journey_id
    md["journey_id"] = journey_id
    md["source_conversation_ids"] = source_ids
    md["source_conversation_count"] = len([x for x in (source_ids or "").split(",") if x.strip()])
    md["total_visible_messages"] = int(len(group))
    if "SENDER_ROLE" in group.columns:
        roles = group["SENDER_ROLE"].fillna("unknown").astype(str).str.lower()
        md["customer_message_count"] = int((roles == "customer").sum())
        md["agent_message_count"] = int((roles == "agent").sum())
        md["unknown_message_count"] = int(((roles != "customer") & (roles != "agent")).sum())
    return md


def message_records_from_group(group: pd.DataFrame, conversation_id: str) -> list[dict]:
    """Return message dicts for a customer journey, in appended order."""
    records: list[dict] = []
    for _, row in group.iterrows():
        msg_index = row.get(MESSAGE_ORDER_COLUMN, row.get(LEGACY_MESSAGE_ORDER_COLUMN))
        records.append(
            {
                "message_id": generate_message_id(conversation_id, msg_index),
                "message_index": int(msg_index) if pd.notna(msg_index) else None,
                "appended_message_index": int(msg_index) if pd.notna(msg_index) else None,
                "source_conversation_id": (
                    str(row.get("CONVERSATION_ID"))
                    if "CONVERSATION_ID" in group.columns and pd.notna(row.get("CONVERSATION_ID"))
                    else None
                ),
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
