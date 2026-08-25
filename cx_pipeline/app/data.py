from __future__ import annotations

import csv
import io
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


JOURNEY_ID_COLUMN = "CUSTOMER_PHONE"
MESSAGE_ORDER_COLUMN = "APPENDED_MESSAGE_INDEX"
LEGACY_MESSAGE_ORDER_COLUMN = "MESSAGE_INDEX"
TEXT_COLUMN = "MESSAGE_TEXT"
ROLE_COLUMN = "SENDER_ROLE"
SOURCE_CONVERSATION_COLUMN = "CONVERSATION_ID"
RAW_ROLE_COLUMN = "RAW_SENDER_ROLE"
REQUIRED_COLUMNS = [
    JOURNEY_ID_COLUMN,
    MESSAGE_ORDER_COLUMN,
    "MESSAGE_TIME",
    ROLE_COLUMN,
    TEXT_COLUMN,
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
    "CONVERSATION_IDS",
    "SOURCE_CONVERSATION_COUNT",
    "TOTAL_VISIBLE_MESSAGES",
    "CUSTOMER_MESSAGE_COUNT",
    "AGENT_MESSAGE_COUNT",
]


@dataclass
class ValidationReport:
    valid: bool
    missing_columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: int = 0
    valid_rows: int = 0
    dropped_rows: int = 0
    journeys: int = 0
    source_conversations: int = 0
    invalid_role_rows: int = 0
    invalid_order_rows: int = 0
    duplicate_order_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "missing_columns": self.missing_columns,
            "errors": self.errors,
            "warnings": self.warnings,
            "rows": self.rows,
            "valid_rows": self.valid_rows,
            "dropped_rows": self.dropped_rows,
            "journeys": self.journeys,
            "source_conversations": self.source_conversations,
            "invalid_role_rows": self.invalid_role_rows,
            "invalid_order_rows": self.invalid_order_rows,
            "duplicate_order_rows": self.duplicate_order_rows,
        }


def load_csv(path: str | Path) -> pd.DataFrame:
    data = Path(path).read_bytes()
    sample = data[:8192].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = ","
    return pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, keep_default_na=False)


def normalize_role(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"agent", "assistant", "bot", "company", "support"}:
        return "agent"
    if raw in {"customer", "consumer", "client", "user"}:
        return "customer"
    if "agent" in raw or "assistant" in raw or "bot" in raw:
        return "agent"
    if "customer" in raw or "consumer" in raw or "client" in raw:
        return "customer"
    return "unknown"


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(valid=True, rows=int(len(df)))
    report.missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if report.missing_columns:
        report.valid = False
        report.errors.append(
            "CSV is missing required columns: " + ", ".join(report.missing_columns)
        )
        return report

    work = df.copy()
    customer_blank = work[JOURNEY_ID_COLUMN].fillna("").astype(str).str.strip() == ""
    text_blank = work[TEXT_COLUMN].fillna("").astype(str).str.strip() == ""
    report.dropped_rows = int((customer_blank | text_blank).sum())
    if report.dropped_rows:
        report.warnings.append(
            f"{report.dropped_rows:,} row(s) have blank {JOURNEY_ID_COLUMN} or {TEXT_COLUMN} and will be ignored."
        )

    order_numeric = pd.to_numeric(work[MESSAGE_ORDER_COLUMN], errors="coerce")
    invalid_order = order_numeric.isna()
    report.invalid_order_rows = int(invalid_order.sum())
    if report.invalid_order_rows:
        report.warnings.append(
            f"{report.invalid_order_rows:,} row(s) have non-numeric {MESSAGE_ORDER_COLUMN}; they will be ordered as 0."
        )

    normalized_roles = work[ROLE_COLUMN].map(normalize_role)
    report.invalid_role_rows = int((normalized_roles == "unknown").sum())
    if report.invalid_role_rows:
        report.warnings.append(
            f"{report.invalid_role_rows:,} row(s) have unrecognized sender roles and will be treated as unknown."
        )

    valid_mask = ~(customer_blank | text_blank)
    valid = work[valid_mask].copy()
    valid[MESSAGE_ORDER_COLUMN] = order_numeric[valid_mask].fillna(0).astype(int)
    report.valid_rows = int(len(valid))
    report.journeys = int(valid[JOURNEY_ID_COLUMN].astype(str).nunique()) if not valid.empty else 0
    if SOURCE_CONVERSATION_COLUMN in valid.columns:
        report.source_conversations = int(valid[SOURCE_CONVERSATION_COLUMN].astype(str).nunique())
    else:
        report.source_conversations = report.journeys
        report.warnings.append(
            f"{SOURCE_CONVERSATION_COLUMN} is missing; each customer journey will be treated as one source conversation."
        )

    if not valid.empty:
        duplicate_mask = valid.duplicated([JOURNEY_ID_COLUMN, MESSAGE_ORDER_COLUMN], keep=False)
        report.duplicate_order_rows = int(duplicate_mask.sum())
        if report.duplicate_order_rows:
            report.warnings.append(
                f"{report.duplicate_order_rows:,} row(s) share the same {JOURNEY_ID_COLUMN}+{MESSAGE_ORDER_COLUMN}; stable CSV order will break ties."
            )

    if report.valid_rows == 0:
        report.valid = False
        report.errors.append("CSV has no valid message rows after removing blank customer/message rows.")

    return report


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    report = validate_dataframe(df)
    if not report.valid:
        raise ValueError("; ".join(report.errors))

    out = df.copy()
    if RAW_ROLE_COLUMN not in out.columns:
        out[RAW_ROLE_COLUMN] = out[ROLE_COLUMN].astype(str)
    out[ROLE_COLUMN] = out[ROLE_COLUMN].map(normalize_role)
    out[MESSAGE_ORDER_COLUMN] = pd.to_numeric(out[MESSAGE_ORDER_COLUMN], errors="coerce")
    out[MESSAGE_ORDER_COLUMN] = out[MESSAGE_ORDER_COLUMN].fillna(0).astype(int)
    if LEGACY_MESSAGE_ORDER_COLUMN not in out.columns:
        out[LEGACY_MESSAGE_ORDER_COLUMN] = out[MESSAGE_ORDER_COLUMN]
    if SOURCE_CONVERSATION_COLUMN not in out.columns:
        out["CONVERSATION_ID"] = out[JOURNEY_ID_COLUMN].astype(str)
    out = out[out[JOURNEY_ID_COLUMN].astype(str).str.strip() != ""].copy()
    out = out[out[TEXT_COLUMN].astype(str).str.strip() != ""].copy()
    return out.sort_values([JOURNEY_ID_COLUMN, MESSAGE_ORDER_COLUMN], kind="stable").reset_index(drop=True)


def journey_selector_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for journey_id, group in df.groupby(JOURNEY_ID_COLUMN, sort=False):
        group = group.sort_values(MESSAGE_ORDER_COLUMN, kind="stable")
        first = group.iloc[0] if not group.empty else pd.Series(dtype=object)
        raw_starter = str(first.get(RAW_ROLE_COLUMN) or first.get(ROLE_COLUMN) or "unknown").strip().lower()
        starter = normalize_role(raw_starter)
        rows.append(
            {
                "journey_id": str(journey_id),
                "journey_starter": starter,
                "message_count": int(len(group)),
                "source_conversation_count": int(group[SOURCE_CONVERSATION_COLUMN].astype(str).nunique())
                if SOURCE_CONVERSATION_COLUMN in group.columns
                else 1,
            }
        )
    return pd.DataFrame(rows)


def proportional_stratified_sample_ids(
    rows: pd.DataFrame,
    sample_size: int,
    *,
    id_column: str = "journey_id",
    stratum_column: str = "journey_starter",
    seed: int | None = None,
) -> list[str]:
    if rows.empty or id_column not in rows.columns:
        return []
    work = rows.copy()
    work[id_column] = work[id_column].astype(str)
    if stratum_column not in work.columns:
        work[stratum_column] = "unknown"
    work[stratum_column] = (
        work[stratum_column].fillna("unknown").astype(str).str.strip().str.lower().replace("", "unknown")
    )
    target = min(max(int(sample_size), 0), len(work))
    if target <= 0:
        return []
    rng = random.Random(seed)
    grouped = {
        stratum: group[id_column].tolist()
        for stratum, group in work.groupby(stratum_column, sort=True)
    }
    exact = {stratum: target * len(ids) / len(work) for stratum, ids in grouped.items()}
    allocations = {stratum: int(exact[stratum]) for stratum in grouped}
    remaining = target - sum(allocations.values())
    order = sorted(
        grouped,
        key=lambda stratum: (-(exact[stratum] - allocations[stratum]), -len(grouped[stratum]), stratum),
    )
    for stratum in order[:remaining]:
        allocations[stratum] += 1
    selected: list[str] = []
    for stratum, ids in grouped.items():
        count = min(allocations[stratum], len(ids))
        if count:
            selected.extend(rng.sample(ids, count))
    rng.shuffle(selected)
    return selected


def conversation_metadata_from_group(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0] if not group.empty else pd.Series(dtype=object)
    last = group.iloc[-1] if not group.empty else pd.Series(dtype=object)
    roles = group[ROLE_COLUMN].fillna("unknown").astype(str).str.lower() if ROLE_COLUMN in group.columns else pd.Series(dtype=str)
    source_ids = []
    if "CONVERSATION_ID" in group.columns:
        for value in group["CONVERSATION_ID"].astype(str):
            if value and value not in source_ids:
                source_ids.append(value)
    return {
        "customer_journey_id": str(first.get(JOURNEY_ID_COLUMN) or ""),
        "customer_phone": str(first.get(JOURNEY_ID_COLUMN) or ""),
        "customer_name": str(first.get("CUSTOMER_NAME") or ""),
        "conversation_start_date": str(first.get("CONVERSATION_START_DATE") or first.get("MESSAGE_TIME") or ""),
        "conversation_end_date": str(last.get("CONVERSATION_END_DATE") or last.get("MESSAGE_TIME") or ""),
        "source_conversation_ids": ", ".join(source_ids),
        "source_conversation_count": len(source_ids),
        "total_visible_messages": int(len(group)),
        "customer_message_count": int((roles == "customer").sum()) if not roles.empty else 0,
        "agent_message_count": int((roles == "agent").sum()) if not roles.empty else 0,
        "unknown_message_count": int(((roles != "customer") & (roles != "agent")).sum()) if not roles.empty else 0,
    }


RAG_CONTEXT_MARKER = "RAG context used for this bot response:"
_RAG_RE = re.compile(r"\s*<rag_retrievals>.*?</rag_retrievals>\s*", re.DOTALL | re.IGNORECASE)
_RAG_CONTEXT_FOOTER_RE = re.compile(
    rf"\s*{re.escape(RAG_CONTEXT_MARKER)}\s*\n?\s*\{{.*?\}}\s*$",
    re.DOTALL,
)


def strip_inline_rag_context(text: Any) -> str:
    cleaned = _RAG_RE.sub(" ", str(text or ""))
    if RAG_CONTEXT_MARKER in cleaned:
        stripped = _RAG_CONTEXT_FOOTER_RE.sub("", cleaned).rstrip()
        if RAG_CONTEXT_MARKER in stripped:
            stripped = stripped.split(RAG_CONTEXT_MARKER, 1)[0].rstrip()
        cleaned = stripped
    return cleaned.strip()


def compute_metadata(message_evaluations: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row.get("parsed_json") or row.get("result") for row in message_evaluations if row.get("parse_status") == "ok"]
    effects = [str((p or {}).get("message_level_effect") or "neutral") for p in parsed]
    frustration = [str((p or {}).get("frustration_level_after_message") or "none") for p in parsed]
    issue_types = Counter(str((p or {}).get("issue_type") or "none") for p in parsed)
    issue_origins = Counter(str((p or {}).get("issue_origin") or "none") for p in parsed)
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "cancellation_risk": 4}
    max_frustration = max(frustration or ["none"], key=lambda value: rank.get(value, 0))
    roles = [str(record.get("sender_role") or "unknown") for record in records]
    return {
        "total_messages": len(records),
        "customer_messages": sum(1 for role in roles if role == "customer"),
        "agent_messages": sum(1 for role in roles if role == "agent"),
        "unknown_messages": sum(1 for role in roles if role not in {"customer", "agent"}),
        "agent_messages_evaluated": len(parsed),
        "max_frustration_level": max_frustration,
        "issue_count": sum(1 for effect in effects if effect in {"minor_issue", "major_issue"}),
        "major_issue_count": effects.count("major_issue"),
        "minor_issue_count": effects.count("minor_issue"),
        "recovered_issue_count": effects.count("recovered_issue"),
        "issue_type_counts": dict(issue_types),
        "issue_origin_counts": dict(issue_origins),
        "customer_side_issue_count": issue_origins.get("customer_side", 0),
        "our_side_issue_count": issue_origins.get("our_side", 0),
    }
