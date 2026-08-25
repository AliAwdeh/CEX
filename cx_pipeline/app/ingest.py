from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.engine import Connection

from .data import (
    JOURNEY_ID_COLUMN,
    LEGACY_MESSAGE_ORDER_COLUMN,
    MESSAGE_ORDER_COLUMN,
    journey_selector_rows,
    conversation_metadata_from_group,
    load_csv,
    normalize_dataframe,
    proportional_stratified_sample_ids,
    validate_dataframe,
)

from .config import get_settings
from .db import as_json, fetch_one


TEXT_COLUMN = "MESSAGE_TEXT"
ROLE_COLUMN = "SENDER_ROLE"
SOURCE_CONVERSATION_COLUMN = "CONVERSATION_ID"
CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}


def _hash_obj(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _message_order_column(df: pd.DataFrame) -> str:
    if MESSAGE_ORDER_COLUMN in df.columns:
        return MESSAGE_ORDER_COLUMN
    return LEGACY_MESSAGE_ORDER_COLUMN


def latest_input_csv(input_dir: str | Path | None = None) -> Path:
    root = Path(input_dir or get_settings().input_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in CSV_EXTENSIONS and not path.name.startswith(".")
    ]
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in input folder: {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_normalized_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_csv(Path(path))
    report = validate_dataframe(raw)
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    return normalize_dataframe(raw), report.to_dict()


def _upsert_customer(conn: Connection, external_customer_id: str, metadata: dict[str, Any]) -> int:
    row = fetch_one(
        conn,
        """
        INSERT INTO customers(external_customer_id, customer_name, metadata, updated_at)
        VALUES(:external_customer_id, :customer_name, CAST(:metadata AS jsonb), now())
        ON CONFLICT(external_customer_id) DO UPDATE SET
            customer_name=COALESCE(EXCLUDED.customer_name, customers.customer_name),
            metadata=customers.metadata || EXCLUDED.metadata,
            updated_at=now()
        RETURNING id
        """,
        {
            "external_customer_id": external_customer_id,
            "customer_name": metadata.get("customer_name"),
            "metadata": as_json(metadata),
        },
    )
    return int(row["id"])


def _insert_source_conversation(
    conn: Connection,
    *,
    customer_id: int,
    source_conversation_id: str,
    content_hash: str,
    metadata: dict[str, Any],
    run_id: str,
    message_count: int,
    first_message_time: str,
    last_message_time: str,
) -> int | None:
    row = fetch_one(
        conn,
        """
        INSERT INTO source_conversations(
            customer_id, source_conversation_id, content_hash, status,
            first_message_time, last_message_time, message_count, metadata,
            first_seen_run_id, updated_at
        )
        VALUES(
            :customer_id, :source_conversation_id, :content_hash, 'new',
            :first_message_time, :last_message_time, :message_count,
            CAST(:metadata AS jsonb), CAST(:run_id AS uuid), now()
        )
        ON CONFLICT(customer_id, source_conversation_id, content_hash) DO NOTHING
        RETURNING id
        """,
        {
            "customer_id": customer_id,
            "source_conversation_id": source_conversation_id,
            "content_hash": content_hash,
            "metadata": as_json(metadata),
            "run_id": run_id,
            "message_count": message_count,
            "first_message_time": first_message_time,
            "last_message_time": last_message_time,
        },
    )
    return int(row["id"]) if row else None


def _insert_message(
    conn: Connection,
    *,
    customer_id: int,
    source_conversation_pk: int,
    customer_message_index: int,
    source_message_index: int | None,
    sender_role: str,
    raw_sender_role: str,
    message_time: str,
    message_text: str,
    raw: dict[str, Any],
) -> int | None:
    row = fetch_one(
        conn,
        """
        INSERT INTO messages(
            customer_id, source_conversation_pk, customer_message_index,
            source_message_index, sender_role, raw_sender_role, message_time,
            message_text, content_hash, raw
        )
        VALUES(
            :customer_id, :source_conversation_pk, :customer_message_index,
            :source_message_index, :sender_role, :raw_sender_role, :message_time,
            :message_text, :content_hash, CAST(:raw AS jsonb)
        )
        ON CONFLICT(source_conversation_pk, customer_message_index, content_hash) DO NOTHING
        RETURNING id
        """,
        {
            "customer_id": customer_id,
            "source_conversation_pk": source_conversation_pk,
            "customer_message_index": customer_message_index,
            "source_message_index": source_message_index,
            "sender_role": sender_role,
            "raw_sender_role": raw_sender_role,
            "message_time": message_time,
            "message_text": message_text,
            "content_hash": _hash_obj(
                {
                    "idx": customer_message_index,
                    "role": sender_role,
                    "text": message_text,
                    "time": message_time,
                }
            ),
            "raw": as_json(raw),
        },
    )
    return int(row["id"]) if row else None


def ingest_csv(
    conn: Connection,
    *,
    csv_path: str | Path | None,
    run_id: str,
    random_journeys: int | None = None,
    random_seed: int | None = None,
    input_dir: str | Path | None = None,
) -> dict[str, Any]:
    selected_path = Path(csv_path) if csv_path else latest_input_csv(input_dir)
    df, validation_report = load_normalized_csv(selected_path)
    if df.empty:
        return {
            "csv_path": str(selected_path),
            "validation": validation_report,
            "customers": 0,
            "source_conversations_inserted": 0,
            "messages_inserted": 0,
            "sampled_journeys": 0,
        }

    selected_ids: list[str] = []
    if random_journeys is not None and int(random_journeys) > 0:
        selector = journey_selector_rows(df)
        selected_ids = proportional_stratified_sample_ids(
            selector,
            int(random_journeys),
            seed=random_seed,
        )
        wanted = set(selected_ids)
        df = df[df[JOURNEY_ID_COLUMN].astype(str).isin(wanted)].copy()

    order_col = _message_order_column(df)
    inserted_sources = 0
    inserted_messages = 0
    touched_customers: set[int] = set()

    for customer_external_id, customer_group in df.groupby(JOURNEY_ID_COLUMN, sort=False):
        customer_external_id = _safe_text(customer_external_id).strip()
        if not customer_external_id:
            continue
        customer_group = customer_group.sort_values(order_col, kind="stable")
        customer_metadata = conversation_metadata_from_group(customer_group)
        customer_id = _upsert_customer(conn, customer_external_id, customer_metadata)
        touched_customers.add(customer_id)

        if SOURCE_CONVERSATION_COLUMN in customer_group.columns:
            grouped_sources = customer_group.groupby(SOURCE_CONVERSATION_COLUMN, sort=False, dropna=False)
        else:
            grouped_sources = [(customer_external_id, customer_group)]

        for source_id, source_group in grouped_sources:
            source_id = _safe_text(source_id).strip() or customer_external_id
            source_group = source_group.sort_values(order_col, kind="stable")
            source_rows = []
            for _, row in source_group.iterrows():
                source_rows.append(
                    {
                        "idx": int(row[order_col]) if pd.notna(row[order_col]) else None,
                        "role": _safe_text(row.get(ROLE_COLUMN)).strip().lower() or "unknown",
                        "text": _safe_text(row.get(TEXT_COLUMN)),
                        "time": _safe_text(row.get("MESSAGE_TIME")),
                    }
                )
            content_hash = _hash_obj(source_rows)
            first_time = source_rows[0]["time"] if source_rows else ""
            last_time = source_rows[-1]["time"] if source_rows else ""
            source_pk = _insert_source_conversation(
                conn,
                customer_id=customer_id,
                source_conversation_id=source_id,
                content_hash=content_hash,
                metadata={"customer_external_id": customer_external_id, "source_conversation_id": source_id},
                run_id=run_id,
                message_count=len(source_rows),
                first_message_time=first_time,
                last_message_time=last_time,
            )
            if source_pk is None:
                continue
            inserted_sources += 1

            for local_index, (_, row) in enumerate(source_group.iterrows(), start=1):
                msg_id = _insert_message(
                    conn,
                    customer_id=customer_id,
                    source_conversation_pk=source_pk,
                    customer_message_index=int(row[order_col]) if pd.notna(row[order_col]) else local_index,
                    source_message_index=local_index,
                    sender_role=_safe_text(row.get(ROLE_COLUMN)).strip().lower() or "unknown",
                    raw_sender_role=_safe_text(row.get("RAW_SENDER_ROLE")),
                    message_time=_safe_text(row.get("MESSAGE_TIME")),
                    message_text=_safe_text(row.get(TEXT_COLUMN)),
                    raw={str(k): _safe_text(v) for k, v in row.to_dict().items()},
                )
                if msg_id is not None:
                    inserted_messages += 1

    return {
        "csv_path": str(selected_path),
        "validation": validation_report,
        "random_journeys_requested": int(random_journeys or 0),
        "random_seed": random_seed,
        "selected_journey_ids": selected_ids,
        "sampled_journeys": len(selected_ids) if selected_ids else int(df[JOURNEY_ID_COLUMN].astype(str).nunique()),
        "customers": len(touched_customers),
        "source_conversations_inserted": inserted_sources,
        "messages_inserted": inserted_messages,
    }
