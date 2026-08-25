from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Connection

from .config import PipelineSettings, get_settings
from .data import compute_metadata, strip_inline_rag_context
from .db import as_json, execute, fetch_all, fetch_one, tx
from .judge import evaluate_message_level, evaluate_ticket_cx, evaluate_ticket_segmentation_once
from .llm import APIConfig, build_client


REOPEN_WINDOW_HOURS = 48
OPEN_TICKET_HIGH_RISK_HOURS = 48
OPEN_TICKET_CRITICAL_IDLE_HOURS = 24
MIN_MESSAGES_FOR_AI_ANALYSIS = 3


def _api_config(settings: PipelineSettings, *, layer: str, concurrency: int = 1) -> APIConfig:
    if layer == "ticket":
        model = settings.ticket_model
        effort = settings.ticket_thinking_effort
    elif layer == "conversation":
        model = settings.conversation_model
        effort = settings.conversation_thinking_effort
    else:
        model = settings.message_model
        effort = settings.message_thinking_effort
    return APIConfig(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=model,
        service_tier=settings.service_tier,
        thinking_effort=effort,
        temperature=settings.temperature,
        top_p=settings.top_p,
        timeout=settings.timeout,
        retries=settings.retries,
        concurrency=concurrency,
        debug_log_calls=settings.debug_log_calls,
    )


def _start_ai_request(
    *,
    run_id: str,
    step_id: int | None,
    worker_id: str | None = None,
    layer: str,
    model: str,
    customer_id: int | None = None,
    ticket_id: int | None = None,
    message_id: int | None = None,
    context: str = "",
) -> int:
    with tx() as conn:
        row = fetch_one(
            conn,
            """
            INSERT INTO ai_requests(
                run_id, step_id, worker_id, layer, model, status, customer_id,
                ticket_id, message_id, context
            )
            VALUES(
                CAST(:run_id AS uuid), :step_id, :worker_id, :layer, :model, 'running',
                :customer_id, :ticket_id, :message_id, :context
            )
            RETURNING id
            """,
            {
                "run_id": run_id,
                "step_id": step_id,
                "worker_id": worker_id,
                "layer": layer,
                "model": model,
                "customer_id": customer_id,
                "ticket_id": ticket_id,
                "message_id": message_id,
                "context": context,
            },
        )
    return int(row["id"])


def _finish_ai_request(
    request_id: int,
    *,
    status: str,
    error: str = "",
    debug: dict[str, Any] | None = None,
) -> None:
    with tx() as conn:
        execute(
            conn,
            """
            UPDATE ai_requests
            SET status=:status,
                finished_at=now(),
                duration_seconds=EXTRACT(EPOCH FROM (now() - started_at)),
                error=:error,
                debug=CAST(:debug AS jsonb)
            WHERE id=:id
            """,
            {
                "id": request_id,
                "status": status,
                "error": error,
                "debug": as_json(debug or {}),
            },
        )


def _ticket_is_closed_sql(alias: str = "t") -> str:
    status = f"lower(COALESCE({alias}.status, ''))"
    return f"(({status} LIKE '%resolved%' AND {status} NOT LIKE '%unresolved%') OR {status} = 'handled')"


def refresh_ticket_lifecycle(conn: Connection, *, ticket_id: int, run_id: str | None = None) -> None:
    watched_columns = """
        ticket_message_count, opened_at, last_message_at, closed_at, reopenable_until,
        lifecycle_risk, lifecycle_reason, analysis_eligible, analysis_skip_reason
    """
    previous = fetch_one(conn, f"SELECT {watched_columns} FROM tickets WHERE id=:ticket_id", {"ticket_id": ticket_id})
    execute(
        conn,
        f"""
        WITH stats AS (
            SELECT
                t.id AS ticket_id,
                count(m.id)::int AS message_count,
                min(NULLIF(m.message_time, '')::timestamptz) AS first_message_at,
                max(NULLIF(m.message_time, '')::timestamptz) AS last_message_at
            FROM tickets t
            LEFT JOIN ticket_messages tm ON tm.ticket_id=t.id
            LEFT JOIN messages m ON m.id=tm.message_id
            WHERE t.id=:ticket_id
            GROUP BY t.id
        )
        UPDATE tickets t
        SET
            ticket_message_count=stats.message_count,
            opened_at=stats.first_message_at,
            last_message_at=stats.last_message_at,
            closed_at=CASE WHEN {_ticket_is_closed_sql('t')} THEN stats.last_message_at ELSE NULL END,
            reopenable_until=CASE
                WHEN {_ticket_is_closed_sql('t')} AND stats.last_message_at IS NOT NULL
                    THEN stats.last_message_at + (:reopen_hours * interval '1 hour')
                ELSE NULL
            END,
            analysis_eligible=stats.message_count >= :min_messages,
            analysis_skip_reason=CASE
                WHEN stats.message_count < :min_messages THEN 'ticket_has_less_than_3_messages'
                ELSE NULL
            END,
            needs_message_analysis=CASE
                WHEN stats.message_count < :min_messages THEN false
                ELSE needs_message_analysis
            END,
            needs_ticket_cx=CASE
                WHEN stats.message_count < :min_messages THEN false
                ELSE needs_ticket_cx
            END,
            lifecycle_risk=CASE
                WHEN NOT {_ticket_is_closed_sql('t')}
                  AND stats.first_message_at IS NOT NULL
                  AND stats.first_message_at <= now() - (:high_risk_hours * interval '1 hour')
                  AND stats.last_message_at <= now() - (:critical_idle_hours * interval '1 hour')
                    THEN 'critical_disregarded'
                WHEN NOT {_ticket_is_closed_sql('t')}
                  AND stats.first_message_at IS NOT NULL
                  AND stats.first_message_at <= now() - (:high_risk_hours * interval '1 hour')
                    THEN 'high_risk_active'
                ELSE 'normal'
            END,
            lifecycle_reason=CASE
                WHEN stats.message_count < :min_messages
                    THEN 'Ticket has fewer than 3 messages, so message and ticket CX analysis are skipped.'
                WHEN NOT {_ticket_is_closed_sql('t')}
                  AND stats.first_message_at IS NOT NULL
                  AND stats.first_message_at <= now() - (:high_risk_hours * interval '1 hour')
                  AND stats.last_message_at <= now() - (:critical_idle_hours * interval '1 hour')
                    THEN 'Open for more than 48 hours with no update in the last 24 hours; customer issue appears disregarded.'
                WHEN NOT {_ticket_is_closed_sql('t')}
                  AND stats.first_message_at IS NOT NULL
                  AND stats.first_message_at <= now() - (:high_risk_hours * interval '1 hour')
                    THEN 'Open for more than 48 hours, but there has been customer activity in the last 24 hours.'
                WHEN {_ticket_is_closed_sql('t')}
                    THEN 'Closed tickets can reopen only within 48 hours of the last ticket message.'
                ELSE NULL
            END,
            lifecycle_updated_at=now()
        FROM stats
        WHERE t.id=stats.ticket_id
        """,
        {
            "ticket_id": ticket_id,
            "reopen_hours": REOPEN_WINDOW_HOURS,
            "high_risk_hours": OPEN_TICKET_HIGH_RISK_HOURS,
            "critical_idle_hours": OPEN_TICKET_CRITICAL_IDLE_HOURS,
            "min_messages": MIN_MESSAGES_FOR_AI_ANALYSIS,
        },
    )
    current = fetch_one(conn, f"SELECT {watched_columns} FROM tickets WHERE id=:ticket_id", {"ticket_id": ticket_id})
    if run_id and previous != current:
        execute(
            conn,
            """
            INSERT INTO run_events(run_id, event_type, message, data)
            SELECT
                CAST(:run_id AS uuid),
                'ticket_lifecycle_refreshed',
                'Ticket lifecycle refreshed',
                jsonb_build_object(
                    'ticket_id', id,
                    'ticket_message_count', ticket_message_count,
                    'opened_at', opened_at,
                    'last_message_at', last_message_at,
                    'closed_at', closed_at,
                    'reopenable_until', reopenable_until,
                    'lifecycle_risk', lifecycle_risk,
                    'analysis_eligible', analysis_eligible,
                    'analysis_skip_reason', analysis_skip_reason,
                    'previous_state', CAST(:previous_state AS jsonb)
                )
            FROM tickets
            WHERE id=:ticket_id
            """,
            {"run_id": run_id, "ticket_id": ticket_id, "previous_state": as_json(previous or {})},
        )


def refresh_customer_ticket_lifecycle(conn: Connection, *, customer_id: int, run_id: str | None = None) -> None:
    rows = fetch_all(conn, "SELECT id FROM tickets WHERE customer_id=:customer_id", {"customer_id": customer_id})
    for row in rows:
        refresh_ticket_lifecycle(conn, ticket_id=int(row["id"]), run_id=run_id)


def refresh_all_ticket_lifecycle(conn: Connection, *, run_id: str | None = None) -> None:
    rows = fetch_all(conn, "SELECT id FROM tickets ORDER BY id")
    for row in rows:
        refresh_ticket_lifecycle(conn, ticket_id=int(row["id"]), run_id=run_id)


def ticket_analysis_eligible(conn: Connection, *, ticket_id: int) -> tuple[bool, str]:
    row = fetch_one(
        conn,
        """
        SELECT analysis_eligible, analysis_skip_reason, ticket_message_count
        FROM tickets
        WHERE id=:ticket_id
        """,
        {"ticket_id": ticket_id},
    )
    if not row:
        return False, "ticket_not_found"
    if bool(row.get("analysis_eligible")):
        return True, ""
    return False, row.get("analysis_skip_reason") or f"ticket_has_less_than_{MIN_MESSAGES_FOR_AI_ANALYSIS}_messages"


def _message_record(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw") or {}
    return {
        "message_id": int(row["id"]),
        "message_index": int(row["customer_message_index"]),
        "appended_message_index": int(row["customer_message_index"]),
        "source_conversation_id": str(row.get("source_conversation_id") or ""),
        "message_time": row.get("message_time") or "",
        "sender_role": str(row.get("sender_role") or "unknown"),
        "raw_sender_role": row.get("raw_sender_role") or "",
        "message_text": strip_inline_rag_context(row.get("message_text") or ""),
        "agent_full_name": raw.get("MESSAGE_AGENT_FULL_NAME"),
        "message_skill": raw.get("MESSAGE_SKILL"),
        "has_rag_retrieval": raw.get("HAS_RAG_RETRIEVAL"),
        "rag_retrieval_count": raw.get("RAG_RETRIEVAL_COUNT"),
        "rag_retrievals": raw.get("RAG_RETRIEVALS"),
        "chunks_fetched": raw.get("CHUNKS_FETCHED"),
        "chunk_justification": raw.get("CHUNK_JUSTIFICATION"),
        "chunk_time": raw.get("CHUNK_TIME"),
    }


def _customer_metadata(conn: Connection, customer_id: int) -> dict[str, Any]:
    row = fetch_one(conn, "SELECT * FROM customers WHERE id=:id", {"id": customer_id}) or {}
    metadata = dict(row.get("metadata") or {})
    metadata["customer_journey_id"] = row.get("external_customer_id")
    metadata["customer_phone"] = row.get("external_customer_id")
    if row.get("customer_name"):
        metadata["customer_name"] = row.get("customer_name")
    return metadata


def customer_records(conn: Connection, customer_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT m.*, sc.source_conversation_id
        FROM messages m
        JOIN source_conversations sc ON sc.id=m.source_conversation_pk
        WHERE m.customer_id=:customer_id
        ORDER BY m.customer_message_index ASC, m.id ASC
        """,
        {"customer_id": customer_id},
    )
    return [_message_record(row) for row in rows]


def new_customer_records(conn: Connection, customer_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT m.*, sc.source_conversation_id
        FROM messages m
        JOIN source_conversations sc ON sc.id=m.source_conversation_pk
        WHERE m.customer_id=:customer_id AND sc.status='new'
        ORDER BY m.customer_message_index ASC, m.id ASC
        """,
        {"customer_id": customer_id},
    )
    return [_message_record(row) for row in rows]


def ticket_records(conn: Connection, ticket_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT m.*, sc.source_conversation_id
        FROM ticket_messages tm
        JOIN messages m ON m.id=tm.message_id
        JOIN source_conversations sc ON sc.id=m.source_conversation_pk
        WHERE tm.ticket_id=:ticket_id
        ORDER BY m.customer_message_index ASC, m.id ASC
        """,
        {"ticket_id": ticket_id},
    )
    return [_message_record(row) for row in rows]


def _existing_ticket_context(conn: Connection, customer_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT t.*, array_agg(m.customer_message_index ORDER BY m.customer_message_index) AS message_indexes
        FROM tickets t
        LEFT JOIN ticket_messages tm ON tm.ticket_id=t.id
        LEFT JOIN messages m ON m.id=tm.message_id
        WHERE t.customer_id=:customer_id
          AND NOT (
            t.closed_at IS NOT NULL
            AND t.closed_at < now() - (:reopen_hours * interval '1 hour')
          )
        GROUP BY t.id
        ORDER BY t.id
        """,
        {"customer_id": customer_id, "reopen_hours": REOPEN_WINDOW_HOURS},
    )
    tickets: list[dict[str, Any]] = []
    for row in rows:
        indexes = [int(v) for v in (row.get("message_indexes") or []) if v is not None]
        seg = dict(row.get("segmentation") or {})
        tickets.append(
            {
                "ticket_id": f"db_ticket_{row['id']}",
                "ticket_category": row.get("category"),
                "request_origin": row.get("request_origin"),
                "ticket_type": row.get("ticket_type"),
                "customer_objective": row.get("objective"),
                "start_message_index": min(indexes) if indexes else None,
                "end_message_index": max(indexes) if indexes else None,
                "included_message_indexes": indexes,
                "status": row.get("status"),
                "opened_at": str(row.get("opened_at") or ""),
                "last_message_at": str(row.get("last_message_at") or ""),
                "closed_at": str(row.get("closed_at") or ""),
                "reopenable_until": str(row.get("reopenable_until") or ""),
                "lifecycle_risk": row.get("lifecycle_risk") or "normal",
                "analysis_eligible": bool(row.get("analysis_eligible")),
                "analysis_skip_reason": row.get("analysis_skip_reason") or "",
                "should_append_future_conversations": row.get("should_append_future"),
                "previous_ticket_id": "",
                "inquiries": seg.get("inquiries") or [],
                "conversation_summaries": seg.get("conversation_summaries") or [],
                "segmentation_reason": seg.get("segmentation_reason") or "",
                "_db_ticket_id": int(row["id"]),
            }
        )
    return tickets


def _source_ids_for_new_records(records: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for record in records:
        source_id = str(record.get("source_conversation_id") or "unknown")
        if source_id not in seen:
            seen.append(source_id)
    return seen


def _match_existing_ticket(
    existing: list[dict[str, Any]],
    ticket: dict[str, Any],
) -> int | None:
    visible_ids = {int(prior["_db_ticket_id"]) for prior in existing}
    raw_ticket_id = str(ticket.get("ticket_id") or "")
    if raw_ticket_id.startswith("db_ticket_"):
        try:
            candidate_id = int(raw_ticket_id.removeprefix("db_ticket_"))
        except ValueError:
            candidate_id = 0
        if candidate_id in visible_ids:
            return candidate_id

    ticket_indexes = set(int(v) for v in ticket.get("included_message_indexes") or [])
    best_id: int | None = None
    best_overlap = 0
    for prior in existing:
        prior_indexes = set(int(v) for v in prior.get("included_message_indexes") or [])
        overlap = len(ticket_indexes & prior_indexes)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = int(prior["_db_ticket_id"])
    return best_id if best_overlap else None


def _message_ids_by_index(conn: Connection, customer_id: int) -> dict[int, int]:
    rows = fetch_all(
        conn,
        "SELECT id, customer_message_index FROM messages WHERE customer_id=:customer_id",
        {"customer_id": customer_id},
    )
    return {int(row["customer_message_index"]): int(row["id"]) for row in rows}


def _upsert_ticket_from_segment(
    conn: Connection,
    *,
    run_id: str,
    customer_id: int,
    existing_context: list[dict[str, Any]],
    segment: dict[str, Any],
) -> int:
    matched_ticket_id = _match_existing_ticket(existing_context, segment)
    params = {
        "customer_id": customer_id,
        "status": segment.get("status") or "pending_unresolved",
        "category": segment.get("ticket_category") or "inquiry",
        "request_origin": segment.get("request_origin") or "customer",
        "ticket_type": segment.get("ticket_type") or "other",
        "objective": segment.get("customer_objective") or "",
        "should_append_future": bool(segment.get("should_append_future_conversations")),
        "model_ticket_id": segment.get("ticket_id") or "",
        "segmentation": as_json(segment),
        "run_id": run_id,
    }
    if matched_ticket_id:
        row = fetch_one(
            conn,
            """
            UPDATE tickets SET
                status=:status, category=:category, request_origin=:request_origin,
                ticket_type=:ticket_type, objective=:objective,
                should_append_future=:should_append_future,
                model_ticket_id=:model_ticket_id, segmentation=CAST(:segmentation AS jsonb),
                latest_ticketing_run_id=CAST(:run_id AS uuid),
                needs_message_analysis=true, updated_at=now()
            WHERE id=:ticket_id
            RETURNING id
            """,
            {**params, "ticket_id": matched_ticket_id},
        )
    else:
        row = fetch_one(
            conn,
            """
            INSERT INTO tickets(
                customer_id, status, category, request_origin, ticket_type,
                objective, should_append_future, model_ticket_id, segmentation,
                latest_ticketing_run_id, needs_message_analysis, needs_ticket_cx
            )
            VALUES(
                :customer_id, :status, :category, :request_origin, :ticket_type,
                :objective, :should_append_future, :model_ticket_id,
                CAST(:segmentation AS jsonb), CAST(:run_id AS uuid), true, false
            )
            RETURNING id
            """,
            params,
        )
    return int(row["id"])


def run_ticketing_step(
    *,
    run_id: str,
    customer_id: int,
    step_id: int | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    with tx() as conn:
        records = new_customer_records(conn, customer_id)
        if not records:
            return {"tickets_created_or_updated": 0, "message_links": 0, "skipped": True}

        refresh_customer_ticket_lifecycle(conn, customer_id=customer_id, run_id=run_id)
        all_records = customer_records(conn, customer_id)
        existing_context = _existing_ticket_context(conn, customer_id)
        metadata = _customer_metadata(conn, customer_id)

    source_ids = _source_ids_for_new_records(records)
    context = {
        "segmentation_mode": "incremental_db_ticketing",
        "instruction": (
            "This is an incremental DB pass. The payload contains only newly ingested "
            "source_conversation_blocks. Use previous_cumulative_ticket_output as the current "
            "ticket map from earlier source conversations. Tickets closed more than 48 hours ago "
            "are intentionally omitted and must not be reopened. Return the complete visible ticket "
            "list, preserving old visible tickets and appending/reopening/updating them when the new "
            "messages continue an old ticket within the 48-hour reopen window. Create a new ticket "
            "for a separate objective or for a continuation of an expired closed ticket."
        ),
        "pass_index": 1,
        "total_passes": 1,
        "current_source_conversation_id": ", ".join(source_ids),
        "processed_previous_source_conversation_ids": [],
        "previous_cumulative_ticket_output": {"tickets": existing_context},
    }
    client = build_client(settings.base_url, settings.api_key)
    api = _api_config(settings, layer="ticket")
    request_id = _start_ai_request(
        run_id=run_id,
        step_id=step_id,
        worker_id=worker_id,
        layer="ticketing",
        model=api.model,
        customer_id=customer_id,
        context=f"ticketing customer {metadata.get('customer_journey_id') or customer_id}",
    )
    try:
        tickets, debug = evaluate_ticket_segmentation_once(
            client,
            api,
            conversation_id=str(metadata.get("customer_journey_id") or customer_id),
            records=records,
            conversation_metadata=metadata,
            truncate_chars=None,
            segmentation_context=context,
            normalization_records=all_records,
        )
        _finish_ai_request(request_id, status="success", debug=debug.get("debug") if isinstance(debug, dict) else {})
    except Exception as exc:
        _finish_ai_request(request_id, status="failed", error=str(exc))
        raise
    new_indexes = {int(record["message_index"]) for record in records}

    with tx() as conn:
        msg_by_index = _message_ids_by_index(conn, customer_id)
        changed_tickets: set[int] = set()
        link_count = 0
        for segment in tickets:
            included = [int(v) for v in segment.get("included_message_indexes") or []]
            if not (set(included) & new_indexes):
                continue
            ticket_id = _upsert_ticket_from_segment(
                conn,
                run_id=run_id,
                customer_id=customer_id,
                existing_context=existing_context,
                segment=segment,
            )
            changed_tickets.add(ticket_id)
            for message_index in included:
                message_id = msg_by_index.get(message_index)
                if not message_id:
                    continue
                execute(
                    conn,
                    """
                    INSERT INTO ticket_messages(ticket_id, message_id)
                    VALUES(:ticket_id, :message_id)
                    ON CONFLICT DO NOTHING
                    """,
                    {"ticket_id": ticket_id, "message_id": message_id},
                )
                link_count += 1

        execute(
            conn,
            """
            UPDATE source_conversations
            SET status='ticketed', ticketed_run_id=CAST(:run_id AS uuid), updated_at=now()
            WHERE customer_id=:customer_id AND status='new'
            """,
            {"run_id": run_id, "customer_id": customer_id},
        )
        for ticket_id in changed_tickets:
            refresh_ticket_lifecycle(conn, ticket_id=ticket_id, run_id=run_id)
            enqueue_message_step(conn, run_id=run_id, ticket_id=ticket_id)

    return {
        "tickets_created_or_updated": len(changed_tickets),
        "message_links": link_count,
        "reopen_window_hours": REOPEN_WINDOW_HOURS,
        "min_messages_for_ai_analysis": MIN_MESSAGES_FOR_AI_ANALYSIS,
        "ticket_calls": 1,
        "debug": debug if settings.save_raw_responses else None,
    }


def _target_messages(records: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("sender_role") == role]


def _history(records: list[dict[str, Any]], up_to_index: int, include_unknown: bool = True) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if int(record["message_index"]) > up_to_index:
            break
        if record.get("sender_role") == "unknown" and not include_unknown:
            continue
        out.append(record)
    return out


def run_message_step(
    *,
    run_id: str,
    ticket_id: int,
    step_id: int | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    with tx() as conn:
        ticket = fetch_one(conn, "SELECT * FROM tickets WHERE id=:id", {"id": ticket_id})
        if not ticket:
            return {"skipped": True, "reason": "ticket not found"}
        refresh_ticket_lifecycle(conn, ticket_id=ticket_id, run_id=run_id)
        eligible, skip_reason = ticket_analysis_eligible(conn, ticket_id=ticket_id)
        if not eligible:
            execute(
                conn,
                """
                UPDATE tickets
                SET needs_message_analysis=false, needs_ticket_cx=false, updated_at=now()
                WHERE id=:ticket_id
                """,
                {"ticket_id": ticket_id},
            )
            return {"skipped": True, "reason": skip_reason, "messages_processed": 0}
        records = ticket_records(conn, ticket_id)
        done_rows = fetch_all(
            conn,
            "SELECT message_id FROM message_results WHERE ticket_id=:ticket_id",
            {"ticket_id": ticket_id},
        )
    role = settings.message_target_role if settings.message_target_role in {"agent", "customer"} else "agent"
    targets = _target_messages(records, role)
    done_ids = {int(row["message_id"]) for row in done_rows}
    pending_targets = [record for record in targets if int(record["message_id"]) not in done_ids]
    if not pending_targets:
        with tx() as conn:
            execute(
                conn,
                """
                UPDATE tickets
                SET needs_message_analysis=false, needs_ticket_cx=true, updated_at=now()
                WHERE id=:ticket_id
                """,
                {"ticket_id": ticket_id},
            )
            enqueue_ticket_cx_step(conn, run_id=run_id, ticket_id=ticket_id)
        return {"messages_processed": 0, "skipped": True, "resume_from_existing_results": len(done_ids)}

    client = build_client(settings.base_url, settings.api_key)
    api = _api_config(settings, layer="message")
    metadata = dict(ticket.get("segmentation") or {})
    metadata["ticket_db_id"] = int(ticket_id)
    metadata["customer_db_id"] = int(ticket["customer_id"])
    processed = 0
    failures = 0
    skipped_existing = len(done_ids)
    for target in pending_targets:
        request_id = _start_ai_request(
            run_id=run_id,
            step_id=step_id,
            worker_id=worker_id,
            layer="message",
            model=api.model,
            customer_id=int(ticket["customer_id"]),
            ticket_id=ticket_id,
            message_id=int(target["message_id"]),
            context=f"message ticket {ticket_id} message {target.get('message_index')}",
        )
        try:
            result = evaluate_message_level(
                client,
                api,
                conversation_id=str(ticket_id),
                target_record=target,
                history_records=_history(records, int(target["message_index"])),
                conversation_metadata=metadata,
                save_raw=settings.save_raw_responses,
                truncate_chars=None,
            )
        except Exception as exc:
            _finish_ai_request(request_id, status="failed", error=str(exc))
            raise
        request_status = "success" if result.get("parse_status") == "ok" else "failed"
        _finish_ai_request(
            request_id,
            status=request_status,
            error=result.get("error_message") or "",
            debug=result.get("debug") if isinstance(result.get("debug"), dict) else {},
        )
        if result.get("parse_status") != "ok":
            failures += 1
        with tx() as conn:
            execute(
                conn,
                """
                INSERT INTO message_results(
                    run_id, ticket_id, message_id, parse_status, result, debug,
                    raw_response, attempts, error, updated_at
                )
                VALUES(
                    CAST(:run_id AS uuid), :ticket_id, :message_id, :parse_status,
                    CAST(:result AS jsonb), CAST(:debug AS jsonb), :raw_response,
                    :attempts, :error, now()
                )
                ON CONFLICT(ticket_id, message_id) DO UPDATE SET
                    run_id=EXCLUDED.run_id,
                    parse_status=EXCLUDED.parse_status,
                    result=EXCLUDED.result,
                    debug=EXCLUDED.debug,
                    raw_response=EXCLUDED.raw_response,
                    attempts=EXCLUDED.attempts,
                    error=EXCLUDED.error,
                    updated_at=now()
                """,
                {
                    "run_id": run_id,
                    "ticket_id": ticket_id,
                    "message_id": int(target["message_id"]),
                    "parse_status": result.get("parse_status") or "failed",
                    "result": as_json(result.get("parsed_json")),
                    "debug": as_json(result.get("debug")),
                    "raw_response": result.get("raw_model_response") if settings.save_raw_responses else None,
                    "attempts": int(result.get("automatic_reruns") or 0) + 1,
                    "error": result.get("error_message"),
                },
            )
        processed += 1
    with tx() as conn:
        execute(
            conn,
            """
            UPDATE tickets
            SET needs_message_analysis=false, needs_ticket_cx=true, updated_at=now()
            WHERE id=:ticket_id
            """,
            {"ticket_id": ticket_id},
        )
        enqueue_ticket_cx_step(conn, run_id=run_id, ticket_id=ticket_id)
    return {
        "messages_processed": processed,
        "message_failures": failures,
        "skipped_existing_message_results": skipped_existing,
        "pending_targets_at_start": len(pending_targets),
        "resume_mode": "only_messages_without_existing_results",
    }


def _message_results_for_ticket(conn: Connection, ticket_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT mr.*, m.customer_message_index, m.message_time, m.message_text, sc.source_conversation_id
        FROM message_results mr
        JOIN messages m ON m.id=mr.message_id
        JOIN source_conversations sc ON sc.id=m.source_conversation_pk
        WHERE mr.ticket_id=:ticket_id
        ORDER BY m.customer_message_index ASC, m.id ASC
        """,
        {"ticket_id": ticket_id},
    )
    out = []
    for row in rows:
        out.append(
            {
                "conversation_id": str(ticket_id),
                "message_index": row.get("customer_message_index"),
                "source_conversation_id": row.get("source_conversation_id"),
                "message_time": row.get("message_time"),
                "target_message_text": row.get("message_text"),
                "parse_status": row.get("parse_status"),
                "parsed_json": row.get("result"),
                "evaluation_output": row.get("result"),
            }
        )
    return out


def run_ticket_cx_step(
    *,
    run_id: str,
    ticket_id: int,
    step_id: int | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    with tx() as conn:
        ticket = fetch_one(conn, "SELECT * FROM tickets WHERE id=:id", {"id": ticket_id})
        if not ticket:
            return {"skipped": True, "reason": "ticket not found"}
        refresh_ticket_lifecycle(conn, ticket_id=ticket_id, run_id=run_id)
        eligible, skip_reason = ticket_analysis_eligible(conn, ticket_id=ticket_id)
        if not eligible:
            execute(
                conn,
                "UPDATE tickets SET needs_ticket_cx=false, updated_at=now() WHERE id=:ticket_id",
                {"ticket_id": ticket_id},
            )
            return {"skipped": True, "reason": skip_reason, "ticket_cx_status": "skipped"}
        records = ticket_records(conn, ticket_id)
        message_results = _message_results_for_ticket(conn, ticket_id)
    computed = compute_metadata(message_results, records)
    computed["evaluation_target_role"] = settings.message_target_role
    computed["target_messages_evaluated"] = sum(1 for row in message_results if row.get("parse_status") == "ok")
    metadata = dict(ticket.get("segmentation") or {})
    metadata["ticket_db_id"] = int(ticket_id)
    metadata["customer_db_id"] = int(ticket["customer_id"])
    client = build_client(settings.base_url, settings.api_key)
    api = _api_config(settings, layer="conversation")
    request_id = _start_ai_request(
        run_id=run_id,
        step_id=step_id,
        worker_id=worker_id,
        layer="ticket_cx",
        model=api.model,
        customer_id=int(ticket["customer_id"]),
        ticket_id=ticket_id,
        context=f"ticket cx {ticket_id}",
    )
    try:
        result = evaluate_ticket_cx(
            client,
            api,
            ticket_id=str(ticket_id),
            conversation_metadata=metadata,
            transcript=records,
            message_level_evaluations=message_results,
            computed_metadata=computed,
            save_raw=settings.save_raw_responses,
            truncate_chars=None,
        )
    except Exception as exc:
        _finish_ai_request(request_id, status="failed", error=str(exc))
        raise
    request_status = "success" if result.get("parse_status") == "ok" else "failed"
    _finish_ai_request(
        request_id,
        status=request_status,
        error=result.get("error_message") or "",
        debug=result.get("debug") if isinstance(result.get("debug"), dict) else {},
    )
    parsed = result.get("parsed_json") or {}
    with tx() as conn:
        execute(
            conn,
            """
            INSERT INTO ticket_cx_results(
                run_id, ticket_id, parse_status, result, computed_metadata, debug,
                raw_response, attempts, error, updated_at
            )
            VALUES(
                CAST(:run_id AS uuid), :ticket_id, :parse_status,
                CAST(:result AS jsonb), CAST(:computed AS jsonb), CAST(:debug AS jsonb),
                :raw_response, :attempts, :error, now()
            )
            ON CONFLICT(ticket_id) DO UPDATE SET
                run_id=EXCLUDED.run_id,
                parse_status=EXCLUDED.parse_status,
                result=EXCLUDED.result,
                computed_metadata=EXCLUDED.computed_metadata,
                debug=EXCLUDED.debug,
                raw_response=EXCLUDED.raw_response,
                attempts=EXCLUDED.attempts,
                error=EXCLUDED.error,
                updated_at=now()
            """,
            {
                "run_id": run_id,
                "ticket_id": ticket_id,
                "parse_status": result.get("parse_status") or "failed",
                "result": as_json(parsed),
                "computed": as_json(computed),
                "debug": as_json(result.get("debug")),
                "raw_response": result.get("raw_model_response") if settings.save_raw_responses else None,
                "attempts": int(result.get("automatic_reruns") or 0) + 1,
                "error": result.get("error_message"),
            },
        )
        execute(
            conn,
            "UPDATE tickets SET needs_ticket_cx=false, updated_at=now() WHERE id=:ticket_id",
            {"ticket_id": ticket_id},
        )
    return {
        "ticket_cx_status": result.get("parse_status"),
        "handled_status": parsed.get("handled_status"),
        "customer_experience": parsed.get("customer_experience"),
    }


def enqueue_ticketing_steps(conn: Connection, *, run_id: str) -> int:
    refresh_all_ticket_lifecycle(conn, run_id=run_id)
    rows = fetch_all(
        conn,
        """
        SELECT DISTINCT customer_id
        FROM source_conversations
        WHERE status='new'
        ORDER BY customer_id
        """,
    )
    count = 0
    for row in rows:
        execute(
            conn,
            """
            INSERT INTO run_steps(run_id, step_type, customer_id, status)
            VALUES(CAST(:run_id AS uuid), 'ticketing', :customer_id, 'pending')
            ON CONFLICT DO NOTHING
            """,
            {"run_id": run_id, "customer_id": int(row["customer_id"])},
        )
        count += 1
    return count


def enqueue_message_step(conn: Connection, *, run_id: str, ticket_id: int) -> None:
    refresh_ticket_lifecycle(conn, ticket_id=ticket_id, run_id=run_id)
    eligible, _skip_reason = ticket_analysis_eligible(conn, ticket_id=ticket_id)
    if not eligible:
        return
    execute(
        conn,
        """
        INSERT INTO run_steps(run_id, step_type, ticket_id, status)
        VALUES(CAST(:run_id AS uuid), 'message', :ticket_id, 'pending')
        ON CONFLICT DO NOTHING
        """,
        {"run_id": run_id, "ticket_id": ticket_id},
    )


def enqueue_ticket_cx_step(conn: Connection, *, run_id: str, ticket_id: int) -> None:
    refresh_ticket_lifecycle(conn, ticket_id=ticket_id, run_id=run_id)
    eligible, _skip_reason = ticket_analysis_eligible(conn, ticket_id=ticket_id)
    if not eligible:
        return
    execute(
        conn,
        """
        INSERT INTO run_steps(run_id, step_type, ticket_id, status)
        VALUES(CAST(:run_id AS uuid), 'ticket_cx', :ticket_id, 'pending')
        ON CONFLICT DO NOTHING
        """,
        {"run_id": run_id, "ticket_id": ticket_id},
    )


def enqueue_missing_downstream(conn: Connection, *, run_id: str) -> dict[str, int]:
    refresh_all_ticket_lifecycle(conn, run_id=run_id)
    message_rows = fetch_all(
        conn,
        """
        SELECT id
        FROM tickets
        WHERE needs_message_analysis=true AND analysis_eligible=true
        ORDER BY id
        """,
    )
    cx_rows = fetch_all(
        conn,
        """
        SELECT id
        FROM tickets
        WHERE needs_ticket_cx=true AND analysis_eligible=true
        ORDER BY id
        """,
    )
    for row in message_rows:
        enqueue_message_step(conn, run_id=run_id, ticket_id=int(row["id"]))
    for row in cx_rows:
        enqueue_ticket_cx_step(conn, run_id=run_id, ticket_id=int(row["id"]))
    return {"message_steps": len(message_rows), "ticket_cx_steps": len(cx_rows)}


def run_kpis(conn: Connection, run_id: str | None = None) -> dict[str, Any]:
    params = {"run_id": run_id}
    run_filter = "WHERE run_id=CAST(:run_id AS uuid)" if run_id else ""
    steps = fetch_all(
        conn,
        f"""
        SELECT step_type, status, count(*) AS count
        FROM run_steps
        {run_filter}
        GROUP BY step_type, status
        ORDER BY step_type, status
        """,
        params if run_id else {},
    )
    step_counts = {}
    for row in steps:
        step_counts[f"{row['step_type']}:{row['status']}"] = int(row["count"])
    basics = fetch_one(
        conn,
        """
        SELECT
            (SELECT count(*) FROM customers) AS customers,
            (SELECT count(*) FROM source_conversations) AS source_conversations,
            (SELECT count(*) FROM source_conversations WHERE status='new') AS unticketed_source_conversations,
            (SELECT count(*) FROM messages) AS messages,
            (SELECT count(*) FROM tickets) AS tickets,
            (SELECT count(*) FROM tickets WHERE analysis_eligible=false) AS analysis_ineligible_tickets,
            (SELECT count(*) FROM tickets WHERE lifecycle_risk='high_risk_active') AS high_risk_active_tickets,
            (SELECT count(*) FROM tickets WHERE lifecycle_risk='critical_disregarded') AS critical_disregarded_tickets,
            (SELECT count(*) FROM message_results) AS message_results,
            (SELECT count(*) FROM ticket_cx_results) AS ticket_cx_results
        """,
    ) or {}
    ticket_status_rows = fetch_all(
        conn,
        "SELECT status, count(*) AS count FROM tickets GROUP BY status ORDER BY status",
    )
    ticket_category_rows = fetch_all(
        conn,
        "SELECT category, count(*) AS count FROM tickets GROUP BY category ORDER BY category",
    )
    cx_rows = fetch_all(
        conn,
        """
        SELECT
            result->>'handled_status' AS handled_status,
            result->>'customer_experience' AS customer_experience,
            count(*) AS count
        FROM ticket_cx_results
        GROUP BY handled_status, customer_experience
        """,
    )
    return {
        **{key: int(value or 0) for key, value in basics.items()},
        "steps": step_counts,
        "ticket_status": {row["status"]: int(row["count"]) for row in ticket_status_rows},
        "ticket_category": {row["category"]: int(row["count"]) for row in ticket_category_rows},
        "cx_breakdown": [
            {
                "handled_status": row.get("handled_status") or "unknown",
                "customer_experience": row.get("customer_experience") or "unknown",
                "count": int(row["count"]),
            }
            for row in cx_rows
        ],
    }
