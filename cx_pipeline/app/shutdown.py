from __future__ import annotations

import os
import sys

from .db import execute, tx


def mark_interrupted(
    *,
    worker_id: str | None = None,
    run_id: str | None = None,
    reason: str = "kill switch",
) -> None:
    filters = []
    params: dict[str, object] = {"reason": reason}
    if worker_id:
        filters.append("locked_by=:worker_id")
        params["worker_id"] = worker_id
    if run_id:
        filters.append("run_id=CAST(:run_id AS uuid)")
        params["run_id"] = run_id
    step_scope = " AND ".join(filters)
    step_scope = f" AND {step_scope}" if step_scope else ""

    request_filters = []
    if worker_id:
        request_filters.append("worker_id=:worker_id")
    if run_id:
        request_filters.append("run_id=CAST(:run_id AS uuid)")
    request_scope = " AND ".join(request_filters)
    request_scope = f" AND {request_scope}" if request_scope else ""

    with tx() as conn:
        execute(
            conn,
            f"""
            UPDATE ai_requests
            SET status='interrupted',
                finished_at=now(),
                duration_seconds=EXTRACT(EPOCH FROM (now() - started_at)),
                error=:reason
            WHERE status='running'{request_scope}
            """,
            params,
        )
        execute(
            conn,
            f"""
            UPDATE run_steps
            SET status='pending',
                locked_by=NULL,
                locked_at=NULL,
                error=:reason,
                updated_at=now()
            WHERE status='running'{step_scope}
            """,
            params,
        )


def hard_exit(
    *,
    code: int = 130,
    worker_id: str | None = None,
    run_id: str | None = None,
    reason: str = "kill switch",
    cleanup: bool = True,
) -> None:
    if cleanup:
        try:
            mark_interrupted(worker_id=worker_id, run_id=run_id, reason=reason)
        except Exception as exc:  # noqa: BLE001
            print(f"Kill switch DB cleanup failed: {exc}", file=sys.stderr, flush=True)
    print("Kill switch activated. Exiting immediately.", file=sys.stderr, flush=True)
    os._exit(code)
