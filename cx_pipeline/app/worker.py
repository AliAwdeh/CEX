from __future__ import annotations

import socket
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from .analysis import run_kpis, run_message_step, run_ticket_cx_step, run_ticketing_step
from .config import PipelineSettings, get_settings
from .db import as_json, execute, fetch_all, fetch_one, tx
from .webhooks import notify_dashboard_run_finished


@dataclass
class WorkerState:
    worker_id: str
    run_id: str | None = None
    running: bool = False
    stop_requested: bool = False
    active_by_step: dict[str, int] = field(default_factory=lambda: {"ticketing": 0, "message": 0, "ticket_cx": 0})
    completed_steps: int = 0
    failed_steps: int = 0
    last_event: str = ""


class PipelineWorker:
    def __init__(self, *, run_id: str | None = None, settings: PipelineSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.state = WorkerState(
            worker_id=f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}",
            run_id=run_id,
        )
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.state.stop_requested = False
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.state.stop_requested = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "worker_id": self.state.worker_id,
                "run_id": self.state.run_id,
                "running": self.state.running,
                "stop_requested": self.state.stop_requested,
                "active_by_step": dict(self.state.active_by_step),
                "completed_steps": self.state.completed_steps,
                "failed_steps": self.state.failed_steps,
                "last_event": self.state.last_event,
            }

    def _set_event(self, message: str) -> None:
        with self._lock:
            self.state.last_event = message

    def _inc_active(self, step_type: str, delta: int) -> None:
        with self._lock:
            self.state.active_by_step[step_type] = max(0, self.state.active_by_step.get(step_type, 0) + delta)

    def _mark_done(self, failed: bool = False) -> None:
        with self._lock:
            if failed:
                self.state.failed_steps += 1
            else:
                self.state.completed_steps += 1

    def _can_submit(self, step_type: str) -> bool:
        active = self.state.active_by_step
        total_active = sum(active.values())
        if total_active >= self.settings.max_total_workers:
            return False
        limits = {
            "ticketing": self.settings.max_ticketing_workers,
            "message": self.settings.max_message_workers,
            "ticket_cx": self.settings.max_ticket_cx_workers,
        }
        return active.get(step_type, 0) < limits[step_type]

    def _claim_step(self, step_type: str) -> dict[str, Any] | None:
        limits = {
            "ticketing": self.settings.max_ticketing_workers,
            "message": self.settings.max_message_workers,
            "ticket_cx": self.settings.max_ticket_cx_workers,
        }
        with tx() as conn:
            execute(
                conn,
                "SELECT pg_advisory_xact_lock(hashtext(:lock_key))",
                {"lock_key": "cx_pipeline_claim_step"},
            )
            row = fetch_one(
                conn,
                """
                WITH active AS (
                    SELECT
                        count(*) AS total_active,
                        count(*) FILTER (WHERE step_type=:step_type) AS step_active
                    FROM run_steps
                    WHERE status='running'
                ),
                candidate AS (
                    SELECT id
                    FROM run_steps, active
                    WHERE status='pending'
                      AND step_type=:step_type
                      AND (:run_id_text = '' OR run_id=CAST(:run_id_text AS uuid))
                      AND active.total_active < :max_total_workers
                      AND active.step_active < :step_limit
                    ORDER BY created_at ASC, id ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE run_steps s
                SET status='running',
                    attempts=attempts + 1,
                    locked_by=:worker_id,
                    locked_at=now(),
                    started_at=COALESCE(started_at, now()),
                    updated_at=now()
                FROM candidate
                WHERE s.id=candidate.id
                RETURNING s.*
                """,
                {
                    "step_type": step_type,
                    "run_id_text": self.state.run_id or "",
                    "worker_id": self.state.worker_id,
                    "max_total_workers": self.settings.max_total_workers,
                    "step_limit": limits[step_type],
                },
            )
            if row:
                execute(
                    conn,
                    """
                    INSERT INTO run_events(run_id, step_id, event_type, message, data)
                    VALUES(:run_id, :step_id, 'step_started', :message, CAST(:data AS jsonb))
                    """,
                    {
                        "run_id": row["run_id"],
                        "step_id": row["id"],
                        "message": f"{step_type} started",
                        "data": "{}",
                    },
                )
            return row

    def _finish_step(self, step: dict[str, Any], *, ok: bool, result: dict[str, Any] | None = None, error: str = "") -> None:
        status = "done" if ok else ("failed" if int(step["attempts"]) >= int(step["max_attempts"]) else "pending")
        with tx() as conn:
            execute(
                conn,
                """
                UPDATE run_steps
                SET status=:status,
                    finished_at=CASE WHEN :terminal THEN now() ELSE finished_at END,
                    locked_by=NULL,
                    locked_at=NULL,
                    error=:error,
                    payload=payload || CAST(:payload AS jsonb),
                    updated_at=now()
                WHERE id=:step_id
                """,
                {
                    "status": status,
                    "terminal": status in {"done", "failed"},
                    "error": error,
                    "payload": "{}" if result is None else __import__("json").dumps(result, default=str),
                    "step_id": step["id"],
                },
            )
            execute(
                conn,
                """
                INSERT INTO run_events(run_id, step_id, event_type, message, data)
                VALUES(:run_id, :step_id, :event_type, :message, CAST(:data AS jsonb))
                """,
                {
                    "run_id": step["run_id"],
                    "step_id": step["id"],
                    "event_type": "step_done" if ok else "step_failed",
                    "message": error if error else f"{step['step_type']} {status}",
                    "data": "{}" if result is None else __import__("json").dumps(result, default=str),
                },
            )

    def _run_step(self, step: dict[str, Any]) -> tuple[dict[str, Any], bool, dict[str, Any] | None, str]:
        try:
            if step["step_type"] == "ticketing":
                result = run_ticketing_step(
                    run_id=str(step["run_id"]),
                    customer_id=int(step["customer_id"]),
                    step_id=int(step["id"]),
                    worker_id=self.state.worker_id,
                )
            elif step["step_type"] == "message":
                result = run_message_step(
                    run_id=str(step["run_id"]),
                    ticket_id=int(step["ticket_id"]),
                    step_id=int(step["id"]),
                    worker_id=self.state.worker_id,
                )
            elif step["step_type"] == "ticket_cx":
                result = run_ticket_cx_step(
                    run_id=str(step["run_id"]),
                    ticket_id=int(step["ticket_id"]),
                    step_id=int(step["id"]),
                    worker_id=self.state.worker_id,
                )
            else:
                raise ValueError(f"Unknown step_type {step['step_type']}")
            return step, True, result, ""
        except Exception as exc:  # noqa: BLE001
            return step, False, None, str(exc)

    def _submit_available(self, executor: ThreadPoolExecutor, futures: dict[Future, str]) -> bool:
        submitted = False
        for step_type in ("ticket_cx", "message", "ticketing"):
            while self._can_submit(step_type):
                step = self._claim_step(step_type)
                if not step:
                    break
                self._inc_active(step_type, 1)
                future = executor.submit(self._run_step, step)
                futures[future] = step_type
                submitted = True
                self._set_event(f"Submitted {step_type} step {step['id']}")
        return submitted

    def _finish_run_if_terminal(self) -> bool:
        if not self.state.run_id:
            return False
        with tx() as conn:
            execute(
                conn,
                "SELECT pg_advisory_xact_lock(hashtext(:lock_key))",
                {"lock_key": f"cx_pipeline_finish_run_{self.state.run_id}"},
            )
            run = fetch_one(
                conn,
                "SELECT * FROM pipeline_runs WHERE id=CAST(:run_id AS uuid) FOR UPDATE",
                {"run_id": self.state.run_id},
            )
            if not run or run.get("status") in {"finished", "failed", "interrupted"}:
                return True
            counts = fetch_all(
                conn,
                """
                SELECT status, count(*) AS count
                FROM run_steps
                WHERE run_id=CAST(:run_id AS uuid)
                GROUP BY status
                """,
                {"run_id": self.state.run_id},
            )
            status_counts = {str(row["status"]): int(row["count"]) for row in counts}
            active = status_counts.get("pending", 0) + status_counts.get("running", 0)
            if active > 0:
                return False

            failed = status_counts.get("failed", 0)
            terminal_status = "failed" if failed else "finished"
            stats = run_kpis(conn, self.state.run_id)
            error = f"{failed} terminal step(s) failed" if failed else ""
            execute(
                conn,
                """
                UPDATE pipeline_runs
                SET status=:status, finished_at=now(), error=:error
                WHERE id=CAST(:run_id AS uuid)
                """,
                {"run_id": self.state.run_id, "status": terminal_status, "error": error},
            )
            execute(
                conn,
                """
                INSERT INTO run_events(run_id, event_type, message, data)
                VALUES(CAST(:run_id AS uuid), 'run_finished', :message, CAST(:data AS jsonb))
                """,
                {
                    "run_id": self.state.run_id,
                    "message": f"Run {terminal_status}",
                    "data": as_json({"status": terminal_status, "stats": stats, "error": error}),
                },
            )

        notified = notify_dashboard_run_finished(
            run_id=self.state.run_id,
            status=terminal_status,
            stats=stats,
            error=error,
            settings=self.settings,
        )
        self._set_event(f"Run {terminal_status}; dashboard webhook {'sent' if notified else 'not sent'}")
        return True

    def run_forever(self, *, idle_sleep: float = 2.0) -> None:
        self.state.running = True
        self._set_event("Worker started")
        try:
            with tx() as conn:
                if self.state.run_id:
                    execute(
                        conn,
                        "UPDATE pipeline_runs SET status='running', started_at=COALESCE(started_at, now()) WHERE id=CAST(:run_id AS uuid)",
                        {"run_id": self.state.run_id},
                    )
            with ThreadPoolExecutor(max_workers=max(1, self.settings.max_total_workers)) as executor:
                futures: dict[Future, str] = {}
                while not self.state.stop_requested:
                    submitted = self._submit_available(executor, futures)
                    if not futures:
                        if not submitted:
                            if self._finish_run_if_terminal():
                                self.state.stop_requested = True
                                break
                            self._set_event("Idle: no pending steps")
                            time.sleep(idle_sleep)
                        continue
                    done, _ = wait(set(futures), timeout=1.0, return_when=FIRST_COMPLETED)
                    for future in done:
                        step_type = futures.pop(future)
                        self._inc_active(step_type, -1)
                        step, ok, result, error = future.result()
                        self._finish_step(step, ok=ok, result=result, error=error)
                        self._mark_done(failed=not ok)
                        self._set_event(f"Finished {step_type} step {step['id']} status={'ok' if ok else 'failed'}")
        finally:
            self.state.running = False
            self._set_event("Worker stopped")
