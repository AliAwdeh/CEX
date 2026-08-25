from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analysis import enqueue_missing_downstream, enqueue_ticketing_steps, run_kpis
from .config import get_settings
from .data import load_csv, validate_dataframe
from .db import as_json, execute, fetch_all, fetch_one, init_db, tx
from .ingest import CSV_EXTENSIONS, ingest_csv, latest_input_csv
from .shutdown import hard_exit, mark_interrupted
from .worker import PipelineWorker


app = FastAPI(title="CX Pipeline API", version="0.1.0")
_workers: dict[str, PipelineWorker] = {}


class CreateRunRequest(BaseModel):
    name: str | None = None
    csv_path: str | None = None
    input_dir: str | None = None
    random_journeys: int | None = None
    random_seed: int | None = None
    mode: Literal["ticket_only", "full"] = "full"
    config: dict[str, Any] = Field(default_factory=dict)
    start_workers: bool = False


class EnqueueResponse(BaseModel):
    run_id: str
    ticketing_steps: int = 0
    message_steps: int = 0
    ticket_cx_steps: int = 0


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.on_event("shutdown")
def _shutdown() -> None:
    active_workers = [worker for worker in _workers.values() if worker.snapshot()["running"]]
    for worker in active_workers:
        snapshot = worker.snapshot()
        worker.stop()
        mark_interrupted(
            worker_id=snapshot["worker_id"],
            run_id=snapshot["run_id"],
            reason="api shutdown kill switch",
        )
    if active_workers:
        hard_exit(code=0, reason="api shutdown kill switch", cleanup=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/input-files")
def input_files(input_dir: str | None = None) -> dict[str, Any]:
    root = Path(input_dir or get_settings().input_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = [
        {
            "path": str(path),
            "name": path.name,
            "bytes": path.stat().st_size,
            "modified_at": path.stat().st_mtime,
        }
        for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
        if path.is_file() and path.suffix.lower() in CSV_EXTENSIONS and not path.name.startswith(".")
    ]
    return {"input_dir": str(root), "files": files}


@app.get("/input-files/validate")
def validate_input_file(csv_path: str | None = None, input_dir: str | None = None) -> dict[str, Any]:
    try:
        path = Path(csv_path) if csv_path else latest_input_csv(input_dir)
        report = validate_dataframe(load_csv(path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"csv_path": str(path), "validation": report.to_dict()}


@app.post("/runs")
def create_run(payload: CreateRunRequest) -> dict[str, Any]:
    with tx() as conn:
        run = fetch_one(
            conn,
            """
            INSERT INTO pipeline_runs(name, mode, status, config, csv_path)
            VALUES(:name, :mode, 'created', CAST(:config AS jsonb), :csv_path)
            RETURNING *
            """,
            {
                "name": payload.name,
                "mode": payload.mode,
                "config": as_json(payload.config),
                "csv_path": payload.csv_path,
            },
        )
        run_id = str(run["id"])
        ingest_counts = ingest_csv(
            conn,
            csv_path=payload.csv_path,
            run_id=run_id,
            random_journeys=payload.random_journeys,
            random_seed=payload.random_seed,
            input_dir=payload.input_dir,
        )
        ticketing_steps = enqueue_ticketing_steps(conn, run_id=run_id)
        downstream = enqueue_missing_downstream(conn, run_id=run_id) if payload.mode == "full" else {}
        execute(
            conn,
            """
            INSERT INTO run_events(run_id, event_type, message, data)
            VALUES(CAST(:run_id AS uuid), 'run_created', 'Run created', CAST(:data AS jsonb))
            """,
            {
                "run_id": run_id,
                "data": as_json(
                    {
                        "ingest": ingest_counts,
                        "ticketing_steps": ticketing_steps,
                        **downstream,
                    }
                ),
            },
        )
    if payload.start_workers:
        start_workers(run_id)
    return {
        "run_id": run_id,
        "run": run,
        "ingest": ingest_counts,
        "ticketing_steps": ticketing_steps,
        **downstream,
    }


@app.post("/runs/{run_id}/enqueue", response_model=EnqueueResponse)
def enqueue_run(run_id: str) -> EnqueueResponse:
    with tx() as conn:
        run = fetch_one(conn, "SELECT * FROM pipeline_runs WHERE id=CAST(:run_id AS uuid)", {"run_id": run_id})
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        ticketing_steps = enqueue_ticketing_steps(conn, run_id=run_id)
        downstream = enqueue_missing_downstream(conn, run_id=run_id)
    return EnqueueResponse(
        run_id=run_id,
        ticketing_steps=ticketing_steps,
        message_steps=int(downstream.get("message_steps") or 0),
        ticket_cx_steps=int(downstream.get("ticket_cx_steps") or 0),
    )


@app.post("/runs/{run_id}/workers/start")
def start_workers(run_id: str) -> dict[str, Any]:
    if run_id in _workers and _workers[run_id].snapshot()["running"]:
        return _workers[run_id].snapshot()
    worker = PipelineWorker(run_id=run_id)
    _workers[run_id] = worker
    worker.start_background()
    return worker.snapshot()


@app.post("/runs/{run_id}/workers/stop")
def stop_workers(run_id: str) -> dict[str, Any]:
    worker = _workers.get(run_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker.stop()
    return worker.snapshot()


@app.get("/workers")
def workers() -> dict[str, Any]:
    return {"workers": [worker.snapshot() for worker in _workers.values()]}


@app.get("/ai-requests")
def ai_requests(
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    where = ["1=1"]
    params: dict[str, Any] = {
        "limit": max(1, min(int(limit), 1000)),
        "offset": max(0, int(offset)),
    }
    if run_id:
        where.append("run_id=CAST(:run_id AS uuid)")
        params["run_id"] = run_id
    if status:
        where.append("status=:status")
        params["status"] = status
    with tx() as conn:
        rows = fetch_all(
            conn,
            f"""
            SELECT
                *,
                CASE
                    WHEN status='running' THEN EXTRACT(EPOCH FROM (now() - started_at))::double precision
                    ELSE duration_seconds::double precision
                END AS elapsed_seconds
            FROM ai_requests
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(finished_at, started_at) DESC, id DESC
            LIMIT :limit OFFSET :offset
            """,
            params,
        )
    return {"ai_requests": rows}


@app.get("/runs")
def list_runs(limit: int = 50) -> dict[str, Any]:
    with tx() as conn:
        rows = fetch_all(
            conn,
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT :limit",
            {"limit": max(1, min(int(limit), 500))},
        )
    return {"runs": rows}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with tx() as conn:
        run = fetch_one(conn, "SELECT * FROM pipeline_runs WHERE id=CAST(:run_id AS uuid)", {"run_id": run_id})
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run": run, "stats": run_kpis(conn, run_id)}


@app.get("/runs/{run_id}/stats")
def get_run_stats(run_id: str) -> dict[str, Any]:
    with tx() as conn:
        return run_kpis(conn, run_id)


@app.get("/tickets")
def list_tickets(customer_id: int | None = None, limit: int = 500) -> dict[str, Any]:
    where = "WHERE t.customer_id=:customer_id" if customer_id is not None else ""
    params: dict[str, Any] = {"limit": max(1, min(int(limit), 5000))}
    if customer_id is not None:
        params["customer_id"] = customer_id
    with tx() as conn:
        rows = fetch_all(
            conn,
            f"""
            SELECT
                t.id,
                c.external_customer_id,
                c.customer_name,
                t.status,
                t.category,
                t.ticket_type,
                t.objective,
                t.should_append_future,
                t.needs_message_analysis,
                t.needs_ticket_cx,
                t.ticket_message_count,
                t.opened_at,
                t.last_message_at,
                t.closed_at,
                t.reopenable_until,
                t.lifecycle_risk,
                t.lifecycle_reason,
                t.analysis_eligible,
                t.analysis_skip_reason,
                cx.result->>'handled_status' AS handled_status,
                cx.result->>'customer_experience' AS customer_experience,
                cx.parse_status AS ticket_cx_parse_status,
                count(tm.message_id) AS message_count,
                t.updated_at
            FROM tickets t
            JOIN customers c ON c.id=t.customer_id
            LEFT JOIN ticket_messages tm ON tm.ticket_id=t.id
            LEFT JOIN ticket_cx_results cx ON cx.ticket_id=t.id
            {where}
            GROUP BY t.id, c.id, cx.id
            ORDER BY t.updated_at DESC
            LIMIT :limit
            """,
            params,
        )
    return {"tickets": rows}


@app.get("/events")
def events(run_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    where = "WHERE (:run_id IS NULL OR run_id=CAST(:run_id AS uuid))"
    with tx() as conn:
        rows = fetch_all(
            conn,
            f"""
            SELECT *
            FROM run_events
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT :limit
            """,
            {"run_id": run_id, "limit": max(1, min(int(limit), 1000))},
        )
    return {"events": rows}
