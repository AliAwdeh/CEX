from __future__ import annotations

import argparse
import json
import signal

from .app.analysis import enqueue_missing_downstream, enqueue_ticketing_steps
from .app.data import load_csv, validate_dataframe
from .app.db import as_json, execute, fetch_one, init_db, tx
from .app.ingest import ingest_csv, latest_input_csv
from .app.shutdown import hard_exit
from .app.worker import PipelineWorker


def cmd_init_db(_: argparse.Namespace) -> None:
    init_db()
    print("Initialized PostgreSQL schema.")


def cmd_create_run(args: argparse.Namespace) -> None:
    init_db()
    with tx() as conn:
        run = fetch_one(
            conn,
            """
            INSERT INTO pipeline_runs(name, mode, status, config, csv_path)
            VALUES(:name, :mode, 'created', CAST(:config AS jsonb), :csv_path)
            RETURNING *
            """,
            {
                "name": args.name,
                "mode": args.mode,
                "config": as_json({}),
                "csv_path": args.csv,
            },
        )
        run_id = str(run["id"])
        ingest_counts = ingest_csv(
            conn,
            csv_path=args.csv,
            run_id=run_id,
            random_journeys=args.random_journeys,
            random_seed=args.random_seed,
            input_dir=args.input_dir,
        )
        ticketing_steps = enqueue_ticketing_steps(conn, run_id=run_id)
        downstream = enqueue_missing_downstream(conn, run_id=run_id) if args.mode == "full" else {}
        execute(
            conn,
            """
            INSERT INTO run_events(run_id, event_type, message, data)
            VALUES(CAST(:run_id AS uuid), 'run_created', 'Run created from CLI', CAST(:data AS jsonb))
            """,
            {
                "run_id": run_id,
                "data": as_json({"ingest": ingest_counts, "ticketing_steps": ticketing_steps, **downstream}),
            },
        )
    print(json.dumps({"run_id": run_id, "ingest": ingest_counts, "ticketing_steps": ticketing_steps, **downstream}, indent=2))


def cmd_validate_input(args: argparse.Namespace) -> None:
    path = args.csv or latest_input_csv(args.input_dir)
    report = validate_dataframe(load_csv(path))
    print(json.dumps({"csv_path": str(path), "validation": report.to_dict()}, indent=2))


def cmd_work(args: argparse.Namespace) -> None:
    init_db()
    worker = PipelineWorker(run_id=args.run_id)

    def kill_switch(signum: int, _: object) -> None:
        worker.stop()
        hard_exit(
            worker_id=worker.state.worker_id,
            run_id=args.run_id,
            reason=f"signal {signum} kill switch",
        )

    signal.signal(signal.SIGINT, kill_switch)
    signal.signal(signal.SIGTERM, kill_switch)
    try:
        worker.run_forever(idle_sleep=float(args.idle_sleep))
    except KeyboardInterrupt:
        worker.stop()
        hard_exit(
            worker_id=worker.state.worker_id,
            run_id=args.run_id,
            reason="keyboard interrupt kill switch",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="CX Pipeline controls")
    sub = parser.add_subparsers(required=True)

    init_p = sub.add_parser("init-db", help="Create/upgrade PostgreSQL tables")
    init_p.set_defaults(func=cmd_init_db)

    run_p = sub.add_parser("create-run", help="Create a run and enqueue work")
    run_p.add_argument("--csv", help="CSV path to ingest")
    run_p.add_argument("--input-dir", help="Folder to load the newest CSV from")
    run_p.add_argument("--random-journeys", type=int, default=None, help="Run only a proportional random sample of journeys")
    run_p.add_argument("--random-seed", type=int, default=None, help="Seed for repeatable random journey sampling")
    run_p.add_argument("--name", default=None)
    run_p.add_argument("--mode", choices=["ticket_only", "full"], default="full")
    run_p.set_defaults(func=cmd_create_run)

    validate_p = sub.add_parser("validate-input", help="Validate a CSV without creating a run")
    validate_p.add_argument("--csv", default=None)
    validate_p.add_argument("--input-dir", default=None)
    validate_p.set_defaults(func=cmd_validate_input)

    work_p = sub.add_parser("work", help="Run workers in this process")
    work_p.add_argument("--run-id", default=None)
    work_p.add_argument("--idle-sleep", default=2.0)
    work_p.set_defaults(func=cmd_work)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
