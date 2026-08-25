from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cx_pipeline.app.db import fetch_all, init_db, tx


st.set_page_config(
    page_title="CX Pipeline Run",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DONE = "#2f9e44"
RUNNING = "#1c7ed6"
PENDING = "#adb5bd"
FAILED = "#e03131"
WARN = "#e8590c"

STAGES = [
    ("ticketing", "Ticketing", "per customer"),
    ("message", "Message", "per ticket"),
    ("ticket_cx", "Ticket CX", "per ticket"),
]

# Requests slower than this are called out as possibly stuck. The client gives
# up at OPENAI_TIMEOUT (600s by default), so this is deliberately well under it.
SLOW_REQUEST_SECONDS = 180


st.markdown(
    """
    <style>
      [data-testid="stMainBlockContainer"] {
        padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px;
      }
      [data-testid="stHeader"] {background: transparent;}
      footer {visibility: hidden;}

      .hero {display:flex; flex-wrap:wrap; align-items:baseline; gap:.55rem; margin:0 0 .15rem;}
      .hero .name {font-size:1.3rem; font-weight:650; line-height:1.3;}
      .hero .sub {font-size:.78rem; opacity:.55; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}

      .pill {display:inline-flex; align-items:center; gap:.35rem; padding:.12rem .55rem;
             border-radius:999px; font-size:.75rem; font-weight:600; line-height:1.6;
             border:1px solid currentColor;}
      .pill .dot {width:.45rem; height:.45rem; border-radius:50%; background:currentColor;}

      .bar {height:.5rem; border-radius:999px; background:rgba(128,128,128,.18);
            overflow:hidden; display:flex; margin:.5rem 0 .35rem;}
      .bar > i {display:block; height:100%;}

      .strip {display:flex; flex-wrap:wrap; gap:.35rem 1.15rem; margin:.15rem 0 .1rem;}
      .strip .s {display:flex; flex-direction:column; min-width:72px;}
      /* inside the narrow stage cards, lay the stats out on an even grid so
         they never wrap into a ragged last row */
      .strip.grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(58px, 1fr));
                   gap:.35rem .6rem;}
      .strip.grid .s {min-width:0;}
      .strip .k {font-size:.66rem; text-transform:uppercase; letter-spacing:.045em; opacity:.55;}
      .strip .v {font-size:1.02rem; font-weight:650; font-variant-numeric:tabular-nums; line-height:1.35;}

      .card-title {display:flex; align-items:center; gap:.4rem; font-weight:650; font-size:.92rem;}
      .card-title .tag {font-size:.68rem; opacity:.5; font-weight:500;}

      .empty {opacity:.55; font-size:.85rem; padding:.3rem 0;}

      @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
          padding-left:.75rem; padding-right:.75rem; padding-top:3rem;
        }
        [data-testid="stHorizontalBlock"] {flex-wrap:wrap; gap:.5rem;}
        [data-testid="stColumn"] {min-width:100% !important; flex:1 1 100% !important;}
        /* keep the run controls compact: picker on its own row, the two small
           controls side by side under it, instead of three stacked rows */
        .st-key-controls [data-testid="stColumn"] {min-width:0 !important;}
        .st-key-controls [data-testid="stColumn"]:first-child {flex:1 1 100% !important;}
        .st-key-controls [data-testid="stColumn"]:not(:first-child) {flex:1 1 40% !important;}
        .hero .name {font-size:1.1rem;}
        .strip {gap:.3rem .9rem;}
        .strip .s {min-width:62px;}
        .strip .v {font-size:.95rem;}
        [data-testid="stMetricValue"] {font-size:1.35rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def q(sql: str, params: dict | None = None) -> pd.DataFrame:
    with tx() as conn:
        return pd.DataFrame(fetch_all(conn, sql, params or {}))


def fmt_int(value) -> str:
    try:
        return f"{int(value or 0):,}"
    except (TypeError, ValueError):
        return "0"


def fmt_seconds(seconds) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "-"
    if total < 0:
        return "-"
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def pill(label: str, color: str) -> str:
    return f'<span class="pill" style="color:{color}"><span class="dot"></span>{label}</span>'


def bar(segments: list[tuple[float, str]]) -> str:
    parts = "".join(
        f'<i style="width:{max(0.0, frac) * 100:.4f}%;background:{color}"></i>'
        for frac, color in segments
        if frac and frac > 0
    )
    return f'<div class="bar">{parts}</div>'


def strip(items: list[tuple[str, str, str | None]], grid: bool = False) -> str:
    cells = "".join(
        f'<div class="s"><span class="k">{key}</span>'
        f'<span class="v"{f" style=color:{color}" if color else ""}>{value}</span></div>'
        for key, value, color in items
    )
    return f'<div class="strip{" grid" if grid else ""}">{cells}</div>'


def status_pill(status: str, has_failures: bool) -> str:
    status = (status or "").lower()
    if status == "running":
        return pill("Running", RUNNING)
    if status == "finished":
        return pill("Finished", FAILED if has_failures else DONE)
    if status == "failed":
        return pill("Failed", FAILED)
    if status == "interrupted":
        return pill("Interrupted", WARN)
    return pill(status.title() or "Created", PENDING)


init_db()

runs = q(
    """
    SELECT id, name, status, mode, created_at, started_at, finished_at, error
    FROM pipeline_runs
    ORDER BY created_at DESC
    LIMIT 100
    """
)

if runs.empty:
    st.markdown('<div class="hero"><span class="name">CX Pipeline</span></div>', unsafe_allow_html=True)
    st.info("No runs yet. Create one with `POST /runs` or `python -m cx_pipeline.cli create-run`.")
    st.stop()


def run_label(row) -> str:
    created = pd.to_datetime(row.created_at).strftime("%b %d %H:%M")
    return f"{row.name or 'Untitled'} · {row.status} · {created} · {str(row.id)[:8]}"


with st.container(key="controls"):
    controls = st.columns([4, 1.4, 1.2], vertical_alignment="center")
    with controls[0]:
        labels = [run_label(row) for row in runs.itertuples()]
        choice = st.selectbox("Run", labels, label_visibility="collapsed", key="run_choice")
        run_id = str(runs.iloc[labels.index(choice)]["id"])
    with controls[1]:
        auto = st.toggle("Live", value=True, key="auto_refresh")
    with controls[2]:
        every = st.selectbox("Every", [2, 5, 10, 30], index=1, format_func=lambda s: f"{s}s",
                             label_visibility="collapsed", key="refresh_every")


def render(run_id: str) -> None:
    run = q(
        """
        SELECT name, mode, status, error, created_at, started_at, finished_at,
               EXTRACT(EPOCH FROM (COALESCE(finished_at, now()) - COALESCE(started_at, created_at)))
                   AS elapsed_seconds
        FROM pipeline_runs
        WHERE id = CAST(:rid AS uuid)
        """,
        {"rid": run_id},
    )
    if run.empty:
        st.warning("That run no longer exists.")
        return
    run = run.iloc[0]

    steps = q(
        """
        SELECT step_type, status, count(*) AS n
        FROM run_steps
        WHERE run_id = CAST(:rid AS uuid)
        GROUP BY step_type, status
        """,
        {"rid": run_id},
    )
    by_stage: dict[str, dict[str, int]] = {key: {} for key, _, _ in STAGES}
    for row in steps.itertuples():
        by_stage.setdefault(row.step_type, {})[row.status] = int(row.n)

    def total_of(status: str) -> int:
        return sum(counts.get(status, 0) for counts in by_stage.values())

    done = total_of("done")
    running = total_of("running")
    pending = total_of("pending")
    failed = total_of("failed")
    known = done + running + pending + failed

    # Steps finished in the last 10 minutes give the live rate. During a lull
    # that window can be empty, so fall back to the run's overall average rather
    # than dropping the ETA entirely. The queue grows as message steps enqueue
    # ticket_cx steps, so this is an ETA for work already queued, not the run.
    rate = q(
        """
        SELECT
            count(*) FILTER (WHERE finished_at >= now() - interval '10 minutes') AS recent,
            count(*) AS total_done,
            EXTRACT(EPOCH FROM (max(finished_at) - min(started_at))) AS span_seconds
        FROM run_steps
        WHERE run_id = CAST(:rid AS uuid) AND status = 'done'
        """,
        {"rid": run_id},
    )
    per_minute = 0.0
    if not rate.empty:
        recent = int(rate.iloc[0]["recent"] or 0)
        total_done = int(rate.iloc[0]["total_done"] or 0)
        span = float(rate.iloc[0]["span_seconds"] or 0)
        if recent:
            per_minute = recent / 10.0
        elif total_done and span > 0:
            per_minute = total_done / (span / 60.0)
    remaining = pending + running
    eta = fmt_seconds(remaining / per_minute * 60) if per_minute > 0 and remaining else "-"

    st.markdown(
        f'<div class="hero"><span class="name">{run["name"] or "Untitled run"}</span>'
        f'{status_pill(str(run["status"]), failed > 0)}'
        f'<span class="sub">{run["mode"]} · {run_id[:8]}</span></div>',
        unsafe_allow_html=True,
    )

    if known:
        st.markdown(
            bar(
                [
                    (done / known, DONE),
                    (running / known, RUNNING),
                    (failed / known, FAILED),
                    (pending / known, PENDING),
                ]
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        strip(
            [
                ("Progress", f"{(done / known * 100) if known else 0:.0f}%", None),
                ("Done", f"{fmt_int(done)} / {fmt_int(known)}", None),
                ("Running", fmt_int(running), RUNNING if running else None),
                ("Pending", fmt_int(pending), None),
                ("Failed", fmt_int(failed), FAILED if failed else None),
                ("Elapsed", fmt_seconds(run["elapsed_seconds"]), None),
                ("Steps/min", f"{per_minute:.1f}", None),
                ("ETA (queued)", eta, None),
            ]
        ),
        unsafe_allow_html=True,
    )

    if run["error"]:
        st.error(str(run["error"]))

    inflight = q(
        """
        SELECT
            layer,
            model,
            worker_id,
            context,
            ticket_id,
            EXTRACT(EPOCH FROM (now() - started_at)) AS elapsed_seconds
        FROM ai_requests
        WHERE run_id = CAST(:rid AS uuid) AND status = 'running'
        ORDER BY started_at ASC
        """,
        {"rid": run_id},
    )

    if pending and not running and inflight.empty:
        st.warning(
            f"{fmt_int(pending)} step(s) pending but nothing is running — no worker is attached. "
            "Start one with `POST /runs/{id}/workers/start` or `cli work --run-id`."
        )

    stale = 0
    if not inflight.empty:
        stale = int((inflight["elapsed_seconds"] > SLOW_REQUEST_SECONDS).sum())
        if stale:
            st.warning(f"{stale} request(s) running longer than {SLOW_REQUEST_SECONDS}s.")

    st.write("")
    for col, (key, title, tag) in zip(st.columns(3), STAGES):
        counts = by_stage.get(key, {})
        s_done = counts.get("done", 0)
        s_running = counts.get("running", 0)
        s_pending = counts.get("pending", 0)
        s_failed = counts.get("failed", 0)
        s_known = s_done + s_running + s_pending + s_failed
        with col, st.container(border=True):
            st.markdown(
                f'<div class="card-title">{title}<span class="tag">{tag}</span></div>',
                unsafe_allow_html=True,
            )
            if s_known:
                st.markdown(
                    bar(
                        [
                            (s_done / s_known, DONE),
                            (s_running / s_known, RUNNING),
                            (s_failed / s_known, FAILED),
                            (s_pending / s_known, PENDING),
                        ]
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="empty">Nothing queued yet.</div>', unsafe_allow_html=True)
            st.markdown(
                strip(
                    [
                        ("Done", f"{fmt_int(s_done)}/{fmt_int(s_known)}", None),
                        ("Running", fmt_int(s_running), RUNNING if s_running else None),
                        ("Pending", fmt_int(s_pending), None),
                        ("Failed", fmt_int(s_failed), FAILED if s_failed else None),
                    ],
                    grid=True,
                ),
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("##### In flight")
    if inflight.empty:
        st.markdown('<div class="empty">No AI requests running right now.</div>', unsafe_allow_html=True)
    else:
        view = inflight.rename(
            columns={
                "layer": "Layer",
                "model": "Model",
                "worker_id": "Worker",
                "context": "Context",
                "ticket_id": "Ticket",
                "elapsed_seconds": "Elapsed",
            }
        )
        st.dataframe(
            view,
            hide_index=True,
            height=min(38 * (len(view) + 1) + 3, 320),
            column_config={
                "Elapsed": st.column_config.ProgressColumn(
                    "Elapsed",
                    format="%.0fs",
                    min_value=0,
                    max_value=float(max(view["Elapsed"].max(), SLOW_REQUEST_SECONDS)),
                ),
                "Ticket": st.column_config.NumberColumn("Ticket", format="%d"),
                "Worker": st.column_config.TextColumn("Worker", width="small"),
                "Context": st.column_config.TextColumn("Context", width="medium"),
            },
        )

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("##### Throughput")
        thr = q(
            """
            SELECT date_trunc('minute', finished_at) AS minute, layer, count(*) AS calls
            FROM ai_requests
            WHERE run_id = CAST(:rid AS uuid)
              AND finished_at >= now() - interval '45 minutes'
            GROUP BY 1, 2
            ORDER BY 1
            """,
            {"rid": run_id},
        )
        if thr.empty:
            st.markdown('<div class="empty">No finished requests in the last 45 minutes.</div>',
                        unsafe_allow_html=True)
        else:
            st.area_chart(
                thr.pivot(index="minute", columns="layer", values="calls").fillna(0),
                height=190,
                color=[RUNNING, DONE, WARN][: thr["layer"].nunique()],
            )

    with right:
        st.markdown("##### Latency by layer")
        lat = q(
            """
            SELECT
                layer,
                count(*) AS calls,
                count(*) FILTER (WHERE status = 'failed') AS failed,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_seconds)::numeric, 1) AS p50,
                round(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_seconds)::numeric, 1) AS p95
            FROM ai_requests
            WHERE run_id = CAST(:rid AS uuid) AND status <> 'running'
            GROUP BY layer
            ORDER BY layer
            """,
            {"rid": run_id},
        )
        if lat.empty:
            st.markdown('<div class="empty">No finished requests yet.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(
                lat.rename(
                    columns={
                        "layer": "Layer",
                        "calls": "Calls",
                        "failed": "Failed",
                        "p50": "p50",
                        "p95": "p95",
                    }
                ),
                hide_index=True,
                column_config={
                    "p50": st.column_config.NumberColumn("p50", format="%.1fs"),
                    "p95": st.column_config.NumberColumn("p95", format="%.1fs"),
                },
            )

    st.markdown("##### Workers")
    workers = q(
        """
        WITH active AS (
            SELECT
                locked_by AS worker,
                count(*) AS active_steps,
                count(*) FILTER (WHERE step_type = 'ticketing') AS ticketing,
                count(*) FILTER (WHERE step_type = 'message') AS message,
                count(*) FILTER (WHERE step_type = 'ticket_cx') AS ticket_cx,
                EXTRACT(EPOCH FROM (now() - min(started_at))) AS oldest_seconds
            FROM run_steps
            WHERE run_id = CAST(:rid AS uuid) AND status = 'running' AND locked_by IS NOT NULL
            GROUP BY locked_by
        ),
        calls AS (
            SELECT
                worker_id AS worker,
                count(*) FILTER (WHERE status = 'success') AS ok,
                count(*) FILTER (WHERE status = 'failed') AS failed
            FROM ai_requests
            WHERE run_id = CAST(:rid AS uuid) AND worker_id IS NOT NULL
            GROUP BY worker_id
        )
        SELECT
            COALESCE(a.worker, c.worker) AS worker,
            COALESCE(a.active_steps, 0) AS active,
            COALESCE(a.ticketing, 0) AS ticketing,
            COALESCE(a.message, 0) AS message,
            COALESCE(a.ticket_cx, 0) AS ticket_cx,
            round(a.oldest_seconds::numeric, 0) AS oldest_s,
            COALESCE(c.ok, 0) AS calls_ok,
            COALESCE(c.failed, 0) AS calls_failed
        FROM active a
        FULL OUTER JOIN calls c ON a.worker = c.worker
        ORDER BY active DESC, calls_ok DESC
        """,
        {"rid": run_id},
    )
    if workers.empty:
        st.markdown('<div class="empty">No worker has claimed a step in this run.</div>',
                    unsafe_allow_html=True)
    else:
        st.dataframe(
            workers.rename(
                columns={
                    "worker": "Worker",
                    "active": "Active",
                    "ticketing": "Tkt",
                    "message": "Msg",
                    "ticket_cx": "CX",
                    "oldest_s": "Oldest",
                    "calls_ok": "OK",
                    "calls_failed": "Failed",
                }
            ),
            hide_index=True,
            column_config={"Oldest": st.column_config.NumberColumn("Oldest", format="%.0fs")},
        )

    results = q(
        """
        SELECT
            COALESCE(result ->> 'handled_status', 'unknown') AS handled,
            COALESCE(result ->> 'customer_experience', 'unknown') AS experience,
            count(*) AS n
        FROM ticket_cx_results
        WHERE run_id = CAST(:rid AS uuid)
        GROUP BY 1, 2
        ORDER BY n DESC
        """,
        {"rid": run_id},
    )
    if not results.empty:
        st.markdown("##### Results so far")
        res_left, res_right = st.columns(2)
        with res_left:
            st.bar_chart(results.groupby("handled")["n"].sum(), height=170, horizontal=True)
        with res_right:
            st.bar_chart(results.groupby("experience")["n"].sum(), height=170, horizontal=True)

    if failed:
        with st.expander(f"Failures ({fmt_int(failed)})", expanded=failed > 0 and known < 50):
            st.dataframe(
                q(
                    """
                    SELECT step_type, attempts, max_attempts, ticket_id, customer_id,
                           finished_at, error
                    FROM run_steps
                    WHERE run_id = CAST(:rid AS uuid) AND status = 'failed'
                    ORDER BY finished_at DESC NULLS LAST
                    LIMIT 100
                    """,
                    {"rid": run_id},
                ),
                hide_index=True,
            )

    with st.expander("Finished requests"):
        st.dataframe(
            q(
                """
                SELECT layer, status, model, worker_id, context, ticket_id,
                       finished_at, round(duration_seconds::numeric, 1) AS duration_s, error
                FROM ai_requests
                WHERE run_id = CAST(:rid AS uuid) AND status <> 'running'
                ORDER BY finished_at DESC, id DESC
                LIMIT 200
                """,
                {"rid": run_id},
            ),
            hide_index=True,
            column_config={"duration_s": st.column_config.NumberColumn("Duration", format="%.1fs")},
        )

    with st.expander("Tickets touched by this run"):
        st.dataframe(
            q(
                """
                SELECT
                    t.id AS ticket,
                    c.external_customer_id AS customer,
                    t.status,
                    t.category,
                    t.ticket_type,
                    t.objective,
                    cx.result ->> 'handled_status' AS handled,
                    cx.result ->> 'customer_experience' AS experience,
                    t.needs_message_analysis AS needs_msg,
                    t.needs_ticket_cx AS needs_cx,
                    t.updated_at
                FROM tickets t
                JOIN customers c ON c.id = t.customer_id
                LEFT JOIN ticket_cx_results cx ON cx.ticket_id = t.id
                WHERE t.latest_ticketing_run_id = CAST(:rid AS uuid)
                   OR cx.run_id = CAST(:rid AS uuid)
                ORDER BY t.updated_at DESC
                LIMIT 300
                """,
                {"rid": run_id},
            ),
            hide_index=True,
        )

    with st.expander("Events"):
        st.dataframe(
            q(
                """
                SELECT created_at, event_type, message
                FROM run_events
                WHERE run_id = CAST(:rid AS uuid)
                ORDER BY created_at DESC, id DESC
                LIMIT 100
                """,
                {"rid": run_id},
            ),
            hide_index=True,
        )


live = st.fragment(run_every=f"{every}s" if auto else None)(render)
live(run_id)
