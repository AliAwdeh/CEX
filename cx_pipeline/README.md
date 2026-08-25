# CX Pipeline

Durable PostgreSQL-backed runner for heavy conversation analysis.

This project is standalone. It does not import the old Streamlit app or its
runtime modules. The LLM client, prompt templates, CSV normalization, ticket
segmentation, message judge, ticket CX judge, metadata computation, API,
workers, and dashboard all live inside `cx_pipeline`.

Prompt templates are loaded by `cx_pipeline/app/prompts.py` from this project's
own copy of `cx_pipeline/correct_prompt_files/`. These are the same prompt
files used by the old CX platform, copied here so this backend is standalone.

## Flow

1. **Ticketing**
   - Ingests new CSV rows into PostgreSQL.
   - Deduplicates source conversations by customer, source conversation id, and
     content hash.
   - Runs ticket segmentation only for customers with new source conversations.
   - Appends new messages to existing tickets when the model links them to old
     ticket context, otherwise creates new tickets.

2. **Message Analysis**
   - Runs only for ticket messages without a saved message result.
   - Runs sequentially inside each ticket because every target message needs
     prior ticket history.
   - Runs concurrently across different tickets.

3. **Ticket CX Analysis**
   - Runs after message analysis finishes for a ticket.
   - Produces the ticket-level CX result: handled/unhandled, bad/good,
     unresolved subtype, scoring signals, summary, actions, and related fields.

## Quick Start

Create a local database:

```bash
createdb cex_pipeline
```

Copy environment values:

```bash
cp cx_pipeline/.env.example cx_pipeline/.env
```

Edit `cx_pipeline/.env` and set:

```bash
DATABASE_URL=postgresql+psycopg://localhost:5432/cex_pipeline
INPUT_CSV_DIR=./cx_pipeline/data/inbox
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://langcc.maidstech.ai/v1
OPENAI_TIMEOUT=600
```

Put input CSV files in:

```bash
cx_pipeline/data/inbox/
```

When a run is created without `csv_path`, the newest CSV in that folder is used.

Initialize tables:

```bash
python -m cx_pipeline.cli init-db
```

Start the API and dashboard with the kill-switch launcher:

```bash
python -m cx_pipeline.run_app
```

Press `Ctrl-C` in this launcher to kill the API, dashboard, and active local workers immediately.

Start the standalone React ticket dashboard:

```bash
cd cx_pipeline/react_dashboard
npm run dev
```

React UI: `http://127.0.0.1:5173`

The React dashboard stats cache is event-driven. The Python worker calls the
dashboard webhook when a run reaches `finished` or `failed`, and the Express API
updates the persisted stats DB for the rows changed by that run.

Create a run:

```bash
curl -X POST http://localhost:8088/runs \
  -H 'Content-Type: application/json' \
  -d '{"name":"test run","mode":"full"}'
```

Create a random performance-test run from the newest inbox CSV:

```bash
curl -X POST http://localhost:8088/runs \
  -H 'Content-Type: application/json' \
  -d '{"name":"50 journey sample","mode":"full","random_journeys":50,"random_seed":42,"start_workers":true}'
```

Run workers from CLI instead of API background workers:

```bash
python -m cx_pipeline.cli work --run-id <RUN_UUID>
```

## API

- `POST /runs`: create a run, optionally ingest a CSV, enqueue work.
- `POST /runs/{run_id}/enqueue`: enqueue missing downstream work for an existing
  run.
- `POST /runs/{run_id}/workers/start`: start background workers inside the API
  process.
- `POST /runs/{run_id}/workers/stop`: stop background workers.
- `GET /runs/{run_id}`: run record and counters.
- `GET /runs/{run_id}/stats`: live step/ticket/message stats.
- `GET /workers`: active worker state.
- `GET /tickets`: compact ticket table for dashboards or a future React app.
- Dashboard webhook from worker completion:
  `POST http://127.0.0.1:8090/api/webhooks/analysis-run-finished`.

## Data-Waste Controls

- Input validation follows the old CX platform's required-column contract:
  `CUSTOMER_PHONE`, `APPENDED_MESSAGE_INDEX`, `MESSAGE_TIME`, `SENDER_ROLE`,
  and `MESSAGE_TEXT`.
- The validator adds a structured report for missing columns, dropped blank
  rows, invalid roles, invalid order values, duplicate per-journey message
  order values, journey count, and source conversation count.
- Exact source conversations are deduped by hash before any AI call.
- AI request timeout is 600 seconds by default. Running AI requests are persisted
  immediately and shown live in the dashboard with elapsed time; completed and
  failed requests move to a paged finished-request table.
- Ticketing receives only new source conversation blocks plus compact previous
  ticket state, not the full historical transcript.
- Message analysis stores integer `message_id` links and sends external IDs only
  when useful in metadata.
- Message analysis reruns only missing/new target messages per ticket.
- Ticket CX reruns only after a ticket receives new message-level output.
- Dashboard aggregates are stored in the React dashboard stats database and are
  recomputed by completion webhook, not during page rendering.
