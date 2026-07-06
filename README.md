# CX Conversation Evaluator

Streamlit application for reviewing customer journeys with an AI judge. It evaluates
individual messages, produces a journey-level assessment, stores runs in SQLite, and
provides management and reviewer views.

## What the application does

The application accepts a CSV containing one visible message per row. Rows are grouped
into customer journeys using `CUSTOMER_PHONE` and ordered using
`APPENDED_MESSAGE_INDEX`.

An evaluation run has two stages:

1. **Message-level evaluation** — evaluates each selected target message using the
   visible history up to that message. The target can be assistant messages or customer
   messages, depending on the run setting.
2. **Conversation-level evaluation** — evaluates the complete appended journey using
   the transcript, successful message-level evaluations, conversation metadata, and
   computed journey statistics.

The active prompt versions define the JSON requested from the model. The application
validates and normalizes known fields used by its dashboards while preserving additional
custom fields in saved results and JSON exports.

## Requirements and startup

- Python 3.10 or newer
- An OpenAI-compatible endpoint that supports chat completions
- An API key if required by that endpoint

Install and start the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The default API base URL is `https://langcc.maidstech.ai/v1`, and the default model
selection is `openai/gpt-5.4-mini`. Both can be changed by a master user.

## Authentication

On first launch, the application asks for a master key. The key can also be supplied
through the `CEX_MASTER_KEY` environment variable or Streamlit secrets:

```toml
# .streamlit/secrets.toml
CEX_MASTER_KEY = "replace-with-a-long-secret"
```

`.streamlit/secrets.toml` is ignored by Git.

A master user can generate and revoke reviewer access keys. Reviewer keys are stored as
salted hashes in SQLite; the plaintext key is shown only when it is created.

### Master access

Master users can access:

- Reviewer Admin
- Upload & Settings
- Prompts
- Run Evaluation
- Overview
- Stats
- Dashboard
- Journey Review
- Exports
- Debug

They also see database and API settings in the sidebar.

### Reviewer access

Reviewer users can access only:

- Overview
- Stats
- Dashboard
- Journey Review

The sidebar is hidden for reviewer accounts.

## Input CSV

The following columns are required:

| Column | Purpose |
| --- | --- |
| `CUSTOMER_PHONE` | Stable key for the complete appended customer journey |
| `APPENDED_MESSAGE_INDEX` | Message order within the journey |
| `MESSAGE_TIME` | Message timestamp |
| `SENDER_ROLE` | Normalized sender role, normally `customer`, `agent`, or `unknown` |
| `MESSAGE_TEXT` | Visible message text |

Useful optional columns include:

- `CONVERSATION_ID`
- `CUSTOMER_NAME`
- `CONVERSATION_IDS`
- `SOURCE_CONVERSATION_COUNT`
- `CONVERSATION_START_DATE`
- `CONVERSATION_END_DATE`
- `CONVERSATION_STATUS`
- `INITIAL_SKILL`
- `LAST_SKILL`
- `JOINED_SKILLS`
- `CONVERSATION_AGENT_FULL_NAME`
- `CONVERSATION_AGENT_LOGIN_NAME`
- `RAW_SENDER_ROLE`
- `MESSAGE_AGENT_FULL_NAME`
- `TOTAL_VISIBLE_MESSAGES`
- `CUSTOMER_MESSAGE_COUNT`
- `AGENT_MESSAGE_COUNT`

`CONVERSATION_ID` is retained as the source conversation ID. The application generates
an internal message ID in the form
`{CUSTOMER_PHONE}-{APPENDED_MESSAGE_INDEX}`.

`RAW_SENDER_ROLE` is used to distinguish human agents, bots, and system broadcasts in
review views. The current loader does not preserve a separate broadcast campaign or
template ID.

The uploaded file should already exclude tool calls, internal tool responses, hidden
analysis, and other rows that should not appear in the customer-visible transcript.

## Evaluation settings

Master users can configure:

- API base URL, API key, and model
- Temperature and top-p
- Maximum response tokens
- Request timeout and retry count
- Message-level concurrency, capped at 100
- Number of journeys to process or run all uploaded journeys
- Maximum target messages per journey
- Whether to evaluate assistant or customer messages
- Optional per-message text truncation
- Whether unknown sender messages are included in history
- Whether a run stops after an API error
- Whether raw model responses are saved

The application sends `temperature`, `top_p`, `max_tokens`, and `timeout` with chat
completion requests. It requests JSON response formatting when the provider supports it
and retries without that option when the provider explicitly rejects it.

The application does **not** currently send a `reasoning_effort` parameter.

## Prompts

Message-level and conversation-level prompts each contain:

- System prompt
- Output schema
- User prompt template

`{output_schema}` can be used inside the system prompt. If it is absent, the schema is
appended automatically. The user prompt template should contain `{payload_json}`.

Saving creates a new prompt version and can activate it for future runs. Prompt versions
used by a run are recorded with that run.

## Saved runs and databases

Runs are saved incrementally to SQLite, including:

- Run configuration
- Prompt version references
- Message-level results
- Conversation-level results
- Transcripts and metadata
- Raw responses when enabled
- Errors

The application uses `cx_evaluator_review_runs_59_57_55.db` when that local file exists;
otherwise it uses `cx_evaluator.db`. A master user can select another local `.db` file
from the sidebar.

SQLite files are intentionally ignored by Git. Pulling code does not provide, update, or
back up application data. Back up and deploy database files separately.

The newest loadable saved run is loaded automatically. Master users can also load,
rename, or delete saved runs from **Run Evaluation**.

## Review pages

### Overview

Summarizes journeys as:

- Handled
- Pending unresolved
- Totally unresolved

It also breaks these groups down by customer experience and frustration origin, then
lists detected issue types and origins.

### Stats

Shows journey counts and percentages. A separate table counts red, yellow, and green
evaluated messages for human agents, bots, and broadcasts.

- **Flagged rate** — red and yellow messages divided by evaluated messages for that
  sender type
- **Share of all flagged** — that sender type's red and yellow messages divided by all
  red and yellow messages

### Dashboard

Provides filtered charts and aggregates for outcomes, issue severity, issue origin,
frustration, and conversation flow.

### Journey Review

Allows reviewers to:

- Filter journeys by outcome, experience, unresolved status, frustration, issue origin,
  issue type, journey starter, human-review requirement, and date
- Search customer and conversation identifiers or summaries
- Sort by conversation score
- Filter for red, yellow, or green messages from agents, bots, and broadcasts
- Show only broadcast-only red-issue journeys
- Open journey analysis and review metrics
- Read the full transcript and inspect message-level assessment details
- Add review comments and mark a journey reviewed
- Move to the previous or next matching journey

General filter groups use AND logic. Selecting several values inside one multiselect uses
OR logic within that field. Agent, bot, and broadcast color filters also combine using
AND logic.

**Only broadcast-only issue journeys** is off by default. When selected, it shows only
journeys where the red issue came exclusively from a system/broadcast message. The
marker is recalculated from saved transcripts and message evaluations when result tables
are built, so it also works with older saved runs.

## Exports

Master users can download:

| File | Contents |
| --- | --- |
| `cx_journey_results.csv` | One row per evaluated customer journey |
| `cx_message_results.csv` | One row per evaluated target message |
| `cx_full_results.json` | Run configuration, journey results, message results, errors, and saved raw responses |

The Exports page also provides journey CSVs filtered into common outcome and customer
experience categories.

The Stats page separately provides:

- `cx_stats_summary.csv`
- `cx_message_flag_stats.csv`

## Local file structure

```text
app.py             Streamlit UI, authentication, navigation, and page orchestration
api_client.py      OpenAI-compatible client and retry behavior
prompts.py         Prompt templates and payload construction
db.py              SQLite schema and persistence
data_loader.py     CSV validation, normalization, grouping, and sampling
evaluator.py       Message-level and conversation-level evaluation
aggregation.py     Computed metadata and table/dashboard aggregation
cost_estimator.py  GPT-5 mini token and cost estimates
exports.py         CSV and JSON export helpers
ui_components.py   Shared review controls and visual components
requirements.txt   Python dependencies
```

## Troubleshooting

### CSV validation fails

Confirm all five required columns are present and use the exact uppercase names shown in
the input table above.

### Models cannot be loaded

Check the base URL, credentials, network access, and whether the endpoint implements
`GET /models`. A model ID can still be entered manually when no model list is available.

### A model call fails

Check the Debug page as a master user. Increase the timeout, lower concurrency, or allow
the run to continue after individual API errors.

### JSON parsing fails

The response is saved as a failed result rather than crashing the complete run. Review
the raw response in Debug and verify that the active prompt and provider produce a JSON
object matching the intended schema.

### The dashboard is empty

Load a saved run or complete a new evaluation run. Review pages read from the currently
loaded run.
