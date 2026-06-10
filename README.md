# CX Conversation Evaluator

AI-as-a-Judge web app for evaluating chatbot/customer conversations from the **customer's perspective**.

The app reads a CSV exported from Snowflake (one row per visible message), then runs a two-layer evaluation:

1. **Message-level**: every agent message is judged using the full visible history up to that point.
2. **Conversation-level**: a final pass over the full transcript, all message-level evaluations, and computed metadata.

The output answers management questions like:

- Was the conversation handled or unhandled?
- Did the customer experience many issues or minimal issues?
- What caused frustration?
- Was the issue from our side, the customer side, shared, or none?
- What should we fix?
- Which conversations need manual review?

---

## 1. Installation

```bash
# Requires Python 3.10+
pip install -r requirements.txt
```

## 2. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints in your browser.

---

## 3. Expected CSV structure

The CSV must contain one row per visible message and these columns:

```
CONVERSATION_ID
CONVERSATION_START_DATE
CONVERSATION_END_DATE
CONVERSATION_STATUS
INITIAL_SKILL
LAST_SKILL
JOINED_SKILLS
CONVERSATION_AGENT_FULL_NAME
CONVERSATION_AGENT_LOGIN_NAME
CUSTOMER_NAME
CUSTOMER_PHONE
MESSAGE_INDEX
MESSAGE_TIME
SENDER_ROLE              -- customer / agent / unknown
RAW_SENDER_ROLE
MESSAGE_AGENT_FULL_NAME
MESSAGE_TEXT
TOTAL_VISIBLE_MESSAGES
CUSTOMER_MESSAGE_COUNT
AGENT_MESSAGE_COUNT
```

The CSV is expected to be **clean**:
- Tool calls removed
- Internal/system tool responses removed
- LLM/internal analysis blobs removed

### Required columns for evaluation

The app will refuse to run if any of these are missing:

```
CONVERSATION_ID
MESSAGE_INDEX
MESSAGE_TIME
SENDER_ROLE
MESSAGE_TEXT
```

Other columns are optional but populate the dashboard and review pages.

---

## 4. API configuration

Settings live in the sidebar:

- **Base URL**: defaults to `https://langcc.maidstech.ai/v1`. Any OpenAI-compatible endpoint will work.
- **API Key**: password-style input.
- **Load available models**: calls `GET /models` through the OpenAI SDK and populates the dropdown.
- **Model**: chosen from the dropdown — never hardcoded.
- Generation parameters: temperature, top_p, max tokens, timeout, retries, concurrency (sequential today).
- Safeguards: max conversations, max agent messages per conversation, optional text truncation, include-unknown toggle, stop-on-error, save-raw-responses.

---

## 5. Evaluation logic

For each conversation, in `MESSAGE_INDEX` order:

```python
for conversation_id, group in df.groupby("CONVERSATION_ID"):
    messages = group.sort_values("MESSAGE_INDEX")

    for each row where SENDER_ROLE == "agent":
        history = all messages where MESSAGE_INDEX <= current MESSAGE_INDEX
        run message-level AI evaluation

    after all agent messages are evaluated:
        compute metadata
        run conversation-level AI evaluation
```

- Customer messages are never evaluated as target messages but are always included in history.
- Unknown messages can be optionally included in history (default: included).
- The app generates a stable message ID per row as `{CONVERSATION_ID}-{MESSAGE_INDEX}`.

### Message-level output

JSON object with:
`message_level_effect`, `frustration_level_after_message`, `frustration_change`,
`customer_effort_level`, `clarity_level`, `context_handling`,
`issue_origin`, `issue_type`, `frustration_cause`, `evidence`,
`business_impact`, `recommended_fix`.

### Conversation-level output

JSON object with:
`customer_objective_type`, `customer_primary_objective`,
`final_classification` (one of *Handled / Unhandled* × *Zero/Minimal / Many Issues*),
`handled_status`, `cx_issue_severity`, `final_customer_sentiment`,
`max_frustration_level`, `main_issue`, `all_detected_issues`,
`positive_signals`, `negative_signals`, `management_summary`,
`recommended_actions`, `manual_review_required`, `manual_review_reason`, `confidence`.

---

## 6. Output files

The **Exports** tab produces three files:

| File | Granularity | Use |
| --- | --- | --- |
| `cx_conversation_results.csv` | One row per conversation | Drop into a BI tool / spreadsheet |
| `cx_message_results.csv` | One row per evaluated agent message | Drill into the agent's turn-by-turn behavior |
| `cx_full_results.json` | Full structured export | Includes raw model responses, debug info, errors, and run config |

---

## 7. App tabs

1. **Upload & Settings** — upload the CSV, see the row/conversation/message summary, verify required columns.
2. **Prompts** — edit the system prompt, output structure, and user-prompt template for both evaluators. Save as new versions, switch active version, reset to default. All versions are stored in SQLite.
3. **Run Evaluation** — see the estimated AI-call count, start the run, watch progress, optionally cancel. Includes a "Past runs" section to load or delete previously saved runs.
4. **Dashboard** — management metrics, classification breakdowns, top issue types, top frustration causes, agent/skill breakdowns.
5. **Conversation Review** — pick a conversation, view its summary card, native chat-bubble transcript, and the message-level evaluation card directly under each agent message.
6. **Exports** — download CSVs and the full JSON.
7. **Debug** — raw prompts, raw responses, parse errors, failed records, sanitized run config.

The main views never expose raw JSON or stack traces — those live only in the Debug tab.

## 7a. Editing prompts

Open the **Prompts** tab. Each layer (Message-Level / Conversation-Level) has three editable fields:

- **System prompt** — the role, instructions, and rules. Use `{output_schema}` as the placeholder where you want the schema block inserted. If the placeholder is missing the schema is appended at the end.
- **Output structure** — the JSON shape the model is told to return. Edit field names, enums, or add new fields.
- **User prompt template** — wraps each per-call payload. Must include `{payload_json}`.

Buttons:

- **Save & Activate** — creates a new version and sets it active.
- **Set selected version active** — switch which saved version is used for new runs.
- **Reset to default** — re-activate the seeded default.
- **Delete selected version** — remove a custom version (the default cannot be deleted).

Custom fields you add to the output structure are preserved through the validators — they appear in the JSON export and Debug view, and any dashboard fields that no longer exist simply render as empty.

## 7b. Saving and reloading runs

Every evaluation run is written to SQLite as it progresses (message-level results, conversation-level results, errors, prompt versions used, and run config). In the **Run Evaluation** tab, the "Past runs" expander lists everything that has ever been saved with id, CSV name, status, and timestamp. Selecting a run and clicking **Load this run** repopulates the Dashboard, Conversation Review, Exports, and Debug tabs from the database — no need to re-run the evaluation.

---

## 8. Cost / token safeguards

- Max conversations to process.
- Max agent messages per conversation.
- Optional truncation of long message text.
- Estimated AI call count shown before running.
- Visible warning for large jobs.
- Partial results preserved in `st.session_state` — you can keep already-completed conversations even if the run is cancelled or errors out.

---

## 9. Troubleshooting

**"This CSV is missing required columns…"**
The CSV does not contain one of `CONVERSATION_ID`, `MESSAGE_INDEX`, `MESSAGE_TIME`, `SENDER_ROLE`, or `MESSAGE_TEXT`. Re-export from Snowflake using the message-level query.

**"Could not load models"**
Check the base URL is reachable, the API key is correct, and that the endpoint exposes the OpenAI `/models` route.

**"JSON parse failed" in Debug**
The model returned content that wasn't valid JSON. The app's JSON extractor tries plain JSON, fenced code blocks, and a greedy `{ ... }` match — anything else is recorded as a parse failure but does not crash the run. Consider lowering temperature or switching models.

**"API call failed"**
A network or provider-side error. Increase the timeout and retry count, or toggle "Stop on API error" off so the run continues past transient failures.

**Empty dashboard**
Run the evaluation first. The dashboard reads from `st.session_state.run_results`.

---

## 10. File structure

```
app.py              # Streamlit entry point and page navigation
api_client.py       # OpenAI-compatible client, /models loader, retry, concurrency cap
prompts.py          # PromptTemplate dataclass + default templates
db.py               # SQLite persistence for prompts, runs, results, errors
data_loader.py      # CSV loading, validation, conversation grouping
evaluator.py        # Message + conversation evaluation orchestration (concurrent)
aggregation.py      # Computed metadata and dashboard aggregates
exports.py          # CSV + JSON exports
ui_components.py    # Metric cards, chat bubbles, inline evaluation cards, filters
requirements.txt
README.md
cx_evaluator.db     # Created on first run; contains all prompts and saved runs
```
# CEX
