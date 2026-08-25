# CX React Dashboard

Ticket review dashboard for the CX pipeline. Reads a source Postgres database
read-only and serves everything the UI needs from a separate, precomputed stats
database, so the browser filters nothing and aggregates nothing.

## Run

```bash
npm install --ignore-scripts
npm run dev
```

UI: `http://127.0.0.1:5173` · Express API: `http://127.0.0.1:8090`

## Using it

The querystring is the whole application state, so any view is a link you can
share and the back button works.

| Key | Action |
| --- | --- |
| `j` / `k` | Move down / up the ticket list |
| `t` | Open the selected ticket's transcript, or return from it |
| `/` | Focus search |
| `Esc` | Leave the transcript, close the ticket pane, or leave the search box |

Filters are multi-select: values inside one facet are OR-ed, and separate facets
are AND-ed. Every option carries its result count under the other active
filters. KPI tiles are filters too — click one to add it.

The transcript is its own full-width page rather than a section of the side
pane. Messages are grouped under the source conversation they came from — a
ticket routinely spans several — with copyable conversation ids, and every
message carries its complete model output beside it. The right-hand rail holds
the ticket-level analysis: parse status, run id, verdict, score breakdown,
friction, and the computed rollup.

Theme (system / light / dark) and row density persist in `localStorage`.

## Front end layout

```
src/
  main.jsx                  mount
  App.jsx                   shell, view switching, keyboard, data wiring
  styles.css                design tokens + all component styles
  lib/
    api.js                  fetch with abort + response memoisation, hooks
    urlState.js             querystring <-> filter state
    facets.js               facet groups, presets, sort options
    useFacets.js            facet counts with per-dimension exclusion
    format.js               enum humanising, semantic tone, dates, deltas
  components/
    FacetRail.jsx           filter rail
    ResultsHead.jsx         KPI strip, trend, toolbar, active filter chips
    TicketList.jsx          result rows
    TicketDetail.jsx        score breakdown, issues, lifecycle
    Transcript.jsx          full-page transcript grouped by source conversation
    Customers.jsx           customer list + their tickets
    ui.jsx                  tags, sections, metric grid, empty states
```

Colours are only ever declared as tokens at the top of `styles.css`, in three
blocks: base light, `prefers-color-scheme: dark`, and `[data-theme="dark"]`.
Nothing else in the stylesheet declares a colour literal.

## Database access

The dashboard has its own Node/Express API and does not call the Python
pipeline app. Point source access at a SELECT-only role:

```bash
CX_DASHBOARD_DATABASE_URL=postgresql://cx_dashboard_readonly:<password>@localhost:5432/cex_pipeline
```

Every source query runs inside `BEGIN READ ONLY`, and non-SELECT SQL is
rejected before it reaches the driver.

Heavy aggregates are refreshed into a separate stats database:

```bash
CX_DASHBOARD_STATS_DATABASE_URL=postgresql://cx_dashboard_stats_writer:<password>@localhost:5432/cex_dashboard_stats
CX_DASHBOARD_WEBHOOK_SECRET=
```

Refresh is event-driven. The server does one full refresh at startup only if the
stats cache is empty; after that the Python analysis engine calls the webhook
when a run finishes and only rows that run touched are recomputed.

## API

```bash
GET  /api/summary            # totals + previous-window baseline, filterable
GET  /api/tickets            # slim rows + total/limit/offset, filterable, sortable
GET  /api/tickets/facets     # per-value counts for every filter dimension
GET  /api/trends             # day/week buckets (available; not currently used by the UI)
GET  /api/tickets/:id        # full ticket, messages, per-message results
GET  /api/customers
GET  /api/customers/:id/journey
GET  /api/filter-options     # superseded by /api/tickets/facets
POST /api/cache/refresh
POST /api/webhooks/analysis-run-finished
```

### Ticket filter parameters

Repeat a parameter to OR its values: `?experience=bad&experience=neutral`.

`status` `category` `ticketType` `requestOrigin` `handled` `experience`
`unresolved` `frustrationOrigin` `maxFrustration` `mainIssue` `messageIssue`
`scoreBand` `culprit` `agent` `manualReview` `runId` `search` `customerId`
`from` `to` `dateField` `scoreMin` `scoreMax` `hasContradiction`
`sort` `limit` `offset`

Webhook payload:

```json
{
  "event": "analysis_run_finished",
  "run_id": "<RUN_UUID>",
  "status": "finished"
}
```
