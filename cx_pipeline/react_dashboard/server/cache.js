import { readonlyQuery } from "./db.js";
import { statsPool, statsQuery } from "./statsDb.js";

let refreshState = {
  running: false,
  lastStartedAt: null,
  lastFinishedAt: null,
  lastError: "",
  lastMode: "",
  lastReason: "",
  lastRunId: "",
  lastChangedRows: 0,
  lastIndexedRows: 0
};

const jsonColumns = new Set([
  "culprits",
  "culprit_agent_names",
  "positive_signals",
  "negative_signals",
  "recommended_actions",
  "all_detected_issues",
  "segmentation",
  "cx_result",
  "computed_metadata"
]);

const ticketIndexColumns = [
  "ticket_id",
  "customer_id",
  "external_customer_id",
  "customer_name",
  "status",
  "open_closed",
  "category",
  "request_origin",
  "ticket_type",
  "objective",
  "should_append_future",
  "start_message_index",
  "end_message_index",
  "segment_messages",
  "linked_messages",
  "source_conversation_count",
  "source_conversation_ids",
  "first_message_at",
  "last_message_at",
  "opened_at",
  "closed_at",
  "reopenable_until",
  "lifecycle_risk",
  "lifecycle_reason",
  "analysis_eligible",
  "analysis_skip_reason",
  "span_seconds",
  "first_response_seconds",
  "analyzed_at",
  "evaluated_messages",
  "ok_messages",
  "failed_messages",
  "handled_status",
  "customer_experience",
  "unhandled_resolution_subtype",
  "frustration_detected",
  "frustration_origin",
  "frustration_timing",
  "max_frustration_level",
  "final_customer_sentiment",
  "customer_started_frustrated",
  "customer_became_frustrated_during_chat",
  "customer_ended_frustrated",
  "manual_review_required",
  "manual_review_reason",
  "customer_objective_type",
  "customer_primary_objective",
  "management_summary",
  "classification_reason",
  "score_final",
  "score_final_100",
  "score_rating",
  "score_band",
  "score_explanation",
  "score_resolution",
  "score_context_understanding",
  "score_customer_effort",
  "score_frustration_risk",
  "score_ai_judgment",
  "score_message_signals",
  "score_raw_total",
  "main_issue_type",
  "main_issue_origin",
  "main_issue_summary",
  "customer_impact",
  "culprits",
  "culprit_agent_names",
  "culprit_reason",
  "positive_signals",
  "negative_signals",
  "recommended_actions",
  "all_detected_issues",
  "message_issue_types",
  "major_issue_count",
  "minor_issue_count",
  "recovered_issue_count",
  "contradiction_count",
  "first_contradiction_message_id",
  "max_message_frustration",
  "high_effort_message_count",
  "unclear_message_count",
  "poor_context_count",
  "culprit_kinds",
  "culprit_agents",
  "handling_agents",
  "customer_message_count",
  "agent_message_count",
  "bot_message_count",
  "broadcast_message_count",
  "latest_run_id",
  "latest_run_at",
  "skills",
  "initial_skill",
  "last_skill",
  "distinct_skill_count",
  "handoff_count",
  "had_unauthorized_skill",
  "unauthorized_skill_count",
  "rag_retrieval_events",
  "messages_with_rag",
  "rag_coverage",
  "confidence",
  "segmentation",
  "cx_result",
  "computed_metadata",
  "updated_at",
  "search_text"
];

function insertValue(row, column) {
  if (!jsonColumns.has(column)) return row[column];
  const fallback = column.endsWith("signals") || column === "recommended_actions" || column === "all_detected_issues" || column === "culprits" || column === "culprit_agent_names" ? [] : {};
  return JSON.stringify(row[column] || fallback);
}

function boolValue(value) {
  if (typeof value === "boolean") return value;
  return String(value ?? "").trim().toLowerCase() === "true";
}

function intValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : null;
}

function numValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function norm(value, fallback = "unknown") {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function statusClass(status) {
  const text = String(status || "").toLowerCase();
  if (text.includes("pending") || text.includes("unresolved") || text === "unhandled") return "open";
  if (text.includes("resolved") || text === "handled") return "closed";
  return text || "unknown";
}

function isoTime(value) {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString();
}

function lifecycleState(row) {
  return {
    ticket_id: Number(row.ticket_id),
    customer_id: row.customer_id == null ? null : Number(row.customer_id),
    status: row.status || null,
    open_closed: row.open_closed || null,
    opened_at: isoTime(row.opened_at),
    closed_at: isoTime(row.closed_at),
    first_message_at: isoTime(row.first_message_at),
    last_message_at: isoTime(row.last_message_at),
    source_conversation_ids: row.source_conversation_ids || [],
    latest_run_id: row.latest_run_id || null,
    reopenable_until: isoTime(row.reopenable_until),
    lifecycle_risk: row.lifecycle_risk || null,
    lifecycle_reason: row.lifecycle_reason || null,
    analysis_eligible: row.analysis_eligible ?? null,
    analysis_skip_reason: row.analysis_skip_reason || null
  };
}

function lifecycleEventType(previous, current) {
  if (!previous) return "initial_snapshot";
  if (previous.open_closed !== current.open_closed) {
    if (previous.open_closed === "closed" && current.open_closed === "open") return "reopened_closed_at_cleared";
    if (current.open_closed === "closed") return "closed_at_set";
    return "status_changed";
  }
  if (previous.closed_at !== current.closed_at) {
    if (!previous.closed_at && current.closed_at) return "closed_at_set";
    if (previous.closed_at && !current.closed_at) return "closed_at_cleared";
    return "closed_at_changed";
  }
  if (previous.opened_at !== current.opened_at) return "opened_at_changed";
  if (previous.lifecycle_risk !== current.lifecycle_risk) return "lifecycle_risk_changed";
  if (previous.analysis_eligible !== current.analysis_eligible) return "analysis_eligibility_changed";
  if (previous.analysis_skip_reason !== current.analysis_skip_reason) return "analysis_skip_reason_changed";
  return "";
}

async function loadPreviousLifecycleStates(client, { changedOnly, ticketIds }) {
  const rows = changedOnly
    ? ticketIds.length
      ? await client.query(
        `
        SELECT
          ticket_id, customer_id, status, open_closed, opened_at, closed_at,
          first_message_at, last_message_at, source_conversation_ids, latest_run_id,
          reopenable_until, lifecycle_risk, lifecycle_reason, analysis_eligible, analysis_skip_reason
        FROM ticket_index
        WHERE ticket_id=ANY($1::bigint[])
        `,
        [ticketIds]
      )
      : { rows: [] }
    : await client.query(
      `
      SELECT
        ticket_id, customer_id, status, open_closed, opened_at, closed_at,
        first_message_at, last_message_at, source_conversation_ids, latest_run_id,
        reopenable_until, lifecycle_risk, lifecycle_reason, analysis_eligible, analysis_skip_reason
      FROM ticket_index
      `
    );
  return new Map(rows.rows.map((row) => [Number(row.ticket_id), lifecycleState(row)]));
}

async function writeLifecycleHistory(client, previousStates, ticketRows) {
  for (const row of ticketRows) {
    const current = lifecycleState(row);
    const previous = previousStates.get(current.ticket_id);
    const eventType = lifecycleEventType(previous, current);
    if (!eventType) continue;
    await client.query(
      `
      INSERT INTO ticket_lifecycle_history(
        ticket_id, customer_id, status, open_closed, opened_at, closed_at,
        first_message_at, last_message_at, source_conversation_ids, latest_run_id,
        event_type, previous_state, current_state
      )
      VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13::jsonb)
      `,
      [
        current.ticket_id,
        current.customer_id,
        current.status,
        current.open_closed,
        current.opened_at,
        current.closed_at,
        current.first_message_at,
        current.last_message_at,
        current.source_conversation_ids,
        current.latest_run_id,
        eventType,
        JSON.stringify(previous || {}),
        JSON.stringify(current)
      ]
    );
  }
}

function sourceSignatureFromRows(ticketRows, customerRows) {
  const maxTicketUpdate = ticketRows.reduce((max, row) => {
    const value = row.updated_at ? new Date(row.updated_at).getTime() : 0;
    return Math.max(max, Number.isFinite(value) ? value : 0);
  }, 0);
  return {
    tickets: ticketRows.length,
    customers: customerRows.length,
    max_ticket_updated_at: maxTicketUpdate ? new Date(maxTicketUpdate).toISOString() : null
  };
}

async function buildSummaryFromStats(client) {
  const totalsResult = await client.query(`
    SELECT
      (SELECT count(*)::int FROM customer_index) AS customers,
      COALESCE((SELECT sum(source_conversations)::int FROM customer_index), 0) AS source_conversations,
      COALESCE((SELECT sum(messages)::int FROM customer_index), 0) AS messages,
      count(*)::int AS tickets,
      count(*) FILTER (WHERE open_closed='closed')::int AS closed_tickets,
      count(*) FILTER (WHERE open_closed='open')::int AS open_tickets,
      COALESCE(sum(evaluated_messages)::int, 0) AS message_results,
      count(*) FILTER (WHERE handled_status <> 'not_analyzed')::int AS ticket_cx_results,
      count(*) FILTER (WHERE manual_review_required)::int AS manual_review_tickets,
      count(*) FILTER (WHERE customer_experience='bad')::int AS bad_experience_tickets,
      count(*) FILTER (WHERE frustration_detected)::int AS frustrated_tickets,
      count(*) FILTER (WHERE max_frustration_level IN ('high', 'cancellation_risk'))::int AS high_frustration_tickets,
      count(*) FILTER (WHERE lifecycle_risk='high_risk_active')::int AS high_risk_active_tickets,
      count(*) FILTER (WHERE lifecycle_risk='critical_disregarded')::int AS critical_disregarded_tickets,
      count(*) FILTER (WHERE analysis_eligible=false)::int AS analysis_ineligible_tickets,
      avg(score_final_100)::float AS avg_score_100
    FROM ticket_index
  `);
  const byStatus = await client.query("SELECT status, count(*)::int AS count FROM ticket_index GROUP BY status ORDER BY count DESC, status");
  const byCategory = await client.query("SELECT category, count(*)::int AS count FROM ticket_index GROUP BY category ORDER BY count DESC, category");
  const byExperience = await client.query(`
    SELECT handled_status, customer_experience, count(*)::int AS count
    FROM ticket_index
    GROUP BY handled_status, customer_experience
    ORDER BY count DESC, handled_status, customer_experience
  `);
  const byFrustrationOrigin = await client.query("SELECT COALESCE(frustration_origin, 'unknown') AS value, count(*)::int AS count FROM ticket_index GROUP BY value ORDER BY count DESC, value");
  const byMainIssueType = await client.query("SELECT COALESCE(main_issue_type, 'unknown') AS value, count(*)::int AS count FROM ticket_index GROUP BY value ORDER BY count DESC, value LIMIT 12");
  const byUnresolvedSubtype = await client.query("SELECT COALESCE(unhandled_resolution_subtype, 'unknown') AS value, count(*)::int AS count FROM ticket_index GROUP BY value ORDER BY count DESC, value");
  const byMaxFrustration = await client.query("SELECT COALESCE(max_frustration_level, 'unknown') AS value, count(*)::int AS count FROM ticket_index GROUP BY value ORDER BY count DESC, value");
  const recent = await client.query(`
    SELECT
      ticket_id AS id,
      ticket_id,
      customer_id,
      customer_name,
      status,
      open_closed,
      category,
      request_origin,
      ticket_type,
      handled_status,
      customer_experience,
      frustration_origin,
      main_issue_type,
      score_final_100 AS score_100,
      score_final_100,
      score_rating,
      score_band,
      manual_review_required,
      linked_messages,
      source_conversation_count,
      failed_messages,
      contradiction_count,
      first_message_at,
      last_message_at,
      opened_at,
      closed_at,
      reopenable_until,
      lifecycle_risk,
      analysis_eligible,
      analysis_skip_reason,
      left(COALESCE(main_issue_summary, ''), 160) AS main_issue_summary,
      left(COALESCE(customer_primary_objective, ''), 120) AS customer_primary_objective
    FROM ticket_index
    ORDER BY last_message_at DESC NULLS LAST, ticket_id DESC
    LIMIT 10
  `);

  return {
    totals: totalsResult.rows[0] || {},
    byStatus: byStatus.rows,
    byCategory: byCategory.rows,
    byExperience: byExperience.rows,
    byFrustrationOrigin: byFrustrationOrigin.rows,
    byMainIssueType: byMainIssueType.rows,
    byUnresolvedSubtype: byUnresolvedSubtype.rows,
    byMaxFrustration: byMaxFrustration.rows,
    recent: recent.rows
  };
}

export function cacheStatus() {
  return { ...refreshState };
}

async function cacheHasRows() {
  const rows = await statsQuery(`
    SELECT
      EXISTS(SELECT 1 FROM summary_cache WHERE id=true) AS has_summary,
      EXISTS(SELECT 1 FROM ticket_index LIMIT 1) AS has_tickets
  `);
  return Boolean(rows[0]?.has_summary && rows[0]?.has_tickets);
}

export async function ensureInitialCache() {
  if (await cacheHasRows()) return cacheStatus();
  return refreshCache({ mode: "full", reason: "startup_empty_cache" });
}

export async function refreshCache({ mode = "full", runId = "", reason = "manual" } = {}) {
  if (refreshState.running) return cacheStatus();
  const changedOnly = mode === "changed" && Boolean(runId) && await cacheHasRows();
  refreshState = {
    ...refreshState,
    running: true,
    lastStartedAt: new Date().toISOString(),
    lastError: "",
    lastMode: changedOnly ? "changed" : "full",
    lastReason: reason,
    lastRunId: runId || ""
  };

  try {
    const sourceParams = changedOnly ? [runId] : [];
    const ticketFilter = changedOnly
      ? `
        WHERE t.id IN (
          SELECT id FROM tickets WHERE latest_ticketing_run_id=$1::uuid
          UNION
          SELECT ticket_id FROM message_results WHERE run_id=$1::uuid
          UNION
          SELECT ticket_id FROM ticket_cx_results WHERE run_id=$1::uuid
          UNION
          SELECT (data->>'ticket_id')::bigint
          FROM run_events
          WHERE run_id=$1::uuid
            AND event_type='ticket_lifecycle_refreshed'
            AND COALESCE(data->>'ticket_id', '') ~ '^\\d+$'
        )
      `
      : "";
    const customerFilter = changedOnly
      ? `
        WHERE c.id IN (
          SELECT customer_id FROM source_conversations WHERE first_seen_run_id=$1::uuid OR ticketed_run_id=$1::uuid
          UNION
          SELECT customer_id FROM tickets WHERE latest_ticketing_run_id=$1::uuid
          UNION
          SELECT t.customer_id FROM tickets t JOIN message_results mr ON mr.ticket_id=t.id WHERE mr.run_id=$1::uuid
          UNION
          SELECT t.customer_id FROM tickets t JOIN ticket_cx_results cx ON cx.ticket_id=t.id WHERE cx.run_id=$1::uuid
          UNION
          SELECT t.customer_id
          FROM tickets t
          JOIN (
            SELECT (data->>'ticket_id')::bigint AS ticket_id
            FROM run_events
            WHERE run_id=$1::uuid
              AND event_type='ticket_lifecycle_refreshed'
              AND COALESCE(data->>'ticket_id', '') ~ '^\\d+$'
          ) e ON e.ticket_id=t.id
        )
      `
      : "";
    const [sourceTickets, sourceCustomers] = await Promise.all([
      readonlyQuery(`
        WITH target_tickets AS (
          SELECT t.*
          FROM tickets t
          ${ticketFilter}
        ),
        ticket_messages_enriched AS (
          SELECT
            tm.ticket_id,
            m.id AS message_id,
            m.customer_message_index,
            NULLIF(m.message_time, '')::timestamptz AS message_at,
            m.raw_sender_role,
            m.sender_role,
            m.raw,
            sc.source_conversation_id
          FROM ticket_messages tm
          JOIN target_tickets tt ON tt.id=tm.ticket_id
          JOIN messages m ON m.id=tm.message_id
          JOIN source_conversations sc ON sc.id=m.source_conversation_pk
        ),
        first_customer AS (
          SELECT ticket_id, min(message_at) AS first_customer_at
          FROM ticket_messages_enriched
          WHERE lower(COALESCE(raw_sender_role, ''))='consumer'
          GROUP BY ticket_id
        ),
        ticket_sources AS (
          SELECT
            e.ticket_id,
            count(e.message_id)::int AS linked_messages,
            count(DISTINCT e.source_conversation_id)::int AS source_conversation_count,
            array_remove(array_agg(DISTINCT e.source_conversation_id ORDER BY e.source_conversation_id), NULL) AS source_conversation_ids,
            min(e.message_at) AS first_message_at,
            max(e.message_at) AS last_message_at,
            EXTRACT(EPOCH FROM (max(e.message_at) - min(e.message_at)))::int AS span_seconds,
            EXTRACT(EPOCH FROM (
              min(e.message_at) FILTER (
                WHERE lower(COALESCE(e.raw_sender_role, '')) IN ('agent', 'bot')
                  AND e.message_at > fc.first_customer_at
              ) - fc.first_customer_at
            ))::int AS first_response_seconds,
            count(*) FILTER (WHERE lower(COALESCE(e.raw_sender_role, ''))='consumer')::int AS customer_message_count,
            count(*) FILTER (WHERE lower(COALESCE(e.raw_sender_role, ''))='agent')::int AS agent_message_count,
            count(*) FILTER (WHERE lower(COALESCE(e.raw_sender_role, ''))='bot')::int AS bot_message_count,
            count(*) FILTER (WHERE lower(COALESCE(e.raw_sender_role, ''))='system')::int AS broadcast_message_count,
            max(CASE WHEN COALESCE(e.raw->>'UNAUTHORIZED_SKILL_COUNT', '') ~ '^\\d+$' THEN (e.raw->>'UNAUTHORIZED_SKILL_COUNT')::int ELSE 0 END)::int AS unauthorized_skill_count,
            bool_or(CASE WHEN COALESCE(e.raw->>'UNAUTHORIZED_SKILL_COUNT', '') ~ '^\\d+$' THEN (e.raw->>'UNAUTHORIZED_SKILL_COUNT')::int ELSE 0 END > 0) AS had_unauthorized_skill,
            sum(CASE WHEN COALESCE(e.raw->>'RAG_RETRIEVAL_COUNT', '') ~ '^\\d+$' THEN (e.raw->>'RAG_RETRIEVAL_COUNT')::int ELSE 0 END)::int AS rag_retrieval_events,
            count(*) FILTER (WHERE upper(COALESCE(e.raw->>'HAS_RAG_RETRIEVAL', ''))='TRUE')::int AS messages_with_rag
          FROM ticket_messages_enriched e
          LEFT JOIN first_customer fc ON fc.ticket_id=e.ticket_id
          GROUP BY e.ticket_id, fc.first_customer_at
        ),
        handling_agent_values AS (
          SELECT ticket_id, NULLIF(raw->>'MESSAGE_AGENT_FULL_NAME', '') AS agent FROM ticket_messages_enriched
          UNION
          SELECT ticket_id, NULLIF(raw->>'CONVERSATION_AGENT_FULL_NAME', '') AS agent FROM ticket_messages_enriched
        ),
        handling_agents AS (
          SELECT ticket_id, array_remove(array_agg(DISTINCT agent ORDER BY agent), NULL) AS handling_agents
          FROM handling_agent_values
          GROUP BY ticket_id
        ),
        skill_values AS (
          SELECT ticket_id, NULLIF(trim(value), '') AS skill
          FROM ticket_messages_enriched, regexp_split_to_table(COALESCE(raw->>'JOINED_SKILLS', ''), ',') value
          UNION
          SELECT ticket_id, NULLIF(raw->>'MESSAGE_SKILL', '') AS skill FROM ticket_messages_enriched
        ),
        skill_rollups AS (
          SELECT ticket_id, array_remove(array_agg(DISTINCT skill ORDER BY skill), NULL) AS skills
          FROM skill_values
          GROUP BY ticket_id
        ),
        skill_order AS (
          SELECT
            ticket_id,
            NULLIF(raw->>'MESSAGE_SKILL', '') AS skill,
            lag(NULLIF(raw->>'MESSAGE_SKILL', '')) OVER (PARTITION BY ticket_id ORDER BY customer_message_index, message_id) AS prev_skill,
            (array_remove(array_agg(NULLIF(raw->>'INITIAL_SKILL', '')) OVER (PARTITION BY ticket_id ORDER BY customer_message_index, message_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), NULL))[1] AS initial_skill,
            (array_remove(array_agg(NULLIF(raw->>'LAST_SKILL', '')) OVER (PARTITION BY ticket_id ORDER BY customer_message_index DESC, message_id DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), NULL))[1] AS last_skill
          FROM ticket_messages_enriched
        ),
        skill_transitions AS (
          SELECT
            ticket_id,
            max(initial_skill) AS initial_skill,
            max(last_skill) AS last_skill,
            count(*) FILTER (WHERE skill IS NOT NULL AND prev_skill IS NOT NULL AND skill <> prev_skill)::int AS handoff_count
          FROM skill_order
          GROUP BY ticket_id
        ),
        message_stats AS (
          SELECT
            mr.ticket_id,
            count(*)::int AS evaluated_messages,
            count(*) FILTER (WHERE mr.parse_status='ok')::int AS ok_messages,
            count(*) FILTER (WHERE mr.parse_status <> 'ok')::int AS failed_messages,
            array_remove(array_agg(DISTINCT CASE WHEN lower(COALESCE(mr.result->>'issue_type', 'none')) <> 'none' THEN mr.result->>'issue_type' END), NULL) AS message_issue_types,
            count(*) FILTER (WHERE mr.result->>'message_level_effect'='major_issue')::int AS major_issue_count,
            count(*) FILTER (WHERE mr.result->>'message_level_effect'='minor_issue')::int AS minor_issue_count,
            count(*) FILTER (WHERE mr.result->>'message_level_effect'='recovered_issue')::int AS recovered_issue_count,
            count(*) FILTER (WHERE lower(COALESCE(mr.result->>'contradiction', 'false'))='true')::int AS contradiction_count,
            (array_remove(array_agg(
              CASE WHEN COALESCE(mr.result->>'first_contradiction_message_id', '') ~ '^\\d+$'
                THEN (mr.result->>'first_contradiction_message_id')::bigint
              END
              ORDER BY m.customer_message_index, m.id
            ), NULL))[1] AS first_contradiction_message_id,
            CASE max(CASE COALESCE(mr.result->>'frustration_level_after_message', 'none')
              WHEN 'cancellation_risk' THEN 5
              WHEN 'high' THEN 4
              WHEN 'medium' THEN 3
              WHEN 'low' THEN 2
              ELSE 1
            END)
              WHEN 5 THEN 'cancellation_risk'
              WHEN 4 THEN 'high'
              WHEN 3 THEN 'medium'
              WHEN 2 THEN 'low'
              ELSE 'none'
            END AS max_message_frustration,
            count(*) FILTER (WHERE mr.result->>'customer_effort_level'='high')::int AS high_effort_message_count,
            count(*) FILTER (WHERE mr.result->>'clarity_level'='unclear')::int AS unclear_message_count,
            count(*) FILTER (WHERE mr.result->>'context_handling'='poor')::int AS poor_context_count
          FROM message_results mr
          JOIN messages m ON m.id=mr.message_id
          GROUP BY mr.ticket_id
        ),
        culprit_kind_values AS (
          SELECT
            tt.id AS ticket_id,
            NULLIF(lower(CASE
              WHEN jsonb_typeof(value)='string' THEN trim(both '"' from value::text)
              ELSE COALESCE(value->>'kind', value->>'culprit_kind', value->>'type', value->>'role')
            END), '') AS culprit_kind
          FROM target_tickets tt
          LEFT JOIN ticket_cx_results cx ON cx.ticket_id=tt.id
          LEFT JOIN LATERAL jsonb_array_elements(COALESCE(cx.result->'culprits', '[]'::jsonb)) value ON true
        ),
        culprit_agent_values AS (
          SELECT tt.id AS ticket_id, NULLIF(trim(both '"' from value::text), '') AS culprit_agent
          FROM target_tickets tt
          LEFT JOIN ticket_cx_results cx ON cx.ticket_id=tt.id
          LEFT JOIN LATERAL jsonb_array_elements(COALESCE(cx.result->'culprit_agent_names', '[]'::jsonb)) value ON true
        ),
        culprit_rollups AS (
          SELECT
            tt.id AS ticket_id,
            COALESCE(array_remove(array_agg(DISTINCT ckv.culprit_kind ORDER BY ckv.culprit_kind), NULL), '{}'::text[]) AS culprit_kinds,
            COALESCE(array_remove(array_agg(DISTINCT cav.culprit_agent ORDER BY cav.culprit_agent), NULL), '{}'::text[]) AS culprit_agents
          FROM target_tickets tt
          LEFT JOIN culprit_kind_values ckv ON ckv.ticket_id=tt.id
          LEFT JOIN culprit_agent_values cav ON cav.ticket_id=tt.id
          GROUP BY tt.id
        )
        SELECT
          t.id AS ticket_id,
          t.customer_id,
          c.external_customer_id,
          c.customer_name,
          t.status,
          t.category,
          t.request_origin,
          t.ticket_type,
          t.objective,
          t.should_append_future,
          NULLIF(t.segmentation->>'start_message_index', '')::int AS start_message_index,
          NULLIF(t.segmentation->>'end_message_index', '')::int AS end_message_index,
          COALESCE(jsonb_array_length(t.segmentation->'included_message_indexes'), 0)::int AS segment_messages,
          COALESCE(ts.linked_messages, 0)::int AS linked_messages,
          COALESCE(ts.source_conversation_count, 0)::int AS source_conversation_count,
          COALESCE(ts.source_conversation_ids, '{}'::text[]) AS source_conversation_ids,
          ts.first_message_at,
          ts.last_message_at,
          t.opened_at AS engine_opened_at,
          t.last_message_at AS engine_last_message_at,
          t.closed_at AS engine_closed_at,
          t.reopenable_until,
          COALESCE(t.lifecycle_risk, 'normal') AS lifecycle_risk,
          t.lifecycle_reason,
          COALESCE(t.analysis_eligible, true) AS analysis_eligible,
          t.analysis_skip_reason,
          COALESCE(ts.span_seconds, 0)::int AS span_seconds,
          ts.first_response_seconds,
          cx.updated_at AS analyzed_at,
          COALESCE(ms.evaluated_messages, 0)::int AS evaluated_messages,
          COALESCE(ms.ok_messages, 0)::int AS ok_messages,
          COALESCE(ms.failed_messages, 0)::int AS failed_messages,
          COALESCE(cx.result->>'handled_status', 'not_analyzed') AS handled_status,
          COALESCE(cx.result->>'customer_experience', 'not_analyzed') AS customer_experience,
          COALESCE(cx.result->>'unhandled_resolution_subtype', 'not_applicable') AS unhandled_resolution_subtype,
          COALESCE(cx.result->>'frustration_origin', 'none') AS frustration_origin,
          COALESCE(cx.result->>'frustration_timing', 'none') AS frustration_timing,
          COALESCE(cx.result->>'max_frustration_level', 'none') AS max_frustration_level,
          COALESCE(cx.result->>'final_customer_sentiment', 'unknown') AS final_customer_sentiment,
          cx.result->>'manual_review_reason' AS manual_review_reason,
          COALESCE(cx.result->>'customer_objective_type', '') AS customer_objective_type,
          COALESCE(cx.result->>'customer_primary_objective', '') AS customer_primary_objective,
          COALESCE(cx.result->>'management_summary', '') AS management_summary,
          COALESCE(cx.result->>'classification_reason', '') AS classification_reason,
          NULLIF(cx.result->'conversation_score'->>'final_score', '')::numeric AS score_final,
          NULLIF(cx.result->'conversation_score'->>'final_score_100', '')::numeric AS score_final_100,
          cx.result->'conversation_score'->>'score_rating' AS score_rating,
          CASE
            WHEN NULLIF(cx.result->'conversation_score'->>'final_score_100', '')::numeric IS NULL THEN 'unscored'
            WHEN NULLIF(cx.result->'conversation_score'->>'final_score_100', '')::numeric < 40 THEN 'critical'
            WHEN NULLIF(cx.result->'conversation_score'->>'final_score_100', '')::numeric < 60 THEN 'poor'
            WHEN NULLIF(cx.result->'conversation_score'->>'final_score_100', '')::numeric < 75 THEN 'fair'
            WHEN NULLIF(cx.result->'conversation_score'->>'final_score_100', '')::numeric < 90 THEN 'good'
            ELSE 'excellent'
          END AS score_band,
          cx.result->'conversation_score'->>'score_explanation' AS score_explanation,
          NULLIF(cx.result->'conversation_score'->>'resolution_score', '')::numeric AS score_resolution,
          NULLIF(cx.result->'conversation_score'->>'context_understanding_score', '')::numeric AS score_context_understanding,
          NULLIF(cx.result->'conversation_score'->>'customer_effort_score', '')::numeric AS score_customer_effort,
          NULLIF(cx.result->'conversation_score'->>'trust_frustration_risk_score', '')::numeric AS score_frustration_risk,
          NULLIF(cx.result->'conversation_score'->>'ai_judgment_score', '')::numeric AS score_ai_judgment,
          NULLIF(cx.result->'conversation_score'->>'message_signal_score', '')::numeric AS score_message_signals,
          NULLIF(cx.result->'conversation_score'->>'raw_total_score', '')::numeric AS score_raw_total,
          COALESCE(cx.result->'main_issue'->>'issue_type', 'none') AS main_issue_type,
          COALESCE(cx.result->'main_issue'->>'issue_origin', 'none') AS main_issue_origin,
          COALESCE(cx.result->'main_issue'->>'issue_summary', '') AS main_issue_summary,
          COALESCE(cx.result->'main_issue'->>'customer_impact', '') AS customer_impact,
          COALESCE(cx.result->'culprits', '[]'::jsonb) AS culprits,
          COALESCE(cx.result->'culprit_agent_names', '[]'::jsonb) AS culprit_agent_names,
          COALESCE(cx.result->>'culprit_reason', '') AS culprit_reason,
          COALESCE(cx.result->'positive_signals', '[]'::jsonb) AS positive_signals,
          COALESCE(cx.result->'negative_signals', '[]'::jsonb) AS negative_signals,
          COALESCE(cx.result->'recommended_actions', '[]'::jsonb) AS recommended_actions,
          COALESCE(cx.result->'all_detected_issues', '[]'::jsonb) AS all_detected_issues,
          COALESCE(ms.message_issue_types, '{}'::text[]) AS message_issue_types,
          COALESCE(ms.major_issue_count, 0)::int AS major_issue_count,
          COALESCE(ms.minor_issue_count, 0)::int AS minor_issue_count,
          COALESCE(ms.recovered_issue_count, 0)::int AS recovered_issue_count,
          COALESCE(ms.contradiction_count, 0)::int AS contradiction_count,
          ms.first_contradiction_message_id,
          COALESCE(ms.max_message_frustration, 'none') AS max_message_frustration,
          COALESCE(ms.high_effort_message_count, 0)::int AS high_effort_message_count,
          COALESCE(ms.unclear_message_count, 0)::int AS unclear_message_count,
          COALESCE(ms.poor_context_count, 0)::int AS poor_context_count,
          COALESCE(cr.culprit_kinds, '{}'::text[]) AS culprit_kinds,
          COALESCE(cr.culprit_agents, '{}'::text[]) AS culprit_agents,
          COALESCE(ha.handling_agents, '{}'::text[]) AS handling_agents,
          COALESCE(ts.customer_message_count, 0)::int AS customer_message_count,
          COALESCE(ts.agent_message_count, 0)::int AS agent_message_count,
          COALESCE(ts.bot_message_count, 0)::int AS bot_message_count,
          COALESCE(ts.broadcast_message_count, 0)::int AS broadcast_message_count,
          CASE
            WHEN pr_cx.finished_at IS NOT NULL
              AND (pr_ticket.finished_at IS NULL OR pr_cx.finished_at >= pr_ticket.finished_at)
              THEN cx.run_id
            ELSE t.latest_ticketing_run_id
          END AS latest_run_id,
          CASE
            WHEN pr_cx.finished_at IS NOT NULL
              AND (pr_ticket.finished_at IS NULL OR pr_cx.finished_at >= pr_ticket.finished_at)
              THEN pr_cx.finished_at
            ELSE pr_ticket.finished_at
          END AS latest_run_at,
          COALESCE(sr.skills, '{}'::text[]) AS skills,
          st.initial_skill,
          st.last_skill,
          COALESCE(array_length(sr.skills, 1), 0)::int AS distinct_skill_count,
          COALESCE(st.handoff_count, 0)::int AS handoff_count,
          COALESCE(ts.had_unauthorized_skill, false) AS had_unauthorized_skill,
          COALESCE(ts.unauthorized_skill_count, 0)::int AS unauthorized_skill_count,
          COALESCE(ts.rag_retrieval_events, 0)::int AS rag_retrieval_events,
          COALESCE(ts.messages_with_rag, 0)::int AS messages_with_rag,
          (COALESCE(ts.messages_with_rag, 0)::numeric / NULLIF(COALESCE(ts.bot_message_count, 0), 0)) AS rag_coverage,
          COALESCE(cx.result->>'confidence', cx.result->>'analysis_confidence', cx.result->>'confidence_level', '') AS confidence,
          t.segmentation,
          COALESCE(cx.result, '{}'::jsonb) AS cx_result,
          COALESCE(cx.computed_metadata, '{}'::jsonb) AS computed_metadata,
          t.updated_at
        FROM target_tickets t
        JOIN customers c ON c.id=t.customer_id
        LEFT JOIN ticket_cx_results cx ON cx.ticket_id=t.id
        LEFT JOIN pipeline_runs pr_ticket ON pr_ticket.id=t.latest_ticketing_run_id
        LEFT JOIN pipeline_runs pr_cx ON pr_cx.id=cx.run_id
        LEFT JOIN ticket_sources ts ON ts.ticket_id=t.id
        LEFT JOIN message_stats ms ON ms.ticket_id=t.id
        LEFT JOIN handling_agents ha ON ha.ticket_id=t.id
        LEFT JOIN skill_rollups sr ON sr.ticket_id=t.id
        LEFT JOIN skill_transitions st ON st.ticket_id=t.id
        LEFT JOIN culprit_rollups cr ON cr.ticket_id=t.id
        ORDER BY t.updated_at DESC, t.id DESC
      `, sourceParams),
      readonlyQuery(`
        SELECT
          c.id AS customer_id,
          c.external_customer_id,
          c.customer_name,
          count(DISTINCT sc.id)::int AS source_conversations,
          count(DISTINCT m.id)::int AS messages,
          count(DISTINCT t.id)::int AS tickets,
          count(DISTINCT t.id) FILTER (WHERE t.status ILIKE '%resolved%' AND t.status NOT ILIKE '%unresolved%')::int AS closed_tickets,
          count(DISTINCT t.id) FILTER (WHERE t.status ILIKE '%unresolved%' OR t.status ILIKE '%pending%')::int AS open_tickets,
          count(DISTINCT t.id) FILTER (WHERE COALESCE(cx.result->>'manual_review_required', 'false') = 'true')::int AS manual_review_tickets,
          count(DISTINCT t.id) FILTER (WHERE COALESCE(cx.result->>'customer_experience', '') = 'bad')::int AS bad_experience_tickets,
          max(GREATEST(t.updated_at, c.updated_at)) AS last_activity
        FROM customers c
        LEFT JOIN source_conversations sc ON sc.customer_id=c.id
        LEFT JOIN messages m ON m.customer_id=c.id
        LEFT JOIN tickets t ON t.customer_id=c.id
        LEFT JOIN ticket_cx_results cx ON cx.ticket_id=t.id
        ${customerFilter}
        GROUP BY c.id
        ORDER BY last_activity DESC NULLS LAST, c.id DESC
      `, sourceParams)
    ]);

    const ticketRows = sourceTickets.map((row) => ({
      ...row,
      ticket_id: Number(row.ticket_id),
      customer_id: Number(row.customer_id),
      open_closed: statusClass(row.status),
      opened_at: row.engine_opened_at || row.first_message_at || null,
      last_message_at: row.engine_last_message_at || row.last_message_at || null,
      closed_at: row.engine_closed_at || (statusClass(row.status) === "closed" ? row.last_message_at || null : null),
      should_append_future: boolValue(row.should_append_future),
      analysis_eligible: boolValue(row.analysis_eligible),
      frustration_detected: boolValue(row.cx_result?.frustration_detected),
      customer_started_frustrated: boolValue(row.cx_result?.customer_started_frustrated),
      customer_became_frustrated_during_chat: boolValue(row.cx_result?.customer_became_frustrated_during_chat),
      customer_ended_frustrated: boolValue(row.cx_result?.customer_ended_frustrated),
      manual_review_required: boolValue(row.cx_result?.manual_review_required),
      start_message_index: intValue(row.start_message_index),
      end_message_index: intValue(row.end_message_index),
      segment_messages: Number(row.segment_messages || 0),
      linked_messages: Number(row.linked_messages || 0),
      source_conversation_count: Number(row.source_conversation_count || 0),
      evaluated_messages: Number(row.evaluated_messages || 0),
      score_final: numValue(row.score_final),
      score_final_100: numValue(row.score_final_100),
      score_resolution: numValue(row.score_resolution),
      score_context_understanding: numValue(row.score_context_understanding),
      score_customer_effort: numValue(row.score_customer_effort),
      score_frustration_risk: numValue(row.score_frustration_risk),
      score_ai_judgment: numValue(row.score_ai_judgment),
      score_message_signals: numValue(row.score_message_signals),
      score_raw_total: numValue(row.score_raw_total),
      span_seconds: intValue(row.span_seconds),
      first_response_seconds: intValue(row.first_response_seconds),
      evaluated_messages: Number(row.evaluated_messages || 0),
      ok_messages: Number(row.ok_messages || 0),
      failed_messages: Number(row.failed_messages || 0),
      major_issue_count: Number(row.major_issue_count || 0),
      minor_issue_count: Number(row.minor_issue_count || 0),
      recovered_issue_count: Number(row.recovered_issue_count || 0),
      contradiction_count: Number(row.contradiction_count || 0),
      high_effort_message_count: Number(row.high_effort_message_count || 0),
      unclear_message_count: Number(row.unclear_message_count || 0),
      poor_context_count: Number(row.poor_context_count || 0),
      customer_message_count: Number(row.customer_message_count || 0),
      agent_message_count: Number(row.agent_message_count || 0),
      bot_message_count: Number(row.bot_message_count || 0),
      broadcast_message_count: Number(row.broadcast_message_count || 0),
      distinct_skill_count: Number(row.distinct_skill_count || 0),
      handoff_count: Number(row.handoff_count || 0),
      had_unauthorized_skill: boolValue(row.had_unauthorized_skill),
      unauthorized_skill_count: Number(row.unauthorized_skill_count || 0),
      rag_retrieval_events: Number(row.rag_retrieval_events || 0),
      messages_with_rag: Number(row.messages_with_rag || 0),
      rag_coverage: numValue(row.rag_coverage),
      search_text: [
        row.ticket_id,
        row.external_customer_id,
        row.customer_name,
        row.status,
        row.category,
        row.request_origin,
        row.ticket_type,
        row.objective,
        row.handled_status,
        row.customer_experience,
        row.unhandled_resolution_subtype,
        row.frustration_origin,
        row.max_frustration_level,
        row.final_customer_sentiment,
        row.customer_primary_objective,
        row.management_summary,
        row.classification_reason,
        row.main_issue_type,
        row.main_issue_origin,
        row.main_issue_summary,
        row.customer_impact,
        row.score_band,
        row.confidence,
        row.lifecycle_risk,
        row.lifecycle_reason,
        row.analysis_skip_reason,
        ...(row.message_issue_types || []),
        ...(row.culprit_kinds || []),
        ...(row.culprit_agents || []),
        ...(row.handling_agents || []),
        ...(row.skills || []),
        ...(row.source_conversation_ids || [])
      ].filter(Boolean).join(" ").toLowerCase()
    }));

    const customerRows = sourceCustomers.map((row) => ({
      ...row,
      customer_id: Number(row.customer_id),
      source_conversations: Number(row.source_conversations || 0),
      messages: Number(row.messages || 0),
      tickets: Number(row.tickets || 0),
      open_tickets: Number(row.open_tickets || 0),
      closed_tickets: Number(row.closed_tickets || 0),
      manual_review_tickets: Number(row.manual_review_tickets || 0),
      bad_experience_tickets: Number(row.bad_experience_tickets || 0),
      search_text: [row.external_customer_id, row.customer_name].filter(Boolean).join(" ").toLowerCase()
    }));

    const client = await statsPool.connect();
    try {
      await client.query("BEGIN");
      const ticketIds = ticketRows.map((row) => row.ticket_id);
      const previousLifecycleStates = await loadPreviousLifecycleStates(client, { changedOnly, ticketIds });
      if (changedOnly) {
        const customerIds = customerRows.map((row) => row.customer_id);
        if (ticketIds.length) {
          await client.query("DELETE FROM ticket_index WHERE ticket_id=ANY($1::bigint[])", [ticketIds]);
        }
        if (customerIds.length) {
          await client.query("DELETE FROM customer_index WHERE customer_id=ANY($1::bigint[])", [customerIds]);
        }
      } else {
        await client.query("TRUNCATE ticket_index, customer_index");
      }
      for (const row of customerRows) {
        await client.query(
          `
          INSERT INTO customer_index(
            customer_id, external_customer_id, customer_name, source_conversations,
            messages, tickets, open_tickets, closed_tickets, manual_review_tickets,
            bad_experience_tickets, last_activity, search_text
          )
          VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
          `,
          [
            row.customer_id,
            row.external_customer_id,
            row.customer_name,
            row.source_conversations,
            row.messages,
            row.tickets,
            row.open_tickets,
            row.closed_tickets,
            row.manual_review_tickets,
            row.bad_experience_tickets,
            row.last_activity,
            row.search_text
          ]
        );
      }
      for (const row of ticketRows) {
        const values = ticketIndexColumns.map((column) => insertValue(row, column));
        await client.query(
          `
          INSERT INTO ticket_index(${ticketIndexColumns.join(", ")})
          VALUES(${ticketIndexColumns.map((_, index) => `$${index + 1}`).join(", ")})
          `,
          values
        );
      }
      await writeLifecycleHistory(client, previousLifecycleStates, ticketRows);
      const summary = await buildSummaryFromStats(client);
      const countRows = await client.query("SELECT count(*)::int AS row_count FROM ticket_index");
      const indexedRowCount = Number(countRows.rows[0]?.row_count || 0);
      const signature = {
        ...sourceSignatureFromRows(ticketRows, customerRows),
        refresh_mode: changedOnly ? "changed" : "full",
        run_id: runId || null,
        changed_tickets: ticketRows.length,
        changed_customers: customerRows.length,
        reason
      };
      await client.query(
        `
        INSERT INTO summary_cache(id, payload, refreshed_at)
        VALUES(true, $1::jsonb, now())
        ON CONFLICT(id) DO UPDATE SET payload=EXCLUDED.payload, refreshed_at=now()
        `,
        [JSON.stringify(summary)]
      );
      await client.query(
        `
        INSERT INTO cache_meta(cache_key, refreshed_at, source_signature, row_count, status, error)
        VALUES('ticket_index', now(), $1::jsonb, $2, 'ready', NULL)
        ON CONFLICT(cache_key) DO UPDATE SET
          refreshed_at=EXCLUDED.refreshed_at,
          source_signature=EXCLUDED.source_signature,
          row_count=EXCLUDED.row_count,
          status='ready',
          error=NULL
        `,
        [JSON.stringify(signature), indexedRowCount]
      );
      await client.query("COMMIT");
      refreshState = {
        running: false,
        lastStartedAt: refreshState.lastStartedAt,
        lastFinishedAt: new Date().toISOString(),
        lastError: "",
        lastMode: changedOnly ? "changed" : "full",
        lastReason: reason,
        lastRunId: runId || "",
        lastChangedRows: ticketRows.length,
        lastIndexedRows: indexedRowCount
      };
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }

    return cacheStatus();
  } catch (error) {
    refreshState = {
      ...refreshState,
      running: false,
      lastFinishedAt: new Date().toISOString(),
      lastError: error.message || String(error)
    };
    await statsQuery(
      `
      INSERT INTO cache_meta(cache_key, refreshed_at, source_signature, row_count, status, error)
      VALUES('ticket_index', now(), '{}'::jsonb, 0, 'error', $1)
      ON CONFLICT(cache_key) DO UPDATE SET refreshed_at=now(), status='error', error=EXCLUDED.error
      `,
      [refreshState.lastError]
    ).catch(() => {});
    throw error;
  }
}
