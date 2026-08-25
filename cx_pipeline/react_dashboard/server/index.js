import express from "express";
import crypto from "node:crypto";
import { cacheStatus, ensureInitialCache, refreshCache } from "./cache.js";
import { asNumber, config, pool, readonlyQuery } from "./db.js";
import { initStatsDb, statsPool, statsQuery } from "./statsDb.js";

const app = express();

app.use(express.json({ limit: "64kb" }));

function numberParam(value, fallback, max) {
  const parsed = Number.parseInt(value ?? "", 10);
  const clean = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(0, Math.min(clean, max));
}

function textParam(value) {
  return String(value ?? "").trim();
}

function listParam(value) {
  const items = Array.isArray(value) ? value : value === undefined ? [] : [value];
  return items.map((item) => textParam(item)).filter(Boolean);
}

function numberValue(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function statusClass(status) {
  if (!status) return "unknown";
  if (status.includes("pending") || status.includes("unresolved") || status === "unhandled") return "open";
  if (status.includes("resolved") || status === "handled") return "closed";
  return status;
}

function phoneHashCandidates(search) {
  const text = textParam(search);
  if (!text) return [];
  const values = new Set([text]);
  const compact = text.replace(/\s+/g, "");
  if (compact) values.add(compact);
  const digits = text.replace(/\D+/g, "");
  if (digits) values.add(digits);
  if (digits.startsWith("00")) values.add(`+${digits.slice(2)}`);
  if (digits && !digits.startsWith("0")) values.add(`+${digits}`);
  return [...values]
    .map((value) => crypto.createHash("sha256").update(value).digest("hex"))
    .filter((value, index, arr) => arr.indexOf(value) === index);
}

function addParam(params, value) {
  params.push(value);
  return `$${params.length}`;
}

function buildTicketFilters(query, { dateOverride = null } = {}) {
  const params = [];
  const where = [];
  const addTextArray = (paramName, column) => {
    const values = listParam(query[paramName]);
    if (values.length) where.push(`${column} = ANY(${addParam(params, values)}::text[])`);
  };

  addTextArray("status", "status");
  addTextArray("category", "category");
  addTextArray("ticketType", "ticket_type");
  addTextArray("requestOrigin", "request_origin");
  addTextArray("handled", "handled_status");
  addTextArray("experience", "customer_experience");
  addTextArray("unresolved", "unhandled_resolution_subtype");
  addTextArray("frustrationOrigin", "frustration_origin");
  addTextArray("maxFrustration", "max_frustration_level");
  addTextArray("mainIssue", "main_issue_type");
  addTextArray("scoreBand", "score_band");
  addTextArray("runId", "latest_run_id::text");
  addTextArray("lifecycleRisk", "lifecycle_risk");

  const manualReview = listParam(query.manualReview);
  if (manualReview.length) {
    const bools = manualReview.map((value) => value === "true").filter((_, index) => ["true", "false"].includes(manualReview[index]));
    if (bools.length) where.push(`manual_review_required = ANY(${addParam(params, bools)}::boolean[])`);
  }

  const analysisEligible = listParam(query.analysisEligible);
  if (analysisEligible.length) {
    const bools = analysisEligible.map((value) => value === "true").filter((_, index) => ["true", "false"].includes(analysisEligible[index]));
    if (bools.length) where.push(`analysis_eligible = ANY(${addParam(params, bools)}::boolean[])`);
  }

  const messageIssues = listParam(query.messageIssue);
  if (messageIssues.length) where.push(`message_issue_types && ${addParam(params, messageIssues)}::text[]`);

  const culprit = listParam(query.culprit);
  if (culprit.length) where.push(`culprit_kinds && ${addParam(params, culprit)}::text[]`);

  const agents = listParam(query.agent);
  if (agents.length) {
    const placeholder = addParam(params, agents);
    where.push(`(culprit_agents && ${placeholder}::text[] OR handling_agents && ${placeholder}::text[])`);
  }

  const hasContradiction = textParam(query.hasContradiction);
  if (hasContradiction === "true") where.push("contradiction_count > 0");
  if (hasContradiction === "false") where.push("contradiction_count = 0");

  const customerId = textParam(query.customerId);
  if (customerId) where.push(`customer_id = ${addParam(params, customerId)}::bigint`);

  const scoreMin = numberValue(query.scoreMin);
  if (scoreMin !== null) where.push(`score_final_100 >= ${addParam(params, scoreMin)}::numeric`);
  const scoreMax = numberValue(query.scoreMax);
  if (scoreMax !== null) where.push(`score_final_100 <= ${addParam(params, scoreMax)}::numeric`);

  const search = textParam(query.search);
  if (search) {
    const hashes = phoneHashCandidates(search);
    const searchPlaceholder = addParam(params, search);
    const hashPlaceholder = addParam(params, hashes);
    where.push(`(search_text ILIKE '%' || lower(${searchPlaceholder}) || '%' OR external_customer_id = ANY(${hashPlaceholder}::text[]))`);
  }

  const requestedDateField = textParam(query.dateField);
  const dateField = {
    first: "first_message_at",
    opened: "opened_at",
    closed: "closed_at"
  }[requestedDateField] || "last_message_at";
  const from = dateOverride?.from ?? textParam(query.from);
  const to = dateOverride?.to ?? textParam(query.to);
  if (from) where.push(`${dateField} >= ${addParam(params, from)}::timestamptz`);
  if (to) where.push(`${dateField} < ${addParam(params, to)}::timestamptz`);

  return {
    params,
    whereSql: where.length ? `WHERE ${where.join(" AND ")}` : "",
    dateField
  };
}

function orderSql(sort, scoreOrder) {
  const chosen = textParam(sort) || (scoreOrder === "best" ? "score_desc" : scoreOrder === "worst" ? "score_asc" : "recent");
  const frustrationRank = "CASE max_frustration_level WHEN 'cancellation_risk' THEN 5 WHEN 'high' THEN 4 WHEN 'medium' THEN 3 WHEN 'low' THEN 2 ELSE 1 END";
  if (chosen === "oldest") return "last_message_at ASC NULLS LAST, ticket_id ASC";
  if (chosen === "score_asc") return "score_final_100 ASC NULLS LAST, last_message_at DESC NULLS LAST, ticket_id DESC";
  if (chosen === "score_desc") return "score_final_100 DESC NULLS LAST, last_message_at DESC NULLS LAST, ticket_id DESC";
  if (chosen === "frustration_desc") return `${frustrationRank} DESC, last_message_at DESC NULLS LAST, ticket_id DESC`;
  if (chosen === "messages_desc") return "linked_messages DESC, last_message_at DESC NULLS LAST, ticket_id DESC";
  return "last_message_at DESC NULLS LAST, ticket_id DESC";
}

async function summaryForFilters(query, dateOverride = null) {
  const { whereSql, params } = buildTicketFilters(query, { dateOverride });
  const rows = await statsQuery(
    `
    SELECT
      count(DISTINCT customer_id)::int AS customers,
      COALESCE(sum(source_conversation_count)::int, 0) AS source_conversations,
      COALESCE(sum(linked_messages)::int, 0) AS messages,
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
      count(*) FILTER (WHERE contradiction_count > 0)::int AS contradiction_tickets,
      avg(score_final_100)::float AS avg_score_100
    FROM ticket_index
    ${whereSql}
    `,
    params
  );
  return rows[0] || {};
}

app.get("/api/health", async (_req, res, next) => {
  try {
    const rows = await readonlyQuery("SELECT now() AS now");
    res.json({ ok: true, database: "readonly", stats_cache: cacheStatus(), now: rows[0]?.now });
  } catch (error) {
    next(error);
  }
});

app.get("/api/summary", async (_req, res, next) => {
  try {
    const hasFilters = Object.keys(_req.query || {}).length > 0;
    if (!hasFilters) {
      const rows = await statsQuery("SELECT payload, refreshed_at FROM summary_cache WHERE id=true");
      res.json({ ...(rows[0]?.payload || {}), previous: null, cache: { refreshed_at: rows[0]?.refreshed_at, ...cacheStatus() } });
      return;
    }
    const totals = await summaryForFilters(_req.query);
    const from = textParam(_req.query.from);
    const to = textParam(_req.query.to);
    let previous = null;
    if (from && to) {
      const fromDate = new Date(from);
      const toDate = new Date(to);
      if (Number.isFinite(fromDate.getTime()) && Number.isFinite(toDate.getTime()) && toDate > fromDate) {
        const windowMs = toDate.getTime() - fromDate.getTime();
        previous = await summaryForFilters(_req.query, {
          from: new Date(fromDate.getTime() - windowMs).toISOString(),
          to: fromDate.toISOString()
        });
      }
    }
    res.json({ totals, previous, cache: cacheStatus() });
  } catch (error) {
    next(error);
  }
});

app.get("/api/cache/status", async (_req, res, next) => {
  try {
    const rows = await statsQuery("SELECT * FROM cache_meta WHERE cache_key='ticket_index'");
    res.json({ memory: cacheStatus(), database: rows[0] || null });
  } catch (error) {
    next(error);
  }
});

app.post("/api/cache/refresh", async (_req, res, next) => {
  try {
    res.json(await refreshCache({ mode: "full", reason: "manual_api_refresh" }));
  } catch (error) {
    next(error);
  }
});

app.post("/api/webhooks/analysis-run-finished", async (req, res, next) => {
  try {
    if (config.webhookSecret) {
      const supplied = req.get("x-cx-dashboard-webhook-secret") || "";
      if (supplied !== config.webhookSecret) {
        res.status(401).json({ error: "Invalid webhook secret" });
        return;
      }
    }
    const runId = textParam(req.body?.run_id || req.body?.runId);
    res.json(await refreshCache({ mode: runId ? "changed" : "full", runId, reason: "analysis_run_finished_webhook" }));
  } catch (error) {
    next(error);
  }
});

app.get("/api/customers", async (req, res, next) => {
  try {
    const search = textParam(req.query.search);
    const hashes = phoneHashCandidates(search);
    const limit = numberParam(req.query.limit, 80, 250);
    const rows = await statsQuery(
      `
      SELECT
        customer_id AS id,
        external_customer_id,
        customer_name,
        source_conversations,
        messages,
        tickets,
        closed_tickets,
        open_tickets,
        manual_review_tickets,
        bad_experience_tickets,
        last_activity
      FROM customer_index
      WHERE (
        $1 = ''
        OR search_text ILIKE '%' || lower($1) || '%'
        OR external_customer_id = ANY($2::text[])
      )
      ORDER BY last_activity DESC NULLS LAST, customer_id DESC
      LIMIT $3
      `,
      [search, hashes, limit]
    );
    res.json({ customers: rows });
  } catch (error) {
    next(error);
  }
});

app.get("/api/tickets", async (req, res, next) => {
  try {
    const { whereSql, params } = buildTicketFilters(req.query);
    const limit = numberParam(req.query.limit, 50, 500);
    const offset = numberParam(req.query.offset, 0, 100000);
    const limitPlaceholder = addParam(params, limit);
    const offsetPlaceholder = addParam(params, offset);

    const rows = await statsQuery(
      `
      SELECT
        ticket_id,
        customer_id,
        customer_name,
        status,
        open_closed,
        category,
        ticket_type,
        request_origin,
        handled_status,
        customer_experience,
        max_frustration_level,
        frustration_origin,
        main_issue_type,
        score_final_100,
        score_rating,
        score_band,
        manual_review_required,
        confidence,
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
        lifecycle_reason,
        analysis_eligible,
        analysis_skip_reason,
        left(COALESCE(main_issue_summary, ''), 160) AS main_issue_summary,
        left(COALESCE(customer_primary_objective, ''), 120) AS customer_primary_objective,
        count(*) OVER()::int AS total_count
      FROM ticket_index
      ${whereSql}
      ORDER BY ${orderSql(req.query.sort, req.query.scoreOrder)}
      LIMIT ${limitPlaceholder} OFFSET ${offsetPlaceholder}
      `,
      params
    );
    const total = rows[0]?.total_count || 0;
    res.json({
      tickets: rows.map(({ total_count: _totalCount, ...row }) => row),
      total,
      limit,
      offset
    });
  } catch (error) {
    next(error);
  }
});

app.get("/api/tickets/facets", async (req, res, next) => {
  try {
    const { whereSql, params } = buildTicketFilters(req.query);
    const rows = await statsQuery(
      `
      WITH filtered AS (
        SELECT *
        FROM ticket_index
        ${whereSql}
      ),
      facet_rows AS (
        SELECT 'status' AS facet, status::text AS value, count(*)::int AS count FROM filtered GROUP BY status
        UNION ALL SELECT 'category', category::text, count(*)::int FROM filtered GROUP BY category
        UNION ALL SELECT 'ticket_type', ticket_type::text, count(*)::int FROM filtered GROUP BY ticket_type
        UNION ALL SELECT 'request_origin', request_origin::text, count(*)::int FROM filtered GROUP BY request_origin
        UNION ALL SELECT 'handled_status', handled_status::text, count(*)::int FROM filtered GROUP BY handled_status
        UNION ALL SELECT 'customer_experience', customer_experience::text, count(*)::int FROM filtered GROUP BY customer_experience
        UNION ALL SELECT 'unhandled_resolution_subtype', unhandled_resolution_subtype::text, count(*)::int FROM filtered GROUP BY unhandled_resolution_subtype
        UNION ALL SELECT 'frustration_origin', frustration_origin::text, count(*)::int FROM filtered GROUP BY frustration_origin
        UNION ALL SELECT 'frustration_timing', frustration_timing::text, count(*)::int FROM filtered GROUP BY frustration_timing
        UNION ALL SELECT 'max_frustration_level', max_frustration_level::text, count(*)::int FROM filtered GROUP BY max_frustration_level
        UNION ALL SELECT 'final_customer_sentiment', final_customer_sentiment::text, count(*)::int FROM filtered GROUP BY final_customer_sentiment
        UNION ALL SELECT 'main_issue_type', main_issue_type::text, count(*)::int FROM filtered GROUP BY main_issue_type
        UNION ALL SELECT 'main_issue_origin', main_issue_origin::text, count(*)::int FROM filtered GROUP BY main_issue_origin
        UNION ALL SELECT 'score_rating', score_rating::text, count(*)::int FROM filtered GROUP BY score_rating
        UNION ALL SELECT 'score_band', score_band::text, count(*)::int FROM filtered GROUP BY score_band
        UNION ALL SELECT 'manual_review_required', manual_review_required::text, count(*)::int FROM filtered GROUP BY manual_review_required
        UNION ALL SELECT 'analysis_eligible', analysis_eligible::text, count(*)::int FROM filtered GROUP BY analysis_eligible
        UNION ALL SELECT 'lifecycle_risk', lifecycle_risk::text, count(*)::int FROM filtered GROUP BY lifecycle_risk
        UNION ALL SELECT 'confidence', confidence::text, count(*)::int FROM filtered GROUP BY confidence
        UNION ALL SELECT 'culprit_kind', value::text, count(*)::int FROM filtered, unnest(culprit_kinds) value GROUP BY value
        UNION ALL SELECT 'message_issue_type', value::text, count(*)::int FROM filtered, unnest(message_issue_types) value GROUP BY value
        UNION ALL SELECT 'run_id', latest_run_id::text, count(*)::int FROM filtered GROUP BY latest_run_id
      ),
      grouped AS (
        SELECT
          facet,
          jsonb_agg(jsonb_build_object('value', value, 'count', count) ORDER BY count DESC, value) AS values
        FROM facet_rows
        WHERE value IS NOT NULL AND value <> ''
        GROUP BY facet
      )
      SELECT
        (SELECT count(*)::int FROM filtered) AS total,
        COALESCE(jsonb_object_agg(facet, values), '{}'::jsonb) AS facets
      FROM grouped
      `,
      params
    );
    res.json({ total: rows[0]?.total || 0, facets: rows[0]?.facets || {}, countMode: "uniform_active_filters" });
  } catch (error) {
    next(error);
  }
});

app.get("/api/trends", async (req, res, next) => {
  try {
    const bucket = textParam(req.query.bucket) === "week" ? "week" : "day";
    const interval = bucket === "week" ? "1 week" : "1 day";
    const { whereSql, params } = buildTicketFilters(req.query);
    const bucketPlaceholder = addParam(params, bucket);
    const intervalPlaceholder = addParam(params, interval);
    const rows = await statsQuery(
      `
      WITH filtered AS (
        SELECT *
        FROM ticket_index
        ${whereSql}
      ),
      bounds AS (
        SELECT
          date_trunc(${bucketPlaceholder}, COALESCE(min(last_message_at), now())) AS start_bucket,
          date_trunc(${bucketPlaceholder}, COALESCE(max(last_message_at), now())) AS end_bucket
        FROM filtered
      ),
      series AS (
        SELECT generate_series(start_bucket, end_bucket, ${intervalPlaceholder}::interval) AS bucket_start
        FROM bounds
      ),
      grouped AS (
        SELECT
          date_trunc(${bucketPlaceholder}, last_message_at) AS bucket_start,
          count(*)::int AS tickets,
          count(*) FILTER (WHERE customer_experience='bad')::int AS bad,
          count(*) FILTER (WHERE handled_status='unhandled')::int AS unhandled,
          count(*) FILTER (WHERE manual_review_required)::int AS manual_review,
          count(*) FILTER (WHERE max_frustration_level IN ('high', 'cancellation_risk'))::int AS high_frustration,
          count(*) FILTER (WHERE contradiction_count > 0)::int AS contradictions,
          avg(score_final_100)::float AS avg_score_100
        FROM filtered
        WHERE last_message_at IS NOT NULL
        GROUP BY 1
      )
      SELECT
        to_char(series.bucket_start, CASE WHEN ${bucketPlaceholder}='week' THEN 'YYYY-MM-DD' ELSE 'YYYY-MM-DD' END) AS bucket,
        COALESCE(grouped.tickets, 0) AS tickets,
        COALESCE(grouped.bad, 0) AS bad,
        COALESCE(grouped.unhandled, 0) AS unhandled,
        COALESCE(grouped.manual_review, 0) AS manual_review,
        COALESCE(grouped.high_frustration, 0) AS high_frustration,
        COALESCE(grouped.contradictions, 0) AS contradictions,
        grouped.avg_score_100
      FROM series
      LEFT JOIN grouped ON grouped.bucket_start=series.bucket_start
      ORDER BY series.bucket_start
      `,
      params
    );
    res.json(rows);
  } catch (error) {
    next(error);
  }
});

app.get("/api/tickets/:id/lifecycle", async (req, res, next) => {
  try {
    const id = asNumber(req.params.id);
    const limit = numberParam(req.query.limit, 100, 500);
    const rows = await statsQuery(
      `
      SELECT
        id,
        ticket_id,
        customer_id,
        status,
        open_closed,
        opened_at,
        closed_at,
        first_message_at,
        last_message_at,
        source_conversation_ids,
        latest_run_id,
        changed_at,
        event_type,
        previous_state,
        current_state
      FROM ticket_lifecycle_history
      WHERE ticket_id=$1
      ORDER BY changed_at DESC, id DESC
      LIMIT $2
      `,
      [id, limit]
    );
    res.json({ ticket_id: id, history: rows });
  } catch (error) {
    next(error);
  }
});

app.get("/api/tickets/:id", async (req, res, next) => {
  try {
    const id = asNumber(req.params.id);
    const [ticketRows, messageRows, resultRows, statsRows, lifecycleRows] = await Promise.all([
      readonlyQuery(
        `
        SELECT
          t.*,
          c.external_customer_id,
          c.customer_name,
          c.metadata AS customer_metadata,
          cx.parse_status AS ticket_cx_parse_status,
          cx.result AS cx_result,
          cx.computed_metadata,
          cx.error AS cx_error
        FROM tickets t
        JOIN customers c ON c.id=t.customer_id
        LEFT JOIN ticket_cx_results cx ON cx.ticket_id=t.id
        WHERE t.id=$1
        `,
        [id]
      ),
      readonlyQuery(
        `
        SELECT
          m.id,
          m.customer_message_index,
          m.source_message_index,
          m.sender_role,
          m.raw_sender_role,
          m.message_time,
          m.message_text,
          m.raw,
          sc.source_conversation_id,
          sc.metadata AS source_metadata,
          mr.parse_status AS message_parse_status,
          mr.result AS message_result,
          mr.error AS message_error
        FROM ticket_messages tm
        JOIN messages m ON m.id=tm.message_id
        JOIN source_conversations sc ON sc.id=m.source_conversation_pk
        LEFT JOIN message_results mr ON mr.ticket_id=tm.ticket_id AND mr.message_id=m.id
        WHERE tm.ticket_id=$1
        ORDER BY m.customer_message_index ASC, m.id ASC
        `,
        [id]
      ),
      readonlyQuery(
        `
        SELECT
          count(*)::int AS evaluated_messages,
          count(*) FILTER (WHERE parse_status='ok')::int AS ok_messages,
          count(*) FILTER (WHERE parse_status <> 'ok')::int AS failed_messages
        FROM message_results
        WHERE ticket_id=$1
        `,
        [id]
      ),
      statsQuery(
        `
        SELECT *
        FROM ticket_index
        WHERE ticket_id=$1
        `,
        [id]
      ),
      statsQuery(
        `
        SELECT
          id,
          ticket_id,
          customer_id,
          status,
          open_closed,
          opened_at,
          closed_at,
          first_message_at,
          last_message_at,
          source_conversation_ids,
          latest_run_id,
          changed_at,
          event_type,
          previous_state,
          current_state
        FROM ticket_lifecycle_history
        WHERE ticket_id=$1
        ORDER BY changed_at DESC, id DESC
        LIMIT 100
        `,
        [id]
      )
    ]);

    if (!ticketRows[0]) {
      res.status(404).json({ error: "Ticket not found" });
      return;
    }
    res.json({
      ticket: ticketRows[0],
      messages: messageRows,
      messageStats: resultRows[0],
      stats: statsRows[0] || null,
      lifecycle: lifecycleRows
    });
  } catch (error) {
    next(error);
  }
});

app.get("/api/customers/:id/journey", async (req, res, next) => {
  try {
    const id = asNumber(req.params.id);
    const [customerRows, ticketRows, messageRows] = await Promise.all([
      readonlyQuery("SELECT * FROM customers WHERE id=$1", [id]),
      readonlyQuery(
        `
        SELECT
          t.id,
          t.status,
          t.category,
          t.ticket_type,
          t.objective,
          t.segmentation,
          COALESCE(cx.result->>'handled_status', 'not_analyzed') AS handled_status,
          COALESCE(cx.result->>'customer_experience', 'not_analyzed') AS customer_experience,
          NULLIF(cx.result->'conversation_score'->>'final_score_100', '')::numeric AS score_100
        FROM tickets t
        LEFT JOIN ticket_cx_results cx ON cx.ticket_id=t.id
        WHERE t.customer_id=$1
        ORDER BY (t.segmentation->>'start_message_index')::int NULLS LAST, t.id
        `,
        [id]
      ),
      readonlyQuery(
        `
        SELECT
          m.id,
          m.customer_message_index,
          m.sender_role,
          m.raw_sender_role,
          m.message_time,
          m.message_text,
          m.raw,
          sc.source_conversation_id,
          sc.metadata AS source_metadata,
          COALESCE(
            jsonb_agg(
              jsonb_build_object(
                'ticket_id', t.id,
                'status', t.status,
                'category', t.category,
                'ticket_type', t.ticket_type,
                'objective', t.objective,
                'handled_status', COALESCE(cx.result->>'handled_status', 'not_analyzed'),
                'customer_experience', COALESCE(cx.result->>'customer_experience', 'not_analyzed')
              )
              ORDER BY t.id
            ) FILTER (WHERE t.id IS NOT NULL),
            '[]'::jsonb
          ) AS tickets
        FROM messages m
        JOIN source_conversations sc ON sc.id=m.source_conversation_pk
        LEFT JOIN ticket_messages tm ON tm.message_id=m.id
        LEFT JOIN tickets t ON t.id=tm.ticket_id
        LEFT JOIN ticket_cx_results cx ON cx.ticket_id=t.id
        WHERE m.customer_id=$1
        GROUP BY m.id, sc.source_conversation_id, sc.metadata
        ORDER BY m.customer_message_index ASC, m.id ASC
        `,
        [id]
      )
    ]);

    if (!customerRows[0]) {
      res.status(404).json({ error: "Customer not found" });
      return;
    }
    res.json({ customer: customerRows[0], tickets: ticketRows, messages: messageRows });
  } catch (error) {
    next(error);
  }
});

app.get("/api/filter-options", async (_req, res, next) => {
  try {
    const rows = await statsQuery(`
      WITH facet_rows AS (
        SELECT 'statuses' AS key, status::text AS value, count(*)::int AS count FROM ticket_index GROUP BY status
        UNION ALL SELECT 'categories', category::text, count(*)::int FROM ticket_index GROUP BY category
        UNION ALL SELECT 'requestOrigins', request_origin::text, count(*)::int FROM ticket_index GROUP BY request_origin
        UNION ALL SELECT 'ticketTypes', ticket_type::text, count(*)::int FROM ticket_index GROUP BY ticket_type
        UNION ALL SELECT 'handled', handled_status::text, count(*)::int FROM ticket_index GROUP BY handled_status
        UNION ALL SELECT 'experiences', customer_experience::text, count(*)::int FROM ticket_index GROUP BY customer_experience
        UNION ALL SELECT 'unresolved', unhandled_resolution_subtype::text, count(*)::int FROM ticket_index GROUP BY unhandled_resolution_subtype
        UNION ALL SELECT 'frustrationOrigins', frustration_origin::text, count(*)::int FROM ticket_index GROUP BY frustration_origin
        UNION ALL SELECT 'maxFrustration', max_frustration_level::text, count(*)::int FROM ticket_index GROUP BY max_frustration_level
        UNION ALL SELECT 'mainIssues', main_issue_type::text, count(*)::int FROM ticket_index GROUP BY main_issue_type
        UNION ALL SELECT 'scoreRatings', score_rating::text, count(*)::int FROM ticket_index GROUP BY score_rating
        UNION ALL SELECT 'scoreBands', score_band::text, count(*)::int FROM ticket_index GROUP BY score_band
        UNION ALL SELECT 'lifecycleRisks', lifecycle_risk::text, count(*)::int FROM ticket_index GROUP BY lifecycle_risk
        UNION ALL SELECT 'analysisEligible', analysis_eligible::text, count(*)::int FROM ticket_index GROUP BY analysis_eligible
        UNION ALL SELECT 'messageIssues', value::text, count(*)::int FROM ticket_index, unnest(message_issue_types) value GROUP BY value
      )
      SELECT key, value, count
      FROM facet_rows
      WHERE value IS NOT NULL AND value <> ''
      ORDER BY key, value
    `);
    const grouped = rows.reduce((out, row) => {
      out[row.key] = out[row.key] || [];
      out[row.key].push({ value: row.value, count: row.count });
      return out;
    }, {});
    res.json({
      ...grouped,
      manualReview: [{ value: "true" }, { value: "false" }],
      analysisEligible: grouped.analysisEligible || [{ value: "true" }, { value: "false" }]
    });
  } catch (error) {
    next(error);
  }
});

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(500).json({ error: error.message || "Unexpected server error" });
});

process.on("SIGINT", async () => {
  await pool.end();
  await statsPool.end();
  process.exit(130);
});

process.on("SIGTERM", async () => {
  await pool.end();
  await statsPool.end();
  process.exit(143);
});

async function start() {
  await initStatsDb();
  await ensureInitialCache().catch((error) => {
    console.error("Initial dashboard cache check failed:", error);
  });
  app.listen(config.port, "127.0.0.1", () => {
    console.log(`CX Express dashboard API listening on http://127.0.0.1:${config.port}`);
    console.log("Dashboard stats cache refresh: webhook/manual only");
  });
}

start().catch((error) => {
  console.error(error);
  process.exit(1);
});
