import pg from "pg";
import { config } from "./db.js";

export const statsPool = new pg.Pool({
  connectionString: config.statsDatabaseUrl,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
  application_name: "cx_react_dashboard_stats_cache"
});

export async function statsQuery(sql, params = []) {
  const result = await statsPool.query(sql, params);
  return result.rows;
}

export async function initStatsDb() {
  await statsPool.query(`
    CREATE TABLE IF NOT EXISTS cache_meta (
      cache_key TEXT PRIMARY KEY,
      refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      source_signature JSONB NOT NULL DEFAULT '{}'::jsonb,
      row_count INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'ready',
      error TEXT
    );

    CREATE TABLE IF NOT EXISTS summary_cache (
      id BOOLEAN PRIMARY KEY DEFAULT true,
      payload JSONB NOT NULL,
      refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      CHECK (id)
    );

    CREATE TABLE IF NOT EXISTS customer_index (
      customer_id BIGINT PRIMARY KEY,
      external_customer_id TEXT NOT NULL,
      customer_name TEXT,
      source_conversations INTEGER NOT NULL DEFAULT 0,
      messages INTEGER NOT NULL DEFAULT 0,
      tickets INTEGER NOT NULL DEFAULT 0,
      open_tickets INTEGER NOT NULL DEFAULT 0,
      closed_tickets INTEGER NOT NULL DEFAULT 0,
      manual_review_tickets INTEGER NOT NULL DEFAULT 0,
      bad_experience_tickets INTEGER NOT NULL DEFAULT 0,
      last_activity TIMESTAMPTZ,
      search_text TEXT NOT NULL DEFAULT '',
      refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS ticket_index (
      ticket_id BIGINT PRIMARY KEY,
      customer_id BIGINT NOT NULL,
      external_customer_id TEXT NOT NULL,
      customer_name TEXT,
      status TEXT,
      open_closed TEXT,
      category TEXT,
      request_origin TEXT,
      ticket_type TEXT,
      objective TEXT,
      should_append_future BOOLEAN NOT NULL DEFAULT false,
      start_message_index INTEGER,
      end_message_index INTEGER,
      segment_messages INTEGER NOT NULL DEFAULT 0,
      linked_messages INTEGER NOT NULL DEFAULT 0,
      source_conversation_count INTEGER NOT NULL DEFAULT 0,
      source_conversation_ids TEXT[] NOT NULL DEFAULT '{}'::text[],
      first_message_at TIMESTAMPTZ,
      last_message_at TIMESTAMPTZ,
      opened_at TIMESTAMPTZ,
      closed_at TIMESTAMPTZ,
      reopenable_until TIMESTAMPTZ,
      lifecycle_risk TEXT NOT NULL DEFAULT 'normal',
      lifecycle_reason TEXT,
      analysis_eligible BOOLEAN NOT NULL DEFAULT true,
      analysis_skip_reason TEXT,
      span_seconds INTEGER,
      first_response_seconds INTEGER,
      analyzed_at TIMESTAMPTZ,
      evaluated_messages INTEGER NOT NULL DEFAULT 0,
      ok_messages INTEGER NOT NULL DEFAULT 0,
      failed_messages INTEGER NOT NULL DEFAULT 0,
      handled_status TEXT,
      customer_experience TEXT,
      unhandled_resolution_subtype TEXT,
      frustration_detected BOOLEAN NOT NULL DEFAULT false,
      frustration_origin TEXT,
      frustration_timing TEXT,
      max_frustration_level TEXT,
      final_customer_sentiment TEXT,
      customer_started_frustrated BOOLEAN NOT NULL DEFAULT false,
      customer_became_frustrated_during_chat BOOLEAN NOT NULL DEFAULT false,
      customer_ended_frustrated BOOLEAN NOT NULL DEFAULT false,
      manual_review_required BOOLEAN NOT NULL DEFAULT false,
      manual_review_reason TEXT,
      customer_objective_type TEXT,
      customer_primary_objective TEXT,
      management_summary TEXT,
      classification_reason TEXT,
      score_final NUMERIC,
      score_final_100 NUMERIC,
      score_rating TEXT,
      score_band TEXT,
      score_explanation TEXT,
      score_resolution NUMERIC,
      score_context_understanding NUMERIC,
      score_customer_effort NUMERIC,
      score_frustration_risk NUMERIC,
      score_ai_judgment NUMERIC,
      score_message_signals NUMERIC,
      score_raw_total NUMERIC,
      main_issue_type TEXT,
      main_issue_origin TEXT,
      main_issue_summary TEXT,
      customer_impact TEXT,
      culprits JSONB NOT NULL DEFAULT '[]'::jsonb,
      culprit_agent_names JSONB NOT NULL DEFAULT '[]'::jsonb,
      culprit_reason TEXT,
      positive_signals JSONB NOT NULL DEFAULT '[]'::jsonb,
      negative_signals JSONB NOT NULL DEFAULT '[]'::jsonb,
      recommended_actions JSONB NOT NULL DEFAULT '[]'::jsonb,
      all_detected_issues JSONB NOT NULL DEFAULT '[]'::jsonb,
      message_issue_types TEXT[] NOT NULL DEFAULT '{}'::text[],
      major_issue_count INTEGER NOT NULL DEFAULT 0,
      minor_issue_count INTEGER NOT NULL DEFAULT 0,
      recovered_issue_count INTEGER NOT NULL DEFAULT 0,
      contradiction_count INTEGER NOT NULL DEFAULT 0,
      first_contradiction_message_id BIGINT,
      max_message_frustration TEXT,
      high_effort_message_count INTEGER NOT NULL DEFAULT 0,
      unclear_message_count INTEGER NOT NULL DEFAULT 0,
      poor_context_count INTEGER NOT NULL DEFAULT 0,
      culprit_kinds TEXT[] NOT NULL DEFAULT '{}'::text[],
      culprit_agents TEXT[] NOT NULL DEFAULT '{}'::text[],
      handling_agents TEXT[] NOT NULL DEFAULT '{}'::text[],
      customer_message_count INTEGER NOT NULL DEFAULT 0,
      agent_message_count INTEGER NOT NULL DEFAULT 0,
      bot_message_count INTEGER NOT NULL DEFAULT 0,
      broadcast_message_count INTEGER NOT NULL DEFAULT 0,
      latest_run_id UUID,
      latest_run_at TIMESTAMPTZ,
      skills TEXT[] NOT NULL DEFAULT '{}'::text[],
      initial_skill TEXT,
      last_skill TEXT,
      distinct_skill_count INTEGER NOT NULL DEFAULT 0,
      handoff_count INTEGER NOT NULL DEFAULT 0,
      had_unauthorized_skill BOOLEAN NOT NULL DEFAULT false,
      unauthorized_skill_count INTEGER NOT NULL DEFAULT 0,
      rag_retrieval_events INTEGER NOT NULL DEFAULT 0,
      messages_with_rag INTEGER NOT NULL DEFAULT 0,
      rag_coverage NUMERIC,
      confidence TEXT,
      segmentation JSONB NOT NULL DEFAULT '{}'::jsonb,
      cx_result JSONB NOT NULL DEFAULT '{}'::jsonb,
      computed_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
      updated_at TIMESTAMPTZ,
      refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      search_text TEXT NOT NULL DEFAULT ''
    );

    ALTER TABLE ticket_index
    ADD COLUMN IF NOT EXISTS evaluated_messages INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS ok_messages INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS failed_messages INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS first_message_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_message_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS opened_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reopenable_until TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS lifecycle_risk TEXT NOT NULL DEFAULT 'normal',
    ADD COLUMN IF NOT EXISTS lifecycle_reason TEXT,
    ADD COLUMN IF NOT EXISTS analysis_eligible BOOLEAN NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS analysis_skip_reason TEXT,
    ADD COLUMN IF NOT EXISTS span_seconds INTEGER,
    ADD COLUMN IF NOT EXISTS first_response_seconds INTEGER,
    ADD COLUMN IF NOT EXISTS analyzed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS score_band TEXT,
    ADD COLUMN IF NOT EXISTS message_issue_types TEXT[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS major_issue_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS minor_issue_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS recovered_issue_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS contradiction_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS first_contradiction_message_id BIGINT,
    ADD COLUMN IF NOT EXISTS max_message_frustration TEXT,
    ADD COLUMN IF NOT EXISTS high_effort_message_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS unclear_message_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS poor_context_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS culprit_kinds TEXT[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS culprit_agents TEXT[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS handling_agents TEXT[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS customer_message_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS agent_message_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS bot_message_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS broadcast_message_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS latest_run_id UUID,
    ADD COLUMN IF NOT EXISTS latest_run_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS skills TEXT[] NOT NULL DEFAULT '{}'::text[],
    ADD COLUMN IF NOT EXISTS initial_skill TEXT,
    ADD COLUMN IF NOT EXISTS last_skill TEXT,
    ADD COLUMN IF NOT EXISTS distinct_skill_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS handoff_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS had_unauthorized_skill BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS unauthorized_skill_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS rag_retrieval_events INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS messages_with_rag INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS rag_coverage NUMERIC,
    ADD COLUMN IF NOT EXISTS confidence TEXT;

    CREATE TABLE IF NOT EXISTS ticket_lifecycle_history (
      id BIGSERIAL PRIMARY KEY,
      ticket_id BIGINT NOT NULL,
      customer_id BIGINT,
      status TEXT,
      open_closed TEXT,
      opened_at TIMESTAMPTZ,
      closed_at TIMESTAMPTZ,
      first_message_at TIMESTAMPTZ,
      last_message_at TIMESTAMPTZ,
      source_conversation_ids TEXT[] NOT NULL DEFAULT '{}'::text[],
      latest_run_id UUID,
      changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      event_type TEXT NOT NULL,
      previous_state JSONB NOT NULL DEFAULT '{}'::jsonb,
      current_state JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE INDEX IF NOT EXISTS idx_ticket_index_filters
    ON ticket_index(status, category, handled_status, customer_experience, frustration_origin, manual_review_required);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_customer
    ON ticket_index(customer_id, updated_at DESC);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_score
    ON ticket_index(score_final_100 ASC NULLS LAST);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_last_msg
    ON ticket_index(last_message_at DESC NULLS LAST, ticket_id DESC);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_first_msg
    ON ticket_index(first_message_at);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_opened_at
    ON ticket_index(opened_at);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_closed_at
    ON ticket_index(closed_at DESC NULLS LAST);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_reopenable_until
    ON ticket_index(reopenable_until DESC NULLS LAST);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_lifecycle_risk
    ON ticket_index(lifecycle_risk);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_analysis_eligible
    ON ticket_index(analysis_eligible);

    CREATE INDEX IF NOT EXISTS idx_ticket_lifecycle_ticket
    ON ticket_lifecycle_history(ticket_id, changed_at DESC, id DESC);

    CREATE INDEX IF NOT EXISTS idx_ticket_lifecycle_run
    ON ticket_lifecycle_history(latest_run_id);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_score_band
    ON ticket_index(score_band);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_msg_issues
    ON ticket_index USING gin(message_issue_types);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_culprit_kinds
    ON ticket_index USING gin(culprit_kinds);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_culprit_agents
    ON ticket_index USING gin(culprit_agents);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_run
    ON ticket_index(latest_run_id);

    CREATE INDEX IF NOT EXISTS idx_ticket_index_skills
    ON ticket_index USING gin(skills);

    CREATE INDEX IF NOT EXISTS idx_customer_index_activity
    ON customer_index(last_activity DESC NULLS LAST);
  `);
}
