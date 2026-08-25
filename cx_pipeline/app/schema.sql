CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    mode TEXT NOT NULL DEFAULT 'full',
    status TEXT NOT NULL DEFAULT 'created',
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    csv_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error TEXT
);

CREATE TABLE IF NOT EXISTS customers (
    id BIGSERIAL PRIMARY KEY,
    external_customer_id TEXT NOT NULL UNIQUE,
    customer_name TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_conversations (
    id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    source_conversation_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    first_message_time TEXT,
    last_message_time TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    ticketed_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(customer_id, source_conversation_id, content_hash)
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    source_conversation_pk BIGINT NOT NULL REFERENCES source_conversations(id) ON DELETE CASCADE,
    customer_message_index INTEGER NOT NULL,
    source_message_index INTEGER,
    sender_role TEXT NOT NULL,
    raw_sender_role TEXT,
    message_time TEXT,
    message_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(source_conversation_pk, customer_message_index, content_hash)
);

CREATE TABLE IF NOT EXISTS tickets (
    id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending_unresolved',
    category TEXT NOT NULL DEFAULT 'inquiry',
    request_origin TEXT NOT NULL DEFAULT 'customer',
    ticket_type TEXT NOT NULL DEFAULT 'other',
    objective TEXT NOT NULL DEFAULT '',
    should_append_future BOOLEAN NOT NULL DEFAULT false,
    previous_ticket_id BIGINT REFERENCES tickets(id) ON DELETE SET NULL,
    model_ticket_id TEXT,
    segmentation JSONB NOT NULL DEFAULT '{}'::jsonb,
    latest_ticketing_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    needs_message_analysis BOOLEAN NOT NULL DEFAULT false,
    needs_ticket_cx BOOLEAN NOT NULL DEFAULT false,
    ticket_message_count INTEGER NOT NULL DEFAULT 0,
    opened_at TIMESTAMPTZ,
    last_message_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    reopenable_until TIMESTAMPTZ,
    lifecycle_risk TEXT NOT NULL DEFAULT 'normal',
    lifecycle_reason TEXT,
    analysis_eligible BOOLEAN NOT NULL DEFAULT true,
    analysis_skip_reason TEXT,
    lifecycle_updated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE tickets
ADD COLUMN IF NOT EXISTS ticket_message_count INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS opened_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS last_message_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS reopenable_until TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS lifecycle_risk TEXT NOT NULL DEFAULT 'normal',
ADD COLUMN IF NOT EXISTS lifecycle_reason TEXT,
ADD COLUMN IF NOT EXISTS analysis_eligible BOOLEAN NOT NULL DEFAULT true,
ADD COLUMN IF NOT EXISTS analysis_skip_reason TEXT,
ADD COLUMN IF NOT EXISTS lifecycle_updated_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS ticket_messages (
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    PRIMARY KEY(ticket_id, message_id)
);

CREATE TABLE IF NOT EXISTS message_results (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    parse_status TEXT NOT NULL,
    result JSONB,
    debug JSONB,
    raw_response TEXT,
    attempts INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(ticket_id, message_id)
);

CREATE TABLE IF NOT EXISTS ticket_cx_results (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    ticket_id BIGINT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE UNIQUE,
    parse_status TEXT NOT NULL,
    result JSONB,
    computed_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    debug JSONB,
    raw_response TEXT,
    attempts INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS run_steps (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    step_type TEXT NOT NULL,
    customer_id BIGINT REFERENCES customers(id) ON DELETE CASCADE,
    ticket_id BIGINT REFERENCES tickets(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    locked_by TEXT,
    locked_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_ticketing_step
ON run_steps(run_id, step_type, customer_id)
WHERE step_type='ticketing' AND status IN ('pending', 'running');

CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_message_step
ON run_steps(run_id, step_type, ticket_id)
WHERE step_type='message' AND status IN ('pending', 'running');

CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_ticket_cx_step
ON run_steps(run_id, step_type, ticket_id)
WHERE step_type='ticket_cx' AND status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS run_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    step_id BIGINT REFERENCES run_steps(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    message TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai_requests (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    step_id BIGINT REFERENCES run_steps(id) ON DELETE SET NULL,
    worker_id TEXT,
    layer TEXT NOT NULL,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    customer_id BIGINT REFERENCES customers(id) ON DELETE SET NULL,
    ticket_id BIGINT REFERENCES tickets(id) ON DELETE SET NULL,
    message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,
    context TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    duration_seconds NUMERIC,
    error TEXT,
    debug JSONB NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE ai_requests
ADD COLUMN IF NOT EXISTS worker_id TEXT;

CREATE INDEX IF NOT EXISTS idx_ai_requests_run_status_started
ON ai_requests(run_id, status, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_ai_requests_finished
ON ai_requests(run_id, finished_at DESC)
WHERE status <> 'running';

CREATE INDEX IF NOT EXISTS idx_ai_requests_worker
ON ai_requests(worker_id, status, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_source_conversations_customer_status
ON source_conversations(customer_id, status);

CREATE INDEX IF NOT EXISTS idx_messages_customer_index
ON messages(customer_id, customer_message_index);

CREATE INDEX IF NOT EXISTS idx_ticket_messages_ticket
ON ticket_messages(ticket_id, message_id);

CREATE INDEX IF NOT EXISTS idx_steps_run_status_type
ON run_steps(run_id, status, step_type);

CREATE INDEX IF NOT EXISTS idx_tickets_customer
ON tickets(customer_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_tickets_reopenable_until
ON tickets(customer_id, reopenable_until DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_tickets_lifecycle_risk
ON tickets(lifecycle_risk);

CREATE INDEX IF NOT EXISTS idx_tickets_analysis_eligible
ON tickets(analysis_eligible, needs_message_analysis, needs_ticket_cx);
