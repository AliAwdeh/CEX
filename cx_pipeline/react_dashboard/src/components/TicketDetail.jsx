import React from "react";
import { ArrowRight, CircleCheck, CircleX, MessagesSquare, X } from "lucide-react";
import { Empty, MetricGrid, RawJson, Section, Tag, ValueTag } from "./ui.jsx";
import { useApi } from "../lib/api.js";
import { cx, dateTime, humanize, score, span } from "../lib/format.js";

const SCORE_AXES = [
  { key: "resolution_score", label: "Resolution" },
  { key: "context_understanding_score", label: "Context" },
  { key: "customer_effort_score", label: "Customer effort" },
  { key: "trust_frustration_risk_score", label: "Trust / risk" },
  { key: "ai_judgment_score", label: "AI judgment" },
  { key: "message_signal_score", label: "Message signals" }
];

function ScoreBreakdown({ conversationScore }) {
  const final = Number(conversationScore.final_score_100);
  if (!Number.isFinite(final) && !SCORE_AXES.some((axis) => conversationScore[axis.key] != null)) return null;

  const band = Number.isFinite(final)
    ? final < 40 ? "critical" : final < 60 ? "poor" : final < 75 ? "fair" : final < 90 ? "good" : "excellent"
    : "unscored";

  return (
    <>
      <div className="scorehero">
        <span className="scorehero-n" style={{ color: `var(--band-${band})` }}>
          {score(final) ?? "—"}
        </span>
        <span className="scorehero-meta">
          <ValueTag value={conversationScore.score_rating || band} />
          <span className="muted" style={{ fontSize: 11.5 }}>out of 100</span>
        </span>
      </div>

      <div className="scorebars">
        {SCORE_AXES.map((axis) => {
          const value = Number(conversationScore[axis.key]);
          if (!Number.isFinite(value)) return null;
          return (
            <div className="scorebar" key={axis.key}>
              <span className="scorebar-label">{axis.label}</span>
              <span className="scorebar-track">
                <span className="scorebar-fill" style={{ width: `${Math.max(0, Math.min(100, value * 10))}%` }} />
              </span>
              <span className="scorebar-val">{score(value)}</span>
            </div>
          );
        })}
      </div>

      {conversationScore.score_explanation && (
        <p className="prose" style={{ marginTop: 12 }}>{conversationScore.score_explanation}</p>
      )}
    </>
  );
}


/* Ticket lifecycle audit rows, newest first. Each row records one state change,
   so the useful thing to show is the diff, not the whole state twice. */

const LIFECYCLE_FIELDS = [
  ["status", "Status"],
  ["open_closed", "Open / closed"],
  ["opened_at", "Opened"],
  ["closed_at", "Closed"],
  ["first_message_at", "First message"],
  ["last_message_at", "Last message"],
  ["latest_run_id", "Analysis run"],
  ["source_conversation_ids", "Source conversations"]
];

function lifecycleValue(field, value) {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return `${value.length} conversation${value.length === 1 ? "" : "s"}`;
  if (field.endsWith("_at")) return dateTime(value);
  if (field === "latest_run_id") return String(value).slice(0, 8);
  return humanize(value);
}

function lifecycleDiff(row) {
  const previous = row.previous_state || {};
  const current = row.current_state || {};
  return LIFECYCLE_FIELDS
    .filter(([field]) => JSON.stringify(previous[field]) !== JSON.stringify(current[field]))
    .map(([field, label]) => ({
      field,
      label,
      from: lifecycleValue(field, previous[field]),
      to: lifecycleValue(field, current[field])
    }));
}

function LifecycleHistory({ rows }) {
  if (!rows?.length) return null;

  return (
    <Section title="Lifecycle" tally={rows.length} open={false}>
      <ol className="lifecycle">
        {rows.map((row) => {
          const changes = lifecycleDiff(row);
          return (
            <li className="lc-row" key={row.id}>
              <div className="lc-head">
                <ValueTag value={row.event_type} />
                <span className="lc-when">{dateTime(row.changed_at)}</span>
              </div>
              {changes.length ? (
                <div className="lc-changes">
                  {changes.map((change) => (
                    <div className="lc-change" key={change.field}>
                      <span className="lc-field">{change.label}</span>
                      <span className="lc-from">{change.from}</span>
                      <ArrowRight size={11} />
                      <span className="lc-to">{change.to}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="lc-nochange">Recorded with no field change.</p>
              )}
            </li>
          );
        })}
      </ol>
    </Section>
  );
}

export default function TicketDetail({ ticketId, onClose, onOpenCustomer, onOpenTranscript }) {
  const { data, loading, error } = useApi(ticketId ? `/api/tickets/${ticketId}` : null, { keepPrevious: false });

  if (!ticketId) {
    return (
      <aside className="detail">
        <Empty title="No ticket selected" text="Pick a ticket from the list, or press j to start moving through them." />
      </aside>
    );
  }
  if (error) {
    return (
      <aside className="detail">
        <Empty error title="Could not load ticket" text={error} />
      </aside>
    );
  }
  if (loading || !data?.ticket) {
    return <aside className="detail"><div className="skel skel-block" /><div className="skel skel-block" /></aside>;
  }

  const { ticket, messages, messageStats } = data;
  const stats = data.stats || {};
  const lifecycle = data.lifecycle || [];
  const result = ticket.cx_result || {};
  const conversationScore = result.conversation_score || {};
  const mainIssue = result.main_issue || {};
  const issues = result.all_detected_issues || [];
  const actions = result.recommended_actions || [];
  const positives = result.positive_signals || [];
  const negatives = result.negative_signals || [];
  const segmentation = ticket.segmentation || {};
  const inquiries = Array.isArray(segmentation.inquiries) ? segmentation.inquiries : [];

  const conversationCount = new Set(messages.map((message) => message.source_conversation_id)).size;

  return (
    <aside className="detail">
      <div className="detail-head">
        <div className="detail-topline">
          <b>#{ticket.id}</b>
          <ValueTag value={ticket.status} />
          <span className="spacer" style={{ flex: 1 }} />
          <button className="iconbtn" type="button" onClick={onClose} aria-label="Close ticket" title="Close (Esc)">
            <X size={15} />
          </button>
        </div>

        <h2 className="detail-title">
          {result.customer_primary_objective || ticket.objective || humanize(ticket.ticket_type)}
        </h2>

        <div className="detail-sub">
          {ticket.customer_name && (
            <button className="linkbtn" type="button" style={{ padding: 0 }} onClick={() => onOpenCustomer(ticket.customer_id)}>
              {ticket.customer_name} <ArrowRight size={11} style={{ verticalAlign: -1 }} />
            </button>
          )}
          <span>{humanize(ticket.category)} · {humanize(ticket.ticket_type)}</span>
        </div>

        <div className="detail-tags">
          <ValueTag value={result.handled_status || "not_analyzed"} />
          <ValueTag value={result.customer_experience || "not_analyzed"} />
          {result.max_frustration_level && result.max_frustration_level !== "none" && (
            <ValueTag value={result.max_frustration_level} prefix="Frustration" />
          )}
          {result.manual_review_required && <Tag kind="warn">Needs review</Tag>}
          {result.confidence && <Tag>{humanize(result.confidence)} confidence</Tag>}
          {stats.lifecycle_risk && stats.lifecycle_risk !== "normal" && <ValueTag value={stats.lifecycle_risk} />}
          {stats.analysis_eligible === false && <Tag kind="warn">Analysis skipped</Tag>}
        </div>
      </div>

      <div className="detail-body scroll">
        {result.management_summary && (
          <Section title="Summary">
            <p className="prose">{result.management_summary}</p>
            {result.classification_reason && (
              <p className="prose"><strong>Why this classification.</strong> {result.classification_reason}</p>
            )}
            {result.manual_review_reason && result.manual_review_reason !== "none" && (
              <p className="prose"><strong>Flagged for review.</strong> {result.manual_review_reason}</p>
            )}
          </Section>
        )}

        <Section title="Score">
          <ScoreBreakdown conversationScore={conversationScore} />
        </Section>

        <Section title="At a glance">
          <MetricGrid
            items={[
              { label: "Messages", value: messages.length, numeric: true },
              { label: "Analysed", value: `${messageStats?.ok_messages ?? 0} / ${messageStats?.evaluated_messages ?? 0}`, numeric: true },
              { label: "Conversations", value: segmentation.conversation_summaries?.length ?? 1, numeric: true },
              { label: "Opened", value: dateTime(stats.opened_at || messages[0]?.message_time) },
              { label: "Closed", value: stats.closed_at ? dateTime(stats.closed_at) : "Open" },
              { label: "Reopen until", value: stats.reopenable_until ? dateTime(stats.reopenable_until) : "N/A" },
              { label: "Lifecycle risk", value: humanize(stats.lifecycle_risk || "normal") },
              { label: "AI eligible", value: stats.analysis_eligible === false ? "No" : "Yes" },
              { label: "Span", value: span(messages[0]?.message_time, messages[messages.length - 1]?.message_time) },
              { label: "Sentiment", value: humanize(result.final_customer_sentiment) },
              { label: "Goal type", value: humanize(result.customer_objective_type) },
              { label: "Raised by", value: humanize(ticket.request_origin) },
              { label: "Frustration origin", value: humanize(result.frustration_origin) }
            ]}
          />
        </Section>

        {mainIssue.issue_exists && (
          <Section title="Main issue">
            <div className={cx("issue", `issue-${String(mainIssue.issue_origin || "").replace("_side", "")}`)}>
              <div className="issue-head">
                <b>{humanize(mainIssue.issue_type)}</b>
                <ValueTag value={mainIssue.issue_origin} />
              </div>
              <p>{mainIssue.issue_summary}</p>
              {mainIssue.customer_impact && mainIssue.customer_impact !== "none" && (
                <p className="ev">{mainIssue.customer_impact}</p>
              )}
            </div>
            {result.culprit_reason && result.culprit_reason !== "none" && (
              <p className="prose"><strong>Responsibility.</strong> {result.culprit_reason}</p>
            )}
            {(result.culprits || []).length > 0 && (
              <div className="detail-tags">
                {result.culprits.map((culprit) => <ValueTag key={culprit} value={culprit} />)}
                {(result.culprit_agent_names || []).map((name) => <Tag kind="bad" key={name}>{name}</Tag>)}
              </div>
            )}
          </Section>
        )}

        {issues.length > 0 && (
          <Section title="All detected issues" tally={issues.length} open={false}>
            {issues.map((issue, index) => (
              <div className={cx("issue", `issue-${String(issue.issue_origin || "").replace("_side", "")}`)} key={`${issue.issue_type}-${index}`}>
                <div className="issue-head">
                  <b>{humanize(issue.issue_type)}</b>
                  <ValueTag value={issue.issue_origin} />
                </div>
                <p>{issue.issue_summary}</p>
                {issue.evidence && <p className="ev">{issue.evidence}</p>}
                {issue.impact && <p>{issue.impact}</p>}
              </div>
            ))}
          </Section>
        )}

        {(positives.length > 0 || negatives.length > 0) && (
          <Section title="Signals" tally={positives.length + negatives.length} open={false}>
            {positives.length > 0 && (
              <ul className="siglist">
                {positives.map((signal, index) => (
                  <li className="pos" key={index}><CircleCheck size={13} />{signal}</li>
                ))}
              </ul>
            )}
            {negatives.length > 0 && (
              <ul className="siglist">
                {negatives.map((signal, index) => (
                  <li className="neg" key={index}><CircleX size={13} />{signal}</li>
                ))}
              </ul>
            )}
          </Section>
        )}

        {actions.length > 0 && (
          <Section title="Recommended actions" tally={actions.length}>
            <ul className="actions">
              {actions.map((action, index) => (
                <li key={index}><ArrowRight size={13} />{action}</li>
              ))}
            </ul>
          </Section>
        )}

        {inquiries.length > 0 && (
          <Section title="Inquiries" tally={inquiries.length} open={false}>
            {inquiries.map((inquiry, index) => (
              <div className="issue" key={inquiry.inquiry_id || index}>
                <div className="issue-head">
                  <b>{inquiry.question || `Inquiry ${index + 1}`}</b>
                  <ValueTag value={inquiry.status} />
                </div>
                {inquiry.answer_summary && <p>{inquiry.answer_summary}</p>}
                {inquiry.unresolved_reason && <p className="ev">{inquiry.unresolved_reason}</p>}
              </div>
            ))}
            {segmentation.segmentation_reason && (
              <p className="prose"><strong>Why these messages.</strong> {segmentation.segmentation_reason}</p>
            )}
          </Section>
        )}

        {(stats.lifecycle_reason || stats.analysis_skip_reason) && (
          <Section title="Lifecycle">
            {stats.lifecycle_reason && <p className="prose">{stats.lifecycle_reason}</p>}
            {stats.analysis_skip_reason && (
              <p className="prose"><strong>Analysis rule.</strong> {humanize(stats.analysis_skip_reason)}</p>
            )}
          </Section>
        )}

        <LifecycleHistory rows={lifecycle} />

        <Section title="Conversation" tally={messages.length}>
          <p className="prose">
            {messages.length} message{messages.length === 1 ? "" : "s"} across{" "}
            {conversationCount} source conversation{conversationCount === 1 ? "" : "s"}
            {messageStats?.evaluated_messages ? `, ${messageStats.evaluated_messages} analysed` : ""}.
          </p>
          <button className="openbtn" type="button" onClick={() => onOpenTranscript(ticket.id)}>
            <MessagesSquare size={14} />
            Open transcript
            <kbd>t</kbd>
          </button>
        </Section>

        <Section title="Raw data" open={false}>
          <RawJson label="CX result" data={ticket.cx_result} />
          <RawJson label="segmentation" data={ticket.segmentation} />
          <RawJson label="computed metadata" data={ticket.computed_metadata} />
          <RawJson label="stats cache" data={stats} />
          <RawJson label="ticket history" data={lifecycle} />
        </Section>
      </div>
    </aside>
  );
}
