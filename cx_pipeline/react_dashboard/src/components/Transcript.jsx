import React from "react";
import {
  ArrowLeft,
  Bot,
  CircleX,
  Copy,
  MessagesSquare,
  Radio,
  UserCog,
  UserRound
} from "lucide-react";
import { Empty, MetricGrid, RawJson, Tag, ValueTag } from "./ui.jsx";
import { useApi } from "../lib/api.js";
import { bandOf, count, cx, dateTime, humanize, score } from "../lib/format.js";

/* A full-width reading surface for one ticket's conversation.
   Two things the side pane could not do: show which source conversation each
   message came from — a ticket routinely spans several — and show the complete
   per-message and per-ticket model output rather than the handful of fields
   that fit in a column. */

const SPEAKER_ICON = { customer: UserRound, agent: UserCog, bot: Bot, broadcast: Radio, unknown: CircleX };

function speakerOf(message) {
  const raw = message.raw || {};
  const rawRole = String(message.raw_sender_role || raw.RAW_SENDER_ROLE || "").toLowerCase();
  const name = String(raw.MESSAGE_AGENT_FULL_NAME || raw.CONVERSATION_AGENT_FULL_NAME || "").trim();
  const skill = String(raw.MESSAGE_SKILL || raw.LAST_SKILL || "").trim();

  if (rawRole.includes("consumer") || rawRole.includes("customer")) return { kind: "customer", label: "Customer", detail: "" };
  if (rawRole.includes("system") || rawRole.includes("broadcast")) return { kind: "broadcast", label: "Broadcast", detail: skill };
  if (rawRole.includes("bot") || skill.toLowerCase().includes("gpt")) return { kind: "bot", label: "Bot", detail: skill };
  if (rawRole.includes("agent")) return { kind: "agent", label: "Agent", detail: name };
  return { kind: "unknown", label: "Unknown", detail: rawRole };
}

function CopyId({ value, label }) {
  return (
    <button
      className="copyid"
      type="button"
      title={`Copy ${label}: ${value}`}
      onClick={(event) => {
        event.stopPropagation();
        navigator.clipboard?.writeText(value);
      }}
    >
      <span className="mono">{value}</span>
      <Copy size={11} />
    </button>
  );
}

/* Group messages into runs of the same source conversation, preserving order,
   so a ticket that jumps between conversations shows that jump rather than
   silently flattening into one thread. */
function groupByConversation(messages) {
  const groups = [];
  for (const message of messages) {
    const id = message.source_conversation_id || "unknown";
    const last = groups[groups.length - 1];
    if (last && last.id === id) last.messages.push(message);
    else groups.push({ id, messages: [message] });
  }
  return groups;
}

const MESSAGE_FIELDS = [
  ["message_level_effect", "Effect"],
  ["issue_type", "Issue type"],
  ["issue_origin", "Issue origin"],
  ["frustration_level_after_message", "Frustration after"],
  ["frustration_change", "Frustration change"],
  ["frustration_cause", "Frustration cause"],
  ["customer_effort_level", "Customer effort"],
  ["clarity_level", "Clarity"],
  ["context_handling", "Context handling"],
  ["contradiction", "Contradiction"]
];

function MessageAnalysis({ message }) {
  const result = message.message_result;
  const failed = message.message_parse_status && message.message_parse_status !== "ok";

  if (failed) {
    return (
      <div className="manalysis manalysis-failed">
        <div className="manalysis-head">Analysis failed</div>
        <p className="manalysis-text">{message.message_error || message.message_parse_status}</p>
      </div>
    );
  }
  if (!result || !Object.keys(result).length) {
    return <div className="manalysis manalysis-none">Not evaluated — this message was not a target for message-level analysis.</div>;
  }

  const values = MESSAGE_FIELDS
    .map(([key, label]) => [label, result[key]])
    .filter(([, value]) => value !== undefined && value !== null && value !== "" && value !== "none" && value !== false);

  return (
    <div className="manalysis">
      <div className="manalysis-head">
        <MessagesSquare size={11} />
        Message analysis
        <span className="manalysis-status">{message.message_parse_status}</span>
      </div>

      <div className="manalysis-tags">
        {values.map(([label, value]) => (
          <span className="atag" key={label}>
            <em>{label}</em>
            <ValueTag value={String(value)} />
          </span>
        ))}
      </div>

      {result.evidence && result.evidence !== "none" && (
        <p className="manalysis-text"><b>Evidence.</b> <span className="quote">{result.evidence}</span></p>
      )}
      {result.business_impact && result.business_impact !== "none" && (
        <p className="manalysis-text"><b>Impact.</b> {result.business_impact}</p>
      )}
      {result.recommended_fix && result.recommended_fix !== "none" && (
        <p className="manalysis-text"><b>Fix.</b> {result.recommended_fix}</p>
      )}
      {result.contradiction_debug_message && result.contradiction_debug_message !== "none" && (
        <p className="manalysis-text"><b>Contradiction.</b> {result.contradiction_debug_message}
          {result.first_contradiction_message_id && result.first_contradiction_message_id !== "none" && (
            <> Source message <span className="mono">{result.first_contradiction_message_id}</span>.</>
          )}
        </p>
      )}

      <RawJson label="model output" data={result} />
    </div>
  );
}

function Message({ message }) {
  const speaker = speakerOf(message);
  const Icon = SPEAKER_ICON[speaker.kind] || CircleX;
  const raw = message.raw || {};
  const effect = message.message_result?.message_level_effect;
  const flagged = ["minor_issue", "major_issue"].includes(effect);

  return (
    <article className={cx("tmsg", `tmsg-${speaker.kind}`, flagged && "tmsg-flagged")} id={`m${message.id}`}>
      <header className="tmsg-head">
        <span className={cx("msg-who", `msg-who-${speaker.kind}`)}>
          <Icon size={12} strokeWidth={2.3} />
          {speaker.label}
        </span>
        {speaker.detail && <span className="tmsg-name">{speaker.detail}</span>}

        <span className="tmsg-ids">
          <span title="Position in the customer's message stream">idx {message.customer_message_index}</span>
          {message.source_message_index != null && (
            <span title="Position within the source conversation">src {message.source_message_index}</span>
          )}
          <span title="Database message id">id {message.id}</span>
        </span>

        <span className="tmsg-when">{dateTime(message.message_time)}</span>
      </header>

      <p className="tmsg-text">{message.message_text}</p>

      <MessageAnalysis message={message} />

      {(raw.LAST_SKILL || raw.MESSAGE_SKILL || raw.HAS_RAG_RETRIEVAL) && (
        <footer className="tmsg-foot">
          {raw.MESSAGE_SKILL && <Tag>skill {raw.MESSAGE_SKILL}</Tag>}
          {!raw.MESSAGE_SKILL && raw.LAST_SKILL && <Tag>skill {raw.LAST_SKILL}</Tag>}
          {raw.HAS_RAG_RETRIEVAL === "TRUE" && <Tag kind="info">RAG {raw.RAG_RETRIEVAL_COUNT || 1}</Tag>}
          {raw.HAS_RAG_RETRIEVAL === "FALSE" && speaker.kind === "bot" && <Tag>no retrieval</Tag>}
          <RawJson label="source fields" data={raw} />
        </footer>
      )}
    </article>
  );
}

function TicketAnalysis({ ticket, messageStats, messages }) {
  const result = ticket.cx_result || {};
  const conversationScore = result.conversation_score || {};
  const computed = ticket.computed_metadata || {};
  const mainIssue = result.main_issue || {};
  const failed = ticket.ticket_cx_parse_status && ticket.ticket_cx_parse_status !== "ok";

  return (
    <aside className="tanalysis scroll">
      <section className="tan-block">
        <h3 className="tan-h">Ticket analysis</h3>
        <div className="tan-kv">
          <span>Parse status</span>
          <ValueTag value={failed ? "bad" : ticket.ticket_cx_parse_status || "not_analyzed"} />
        </div>
        {ticket.cx_error && <p className="tan-err">{ticket.cx_error}</p>}
        <div className="tan-kv"><span>Confidence</span><ValueTag value={result.confidence || "unknown"} /></div>
        <div className="tan-kv"><span>Run</span><CopyId value={ticket.latest_ticketing_run_id || "—"} label="run id" /></div>
        <div className="tan-kv"><span>Model ticket id</span><span className="mono">{ticket.model_ticket_id || "—"}</span></div>
        <div className="tan-kv"><span>Re-analysis queued</span>
          <span className="mono">{ticket.needs_message_analysis ? "messages " : ""}{ticket.needs_ticket_cx ? "ticket" : ""}{!ticket.needs_message_analysis && !ticket.needs_ticket_cx ? "no" : ""}</span>
        </div>
      </section>

      <section className="tan-block">
        <h3 className="tan-h">Verdict</h3>
        <div className="tan-tags">
          <ValueTag value={result.handled_status || "not_analyzed"} />
          <ValueTag value={result.customer_experience || "not_analyzed"} />
          <ValueTag value={result.unhandled_resolution_subtype} />
          <ValueTag value={result.final_customer_sentiment} />
          {result.manual_review_required && <Tag kind="warn">Needs review</Tag>}
        </div>
        {result.classification_reason && <p className="tan-text">{result.classification_reason}</p>}
        {result.manual_review_reason && result.manual_review_reason !== "none" && (
          <p className="tan-text"><b>Review reason.</b> {result.manual_review_reason}</p>
        )}
      </section>

      <section className="tan-block">
        <h3 className="tan-h">Score</h3>
        <div className="tan-score">
          <span className="scorehero-n" style={{ color: `var(--band-${bandOf({ score_band: null, score_final_100: conversationScore.final_score_100 })})` }}>
            {score(conversationScore.final_score_100) ?? "—"}
          </span>
          <ValueTag value={conversationScore.score_rating || "unscored"} />
        </div>
        <MetricGrid
          items={[
            { label: "Resolution", value: score(conversationScore.resolution_score), numeric: true },
            { label: "Context", value: score(conversationScore.context_understanding_score), numeric: true },
            { label: "Effort", value: score(conversationScore.customer_effort_score), numeric: true },
            { label: "Trust / risk", value: score(conversationScore.trust_frustration_risk_score), numeric: true },
            { label: "AI judgment", value: score(conversationScore.ai_judgment_score), numeric: true },
            { label: "Msg signals", value: score(conversationScore.message_signal_score), numeric: true }
          ]}
        />
        {conversationScore.score_explanation && <p className="tan-text">{conversationScore.score_explanation}</p>}
      </section>

      <section className="tan-block">
        <h3 className="tan-h">Friction</h3>
        <div className="tan-tags">
          <ValueTag value={result.max_frustration_level} prefix="Max" />
          <ValueTag value={result.frustration_origin} prefix="Origin" />
          <ValueTag value={result.frustration_timing} prefix="Timing" />
        </div>
        <div className="tan-kv"><span>Started frustrated</span><span className="mono">{String(result.customer_started_frustrated ?? false)}</span></div>
        <div className="tan-kv"><span>Became frustrated</span><span className="mono">{String(result.customer_became_frustrated_during_chat ?? false)}</span></div>
        <div className="tan-kv"><span>Ended frustrated</span><span className="mono">{String(result.customer_ended_frustrated ?? false)}</span></div>
      </section>

      {mainIssue.issue_exists && (
        <section className="tan-block">
          <h3 className="tan-h">Main issue</h3>
          <div className="tan-tags">
            <ValueTag value={mainIssue.issue_type} />
            <ValueTag value={mainIssue.issue_origin} />
          </div>
          <p className="tan-text">{mainIssue.issue_summary}</p>
          {result.culprit_reason && result.culprit_reason !== "none" && (
            <p className="tan-text"><b>Responsibility.</b> {result.culprit_reason}</p>
          )}
          <div className="tan-tags">
            {(result.culprits || []).map((culprit) => <ValueTag key={culprit} value={culprit} />)}
            {(result.culprit_agent_names || []).map((name) => <Tag kind="bad" key={name}>{name}</Tag>)}
          </div>
        </section>
      )}

      <section className="tan-block">
        <h3 className="tan-h">Computed rollup</h3>
        <MetricGrid
          items={[
            { label: "Messages", value: computed.total_messages ?? messages.length, numeric: true },
            { label: "Customer", value: computed.customer_messages, numeric: true },
            { label: "Agent", value: computed.agent_messages, numeric: true },
            { label: "Evaluated", value: computed.target_messages_evaluated ?? messageStats?.evaluated_messages, numeric: true },
            { label: "Failed", value: messageStats?.failed_messages, numeric: true },
            { label: "Target role", value: computed.evaluation_target_role },
            { label: "Issues", value: computed.issue_count, numeric: true },
            { label: "Major", value: computed.major_issue_count, numeric: true },
            { label: "Minor", value: computed.minor_issue_count, numeric: true },
            { label: "Recovered", value: computed.recovered_issue_count, numeric: true },
            { label: "Our side", value: computed.our_side_issue_count, numeric: true },
            { label: "Customer side", value: computed.customer_side_issue_count, numeric: true }
          ]}
        />
        {computed.issue_type_counts && (
          <div className="tan-tags">
            {Object.entries(computed.issue_type_counts).map(([type, n]) => (
              <Tag key={type} kind={type === "none" ? undefined : "warn"}>{humanize(type)} {n}</Tag>
            ))}
          </div>
        )}
      </section>

      <section className="tan-block">
        <h3 className="tan-h">Raw</h3>
        <RawJson label="cx_result" data={ticket.cx_result} />
        <RawJson label="computed_metadata" data={ticket.computed_metadata} />
        <RawJson label="segmentation" data={ticket.segmentation} />
      </section>
    </aside>
  );
}

export default function Transcript({ ticketId, onBack, onOpenTicket }) {
  const { data, loading, error } = useApi(ticketId ? `/api/tickets/${ticketId}` : null, { keepPrevious: false });

  if (!ticketId) return <div className="transcript-page"><Empty title="No ticket selected" /></div>;
  if (error) return <div className="transcript-page"><Empty error title="Could not load transcript" text={error} /></div>;
  if (loading || !data?.ticket) {
    return <div className="transcript-page"><div className="skel skel-block" /><div className="skel skel-block" /></div>;
  }

  const { ticket, messages, messageStats } = data;
  const result = ticket.cx_result || {};
  const groups = groupByConversation(messages);
  const distinctConversations = new Set(messages.map((message) => message.source_conversation_id)).size;

  return (
    <div className="transcript-page">
      <header className="tp-head">
        <button className="backbtn" type="button" onClick={onBack}>
          <ArrowLeft size={14} />
          Back
        </button>

        <div className="tp-title">
          <div className="tp-titleline">
            <b className="mono">#{ticket.id}</b>
            <h1>{result.customer_primary_objective || ticket.objective || humanize(ticket.ticket_type)}</h1>
          </div>
          <div className="tp-sub">
            <span>{ticket.customer_name || "Unnamed customer"}</span>
            <span className="sep">·</span>
            <span>{humanize(ticket.category)} · {humanize(ticket.ticket_type)}</span>
            <span className="sep">·</span>
            <span>{count(messages.length)} messages across {count(distinctConversations)} conversation{distinctConversations === 1 ? "" : "s"}</span>
          </div>
        </div>

        <button className="linkbtn" type="button" onClick={() => onOpenTicket(ticket.id)}>Open in Tickets</button>
      </header>

      <div className="tp-body">
        <main className="tp-stream scroll">
          {groups.map((group, index) => (
            <section className="convgroup" key={`${group.id}-${index}`}>
              <header className="convhead">
                <span className="convhead-label">Conversation</span>
                <CopyId value={group.id} label="conversation id" />
                <span className="convhead-meta">
                  {count(group.messages.length)} message{group.messages.length === 1 ? "" : "s"}
                  {group.messages[0]?.message_time && <> · {dateTime(group.messages[0].message_time)}</>}
                </span>
              </header>
              {group.messages.map((message) => <Message key={message.id} message={message} />)}
            </section>
          ))}
        </main>

        <TicketAnalysis ticket={ticket} messageStats={messageStats} messages={messages} />
      </div>
    </div>
  );
}
