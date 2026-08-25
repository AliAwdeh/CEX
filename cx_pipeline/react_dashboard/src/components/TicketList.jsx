import React, { useEffect, useRef } from "react";
import { AlertTriangle, Flame, GitCompareArrows, MessageSquare, ShieldAlert } from "lucide-react";
import { Empty, Skeleton, Tag, ValueTag } from "./ui.jsx";
import { bandOf, count, humanize, score, span } from "../lib/format.js";

/* Rows use content-visibility so the browser skips layout and paint for
   off-screen rows. With server-side paging capped at 500 that gives the win
   of virtualisation without the fragility of fixed row heights. */

function TicketRow({ ticket, active, onOpen, rowRef }) {
  const band = bandOf(ticket);
  const value = score(ticket.score_final_100);

  return (
    <button
      className="trow"
      type="button"
      ref={rowRef}
      data-band={band}
      aria-current={active}
      onClick={() => onOpen(ticket.ticket_id)}
    >
      <span className="trow-stripe" />

      <span className="trow-score">
        <span className="score-n">{value ?? "—"}</span>
        <span className="score-band-label">{band === "unscored" ? "no score" : band}</span>
      </span>

      <span className="trow-main">
        <span className="trow-id">
          <b>#{ticket.ticket_id}</b>
          {ticket.customer_name && <><span className="sep">·</span><span className="who">{ticket.customer_name}</span></>}
          <span className="sep">·</span>
          <span>{humanize(ticket.ticket_type)}</span>
        </span>

        <span className="trow-obj">
          {ticket.customer_primary_objective || ticket.main_issue_summary || "No objective recorded"}
        </span>

        <span className="trow-tags">
          <ValueTag value={ticket.handled_status} />
          <ValueTag value={ticket.customer_experience} />
          {ticket.max_frustration_level && ticket.max_frustration_level !== "none" && (
            <Tag kind="bad" icon={Flame}>{humanize(ticket.max_frustration_level)}</Tag>
          )}
          {ticket.main_issue_type && ticket.main_issue_type !== "none" && (
            <Tag kind="warn" icon={AlertTriangle} optional>{humanize(ticket.main_issue_type)}</Tag>
          )}
          {ticket.contradiction_count > 0 && (
            <Tag kind="bad" icon={GitCompareArrows} title="Contradicting statements detected">
              {ticket.contradiction_count}
            </Tag>
          )}
          {ticket.manual_review_required && <Tag kind="warn" icon={ShieldAlert}>Review</Tag>}
          {ticket.failed_messages > 0 && (
            <Tag kind="bad" optional title="Messages that failed to parse">{ticket.failed_messages} failed</Tag>
          )}
        </span>
      </span>

      <span className="trow-side">
        <span className="trow-when">{span(ticket.first_message_at, ticket.last_message_at)}</span>
        <span className="trow-meta">
          <MessageSquare size={11} />
          {count(ticket.linked_messages)}
          {ticket.source_conversation_count > 1 && <span>· {ticket.source_conversation_count} convs</span>}
        </span>
      </span>
    </button>
  );
}

export default function TicketList({ tickets, loading, error, selectedId, onOpen, filtered, onClear }) {
  const listRef = useRef(null);
  const activeRef = useRef(null);

  /* Keep the keyboard selection in view when j/k moves it. */
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);

  if (error) {
    return <div className="ticketlist"><Empty error title="Could not load tickets" text={error} /></div>;
  }
  if (loading && !tickets?.length) {
    return <div className="ticketlist scroll"><Skeleton rows={8} /></div>;
  }
  if (!tickets?.length) {
    return (
      <div className="ticketlist">
        <Empty
          title={filtered ? "No tickets match these filters" : "No tickets yet"}
          text={filtered ? "Loosen a filter, or clear them all to start again." : "Run the pipeline, then refresh the cache."}
        />
        {filtered && (
          <div style={{ display: "grid", placeItems: "center", paddingBottom: 24 }}>
            <button className="preset" type="button" onClick={onClear}>Clear all filters</button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="ticketlist scroll" ref={listRef}>
      {tickets.map((ticket) => {
        const active = String(ticket.ticket_id) === String(selectedId);
        return (
          <TicketRow
            key={ticket.ticket_id}
            ticket={ticket}
            active={active}
            rowRef={active ? activeRef : undefined}
            onOpen={onOpen}
          />
        );
      })}
    </div>
  );
}
