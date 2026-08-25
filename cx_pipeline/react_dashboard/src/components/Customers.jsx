import React, { useEffect } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Flame,
  GitCompareArrows,
  MessageSquare,
  Search,
  ShieldAlert,
  Split
} from "lucide-react";
import { Empty, Skeleton, Tag, ValueTag } from "./ui.jsx";
import { useApi, useDebounced } from "../lib/api.js";
import { bandOf, count, humanize, score, span } from "../lib/format.js";

/* This view answers "what has happened to this customer", which is a question
   about their tickets — not about individual messages. The transcript lives in
   the ticket pane, where each message sits next to the analysis that judged it;
   showing a flat message stream here read as the whole story when it was really
   several separate ones interleaved. Each card opens that ticket in Tickets. */

function TicketCard({ ticket, onOpen }) {
  const band = bandOf(ticket);
  const value = score(ticket.score_final_100);

  return (
    <button className="tcard" type="button" data-band={band} onClick={() => onOpen(ticket.ticket_id)}>
      <span className="tcard-stripe" />

      <span className="tcard-body">
        <span className="tcard-head">
          <span className="tcard-score">
            <span className="score-n">{value ?? "—"}</span>
            <span className="score-band-label">{band === "unscored" ? "no score" : band}</span>
          </span>

          <span className="tcard-ident">
            <span className="tcard-title">
              <b>#{ticket.ticket_id}</b>
              <span>{humanize(ticket.ticket_type)}</span>
            </span>
            <span className="tcard-meta">
              <span>{span(ticket.first_message_at, ticket.last_message_at)}</span>
              <span className="sep">·</span>
              <span><MessageSquare size={11} /> {count(ticket.linked_messages)}</span>
              {ticket.source_conversation_count > 1 && (
                <>
                  <span className="sep">·</span>
                  <span title="Spans more than one source conversation">
                    <Split size={11} /> {ticket.source_conversation_count} convs
                  </span>
                </>
              )}
            </span>
          </span>

          <span className="tcard-go"><ArrowRight size={14} /></span>
        </span>

        <span className="tcard-objective">
          {ticket.customer_primary_objective || ticket.main_issue_summary || "No objective recorded"}
        </span>

        {ticket.main_issue_type && ticket.main_issue_type !== "none" && ticket.main_issue_summary && (
          <span className="tcard-issue">{ticket.main_issue_summary}</span>
        )}

        <span className="tcard-tags">
          <ValueTag value={ticket.status} />
          <ValueTag value={ticket.handled_status} />
          <ValueTag value={ticket.customer_experience} />
          {ticket.max_frustration_level && ticket.max_frustration_level !== "none" && (
            <Tag kind="bad" icon={Flame}>{humanize(ticket.max_frustration_level)}</Tag>
          )}
          {ticket.main_issue_type && ticket.main_issue_type !== "none" && (
            <Tag kind="warn" icon={AlertTriangle}>{humanize(ticket.main_issue_type)}</Tag>
          )}
          {ticket.contradiction_count > 0 && (
            <Tag kind="bad" icon={GitCompareArrows} title="Contradicting statements detected">
              {ticket.contradiction_count} contradiction{ticket.contradiction_count > 1 ? "s" : ""}
            </Tag>
          )}
          {ticket.manual_review_required && <Tag kind="warn" icon={ShieldAlert}>Needs review</Tag>}
          {ticket.failed_messages > 0 && (
            <Tag kind="bad" title="Messages that failed to parse">{ticket.failed_messages} failed</Tag>
          )}
        </span>
      </span>
    </button>
  );
}

function CustomerHeader({ customer, totals }) {
  const stats = [
    { label: "Tickets", value: count(totals.tickets ?? customer?.tickets ?? 0) },
    { label: "Avg score", value: totals.avg_score_100 == null ? "—" : score(totals.avg_score_100) },
    { label: "Bad", value: count(totals.bad_experience_tickets ?? 0), tone: totals.bad_experience_tickets ? "bad" : null },
    { label: "Still open", value: count(totals.open_tickets ?? 0), tone: totals.open_tickets ? "warn" : null },
    { label: "Needs review", value: count(totals.manual_review_tickets ?? 0), tone: totals.manual_review_tickets ? "warn" : null },
    { label: "Messages", value: count(totals.messages ?? customer?.messages ?? 0) },
    { label: "Conversations", value: count(totals.source_conversations ?? customer?.source_conversations ?? 0) }
  ];

  return (
    <div className="journey-head">
      <h2>{customer?.customer_name || "Unnamed customer"}</h2>
      <div className="cust-stats">
        {stats.map((stat) => (
          <div className="cust-stat" key={stat.label}>
            <span className={stat.tone ? `cust-stat-v tone-${stat.tone}` : "cust-stat-v"}>{stat.value}</span>
            <span className="cust-stat-k">{stat.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CustomerTickets({ customer, customerId, onOpenTicket }) {
  const tickets = useApi(
    customerId ? `/api/tickets?customerId=${customerId}&sort=recent&limit=200` : null,
    { keepPrevious: false }
  );
  const summary = useApi(customerId ? `/api/summary?customerId=${customerId}` : null, { keepPrevious: false });

  if (!customerId) {
    return (
      <div className="journey">
        <Empty title="No customer selected" text="Pick someone from the list to see every ticket raised on their account." />
      </div>
    );
  }
  if (tickets.error) {
    return <div className="journey"><Empty error title="Could not load tickets" text={tickets.error} /></div>;
  }

  const rows = tickets.data?.tickets || [];

  return (
    <div className="journey">
      <CustomerHeader customer={customer} totals={summary.data?.totals || {}} />

      <div className="journey-body scroll">
        {tickets.loading && !rows.length && <><div className="skel skel-block" /><div className="skel skel-block" /></>}

        {!tickets.loading && !rows.length && (
          <Empty title="No tickets" text="Nothing has been ticketed for this customer yet." />
        )}

        <div className="tcards">
          {rows.map((ticket) => (
            <TicketCard key={ticket.ticket_id} ticket={ticket} onOpen={onOpenTicket} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Customers({ state, update, onOpenTicket }) {
  const search = useDebounced(state.search, 300);
  const { data, loading, error } = useApi(`/api/customers?limit=200&search=${encodeURIComponent(search)}`);
  const customers = data?.customers || [];
  const selectedId = state.customer;
  const selected = customers.find((customer) => String(customer.id) === String(selectedId));

  useEffect(() => {
    if (!selectedId && customers.length) update({ customer: String(customers[0].id) }, { replace: true });
  }, [selectedId, customers, update]);

  return (
    <div className="custlayout">
      <div className="custlist scroll">
        {loading && !customers.length && <Skeleton rows={6} />}
        {error && <Empty error title="Could not load customers" text={error} />}
        {!loading && !customers.length && !error && (
          <Empty
            title="No customers found"
            text={search ? "Try a different name, or paste the full phone number." : "Nothing ingested yet."}
            icon={Search}
          />
        )}
        {customers.map((customer) => (
          <button
            className="custrow"
            type="button"
            key={customer.id}
            aria-current={String(customer.id) === String(selectedId)}
            onClick={() => update({ customer: String(customer.id) })}
          >
            <div className="custrow-name">{customer.customer_name || "Unnamed"}</div>
            <div className="custrow-tags">
              <Tag>{count(customer.tickets)} tickets</Tag>
              {customer.open_tickets > 0 && <Tag kind="warn">{customer.open_tickets} open</Tag>}
              {customer.bad_experience_tickets > 0 && <Tag kind="bad">{customer.bad_experience_tickets} bad</Tag>}
              {customer.manual_review_tickets > 0 && <Tag kind="warn">{customer.manual_review_tickets} review</Tag>}
            </div>
          </button>
        ))}
      </div>

      <CustomerTickets customer={selected} customerId={selectedId} onOpenTicket={onOpenTicket} />
    </div>
  );
}
