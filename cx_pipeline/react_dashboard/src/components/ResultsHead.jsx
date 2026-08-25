import React from "react";
import { ChevronDown, ChevronLeft, ChevronRight, PanelLeftClose, PanelLeftOpen, X } from "lucide-react";
import { SORTS, facetByKey } from "../lib/facets.js";
import { MULTI_KEYS } from "../lib/urlState.js";
import { count, cx, delta, humanize, score } from "../lib/format.js";

/* KPI tiles double as filters — the number you're looking at is the filter you
   probably want next, so clicking it applies that filter rather than making
   you find the matching checkbox in the rail. */

function Kpi({ label, value, previous, asPercent, filter, state, update }) {
  const change = previous == null ? null : delta(value, previous, { asPercent });
  const active = filter
    ? Object.entries(filter).every(([key, patch]) => (Array.isArray(patch)
      ? patch.every((item) => (state[key] || []).includes(item))
      : String(state[key] ?? "") === String(patch)))
    : false;

  const content = (
    <>
      <div className="kpi-val">
        <span>{value == null ? "—" : asPercent ? score(value) ?? "—" : count(value)}</span>
        {change && change.direction !== "flat" && (
          <span className={cx("delta", `delta-${change.direction}`)}>{change.text}</span>
        )}
      </div>
      <div className="kpi-label">{label}</div>
    </>
  );

  if (!filter) return <div className="kpi">{content}</div>;

  return (
    <button
      className="kpi"
      type="button"
      data-clickable="true"
      aria-pressed={active}
      title={active ? `Remove ${label.toLowerCase()} filter` : `Filter to ${label.toLowerCase()}`}
      onClick={() => update((prev) => {
        const next = { ...prev, page: 0 };
        for (const [key, patch] of Object.entries(filter)) {
          next[key] = active ? (Array.isArray(patch) ? [] : "") : patch;
        }
        return next;
      })}
    >
      {content}
    </button>
  );
}

function ActiveChips({ state, update, toggle }) {
  const chips = [];

  for (const key of MULTI_KEYS) {
    const facet = facetByKey(key);
    for (const value of state[key] || []) {
      chips.push({
        id: `${key}:${value}`,
        label: facet?.label || key,
        value: facet?.mono ? String(value).slice(0, 8) : humanize(value),
        remove: () => toggle(key, value)
      });
    }
  }
  if (state.scoreMin) chips.push({ id: "scoreMin", label: "Score ≥", value: state.scoreMin, remove: () => update({ scoreMin: "", page: 0 }) });
  if (state.scoreMax) chips.push({ id: "scoreMax", label: "Score ≤", value: state.scoreMax, remove: () => update({ scoreMax: "", page: 0 }) });
  if (state.hasContradiction) chips.push({ id: "contra", label: "Contradiction", value: humanize(state.hasContradiction), remove: () => update({ hasContradiction: "", page: 0 }) });
  if (state.search) chips.push({ id: "search", label: "Search", value: state.search, remove: () => update({ search: "", page: 0 }) });
  if (state.customerId) chips.push({ id: "customerId", label: "Customer", value: `#${state.customerId}`, remove: () => update({ customerId: "", page: 0 }) });

  if (!chips.length) return null;

  return (
    <div className="chipbar">
      {chips.map((chip) => (
        <span className="fchip" key={chip.id}>
          <em>{chip.label}</em>
          <span className="truncate">{chip.value}</span>
          <button className="x" type="button" onClick={chip.remove} aria-label={`Remove ${chip.label} ${chip.value}`}>
            <X size={11} strokeWidth={2.6} />
          </button>
        </span>
      ))}
    </div>
  );
}

export default function ResultsHead({
  state,
  update,
  toggle,
  summary,
  total,
  loading,
  railCollapsed,
  onToggleRail
}) {
  const totals = summary?.totals || {};
  const previous = summary?.previous || {};
  const limit = state.limit || 50;
  const page = state.page || 0;
  const first = total ? page * limit + 1 : 0;
  const last = Math.min((page + 1) * limit, total || 0);
  const lastPage = Math.max(0, Math.ceil((total || 0) / limit) - 1);

  return (
    <div className="results-head">
      {loading && <div className="busy" />}

      <div className="kpis">
        <Kpi label="Tickets" value={totals.tickets} previous={previous.tickets} />
        <Kpi label="Avg score" value={totals.avg_score_100} previous={previous.avg_score_100} asPercent />
        <Kpi label="Bad experience" value={totals.bad_experience_tickets} previous={previous.bad_experience_tickets} filter={{ experience: ["bad"] }} state={state} update={update} />
        <Kpi label="Still open" value={totals.open_tickets} previous={previous.open_tickets} filter={{ status: ["pending_unresolved"] }} state={state} update={update} />
        <Kpi label="Needs review" value={totals.manual_review_tickets} previous={previous.manual_review_tickets} filter={{ manualReview: ["true"] }} state={state} update={update} />
        <Kpi label="High frustration" value={totals.high_frustration_tickets} previous={previous.high_frustration_tickets} filter={{ maxFrustration: ["high", "cancellation_risk"] }} state={state} update={update} />
        <Kpi label="Contradictions" value={totals.contradiction_tickets} previous={previous.contradiction_tickets} filter={{ hasContradiction: "true" }} state={state} update={update} />
      </div>

      <div className="toolbar">
        <button
          className="iconbtn"
          type="button"
          onClick={onToggleRail}
          aria-pressed={!railCollapsed}
          title={railCollapsed ? "Show filters" : "Hide filters"}
        >
          {railCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
        </button>

        <span className="toolbar-count">
          {total ? <><b>{count(first)}–{count(last)}</b> of <b>{count(total)}</b></> : "No tickets"}
        </span>

        <span className="spacer" />

        <label className="selectwrap">
          <select
            value={state.sort}
            aria-label="Sort tickets"
            onChange={(event) => update({ sort: event.target.value, page: 0 })}
          >
            {SORTS.map((sort) => <option key={sort.value} value={sort.value}>{sort.label}</option>)}
          </select>
          <ChevronDown className="chev" size={13} />
        </label>

        <label className="selectwrap">
          <select
            value={String(limit)}
            aria-label="Rows per page"
            onChange={(event) => update({ limit: Number(event.target.value), page: 0 })}
          >
            {/* Include the live value so a hand-written ?limit= in the URL
                shows what is actually in effect instead of falling back. */}
            {[...new Set([25, 50, 100, 200, 500, limit])].sort((a, b) => a - b).map((size) => (
              <option key={size} value={size}>{size} / page</option>
            ))}
          </select>
          <ChevronDown className="chev" size={13} />
        </label>

        <span className="pager">
          <button type="button" disabled={page <= 0} onClick={() => update({ page: page - 1 })} aria-label="Previous page">
            <ChevronLeft size={15} />
          </button>
          <button type="button" disabled={page >= lastPage} onClick={() => update({ page: page + 1 })} aria-label="Next page">
            <ChevronRight size={15} />
          </button>
        </span>
      </div>

      <ActiveChips state={state} update={update} toggle={toggle} />
    </div>
  );
}
