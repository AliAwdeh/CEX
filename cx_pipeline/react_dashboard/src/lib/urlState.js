import { useCallback, useEffect, useRef, useState } from "react";
import { daysAgo, tomorrow } from "./format.js";

/* The querystring is the single source of truth for what the user is looking
   at. Back/forward work, a refresh keeps the view, and a filtered list is a
   link you can paste to someone. Nothing derives filter state from component
   state, so the list and the facet counts can never disagree. */

export const MULTI_KEYS = [
  "status",
  "category",
  "ticketType",
  "requestOrigin",
  "handled",
  "experience",
  "unresolved",
  "frustrationOrigin",
  "maxFrustration",
  "mainIssue",
  "messageIssue",
  "scoreBand",
  "culprit",
  "agent",
  "manualReview",
  "runId"
];

export const SINGLE_KEYS = [
  "search",
  "from",
  "to",
  "dateField",
  "scoreMin",
  "scoreMax",
  "hasContradiction",
  "customerId"
];

const UI_KEYS = ["view", "ticket", "customer", "page", "sort", "limit", "range"];

export const DEFAULTS = {
  view: "tickets",
  ticket: "",
  customer: "",
  page: 0,
  limit: 50,
  sort: "recent",
  range: "30d",
  dateField: "last"
};

/* Every filter key must always be present with a defined value. React inputs
   bound to `state.from` or `state.scoreMin` flip from controlled to
   uncontrolled the moment one of them goes undefined, so state is always built
   from this base rather than spread over DEFAULTS alone. */
export function emptyFilters() {
  const base = {};
  for (const key of SINGLE_KEYS) base[key] = "";
  for (const key of MULTI_KEYS) base[key] = [];
  return base;
}

export function parseState(searchString) {
  const params = new URLSearchParams(searchString);
  const state = { ...emptyFilters(), ...DEFAULTS };

  for (const key of UI_KEYS) {
    const value = params.get(key);
    if (value === null) continue;
    state[key] = key === "page" || key === "limit" ? Math.max(0, Number.parseInt(value, 10) || 0) : value;
  }
  if (!state.limit) state.limit = DEFAULTS.limit;

  for (const key of SINGLE_KEYS) state[key] = params.get(key) || "";
  for (const key of MULTI_KEYS) state[key] = params.getAll(key);

  return state;
}

export function serializeState(state) {
  const params = new URLSearchParams();

  for (const key of UI_KEYS) {
    const value = state[key];
    if (value === undefined || value === "" || value === null) continue;
    if (String(value) === String(DEFAULTS[key])) continue;
    params.set(key, String(value));
  }
  for (const key of SINGLE_KEYS) {
    if (!state[key]) continue;
    if (DEFAULTS[key] !== undefined && String(state[key]) === String(DEFAULTS[key])) continue;
    params.set(key, state[key]);
  }
  for (const key of MULTI_KEYS) {
    for (const value of state[key] || []) params.append(key, value);
  }

  const text = params.toString();
  return text ? `?${text}` : "";
}

/** Resolve the `range` preset into concrete from/to, unless a custom range is set. */
export function resolveDates(state) {
  if (state.range === "custom") return { from: state.from, to: state.to };
  if (state.range === "all") return { from: "", to: "" };
  const days = { "7d": 7, "30d": 30, "90d": 90 }[state.range];
  if (!days) return { from: "", to: "" };
  return { from: daysAgo(days), to: tomorrow() };
}

/** The subset of state the API cares about, as a querystring. */
export function filterQuery(state, { includePaging = false } = {}) {
  const params = new URLSearchParams();
  const { from, to } = resolveDates(state);

  for (const key of MULTI_KEYS) {
    for (const value of state[key] || []) params.append(key, value);
  }
  for (const key of SINGLE_KEYS) {
    if (key === "from" || key === "to") continue;
    if (state[key]) params.set(key, state[key]);
  }
  if (from) params.set("from", from);
  if (to) params.set("to", to);

  if (includePaging) {
    params.set("limit", String(state.limit || DEFAULTS.limit));
    params.set("offset", String((state.page || 0) * (state.limit || DEFAULTS.limit)));
    params.set("sort", state.sort || DEFAULTS.sort);
  }

  return params.toString();
}

export function activeFilterCount(state) {
  let total = 0;
  for (const key of MULTI_KEYS) total += (state[key] || []).length;
  for (const key of SINGLE_KEYS) {
    if (key === "from" || key === "to" || key === "dateField") continue;
    if (state[key]) total += 1;
  }
  if (state.range && state.range !== "all") total += 1;
  return total;
}

export function isFiltered(state) {
  return activeFilterCount(state) > 0;
}

/** Reactive wrapper around history state. */
export function useUrlState() {
  const [state, setState] = useState(() => parseState(window.location.search));
  const modeRef = useRef("push");
  const syncedRef = useRef(serializeState(parseState(window.location.search)));

  useEffect(() => {
    const onPop = () => {
      const next = parseState(window.location.search);
      syncedRef.current = serializeState(next);
      setState(next);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  /* The history write lives here rather than inside the state updater: an
     updater must be pure (StrictMode calls it twice), and doing it after the
     batch settles means N rapid updates produce one entry, not N. */
  useEffect(() => {
    const query = serializeState(state);
    if (query === syncedRef.current) return;
    syncedRef.current = query;
    const url = `${window.location.pathname}${query}`;
    if (modeRef.current === "replace") window.history.replaceState(null, "", url);
    else window.history.pushState(null, "", url);
    modeRef.current = "push";
  }, [state]);

  const update = useCallback((patch, { replace = false } = {}) => {
    if (replace) modeRef.current = "replace";
    setState((prev) => (typeof patch === "function" ? patch(prev) : { ...prev, ...patch }));
  }, []);

  /** Toggle one value inside a multi-select facet, and reset paging. */
  const toggle = useCallback((key, value) => {
    update((prev) => {
      const current = prev[key] || [];
      const next = current.includes(value) ? current.filter((item) => item !== value) : [...current, value];
      return { ...prev, [key]: next, page: 0 };
    });
  }, [update]);

  const clearAll = useCallback(() => {
    update((prev) => ({ ...emptyFilters(), ...DEFAULTS, view: prev.view, ticket: prev.ticket, customer: prev.customer, range: "all" }));
  }, [update]);

  return { state, update, toggle, clearAll };
}
