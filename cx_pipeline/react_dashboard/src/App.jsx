import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Gauge,
  Moon,
  RefreshCw,
  Rows2,
  Rows3,
  Search,
  Sun,
  Ticket,
  UserRound
} from "lucide-react";
import FacetRail from "./components/FacetRail.jsx";
import ResultsHead from "./components/ResultsHead.jsx";
import TicketList from "./components/TicketList.jsx";
import TicketDetail from "./components/TicketDetail.jsx";
import Customers from "./components/Customers.jsx";
import Transcript from "./components/Transcript.jsx";
import { apiPost, clearCache, useApi, useDebounced } from "./lib/api.js";
import { useFacets } from "./lib/useFacets.js";
import { filterQuery, isFiltered, useUrlState } from "./lib/urlState.js";
import { cx } from "./lib/format.js";

/* ---------- persisted display preferences ---------- */

function usePref(key, fallback) {
  const [value, setValue] = useState(() => localStorage.getItem(key) || fallback);
  useEffect(() => {
    localStorage.setItem(key, value);
    document.documentElement.setAttribute(`data-${key}`, value);
  }, [key, value]);
  return [value, setValue];
}

function Topbar({ state, update, theme, setTheme, density, setDensity, searchText, setSearchText, cacheStamp, onRefresh, refreshing }) {
  const views = [
    { id: "tickets", label: "Tickets", icon: Ticket },
    { id: "customers", label: "Customers", icon: UserRound }
  ];

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark"><Gauge size={16} strokeWidth={2.2} /></span>
        <span className="brand-name">CX Review</span>
      </div>

      <nav className="viewtabs">
        {views.map((view) => {
          const Icon = view.icon;
          return (
            <button
              key={view.id}
              className="viewtab"
              type="button"
              aria-current={state.view === view.id ? "page" : undefined}
              onClick={() => update({ view: view.id })}
            >
              <Icon size={14} />
              {view.label}
            </button>
          );
        })}
      </nav>

      <label className="topsearch">
        <Search size={14} />
        <input
          id="global-search"
          value={searchText}
          onChange={(event) => setSearchText(event.target.value)}
          placeholder="Search objective, issue, name, or paste a full phone number"
          aria-label="Search tickets"
        />
        {!searchText && <kbd>/</kbd>}
      </label>

      <div className="topright">
        {cacheStamp && <span className="cachestamp">{cacheStamp}</span>}
        <button
          className="iconbtn"
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
          title="Rebuild the stats cache from the source database"
          aria-label="Refresh cache"
        >
          <RefreshCw size={15} className={refreshing ? "spin" : undefined} />
        </button>
        <button
          className="iconbtn"
          type="button"
          aria-pressed={density === "compact"}
          onClick={() => setDensity(density === "compact" ? "comfortable" : "compact")}
          title={density === "compact" ? "Comfortable rows" : "Compact rows"}
          aria-label="Toggle row density"
        >
          {density === "compact" ? <Rows3 size={15} /> : <Rows2 size={15} />}
        </button>
        <button
          className="iconbtn"
          type="button"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "Light theme" : "Dark theme"}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>
    </header>
  );
}

export default function App() {
  const { state, update, toggle, clearAll } = useUrlState();
  const [theme, setTheme] = usePref("theme", window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const [density, setDensity] = usePref("density", "comfortable");
  const [railCollapsed, setRailCollapsed] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  /* Search is typed locally and only reaches the URL (and the network) once it
     settles, so a name is one request instead of one per keystroke. */
  const [searchText, setSearchText] = useState(state.search);
  const debouncedSearch = useDebounced(searchText, 300);

  useEffect(() => {
    if (debouncedSearch !== state.search) update({ search: debouncedSearch, page: 0 }, { replace: true });
  }, [debouncedSearch]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setSearchText((current) => (current === state.search ? current : state.search));
  }, [state.search]);

  const listQuery = useMemo(() => filterQuery(state, { includePaging: true }), [state]);
  const facetQuery = useMemo(() => filterQuery(state), [state]);

  const onTickets = state.view === "tickets";
  const tickets = useApi(onTickets ? `/api/tickets?${listQuery}` : null);
  const facets = useFacets(state, { enabled: onTickets });
  const summary = useApi(onTickets ? `/api/summary?${facetQuery}` : null);

  const rows = tickets.data?.tickets || [];
  const total = tickets.data?.total ?? 0;

  const openTicket = useCallback((id) => update({ ticket: String(id) }), [update]);
  const closeTicket = useCallback(() => update({ ticket: "" }), [update]);
  const openCustomer = useCallback((id) => update({ view: "customers", customer: String(id) }), [update]);
  const openTranscript = useCallback((id) => update({ view: "transcript", ticket: String(id) }), [update]);
  const backFromTranscript = useCallback(() => update({ view: "tickets" }), [update]);
  /* Widen the window on the way in: the customer view is not date-scoped, so a
     ticket older than the default range would open in the pane while being
     absent from the list behind it, and j/k would jump somewhere unrelated. */
  const openTicketFromCustomer = useCallback(
    (id) => update({ view: "tickets", ticket: String(id), range: "all", page: 0 }),
    [update]
  );

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await apiPost("/api/cache/refresh");
      clearCache();
      tickets.reload();
      summary.reload();
    } catch (error) {
      console.error("Cache refresh failed:", error);
    } finally {
      setRefreshing(false);
    }
  }, [tickets, summary]);

  /* Keyboard: j/k walk the list, Enter opens, Esc closes, / focuses search. */
  useEffect(() => {
    const onKey = (event) => {
      const target = event.target;
      const typing = target instanceof HTMLElement
        && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);

      if (event.key === "/" && !typing) {
        event.preventDefault();
        document.getElementById("global-search")?.focus();
        return;
      }
      if (event.key === "Escape") {
        if (typing) target.blur();
        else if (state.view === "transcript") backFromTranscript();
        else if (state.ticket) closeTicket();
        return;
      }
      if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "t" && state.ticket) {
        event.preventDefault();
        update({ view: state.view === "transcript" ? "tickets" : "transcript" });
        return;
      }
      if (state.view !== "tickets" || !rows.length) return;

      if (event.key === "j" || event.key === "k") {
        event.preventDefault();
        const step = event.key === "j" ? 1 : -1;
        /* Resolve against `prev`, not the captured render state — holding j
           repeats faster than React can re-bind this listener, and reading the
           closure would make every repeat land on the same row. */
        update((prev) => {
          const index = rows.findIndex((row) => String(row.ticket_id) === String(prev.ticket));
          const nextIndex = index === -1 ? (step === 1 ? 0 : rows.length - 1) : index + step;
          const next = rows[Math.max(0, Math.min(rows.length - 1, nextIndex))];
          return next ? { ...prev, ticket: String(next.ticket_id) } : prev;
        }, { replace: true });
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows, state.ticket, state.view, closeTicket, backFromTranscript, update]);

  const cacheStamp = summary.data?.cache?.refreshed_at
    ? `cache ${new Date(summary.data.cache.refreshed_at).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`
    : "";

  const busy = tickets.loading || facets.loading;

  return (
    <div className="app">
      <Topbar
        state={state}
        update={update}
        theme={theme}
        setTheme={setTheme}
        density={density}
        setDensity={setDensity}
        searchText={searchText}
        setSearchText={setSearchText}
        cacheStamp={cacheStamp}
        onRefresh={onRefresh}
        refreshing={refreshing}
      />

      {state.view === "transcript" ? (
        <Transcript ticketId={state.ticket} onBack={backFromTranscript} onOpenTicket={openTicket} />
      ) : state.view === "customers" ? (
        <Customers state={state} update={update} onOpenTicket={openTicketFromCustomer} />
      ) : (
        <div className={cx("workspace", railCollapsed && "rail-collapsed", !state.ticket && "no-detail")}>
          {!railCollapsed && (
            <FacetRail
              state={state}
              update={update}
              toggle={toggle}
              clearAll={clearAll}
              facets={facets.facets}
              loading={facets.loading}
            />
          )}

          <section className="results">
            <ResultsHead
              state={state}
              update={update}
              toggle={toggle}
              summary={summary.data}
              total={total}
              loading={busy}
              railCollapsed={railCollapsed}
              onToggleRail={() => setRailCollapsed((value) => !value)}
            />
            <TicketList
              tickets={rows}
              loading={tickets.loading}
              error={tickets.error}
              selectedId={state.ticket}
              onOpen={openTicket}
              filtered={isFiltered(state)}
              onClear={clearAll}
            />
          </section>

          {state.ticket && (
            <TicketDetail
              ticketId={state.ticket}
              onClose={closeTicket}
              onOpenCustomer={openCustomer}
              onOpenTranscript={openTranscript}
            />
          )}
        </div>
      )}
    </div>
  );
}
