import React, { useState } from "react";
import { Check, ChevronRight, Info } from "lucide-react";
import { FACET_GROUPS, PRESETS, RANGES, orderOptions, presetActive } from "../lib/facets.js";
import { activeFilterCount } from "../lib/urlState.js";
import { count, cx, humanize } from "../lib/format.js";

function Option({ option, selected, disabled, swatch, mono, onToggle }) {
  return (
    <button
      className="opt"
      type="button"
      aria-pressed={selected}
      disabled={disabled && !selected}
      onClick={onToggle}
      title={humanize(option.value)}
    >
      <span className="opt-box">{selected && <Check size={10} strokeWidth={3.5} />}</span>
      {swatch && <span className="opt-swatch" style={{ background: `var(--band-${option.value}, var(--rule-firm))` }} />}
      <span className={cx("opt-name", mono && "mono")}>
        {mono ? String(option.value).slice(0, 8) : humanize(option.value)}
      </span>
      <span className="opt-count num">{count(option.count)}</span>
    </button>
  );
}

function Facet({ facet, options, selected, onToggle }) {
  const [expanded, setExpanded] = useState(false);
  const rows = orderOptions(options, facet);
  const limit = facet.limit || 6;
  const overflow = rows.length > limit + 1;
  const visible = expanded || !overflow ? rows : rows.slice(0, limit);

  if (!rows.length && !selected.length) return null;

  return (
    <div className="facet">
      <div className="facet-label">
        {facet.label}
        {facet.hint && <Info size={11} className="muted" aria-label={facet.hint} />}
      </div>
      {visible.map((option) => (
        <Option
          key={String(option.value)}
          option={option}
          selected={selected.includes(String(option.value))}
          disabled={!option.count}
          swatch={facet.swatch}
          mono={facet.mono}
          onToggle={() => onToggle(facet.key, String(option.value))}
        />
      ))}
      {overflow && (
        <button className="facet-more" type="button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Show fewer" : `Show ${rows.length - limit} more`}
        </button>
      )}
    </div>
  );
}

function DateFacet({ state, update }) {
  return (
    <div className="facetgroup">
      <div className="facet-label" style={{ padding: "10px 12px 5px" }}>Date range</div>
      <div className="daterow">
        {RANGES.map((range) => (
          <button
            key={range.value}
            className="datechip"
            type="button"
            aria-pressed={state.range === range.value}
            onClick={() => update({ range: range.value, page: 0 })}
          >
            {range.label}
          </button>
        ))}
      </div>
      {state.range === "custom" && (
        <div className="datefields">
          <input
            type="date"
            aria-label="From date"
            value={state.from}
            onChange={(event) => update({ from: event.target.value, page: 0 })}
          />
          <span>to</span>
          <input
            type="date"
            aria-label="To date"
            value={state.to}
            onChange={(event) => update({ to: event.target.value, page: 0 })}
          />
        </div>
      )}
    </div>
  );
}

function ScoreRange({ state, update }) {
  return (
    <div className="facet">
      <div className="facet-label">Score range</div>
      <div className="rangefields">
        <input
          type="number"
          min="0"
          max="100"
          placeholder="min"
          aria-label="Minimum score"
          value={state.scoreMin}
          onChange={(event) => update({ scoreMin: event.target.value, page: 0 })}
        />
        <input
          type="number"
          min="0"
          max="100"
          placeholder="max"
          aria-label="Maximum score"
          value={state.scoreMax}
          onChange={(event) => update({ scoreMax: event.target.value, page: 0 })}
        />
      </div>
    </div>
  );
}

function ContradictionFacet({ state, update }) {
  const on = state.hasContradiction === "true";
  return (
    <div className="facet">
      <button
        className="opt"
        type="button"
        aria-pressed={on}
        onClick={() => update({ hasContradiction: on ? "" : "true", page: 0 })}
      >
        <span className="opt-box">{on && <Check size={10} strokeWidth={3.5} />}</span>
        <span className="opt-name">Contains a contradiction</span>
      </button>
    </div>
  );
}

export default function FacetRail({ state, update, toggle, clearAll, facets, loading }) {
  const active = activeFilterCount(state);
  const counts = facets || {};

  return (
    <aside className="rail">
      <div className="rail-head">
        <h2>Filters</h2>
        <button className="linkbtn" type="button" onClick={clearAll} disabled={!active}>
          Clear{active ? ` (${active})` : ""}
        </button>
      </div>

      <div className="rail-body scroll">
        <div className="presets">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              className="preset"
              type="button"
              aria-pressed={presetActive(preset, state)}
              onClick={() => update((prev) => (presetActive(preset, state)
                ? { ...prev, ...Object.fromEntries(Object.keys(preset.patch).map((key) => [key, Array.isArray(preset.patch[key]) ? [] : ""])), page: 0, range: prev.range }
                : { ...prev, ...preset.patch }))}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <DateFacet state={state} update={update} />

        {FACET_GROUPS.map((group) => {
          const groupActive = group.facets.reduce((total, facet) => total + (state[facet.key] || []).length, 0)
            + (group.id === "outcome" && (state.scoreMin || state.scoreMax) ? 1 : 0)
            + (group.id === "friction" && state.hasContradiction ? 1 : 0);

          return (
            <details className="facetgroup" key={group.id} open={group.open || groupActive > 0}>
              <summary>
                <ChevronRight className="chev" size={13} strokeWidth={2.5} />
                {group.label}
                {groupActive > 0 && <span className="count-dot num">{groupActive}</span>}
              </summary>
              <div className="facetgroup-body">
                {group.facets.map((facet) => (
                  <Facet
                    key={facet.key}
                    facet={facet}
                    options={counts[facet.facet]}
                    selected={state[facet.key] || []}
                    onToggle={toggle}
                  />
                ))}
                {group.id === "outcome" && <ScoreRange state={state} update={update} />}
                {group.id === "friction" && <ContradictionFacet state={state} update={update} />}
              </div>
            </details>
          );
        })}

        {loading && !Object.keys(counts).length && (
          <div className="facet muted" style={{ fontSize: 12 }}>Loading counts…</div>
        )}
      </div>
    </aside>
  );
}
