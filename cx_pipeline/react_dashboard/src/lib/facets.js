import { BAND_ORDER, FRUSTRATION_ORDER } from "./format.js";

/* Maps each filter parameter to the facet the API counts it under.
   `order` pins a scale to its natural sequence instead of letting count
   frequency shuffle it — a severity ramp read out of order is unreadable. */

export const FACET_GROUPS = [
  {
    id: "outcome",
    label: "Outcome",
    open: true,
    facets: [
      { key: "handled", facet: "handled_status", label: "Handled" },
      { key: "experience", facet: "customer_experience", label: "Experience" },
      { key: "scoreBand", facet: "score_band", label: "Score band", order: BAND_ORDER, swatch: true },
      { key: "unresolved", facet: "unhandled_resolution_subtype", label: "Resolution" }
    ]
  },
  {
    id: "friction",
    label: "Friction",
    open: true,
    facets: [
      { key: "maxFrustration", facet: "max_frustration_level", label: "Frustration level", order: FRUSTRATION_ORDER },
      { key: "frustrationOrigin", facet: "frustration_origin", label: "Frustration origin" }
    ]
  },
  {
    id: "issue",
    label: "Issue",
    open: true,
    facets: [
      { key: "mainIssue", facet: "main_issue_type", label: "Main issue" },
      { key: "messageIssue", facet: "message_issue_type", label: "Any message issue", hint: "Matches tickets containing this issue on any message, not just the main one." }
    ]
  },
  {
    id: "ticket",
    label: "Ticket",
    facets: [
      { key: "status", facet: "status", label: "Status" },
      { key: "category", facet: "category", label: "Category" },
      { key: "ticketType", facet: "ticket_type", label: "Type", limit: 8 },
      { key: "requestOrigin", facet: "request_origin", label: "Raised by" }
    ]
  },
  {
    id: "blame",
    label: "Responsibility",
    facets: [
      { key: "culprit", facet: "culprit_kind", label: "At fault" }
    ]
  },
  {
    id: "ops",
    label: "Operations",
    facets: [
      { key: "manualReview", facet: "manual_review_required", label: "Manual review" },
      { key: "runId", facet: "run_id", label: "Analysis run", limit: 5, mono: true }
    ]
  }
];

export const ALL_FACETS = FACET_GROUPS.flatMap((group) => group.facets);

export function facetByKey(key) {
  return ALL_FACETS.find((facet) => facet.key === key);
}

export const SORTS = [
  { value: "recent", label: "Most recent" },
  { value: "oldest", label: "Oldest first" },
  { value: "score_asc", label: "Worst score" },
  { value: "score_desc", label: "Best score" },
  { value: "frustration_desc", label: "Most frustrated" },
  { value: "messages_desc", label: "Longest" }
];

export const RANGES = [
  { value: "7d", label: "7d" },
  { value: "30d", label: "30d" },
  { value: "90d", label: "90d" },
  { value: "all", label: "All" },
  { value: "custom", label: "Custom" }
];

/* Presets are just patches. `matches` decides whether the pill reads as on. */
export const PRESETS = [
  {
    id: "review",
    label: "Needs review",
    patch: { manualReview: ["true"], range: "all", page: 0 }
  },
  {
    id: "bad",
    label: "Bad experience",
    patch: { experience: ["bad"], page: 0 }
  },
  {
    id: "low",
    label: "Score under 60",
    patch: { scoreMax: "59", page: 0 }
  },
  {
    id: "unhandled",
    label: "Unhandled",
    patch: { handled: ["unhandled"], page: 0 }
  },
  {
    id: "frustrated",
    label: "High frustration",
    patch: { maxFrustration: ["high", "cancellation_risk"], page: 0 }
  },
  {
    id: "contradiction",
    label: "Contradictions",
    patch: { hasContradiction: "true", page: 0 }
  }
];

export function presetActive(preset, state) {
  return Object.entries(preset.patch).every(([key, value]) => {
    if (key === "page") return true;
    if (Array.isArray(value)) {
      const current = state[key] || [];
      return value.every((item) => current.includes(item)) && current.length === value.length;
    }
    return String(state[key] ?? "") === String(value);
  });
}

/** Order a facet's options: pinned scale first if given, else by count. */
export function orderOptions(options, facet) {
  const rows = [...(options || [])];
  if (facet.order) {
    return rows.sort((a, b) => {
      const ai = facet.order.indexOf(a.value);
      const bi = facet.order.indexOf(b.value);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });
  }
  return rows.sort((a, b) => b.count - a.count || String(a.value).localeCompare(String(b.value)));
}
