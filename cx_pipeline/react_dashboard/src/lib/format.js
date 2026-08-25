/* Display formatting. Enum values are humanised once, here, so the raw value
   stays the thing we filter by and the label stays the thing we render. */

const LABELS = {
  not_analyzed: "Not analysed",
  not_applicable: "N/A",
  pending_unresolved: "Pending",
  totally_unresolved: "Unresolved",
  cancellation_risk: "Cancellation risk",
  our_side: "Our side",
  customer_side: "Customer side",
  tool_or_system_failure: "Tool / system failure",
  wrong_info: "Wrong info",
  unclear_guidance: "Unclear guidance",
  ignored_context: "Ignored context",
  missing_next_step: "Missing next step",
  dead_end: "Dead end",
  poor_tone: "Poor tone",
  high_risk_active: "High risk",
  critical_disregarded: "Critical",
  ticket_has_less_than_3_messages: "Less than 3 messages",
  true: "Yes",
  false: "No"
};

export function humanize(value) {
  if (value === null || value === undefined || value === "") return "None";
  const key = String(value);
  if (LABELS[key]) return LABELS[key];
  return key.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function count(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0";
  return n.toLocaleString();
}

export function score(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return n % 1 ? n.toFixed(1) : String(n);
}

export const BAND_ORDER = ["critical", "poor", "fair", "good", "excellent", "unscored"];
export const FRUSTRATION_ORDER = ["cancellation_risk", "high", "medium", "low", "none"];

export function bandOf(ticket) {
  return ticket.score_band || (ticket.score_final_100 == null ? "unscored" : "fair");
}

/* Semantic tone. Everything that maps a domain value to good/bad/warn lives
   here so the same value never reads green in one place and red in another. */
export function tone(value) {
  const t = String(value ?? "").toLowerCase();
  if (["good", "handled", "resolved", "satisfied", "excellent", "closed", "ok", "none", "not_applicable", "false"].includes(t)) {
    return t === "none" || t === "not_applicable" || t === "false" ? "neutral" : "good";
  }
  if (["bad", "unhandled", "frustrated", "dissatisfied", "critical", "poor", "cancellation_risk", "high", "totally_unresolved", "our_side"].includes(t)) return "bad";
  if (["critical_disregarded"].includes(t)) return "bad";
  if (["pending_unresolved", "medium", "confused", "fair", "shared", "true", "low", "high_risk_active"].includes(t)) return "warn";
  if (["neutral", "customer_side", "open"].includes(t)) return "info";
  return "neutral";
}

/* ---------- dates ---------- */

const DAY = 86400000;

export function isoDay(date) {
  return new Date(date).toISOString().slice(0, 10);
}

export function daysAgo(n) {
  return isoDay(Date.now() - n * DAY);
}

export function tomorrow() {
  return isoDay(Date.now() + DAY);
}

export function shortDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

export function dateTime(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function duration(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
}

export function span(from, to) {
  if (!from && !to) return "—";
  const a = shortDate(from);
  const b = shortDate(to);
  return a === b ? a : `${a} → ${b}`;
}

/* ---------- deltas ---------- */

export function delta(current, previous, { asPercent = false } = {}) {
  const a = Number(current);
  const b = Number(previous);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  const diff = a - b;
  if (Math.abs(diff) < (asPercent ? 0.05 : 0.5)) return { direction: "flat", text: "—" };
  const sign = diff > 0 ? "+" : "−";
  const magnitude = asPercent ? Math.abs(diff).toFixed(1) : count(Math.abs(Math.round(diff)));
  return { direction: diff > 0 ? "up" : "down", text: `${sign}${magnitude}` };
}

/* ---------- speaker identity ---------- */

export function speakerOf(message) {
  const raw = message.raw || {};
  const rawRole = String(message.raw_sender_role || raw.RAW_SENDER_ROLE || "").toLowerCase();
  const name = String(raw.MESSAGE_AGENT_FULL_NAME || raw.CONVERSATION_AGENT_FULL_NAME || "").trim();
  const skill = String(raw.MESSAGE_SKILL || raw.LAST_SKILL || "").trim();

  if (rawRole.includes("consumer") || rawRole.includes("customer") || message.sender_role === "customer") {
    return { kind: "customer", label: "Customer", detail: "" };
  }
  if (rawRole.includes("system") || rawRole.includes("broadcast")) {
    return { kind: "broadcast", label: "Broadcast", detail: skill };
  }
  if (rawRole.includes("bot") || skill.toLowerCase().includes("gpt")) {
    return { kind: "bot", label: "Bot", detail: skill };
  }
  if (rawRole.includes("agent") || message.sender_role === "agent") {
    return { kind: "agent", label: "Agent", detail: name };
  }
  return { kind: "unknown", label: "Unknown", detail: rawRole };
}

export function cx(...items) {
  return items.filter(Boolean).join(" ");
}
