import React, { useState } from "react";
import { AlertCircle, Braces, ChevronRight, Inbox } from "lucide-react";
import { cx, humanize, tone } from "../lib/format.js";

export function Tag({ children, kind, icon: Icon, title, optional = false }) {
  return (
    <span className={cx("tag", kind && `tag-${kind}`, optional && "tag-optional")} title={title}>
      {Icon && <Icon size={11} strokeWidth={2.2} />}
      <span>{children}</span>
    </span>
  );
}

/** Tag whose colour is derived from the domain value itself. */
export function ValueTag({ value, icon, prefix, optional }) {
  if (value === null || value === undefined || value === "") return null;
  const label = humanize(value);
  return (
    <Tag kind={tone(value)} icon={icon} optional={optional} title={prefix ? `${prefix}: ${label}` : label}>
      {prefix ? `${prefix} ${label.toLowerCase()}` : label}
    </Tag>
  );
}

export function Empty({ title, text, error = false, icon: Icon }) {
  const Glyph = Icon || (error ? AlertCircle : Inbox);
  return (
    <div className={cx("empty", error && "empty-err")}>
      <Glyph size={26} strokeWidth={1.6} />
      <strong>{title}</strong>
      {text && <span>{text}</span>}
    </div>
  );
}

export function Skeleton({ rows = 6 }) {
  return (
    <div aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, index) => (
        <div className="skel skel-row" key={index} />
      ))}
    </div>
  );
}

export function Section({ title, tally, children, open = true, id }) {
  return (
    <details className="dsection" open={open} name={id ? undefined : undefined}>
      <summary>
        <ChevronRight className="chev" size={13} strokeWidth={2.5} />
        {title}
        {tally != null && <span className="tally">{tally}</span>}
      </summary>
      <div className="dsection-body">{children}</div>
    </details>
  );
}

export function MetricGrid({ items }) {
  const rows = items.filter((item) => item && item.value !== null && item.value !== undefined && item.value !== "");
  if (!rows.length) return null;
  return (
    <div className="mgrid">
      {rows.map((item) => (
        <div className="mcell" key={item.label}>
          <div className="mcell-k">{item.label}</div>
          <div className={cx("mcell-v", item.numeric && "num")} title={String(item.value)}>
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
}

export function RawJson({ label = "raw data", data }) {
  const [open, setOpen] = useState(false);
  if (!data || (typeof data === "object" && !Object.keys(data).length)) return null;
  return (
    <div>
      <button className="rawtoggle" type="button" onClick={() => setOpen((value) => !value)}>
        <Braces size={12} />
        {open ? `Hide ${label}` : `Show ${label}`}
      </button>
      {open && <pre className="rawjson">{JSON.stringify(data, null, 2)}</pre>}
    </div>
  );
}
