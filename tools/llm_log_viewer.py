"""Inspect logs/llm_calls.jsonl in the browser.

Standalone so it never touches app.py:

    streamlit run tools/llm_log_viewer.py --server.port 8502

The log is hundreds of MB, so the file is indexed by byte offset once per
(size, mtime) and individual records are seeked on demand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "llm_calls.jsonl"

CONTEXT_RE = re.compile(
    r"^(?P<kind>[a-z_]+):(?P<contract>[0-9a-f]+):pass(?P<n>\d+)/(?P<total>\d+):(?P<source>.+)$"
)


@st.cache_data(show_spinner="Indexing log file...")
def build_index(path_str: str, size: int, mtime: float) -> list[dict]:
    """Stream the log once, keeping only byte offsets and light metadata."""
    entries: list[dict] = []
    offset = 0
    with open(path_str, "rb") as handle:
        for raw in handle:
            start, offset = offset, offset + len(raw)
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8", "replace"))
            except Exception:
                continue
            usage = rec.get("usage") or {}
            context = str(rec.get("context") or "")
            match = CONTEXT_RE.match(context)
            entries.append(
                {
                    "offset": start,
                    "length": len(raw),
                    "timestamp": str(rec.get("timestamp") or "")[:19],
                    "context": context,
                    "kind": match.group("kind") if match else context.split(":")[0],
                    "contract": match.group("contract") if match else "",
                    "pass": int(match.group("n")) if match else 0,
                    "total_passes": int(match.group("total")) if match else 0,
                    "source_conversation_id": match.group("source") if match else "",
                    "model": rec.get("model") or "",
                    "thinking": rec.get("thinking_effort") or "",
                    "prompt_tokens": usage.get("prompt_tokens") or 0,
                    "completion_tokens": usage.get("completion_tokens") or 0,
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                    "elapsed_s": round(float(rec.get("elapsed_seconds") or 0), 1),
                    "system_prompt_chars": len(str(rec.get("system_prompt") or "")),
                    "success": bool(rec.get("success")),
                }
            )
    return entries


def load_record(path_str: str, offset: int, length: int) -> dict:
    with open(path_str, "rb") as handle:
        handle.seek(offset)
        return json.loads(handle.read(length).decode("utf-8", "replace"))


def parse_tickets(response_text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?", "", (response_text or "").strip()).rstrip("`").strip()
    try:
        payload = json.loads(text)
    except Exception:
        return []
    tickets = payload.get("tickets") if isinstance(payload, dict) else None
    return tickets if isinstance(tickets, list) else []


def ticket_rows(tickets: list[dict]) -> list[dict]:
    rows = []
    for ticket in tickets:
        if not isinstance(ticket, dict):
            continue
        idxs = [
            int(v)
            for v in (ticket.get("included_message_indexes") or [])
            if str(v).strip().lstrip("-").isdigit()
        ]
        rows.append(
            {
                "ticket_id": ticket.get("ticket_id"),
                "category": ticket.get("ticket_category"),
                "type": ticket.get("ticket_type"),
                "objective": ticket.get("customer_objective"),
                "status": ticket.get("status"),
                "n_msgs": len(idxs),
                "range": f"{min(idxs)}-{max(idxs)}" if idxs else "",
                "prev": ticket.get("previous_ticket_id") or "",
            }
        )
    return rows


def main() -> None:
    st.set_page_config(page_title="LLM call log", layout="wide")
    st.title("LLM call log viewer")

    if not LOG_PATH.exists():
        st.error(f"No log at {LOG_PATH}. Enable 'debug_log_llm_calls' in the main app and re-run.")
        st.stop()

    stat = LOG_PATH.stat()
    st.caption(
        f"{LOG_PATH}  —  {stat.st_size / 1e6:,.0f} MB  —  "
        f"modified {pd.Timestamp(stat.st_mtime, unit='s')}"
    )

    index = build_index(str(LOG_PATH), stat.st_size, stat.st_mtime)
    if not index:
        st.warning("Log parsed to zero records.")
        st.stop()

    frame = pd.DataFrame(index)
    kinds = sorted(frame["kind"].unique())
    default_kind = kinds.index("ticket_segmentation") if "ticket_segmentation" in kinds else 0
    kind = st.sidebar.selectbox("Call kind", kinds, index=default_kind)
    scoped = frame[frame["kind"] == kind]

    search = st.sidebar.text_input("Filter by contract hash or source id")
    if search:
        scoped = scoped[scoped["context"].str.contains(search, case=False, na=False)]

    contracts = (
        scoped[scoped["contract"] != ""]
        .groupby("contract")
        .agg(passes=("pass", "max"), calls=("pass", "size"), last=("timestamp", "max"))
        .sort_values("last", ascending=False)
        .reset_index()
    )
    st.sidebar.markdown(f"**{len(scoped)} calls / {len(contracts)} contracts**")

    if contracts.empty:
        st.warning("No contracts match.")
        st.stop()

    labels = [f"{r.contract[:12]}…  ({r.passes} passes, last {r.last})" for r in contracts.itertuples()]
    picked = st.sidebar.radio("Contract", range(len(labels)), format_func=lambda i: labels[i])
    contract_id = contracts.iloc[picked]["contract"]

    journey = (
        scoped[scoped["contract"] == contract_id]
        .sort_values(["pass", "timestamp"])
        .reset_index(drop=True)
    )

    st.subheader(f"Contract {contract_id}")
    cols = st.columns(4)
    cols[0].metric("Calls", len(journey))
    cols[1].metric("Max prompt tokens", f"{journey['prompt_tokens'].max():,}")
    cols[2].metric("Total completion", f"{journey['completion_tokens'].sum():,}")
    cols[3].metric("Wall time", f"{journey['elapsed_s'].sum() / 60:.1f} min")

    spikes = journey[journey["prompt_tokens"] > 100_000]
    if not spikes.empty:
        st.error(
            f"{len(spikes)} pass(es) exceeded 100k prompt tokens (max {spikes['prompt_tokens'].max():,}). "
            "Those inputs were likely truncated by the provider."
        )

    versions = sorted(journey["system_prompt_chars"].unique())
    if len(versions) > 1:
        st.warning(
            f"This journey used {len(versions)} different system-prompt versions "
            f"({', '.join(f'{v:,}' for v in versions)} chars). Passes are not comparable."
        )

    tab_passes, tab_evolution, tab_detail = st.tabs(["Passes", "Ticket evolution", "Call detail"])

    with tab_passes:
        st.dataframe(
            journey[
                ["pass", "total_passes", "timestamp", "model", "thinking", "prompt_tokens",
                 "cached_tokens", "completion_tokens", "elapsed_s", "system_prompt_chars",
                 "success", "source_conversation_id"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_evolution:
        st.caption("Tickets produced by each pass. Use this to find the pass where a merge or split went wrong.")
        limit = st.slider("Passes to render", 1, max(1, len(journey)), min(12, len(journey)))
        for row in journey.head(limit).itertuples():
            rec = load_record(str(LOG_PATH), int(row.offset), int(row.length))
            rows = ticket_rows(parse_tickets(str(rec.get("response_text") or "")))
            pass_no = getattr(row, "pass")
            with st.expander(f"pass {pass_no}/{row.total_passes} — {len(rows)} tickets — {row.timestamp}"):
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No parseable tickets in this response.")

    with tab_detail:
        sel = st.selectbox(
            "Pass",
            list(range(len(journey))),
            format_func=lambda i: (
                f"pass {journey.iloc[i]['pass']}/{journey.iloc[i]['total_passes']} "
                f"— {journey.iloc[i]['timestamp']}"
            ),
        )
        row = journey.iloc[sel]
        rec = load_record(str(LOG_PATH), int(row["offset"]), int(row["length"]))

        system_prompt = str(rec.get("system_prompt") or "")
        st.markdown(f"**System prompt used for this call** — {len(system_prompt):,} chars")
        probe = st.text_input("Was this rule live in this call?", "FINAL FIELD CHECK")
        if probe:
            st.success("Present") if probe in system_prompt else st.error("NOT present in this call's prompt")

        with st.expander("System prompt"):
            st.text(system_prompt)
        with st.expander("User prompt (this pass's input)", expanded=True):
            st.text(str(rec.get("user_prompt") or ""))
        with st.expander("Response", expanded=True):
            st.code(str(rec.get("response_text") or ""), language="json")
        if rec.get("reasoning"):
            with st.expander("Reasoning"):
                st.text(str(rec.get("reasoning")))
        st.json({"usage": rec.get("usage"), "errors": rec.get("errors"), "attempts": rec.get("attempts")})


if __name__ == "__main__":
    main()
