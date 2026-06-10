"""Reusable Streamlit UI components: metric cards, transcript bubbles, evaluation panels."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st


# ----- Color hints for severity / sentiment -----

_FRUSTRATION_COLORS = {
    "none": "#16a34a",
    "low": "#65a30d",
    "medium": "#d97706",
    "high": "#dc2626",
    "cancellation_risk": "#7f1d1d",
}

_SENTIMENT_COLORS = {
    "satisfied": "#16a34a",
    "neutral": "#6b7280",
    "frustrated": "#dc2626",
    "confused": "#d97706",
    "dissatisfied": "#b91c1c",
    "unknown": "#6b7280",
}

_EFFECT_COLORS = {
    "helped": "#16a34a",
    "neutral": "#6b7280",
    "recovered_issue": "#0891b2",
    "minor_issue": "#d97706",
    "major_issue": "#dc2626",
}


def _badge(label: str, value: str, color: str) -> str:
    """Render a colored pill label/value badge."""
    safe_value = html.escape(str(value))
    safe_label = html.escape(str(label))
    return (
        f"<span style=\"display:inline-block;padding:2px 8px;margin:2px 6px 2px 0;"
        f"border-radius:9999px;background:{color}20;color:{color};"
        f"border:1px solid {color}55;font-size:0.78rem;font-weight:600;\">"
        f"{safe_label}: {safe_value}</span>"
    )


def metric_row(metrics: list[tuple[str, Any, str | None]]) -> None:
    """Render a row of st.metric cards from (label, value, delta) tuples."""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            if delta is None:
                st.metric(label, value)
            else:
                st.metric(label, value, delta)


def render_transcript(messages: list[dict]) -> None:
    """Render a clean chat-bubble transcript view."""
    if not messages:
        st.info("No messages to display.")
        return

    css = """
    <style>
      .chat-wrap { display:flex; flex-direction:column; gap:6px; padding:6px 0; }
      .bubble-row { display:flex; width:100%; }
      .bubble-row.customer { justify-content:flex-start; }
      .bubble-row.agent { justify-content:flex-end; }
      .bubble-row.unknown { justify-content:center; }
      .bubble {
        max-width:78%;
        padding:8px 12px;
        border-radius:14px;
        font-size:0.92rem;
        line-height:1.35;
        white-space:pre-wrap;
        word-break:break-word;
        border:1px solid transparent;
      }
      .bubble.customer { background:#eff6ff; border-color:#bfdbfe; color:#0f172a; border-top-left-radius:4px; }
      .bubble.agent    { background:#ecfdf5; border-color:#a7f3d0; color:#064e3b; border-top-right-radius:4px; }
      .bubble.unknown  { background:#f3f4f6; border-color:#e5e7eb; color:#374151; font-style:italic; }
      .bubble .meta { font-size:0.72rem; color:#6b7280; margin-bottom:3px; }
    </style>
    """
    html_parts = [css, "<div class=\"chat-wrap\">"]
    for m in messages:
        role = (m.get("sender_role") or "unknown").lower()
        klass = role if role in ("customer", "agent") else "unknown"
        idx = m.get("message_index")
        when = m.get("message_time") or ""
        text = html.escape(str(m.get("message_text", "")))
        meta_bits = []
        if idx is not None:
            meta_bits.append(f"#{idx}")
        meta_bits.append(role.capitalize())
        if when:
            meta_bits.append(html.escape(str(when)))
        meta = " • ".join(meta_bits)
        html_parts.append(
            f"<div class=\"bubble-row {klass}\"><div class=\"bubble {klass}\">"
            f"<div class=\"meta\">{meta}</div>{text}</div></div>"
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _highlight_box(color: str, text: str) -> str:
    """Compact colored callout used to flag major issues / cancellation risk."""
    return (
        f"<div style=\"background:{color}15;border-left:4px solid {color};"
        f"padding:6px 10px;border-radius:4px;color:#111;font-weight:600;"
        f"font-size:0.82rem;margin-top:4px;\">"
        f"{html.escape(text)}</div>"
    )


def render_inline_evaluation(message_result: dict) -> None:
    """Render a compact evaluation card to display directly under an agent message.

    Includes colored badges (effect, frustration, change, issue type, origin) plus
    evidence / impact / fix / cause as short labelled lines. Designed to live
    inside an ``st.chat_message`` block so it sits next to its message.
    """
    pj = message_result.get("parsed_json") or {}
    status = message_result.get("parse_status", "ok")

    if status != "ok":
        st.markdown(
            _highlight_box("#dc2626", f"Evaluation could not be parsed ({status})"),
            unsafe_allow_html=True,
        )
        with st.expander("Show error"):
            st.code(message_result.get("error_message") or "—")
        return

    eff = pj.get("message_level_effect", "neutral")
    fl = pj.get("frustration_level_after_message", "none")
    fc = pj.get("frustration_change", "unchanged")
    it = pj.get("issue_type") or "none"
    io = pj.get("issue_origin") or "none"

    fc_color = {
        "decreased": "#16a34a",
        "unchanged": "#6b7280",
        "increased": "#d97706",
        "created": "#dc2626",
    }.get(fc, "#6b7280")

    badges = [
        _badge("Effect", eff.replace("_", " "), _EFFECT_COLORS.get(eff, "#6b7280")),
        _badge("Frustration", fl.replace("_", " "), _FRUSTRATION_COLORS.get(fl, "#6b7280")),
        _badge("Change", fc.replace("_", " "), fc_color),
    ]
    if it and it != "none":
        badges.append(_badge("Issue", it.replace("_", " "), "#b91c1c"))
    if io and io != "none":
        badges.append(_badge("Origin", io.replace("_", " "), "#475569"))

    # Eye-catching banners for the worst categories.
    banners = []
    if eff == "major_issue":
        banners.append(_highlight_box("#dc2626", "Major issue detected"))
    elif eff == "recovered_issue":
        banners.append(_highlight_box("#0891b2", "This message recovered from a prior issue"))
    if fl == "cancellation_risk":
        banners.append(_highlight_box("#7f1d1d", "Cancellation risk after this message"))
    elif fl == "high":
        banners.append(_highlight_box("#dc2626", "High frustration after this message"))

    st.markdown("".join(badges) + "".join(banners), unsafe_allow_html=True)

    cause = pj.get("frustration_cause") or "none"
    evidence = pj.get("evidence") or ""
    impact = pj.get("business_impact") or ""
    fix = pj.get("recommended_fix") or ""

    lines = []
    if evidence:
        lines.append(f"- **Evidence:** {evidence}")
    if impact:
        lines.append(f"- **Customer impact:** {impact}")
    if fix:
        lines.append(f"- **Recommended fix:** {fix}")
    if cause and cause.lower() != "none":
        lines.append(f"- **Frustration cause:** {cause}")
    if lines:
        st.markdown("\n".join(lines))


def render_conversation_transcript_with_evals(
    transcript: list[dict],
    message_results: list[dict] | None,
) -> None:
    """Render a transcript using native chat bubbles, with the message-level
    evaluation card attached directly under each agent message.

    Each bubble's header shows the raw sender role from the CSV (preferring
    ``raw_sender_role`` when present, otherwise the normalized ``sender_role``).
    No customer or agent name is rendered.
    """
    if not transcript:
        st.info("No messages to display.")
        return

    eval_by_idx = {m.get("message_index"): m for m in (message_results or [])}

    n_evals = sum(1 for m in (message_results or []) if m.get("parsed_json"))
    n_major = sum(
        1 for m in (message_results or [])
        if (m.get("parsed_json") or {}).get("message_level_effect") == "major_issue"
    )
    n_minor = sum(
        1 for m in (message_results or [])
        if (m.get("parsed_json") or {}).get("message_level_effect") == "minor_issue"
    )
    n_recovered = sum(
        1 for m in (message_results or [])
        if (m.get("parsed_json") or {}).get("message_level_effect") == "recovered_issue"
    )
    st.caption(
        f"{len(transcript)} messages • {n_evals} agent messages evaluated • "
        f"{n_major} major · {n_minor} minor · {n_recovered} recovered"
    )

    for msg in transcript:
        role = (msg.get("sender_role") or "unknown").lower()
        raw_role = msg.get("raw_sender_role")
        display_role = str(raw_role) if raw_role else (msg.get("sender_role") or "unknown")
        idx = msg.get("message_index")
        when = msg.get("message_time") or ""
        text = str(msg.get("message_text", "") or "")

        if role == "customer":
            chat_role, avatar = "user", "🧑"
        elif role == "agent":
            chat_role, avatar = "assistant", "🤖"
        else:
            chat_role, avatar = "ai", "❔"

        with st.chat_message(chat_role, avatar=avatar):
            header_bits = [f"**{display_role}**"]
            if idx is not None:
                header_bits.append(f"#{idx}")
            if when:
                header_bits.append(str(when))
            st.markdown(" · ".join(header_bits))
            st.write(text if text else "_(empty message)_")

            if role == "agent":
                eval_record = eval_by_idx.get(idx)
                if eval_record:
                    st.markdown(
                        "<div style=\"height:1px;background:#e5e7eb;margin:8px 0;\"></div>",
                        unsafe_allow_html=True,
                    )
                    render_inline_evaluation(eval_record)
                else:
                    st.caption("_No evaluation was recorded for this message._")


def render_message_evaluation_panel(message_result: dict) -> None:
    """Render a single message-level evaluation panel inside an expander or container."""
    pj = message_result.get("parsed_json") or {}
    status = message_result.get("parse_status", "ok")
    text = message_result.get("target_message_text", "")
    idx = message_result.get("message_index")

    badges = []
    eff = pj.get("message_level_effect", "neutral")
    badges.append(_badge("Effect", eff.replace("_", " "), _EFFECT_COLORS.get(eff, "#6b7280")))
    fl = pj.get("frustration_level_after_message", "none")
    badges.append(_badge("Frustration", fl.replace("_", " "), _FRUSTRATION_COLORS.get(fl, "#6b7280")))
    fc = pj.get("frustration_change", "unchanged")
    fc_color = {"decreased": "#16a34a", "unchanged": "#6b7280", "increased": "#d97706", "created": "#dc2626"}.get(fc, "#6b7280")
    badges.append(_badge("Change", fc.replace("_", " "), fc_color))
    if pj.get("issue_type") and pj["issue_type"] != "none":
        badges.append(_badge("Issue", pj["issue_type"].replace("_", " "), "#b91c1c"))
    if pj.get("issue_origin") and pj["issue_origin"] != "none":
        badges.append(_badge("Origin", pj["issue_origin"].replace("_", " "), "#475569"))

    st.markdown("".join(badges), unsafe_allow_html=True)

    if idx is not None:
        st.caption(f"Message index: {idx}")

    st.markdown("**Agent message**")
    st.write(text or "_(empty)_")

    if status != "ok":
        st.warning(
            f"This evaluation could not be parsed ({status}). The conversation was still summarized."
        )
        with st.expander("Error details"):
            st.code(message_result.get("error_message", "Unknown error"))
        return

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Evidence**")
        st.write(pj.get("evidence") or "_(none)_")
        st.markdown("**Business impact**")
        st.write(pj.get("business_impact") or "_(none)_")
    with cols[1]:
        st.markdown("**Recommended fix**")
        st.write(pj.get("recommended_fix") or "_(none)_")
        st.markdown("**Frustration cause**")
        st.write(pj.get("frustration_cause") or "_(none)_")


def render_conversation_summary_card(conv_result: dict) -> None:
    """Render a clean summary card for a conversation."""
    pj = conv_result.get("parsed_json") or {}
    md = conv_result.get("conversation_metadata") or {}
    cm = conv_result.get("computed_metadata") or {}

    classification = pj.get("final_classification", "Unknown")
    handled = pj.get("handled_status", "unknown")
    severity = pj.get("cx_issue_severity", "")
    sentiment = pj.get("final_customer_sentiment", "unknown")
    max_fl = pj.get("max_frustration_level", "none")
    main = pj.get("main_issue") or {}

    head_colors = {
        "Handled with Zero/Minimal Issues": "#16a34a",
        "Handled with Many Issues": "#d97706",
        "Unhandled with Zero/Minimal Issues": "#0ea5e9",
        "Unhandled with Many Issues": "#dc2626",
    }
    color = head_colors.get(classification, "#6b7280")

    st.markdown(
        f"""
        <div style="border-left:6px solid {color}; padding:10px 14px; background:#f9fafb; border-radius:6px; margin-bottom:8px;">
          <div style="font-size:0.95rem; color:#374151;">Final Classification</div>
          <div style="font-size:1.25rem; font-weight:700; color:{color};">{html.escape(classification)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    badges = []
    badges.append(_badge("Handled", handled, "#16a34a" if handled == "handled" else "#dc2626"))
    badges.append(_badge("CX Severity", severity.replace("_", " ") or "—", "#d97706" if severity == "many" else "#16a34a"))
    badges.append(_badge("Sentiment", sentiment, _SENTIMENT_COLORS.get(sentiment, "#6b7280")))
    badges.append(_badge("Frustration", max_fl.replace("_", " "), _FRUSTRATION_COLORS.get(max_fl, "#6b7280")))
    if pj.get("manual_review_required"):
        badges.append(_badge("Manual review", "required", "#dc2626"))
    badges.append(_badge("Confidence", pj.get("confidence", "—"), "#475569"))
    st.markdown("".join(badges), unsafe_allow_html=True)

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown("**Conversation ID**")
        st.write(conv_result.get("conversation_id", ""))
        st.markdown("**Customer**")
        st.write(md.get("customer_name") or "—")
        st.markdown("**Phone**")
        st.write(md.get("customer_phone") or "—")
    with cols[1]:
        st.markdown("**Start**")
        st.write(md.get("conversation_start_date") or "—")
        st.markdown("**End**")
        st.write(md.get("conversation_end_date") or "—")
        st.markdown("**Status**")
        st.write(md.get("conversation_status") or "—")
    with cols[2]:
        st.markdown("**Initial skill**")
        st.write(md.get("initial_skill") or "—")
        st.markdown("**Last skill**")
        st.write(md.get("last_skill") or "—")
        st.markdown("**Agent**")
        st.write(md.get("conversation_agent_full_name") or "—")

    st.markdown("---")
    st.markdown("### Main CX Issue")
    if main.get("issue_exists"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Issue type:** {main.get('issue_type', '—').replace('_', ' ')}")
            st.markdown(f"**Issue origin:** {main.get('issue_origin', '—').replace('_', ' ')}")
            st.markdown("**Summary**")
            st.write(main.get("issue_summary") or "—")
        with cols[1]:
            st.markdown("**Customer impact**")
            st.write(main.get("customer_impact") or "—")
    else:
        st.success("No significant CX issue detected.")

    st.markdown("### Management Summary")
    st.write(pj.get("management_summary") or "—")

    rec = pj.get("recommended_actions") or []
    if rec:
        st.markdown("### Recommended Actions")
        for r in rec:
            st.write(f"- {r}")

    if pj.get("manual_review_required"):
        reason = pj.get("manual_review_reason") or "Flagged by evaluator."
        st.warning(f"Manual review required: {reason}")

    with st.expander("Computed metadata"):
        st.json(cm, expanded=False)


def conversation_filters(conv_df: pd.DataFrame) -> dict:
    """Render filter widgets and return the active filter values."""
    if conv_df.empty:
        return {}

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            classifications = sorted(
                [c for c in conv_df.get("final_classification", pd.Series(dtype=str)).dropna().unique()]
            )
            sel_class = st.multiselect("Classification", classifications, default=[])
            handled = sorted([h for h in conv_df.get("handled_status", pd.Series(dtype=str)).dropna().unique()])
            sel_handled = st.multiselect("Handled status", handled, default=[])
            severity = sorted([s for s in conv_df.get("cx_issue_severity", pd.Series(dtype=str)).dropna().unique()])
            sel_severity = st.multiselect("CX issue severity", severity, default=[])
        with c2:
            frustration_levels = sorted(
                [f for f in conv_df.get("max_frustration_level", pd.Series(dtype=str)).dropna().unique()]
            )
            sel_frustration = st.multiselect("Max frustration", frustration_levels, default=[])
            origins = sorted([o for o in conv_df.get("main_issue_origin", pd.Series(dtype=str)).dropna().unique()])
            sel_origin = st.multiselect("Issue origin", origins, default=[])
            issue_types = sorted([t for t in conv_df.get("main_issue_type", pd.Series(dtype=str)).dropna().unique()])
            sel_issue_type = st.multiselect("Issue type", issue_types, default=[])
        with c3:
            mr_options = ["Any", "Only manual review", "Only no manual review"]
            sel_mr = st.selectbox("Manual review", mr_options, index=0)
            initial_skills = sorted([s for s in conv_df.get("initial_skill", pd.Series(dtype=str)).dropna().unique()])
            sel_initial_skill = st.multiselect("Initial skill", initial_skills, default=[])
            last_skills = sorted([s for s in conv_df.get("last_skill", pd.Series(dtype=str)).dropna().unique()])
            sel_last_skill = st.multiselect("Last skill", last_skills, default=[])

        c4, c5 = st.columns(2)
        with c4:
            agents = sorted(
                [a for a in conv_df.get("conversation_agent_full_name", pd.Series(dtype=str)).dropna().unique()]
            )
            sel_agents = st.multiselect("Agent name", agents, default=[])
        with c5:
            date_range = None
            if "conversation_start_date" in conv_df.columns:
                parsed = pd.to_datetime(conv_df["conversation_start_date"], errors="coerce")
                non_null = parsed.dropna()
                if len(non_null) >= 2:
                    min_d = non_null.min().date()
                    max_d = non_null.max().date()
                    date_range = st.date_input(
                        "Date range",
                        value=(min_d, max_d),
                        min_value=min_d,
                        max_value=max_d,
                    )

    return {
        "classification": sel_class,
        "handled_status": sel_handled,
        "cx_issue_severity": sel_severity,
        "max_frustration_level": sel_frustration,
        "main_issue_origin": sel_origin,
        "main_issue_type": sel_issue_type,
        "manual_review": sel_mr,
        "initial_skill": sel_initial_skill,
        "last_skill": sel_last_skill,
        "agent": sel_agents,
        "date_range": date_range,
    }


def apply_conversation_filters(conv_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the active filters returned from ``conversation_filters``."""
    if conv_df.empty or not filters:
        return conv_df
    df = conv_df.copy()

    def in_filter(col: str, key: str):
        nonlocal df
        sel = filters.get(key) or []
        if sel and col in df.columns:
            df = df[df[col].isin(sel)]

    in_filter("final_classification", "classification")
    in_filter("handled_status", "handled_status")
    in_filter("cx_issue_severity", "cx_issue_severity")
    in_filter("max_frustration_level", "max_frustration_level")
    in_filter("main_issue_origin", "main_issue_origin")
    in_filter("main_issue_type", "main_issue_type")
    in_filter("initial_skill", "initial_skill")
    in_filter("last_skill", "last_skill")
    in_filter("conversation_agent_full_name", "agent")

    mr = filters.get("manual_review")
    if mr == "Only manual review" and "manual_review_required" in df.columns:
        df = df[df["manual_review_required"].fillna(False).astype(bool)]
    elif mr == "Only no manual review" and "manual_review_required" in df.columns:
        df = df[~df["manual_review_required"].fillna(False).astype(bool)]

    dr = filters.get("date_range")
    if dr and "conversation_start_date" in df.columns:
        try:
            start, end = dr
            parsed = pd.to_datetime(df["conversation_start_date"], errors="coerce")
            df = df[(parsed.dt.date >= start) & (parsed.dt.date <= end)]
        except Exception:
            pass

    return df
