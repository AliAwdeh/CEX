"""Reusable Streamlit UI components: metric cards, transcript bubbles, evaluation panels."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from aggregation import humanize_label


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
    """Render a compact evaluation card to display directly under an assistant message.

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
        _badge("Effect", humanize_label(eff), _EFFECT_COLORS.get(eff, "#6b7280")),
        _badge("Frustration", humanize_label(fl), _FRUSTRATION_COLORS.get(fl, "#6b7280")),
        _badge("Change", humanize_label(fc), fc_color),
    ]
    if it and it != "none":
        badges.append(_badge("Issue", humanize_label(it), "#b91c1c"))
    if io and io != "none":
        badges.append(_badge("Origin", humanize_label(io), "#475569"))

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
    evaluation card attached directly under each assistant message.

    Header label rules:
    - Customer / unknown rows show the raw sender role from the CSV.
    - Assistant rows are shown with a generic assistant label; names are not
      shown in the frontend.
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
        f"{len(transcript)} messages • {n_evals} messages evaluated • "
        f"{n_major} major · {n_minor} minor · {n_recovered} recovered"
    )

    for msg in transcript:
        role = (msg.get("sender_role") or "unknown").lower()
        raw_role = msg.get("raw_sender_role")
        raw_role_norm = str(raw_role).strip().lower() if raw_role else ""
        idx = msg.get("message_index")
        when = msg.get("message_time") or ""
        text = str(msg.get("message_text", "") or "")

        if role == "customer":
            chat_role, avatar = "user", "🧑"
            display_role = str(raw_role) if raw_role else "customer"
        elif role == "agent":
            chat_role, avatar = "assistant", "🤖"
            if raw_role_norm == "agent":
                display_role = "Assistant"
            else:
                display_role = "Assistant"
        else:
            chat_role, avatar = "ai", "❔"
            display_role = str(raw_role) if raw_role else "unknown"

        with st.chat_message(chat_role, avatar=avatar):
            header_bits = [f"**{display_role}**"]
            if idx is not None:
                header_bits.append(f"#{idx}")
            if when:
                header_bits.append(str(when))
            st.markdown(" · ".join(header_bits))
            st.write(text if text else "_(empty message)_")

            # Attach the message-level evaluation card to whichever message
            # was actually evaluated (assistant in assistant mode, customer in
            # customer mode). ``eval_by_idx`` only contains entries for
            # target messages, so this works for both modes.
            eval_record = eval_by_idx.get(idx)
            if eval_record:
                st.markdown(
                    "<div style=\"height:1px;background:#e5e7eb;margin:8px 0;\"></div>",
                    unsafe_allow_html=True,
                )
                render_inline_evaluation(eval_record)


def render_message_evaluation_panel(message_result: dict) -> None:
    """Render a single message-level evaluation panel inside an expander or container."""
    pj = message_result.get("parsed_json") or {}
    status = message_result.get("parse_status", "ok")
    text = message_result.get("target_message_text", "")
    idx = message_result.get("message_index")

    badges = []
    eff = pj.get("message_level_effect", "neutral")
    badges.append(_badge("Effect", humanize_label(eff), _EFFECT_COLORS.get(eff, "#6b7280")))
    fl = pj.get("frustration_level_after_message", "none")
    badges.append(_badge("Frustration", humanize_label(fl), _FRUSTRATION_COLORS.get(fl, "#6b7280")))
    fc = pj.get("frustration_change", "unchanged")
    fc_color = {"decreased": "#16a34a", "unchanged": "#6b7280", "increased": "#d97706", "created": "#dc2626"}.get(fc, "#6b7280")
    badges.append(_badge("Change", humanize_label(fc), fc_color))
    if pj.get("issue_type") and pj["issue_type"] != "none":
        badges.append(_badge("Issue", humanize_label(pj["issue_type"]), "#b91c1c"))
    if pj.get("issue_origin") and pj["issue_origin"] != "none":
        badges.append(_badge("Origin", humanize_label(pj["issue_origin"]), "#475569"))

    st.markdown("".join(badges), unsafe_allow_html=True)

    if idx is not None:
        st.caption(f"Message index: {idx}")

    st.markdown("**Assistant message**")
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
    severity_display = humanize_label(severity)
    frustration_detected = bool(pj.get("frustration_detected", False))
    frustration_timing = pj.get("frustration_timing", "none")
    subtype = pj.get("unhandled_resolution_subtype", "")
    sentiment = pj.get("final_customer_sentiment", "unknown")
    max_fl = pj.get("max_frustration_level", "none")
    main = pj.get("main_issue") or {}

    head_colors = {
        "Handled with Minimal Issues": "#16a34a",
        "Handled with Many Issues": "#d97706",
        "Handled with Minimal Issues and Frustration": "#0f766e",
        "Handled with Many Issues and Frustration": "#b45309",
        "Handled with Minimal Caused Issues and Frustration": "#0f766e",
        "Handled with Many Caused Issues and Frustration": "#c2410c",
        "Not Handled with Minimal Issues": "#0284c7",
        "Not Handled with Many Issues": "#dc2626",
        "Not Handled with Minimal Issues and Frustration": "#0369a1",
        "Not Handled with Many Issues and Frustration": "#b91c1c",
        "Not Handled with Minimal Caused Issues and Frustration": "#075985",
        "Not Handled with Many Caused Issues and Frustration": "#991b1b",
    }
    color = head_colors.get(classification, "#6b7280")
    subtype_display = humanize_label(subtype) or "n/a"
    show_unresolved_header_badge = handled == "unhandled"
    subtype_color = "#0f766e" if subtype == "pending_unresolved" else "#b45309"
    unresolved_html = ""
    if show_unresolved_header_badge:
        unresolved_html = f"""
          <div style="margin-top:12px; padding:14px 16px; border-radius:8px; background:#fff7ed; border:1px solid #fed7aa;">
            <div style="font-size:0.8rem; font-weight:800; color:#7c2d12; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px;">
              Unresolved status
            </div>
            <div style="font-size:1.18rem; font-weight:900; color:{subtype_color};">
              {html.escape(subtype_display)}
            </div>
          </div>
        """

    st.markdown(
        f"""
        <div style="border-left:6px solid {color}; padding:14px 18px; background:#f9fafb; border-radius:6px; margin-bottom:12px;">
          <div style="font-size:0.95rem; color:#374151;">Overall result</div>
          <div style="font-size:1.25rem; font-weight:700; color:{color};">{html.escape(classification)}</div>
          {unresolved_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    badges = []
    badges.append(_badge("Outcome", humanize_label(handled), "#16a34a" if handled == "handled" else "#dc2626"))
    badges.append(_badge("Journey quality", severity_display or "n/a", "#d97706" if severity == "many" else "#16a34a"))
    badges.append(
        _badge("Customer frustration", "yes" if frustration_detected else "no", "#b91c1c" if frustration_detected else "#475569")
    )
    if frustration_detected:
        badges.append(_badge("When frustration appeared", humanize_label(frustration_timing) or "n/a", "#475569"))
    if not show_unresolved_header_badge:
        badges.append(_badge("Unresolved status", subtype_display, "#475569"))
    badges.append(_badge("Customer feeling at end", humanize_label(sentiment), _SENTIMENT_COLORS.get(sentiment, "#6b7280")))
    badges.append(_badge("Highest frustration level", humanize_label(max_fl), _FRUSTRATION_COLORS.get(max_fl, "#6b7280")))
    if pj.get("manual_review_required"):
        badges.append(_badge("Needs human review", "yes", "#dc2626"))
    badges.append(_badge("Confidence", pj.get("confidence", "—"), "#475569"))
    st.markdown("".join(badges), unsafe_allow_html=True)

    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown("**ID**")
        st.write(conv_result.get("conversation_id", ""))
        st.markdown("**Customer**")
        st.write(md.get("customer_name") or "—")
        st.markdown("**Phone**")
        st.write(md.get("customer_phone") or "—")
    with cols[1]:
        st.markdown("**Started**")
        st.write(md.get("conversation_start_date") or "—")
        st.markdown("**Ended**")
        st.write(md.get("conversation_end_date") or "—")
        st.markdown("**Conversation status**")
        st.write(md.get("conversation_status") or "—")
    st.markdown("---")
    st.markdown("### Main Customer Problem")
    if main.get("issue_exists"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Problem type:** {humanize_label(main.get('issue_type')) or 'n/a'}")
            st.markdown(f"**Where it came from:** {humanize_label(main.get('issue_origin')) or 'n/a'}")
            st.markdown("**What happened**")
            st.write(main.get("issue_summary") or "—")
        with cols[1]:
            st.markdown("**Impact on the customer**")
            st.write(main.get("customer_impact") or "—")
    else:
        st.success("No major customer problem was detected.")

    st.markdown("### Business Summary")
    st.write(pj.get("management_summary") or "—")

    positives = pj.get("positive_signals") or []
    negatives = pj.get("negative_signals") or []
    if positives or negatives:
        sig_cols = st.columns(2)
        with sig_cols[0]:
            st.markdown("### What Went Well")
            if positives:
                for item in positives:
                    st.write(f"- {item}")
            else:
                st.write("n/a")
        with sig_cols[1]:
            st.markdown("### What Went Wrong")
            if negatives:
                for item in negatives:
                    st.write(f"- {item}")
            else:
                st.write("n/a")

    issues = pj.get("all_detected_issues") or []
    if issues:
        with st.expander("All customer issues found", expanded=False):
            issue_df = pd.DataFrame(issues)
            for col in ("issue_origin", "issue_type"):
                if col in issue_df.columns:
                    issue_df[col] = issue_df[col].apply(humanize_label)
            issue_df = issue_df.rename(columns={c: humanize_label(c) for c in issue_df.columns})
            st.dataframe(issue_df, use_container_width=True, hide_index=True)

    quantifiable_metrics = pj.get("quantifiable_metrics") or []
    if quantifiable_metrics:
        metric_rows = []
        for category_obj in quantifiable_metrics:
            if not isinstance(category_obj, dict):
                continue
            category = category_obj.get("category") or "Metrics"
            metrics = category_obj.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                metric_rows.append(
                    {
                        "Category": category,
                        "Metric": humanize_label(name),
                        "Value": value,
                    }
                )
        if metric_rows:
            with st.expander("Detailed counts", expanded=False):
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    rec = pj.get("recommended_actions") or []
    if rec:
        st.markdown("### Recommended Next Steps")
        for r in rec:
            st.write(f"- {r}")

    if pj.get("manual_review_required"):
        reason = pj.get("manual_review_reason") or "This conversation needs a closer human check."
        st.warning(f"Human review recommended: {reason}")

    with st.expander("Technical details"):
        visible_cm = {
            k: v for k, v in cm.items()
            if k not in {"agent_messages", "agent_messages_evaluated"}
        }
        st.json(visible_cm, expanded=False)


def conversation_filters(conv_df: pd.DataFrame, key_prefix: str = "conv_filters") -> dict:
    """Render filter widgets and return the active filter values."""
    if conv_df.empty:
        return {}

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            classifications = sorted(
                [c for c in conv_df.get("final_classification", pd.Series(dtype=str)).dropna().unique()]
            )
            sel_class = st.multiselect(
                "Overall result",
                classifications,
                default=[],
                key=f"{key_prefix}_classification",
            )
            handled = sorted([h for h in conv_df.get("handled_status", pd.Series(dtype=str)).dropna().unique()])
            sel_handled = st.multiselect(
                "Outcome",
                handled,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_handled",
            )
            severity = sorted([s for s in conv_df.get("cx_issue_severity", pd.Series(dtype=str)).dropna().unique()])
            sel_severity = st.multiselect(
                "Journey quality",
                severity,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_severity",
            )
        with c2:
            subtypes = sorted(
                [
                    s for s in conv_df.get("unhandled_resolution_subtype", pd.Series(dtype=str)).dropna().unique()
                    if str(s).strip().lower() != "not_applicable"
                ]
            )
            sel_subtype = st.multiselect(
                "Unresolved status",
                subtypes,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_subtype",
            )
            frustration_levels = sorted(
                [f for f in conv_df.get("max_frustration_level", pd.Series(dtype=str)).dropna().unique()]
            )
            sel_frustration = st.multiselect(
                "Highest frustration level",
                frustration_levels,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_frustration",
            )
            origins = sorted([o for o in conv_df.get("main_issue_origin", pd.Series(dtype=str)).dropna().unique()])
            sel_origin = st.multiselect(
                "Where the main problem came from",
                origins,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_origin",
            )
            issue_types = sorted([t for t in conv_df.get("main_issue_type", pd.Series(dtype=str)).dropna().unique()])
            sel_issue_type = st.multiselect(
                "Main problem type",
                issue_types,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_issue_type",
            )
        with c3:
            mr_options = ["Any", "Only manual review", "Only no manual review"]
            sel_mr = st.selectbox("Human review", mr_options, index=0, key=f"{key_prefix}_manual_review")
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
                        key=f"{key_prefix}_date_range",
                    )

    return {
        "classification": sel_class,
        "handled_status": sel_handled,
        "cx_issue_severity": sel_severity,
        "unhandled_resolution_subtype": sel_subtype,
        "max_frustration_level": sel_frustration,
        "main_issue_origin": sel_origin,
        "main_issue_type": sel_issue_type,
        "manual_review": sel_mr,
        "date_range": date_range,
    }


def apply_conversation_filters(conv_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply active filters using AND semantics across all filter groups."""
    if conv_df.empty or not filters:
        return conv_df
    mask = pd.Series(True, index=conv_df.index)

    def in_filter(col: str, key: str) -> None:
        nonlocal mask
        sel = filters.get(key) or []
        if sel and col in conv_df.columns:
            mask &= conv_df[col].isin(sel)

    in_filter("final_classification", "classification")
    in_filter("handled_status", "handled_status")
    in_filter("cx_issue_severity", "cx_issue_severity")
    in_filter("unhandled_resolution_subtype", "unhandled_resolution_subtype")
    in_filter("max_frustration_level", "max_frustration_level")
    in_filter("main_issue_origin", "main_issue_origin")
    in_filter("main_issue_type", "main_issue_type")
    mr = filters.get("manual_review")
    if mr == "Only manual review" and "manual_review_required" in conv_df.columns:
        mask &= conv_df["manual_review_required"].fillna(False).astype(bool)
    elif mr == "Only no manual review" and "manual_review_required" in conv_df.columns:
        mask &= ~conv_df["manual_review_required"].fillna(False).astype(bool)

    dr = filters.get("date_range")
    if dr and "conversation_start_date" in conv_df.columns:
        try:
            start, end = dr
            parsed = pd.to_datetime(conv_df["conversation_start_date"], errors="coerce")
            mask &= (parsed.dt.date >= start) & (parsed.dt.date <= end)
        except Exception:
            pass

    return conv_df[mask].copy()
