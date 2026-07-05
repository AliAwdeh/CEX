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

_SEVERITY_STYLES = {
    "green": {
        "bg": "#0f3f2e",
        "border": "#22c55e",
        "text": "#ecfdf5",
        "label": "Clean",
    },
    "yellow": {
        "bg": "#4a3410",
        "border": "#f59e0b",
        "text": "#fffbeb",
        "label": "Watch",
    },
    "red": {
        "bg": "#4c1111",
        "border": "#ef4444",
        "text": "#fef2f2",
        "label": "Issue",
    },
    "gray": {
        "bg": "#1f2937",
        "border": "#64748b",
        "text": "#f8fafc",
        "label": "Review",
    },
}


def _message_severity(message_result: dict | None) -> str:
    """Return green/yellow/red/gray for the side review control."""
    if not message_result:
        return "gray"
    if message_result.get("parse_status") != "ok":
        return "red"
    pj = message_result.get("parsed_json") or {}
    effect = pj.get("message_level_effect")
    frustration = pj.get("frustration_level_after_message")
    change = pj.get("frustration_change")
    issue_type = pj.get("issue_type") or "none"
    issue_origin = pj.get("issue_origin") or "none"
    has_issue = effect in {"minor_issue", "major_issue"} or issue_type != "none" or issue_origin != "none"
    if effect == "major_issue" or frustration in {"high", "cancellation_risk"}:
        return "red"
    if change == "created" and has_issue:
        return "red"
    if effect == "minor_issue" or (has_issue and frustration in {"medium", "low"}) or change == "increased":
        return "yellow"
    return "green"


def _severity_badge(severity: str, text: str | None = None) -> str:
    style = _SEVERITY_STYLES.get(severity, _SEVERITY_STYLES["gray"])
    label = text or style["label"]
    return (
        f"<div style=\"display:inline-flex;align-items:center;justify-content:center;"
        f"min-width:92px;padding:6px 10px;border-radius:8px;background:{style['bg']};"
        f"border:1px solid {style['border']};color:{style['text']};font-weight:800;"
        f"font-size:0.78rem;letter-spacing:0;text-align:center;\">"
        f"{html.escape(label)}</div>"
    )


def _badge(label: str, value: str, color: str) -> str:
    """Render a colored pill label/value badge."""
    safe_value = html.escape(str(value))
    safe_label = html.escape(str(label))
    palette = {
        "#16a34a": ("#0f3f2e", "#ecfdf5"),
        "#65a30d": ("#365314", "#f7fee7"),
        "#d97706": ("#4a3410", "#fffbeb"),
        "#dc2626": ("#4c1111", "#fef2f2"),
        "#b91c1c": ("#4c1111", "#fef2f2"),
        "#7f1d1d": ("#4c1111", "#fef2f2"),
        "#0891b2": ("#164e63", "#ecfeff"),
        "#475569": ("#1f2937", "#f8fafc"),
        "#6b7280": ("#1f2937", "#f8fafc"),
    }
    bg, fg = palette.get(color, ("#f8fafc", "#111827"))
    return (
        f"<span style=\"display:inline-block;padding:2px 8px;margin:2px 6px 2px 0;"
        f"border-radius:9999px;background:{bg};color:{fg};"
        f"border:1px solid {color};font-size:0.78rem;font-weight:700;\">"
        f"{safe_label}: {safe_value}</span>"
    )


def _format_score(value: Any, maximum: int) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if num.is_integer():
        return f"{int(num)} / {maximum}"
    return f"{num:.1f} / {maximum}"


def _score_color(value: Any, rating: str | None = None) -> str:
    rating_key = str(rating or "").strip().lower()
    if rating_key == "excellent":
        return "#16a34a"
    if rating_key == "good":
        return "#65a30d"
    if rating_key == "fair":
        return "#d97706"
    if rating_key == "poor":
        return "#dc2626"
    if rating_key == "critical":
        return "#7f1d1d"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "#6b7280"
    if num >= 90:
        return "#16a34a"
    if num >= 75:
        return "#65a30d"
    if num >= 60:
        return "#d97706"
    if num >= 40:
        return "#dc2626"
    return "#7f1d1d"


def _has_real_conversation_score(score: dict) -> bool:
    values = [
        score.get("resolution_score"),
        score.get("context_understanding_score"),
        score.get("customer_effort_score"),
        score.get("trust_frustration_risk_score", score.get("frustration_risk_score")),
        score.get("raw_total_score"),
        score.get("final_score"),
    ]
    if not any(v not in (None, "", "none", "None") for v in values):
        return False
    all_zero = True
    for value in values:
        if value in (None, "", "none", "None"):
            continue
        try:
            if float(value) != 0.0:
                all_zero = False
                break
        except (TypeError, ValueError):
            all_zero = False
            break
    return not (all_zero and not str(score.get("score_explanation", "") or "").strip())


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
        source_id = m.get("source_conversation_id")
        if source_id:
            meta_bits.append(f"Source {html.escape(str(source_id))}")
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
        f"<div style=\"background:#111827;border-left:4px solid {color};"
        f"padding:6px 10px;border-radius:4px;color:#f8fafc;font-weight:700;"
        f"font-size:0.82rem;margin-top:4px;\">"
        f"{html.escape(text)}</div>"
    )


def _bubble_html(msg: dict, display_role: str, *, align: str, accent: str | None = None) -> str:
    """Return a high-contrast message bubble."""
    role = (msg.get("sender_role") or "unknown").lower()
    idx = msg.get("message_index")
    when = str(msg.get("message_time") or "")
    text = str(msg.get("message_text", "") or "")
    palette = {
        "customer": {
            "bg": "#1e293b",
            "border": "#64748b",
            "text": "#f8fafc",
            "meta": "#cbd5e1",
        },
        "agent": {
            "bg": "#0f3a5f",
            "border": "#38bdf8",
            "text": "#f8fafc",
            "meta": "#bae6fd",
        },
        "unknown": {
            "bg": "#312e81",
            "border": "#818cf8",
            "text": "#f8fafc",
            "meta": "#c7d2fe",
        },
    }.get(role, {
        "bg": "#f1f5f9",
        "border": "#cbd5e1",
        "text": "#1f2937",
        "meta": "#475569",
    })
    bits = [html.escape(display_role)]
    if idx is not None:
        bits.append(f"#{html.escape(str(idx))}")
    if when:
        bits.append(html.escape(when))
    radius = "16px 16px 16px 4px" if align == "left" else "16px 16px 4px 16px"
    accent_style = ""
    if accent:
        side = "border-left" if align == "left" else "border-right"
        accent_style = f"{side}:4px solid {accent};"
    return (
        f"<div style=\"display:flex;justify-content:{'flex-start' if align == 'left' else 'flex-end'};"
        f"margin:4px 0;\">"
        f"<div style=\"max-width:980px;width:fit-content;padding:12px 15px;border-radius:{radius};"
        f"background:{palette['bg']};border:1px solid {palette['border']};color:{palette['text']} !important;"
        f"{accent_style}box-shadow:0 10px 22px rgba(0,0,0,0.22);\">"
        f"<div style=\"font-size:0.78rem;font-weight:900;color:{palette['meta']};margin-bottom:6px;\">"
        f"{' · '.join(bits)}</div>"
        f"<div style=\"white-space:pre-wrap;word-break:break-word;line-height:1.5;font-size:0.98rem;color:{palette['text']} !important;\">"
        f"{html.escape(text) if text else '<em>(empty message)</em>'}</div>"
        f"</div></div>"
    )


def _source_divider(source_id: Any) -> str:
    label = f"Source conversation {source_id}" if source_id else "Source conversation unknown"
    return (
        "<div style=\"display:flex;align-items:center;gap:12px;margin:18px 0 12px 0;\">"
        "<div style=\"height:1px;background:#475569;flex:1;\"></div>"
        "<div style=\"background:#111827;color:#e5e7eb;border:1px solid #475569;"
        "border-radius:999px;padding:4px 12px;font-size:0.78rem;font-weight:800;\">"
        f"{html.escape(str(label))}</div>"
        "<div style=\"height:1px;background:#475569;flex:1;\"></div>"
        "</div>"
    )


def _render_message_run_details(message_result: dict) -> None:
    """Render the full hidden message run detail panel."""
    pj = message_result.get("parsed_json") or {}
    status = message_result.get("parse_status", "ok")
    idx = message_result.get("message_index")
    source_id = message_result.get("source_conversation_id")

    metric_row(
        [
            ("Status", status, None),
            ("Appended index", idx if idx is not None else "—", None),
            ("Source conversation", source_id or "—", None),
        ]
    )

    if status != "ok":
        st.warning(message_result.get("error_message") or "Message evaluation failed.")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Effect**")
        st.write(humanize_label(pj.get("message_level_effect")) or "—")
        st.markdown("**Frustration after message**")
        st.write(humanize_label(pj.get("frustration_level_after_message")) or "—")
        st.markdown("**Issue type / origin**")
        st.write(
            f"{humanize_label(pj.get('issue_type')) or '—'} / "
            f"{humanize_label(pj.get('issue_origin')) or '—'}"
        )
    with cols[1]:
        st.markdown("**Evidence**")
        st.write(pj.get("evidence") or "—")
        st.markdown("**Business impact**")
        st.write(pj.get("business_impact") or "—")
        st.markdown("**Recommended fix**")
        st.write(pj.get("recommended_fix") or "—")

    with st.expander("Message run JSON", expanded=False):
        st.json(message_result, expanded=False)


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
    """Render a centered transcript with hidden message-run details.

    Header label rules:
    - RAW_SENDER_ROLE=System shows Broadcast.
    - RAW_SENDER_ROLE=Bot shows Assistant.
    - RAW_SENDER_ROLE=Agent shows MESSAGE_AGENT_FULL_NAME when available.
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

    st.markdown(
        """
        <style>
          div[data-testid="stExpander"] summary {
            font-weight: 800;
          }
          div[data-testid="stExpander"] {
            border-color: #334155;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    previous_source = object()
    for msg in transcript:
        role = (msg.get("sender_role") or "unknown").lower()
        raw_role = msg.get("raw_sender_role")
        raw_role_norm = str(raw_role).strip().lower() if raw_role else ""
        idx = msg.get("message_index")
        source_id = msg.get("source_conversation_id")

        if source_id != previous_source:
            st.markdown(_source_divider(source_id), unsafe_allow_html=True)
            previous_source = source_id

        if role == "customer":
            if raw_role_norm == "system":
                display_role = "Broadcast"
            elif raw_role_norm == "bot":
                display_role = "Assistant"
            elif raw_role_norm == "agent":
                display_role = str(msg.get("agent_full_name") or "").strip() or "Agent"
            else:
                display_role = str(raw_role) if raw_role else "Customer"
            align = "left"
        elif role == "agent":
            if raw_role_norm == "system":
                display_role = "Broadcast"
            elif raw_role_norm == "bot":
                display_role = "Assistant"
            elif raw_role_norm == "agent":
                display_role = str(msg.get("agent_full_name") or "").strip() or "Agent"
            else:
                display_role = str(msg.get("agent_full_name") or "").strip() or "Assistant"
            align = "right"
        else:
            if raw_role_norm == "system":
                display_role = "Broadcast"
            elif raw_role_norm == "bot":
                display_role = "Assistant"
            elif raw_role_norm == "agent":
                display_role = str(msg.get("agent_full_name") or "").strip() or "Agent"
            else:
                display_role = str(raw_role) if raw_role else "Unknown"
            align = "left"

        eval_record = eval_by_idx.get(idx)
        severity = _message_severity(eval_record)
        style = _SEVERITY_STYLES.get(severity, _SEVERITY_STYLES["gray"])
        accent = style["border"] if eval_record else None

        bubble_col, review_col = st.columns([10.5, 0.5], vertical_alignment="center")

        with bubble_col:
            st.markdown(_bubble_html(msg, display_role, align=align, accent=accent), unsafe_allow_html=True)

        with review_col:
            if eval_record:
                with st.popover("i", use_container_width=True):
                    st.markdown(_severity_badge(severity), unsafe_allow_html=True)
                    _render_message_run_details(eval_record)
            else:
                st.markdown(
                    "<div style=\"width:1px;height:1px;\"></div>",
                    unsafe_allow_html=True,
                )


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
        source_id = message_result.get("source_conversation_id")
        source_text = f" • Source conversation: {source_id}" if source_id else ""
        st.caption(f"Appended message index: {idx}{source_text}")

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


def render_conversation_summary_card(
    conv_result: dict,
    show_details: bool = True,
) -> None:
    """Render a journey's overall result, with optional supporting details."""
    pj = conv_result.get("parsed_json") or {}
    md = conv_result.get("conversation_metadata") or {}
    cm = conv_result.get("computed_metadata") or {}

    handled = pj.get("handled_status", "unknown")
    experience = pj.get("customer_experience", "unknown")
    experience_display = humanize_label(experience)
    frustration_detected = bool(pj.get("frustration_detected", False))
    frustration_origin = pj.get("frustration_origin", "none")
    frustration_timing = pj.get("frustration_timing", "none")
    subtype = pj.get("unhandled_resolution_subtype", "")
    sentiment = pj.get("final_customer_sentiment", "unknown")
    max_fl = pj.get("max_frustration_level", "none")
    main = pj.get("main_issue") or {}
    score = pj.get("conversation_score") or {}
    if not isinstance(score, dict):
        score = {}
    if score and not _has_real_conversation_score(score):
        score = {}

    color = "#16a34a" if handled == "handled" and experience == "good" else "#dc2626" if handled == "unhandled" and experience == "bad" else "#d97706"
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
          <div style="font-size:1.25rem; font-weight:700; color:{color};">{html.escape(humanize_label(handled))} / {html.escape(experience_display)}</div>
          {unresolved_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not show_details:
        return

    badges = []
    badges.append(_badge("Outcome", humanize_label(handled), "#16a34a" if handled == "handled" else "#dc2626"))
    badges.append(_badge("Customer experience", experience_display or "n/a", "#d97706" if experience == "bad" else "#16a34a"))
    badges.append(
        _badge("Customer frustration", "yes" if frustration_detected else "no", "#b91c1c" if frustration_detected else "#475569")
    )
    if frustration_detected:
        badges.append(_badge("When frustration appeared", humanize_label(frustration_timing) or "n/a", "#475569"))
        badges.append(_badge("Frustration origin", humanize_label(frustration_origin) or "n/a", "#475569"))
    if not show_unresolved_header_badge:
        badges.append(_badge("Unresolved status", subtype_display, "#475569"))
    badges.append(_badge("Customer feeling at end", humanize_label(sentiment), _SENTIMENT_COLORS.get(sentiment, "#6b7280")))
    badges.append(_badge("Highest frustration level", humanize_label(max_fl), _FRUSTRATION_COLORS.get(max_fl, "#6b7280")))
    if pj.get("manual_review_required"):
        badges.append(_badge("Needs human review", "yes", "#dc2626"))
    badges.append(_badge("Confidence", pj.get("confidence", "—"), "#475569"))
    st.markdown("".join(badges), unsafe_allow_html=True)

    if score:
        final_score = score.get("final_score")
        raw_total = score.get("raw_total_score")
        rating = score.get("score_rating") or "-"
        score_color = _score_color(final_score, rating)
        st.markdown(
            f"""
            <div style="margin:14px 0 10px 0; padding:14px 18px; border-radius:10px;
                        background:#0f172a; border:1px solid {score_color};">
              <div style="font-size:0.82rem; color:#cbd5e1; font-weight:800; text-transform:uppercase; letter-spacing:0.04em;">
                Conversation score
              </div>
              <div style="display:flex; align-items:baseline; gap:10px; margin-top:4px;">
                <span style="font-size:2.1rem; font-weight:900; color:{score_color};">{html.escape(_format_score(final_score, 100))}</span>
                <span style="font-size:1rem; color:#e5e7eb; font-weight:800;">{html.escape(str(rating))}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_row(
            [
                ("Resolution", _format_score(score.get("resolution_score"), 20), None),
                ("Context & Understanding", _format_score(score.get("context_understanding_score"), 20), None),
                ("Customer Effort", _format_score(score.get("customer_effort_score"), 20), None),
                (
                    "Frustration & Risk",
                    _format_score(
                        score.get("trust_frustration_risk_score", score.get("frustration_risk_score")),
                        40,
                    ),
                    None,
                ),
                ("Raw total", _format_score(raw_total, 100), None),
            ]
        )
        if score.get("score_explanation"):
            st.markdown("**Why this score was assigned**")
            st.write(score.get("score_explanation"))

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown("**ID**")
        st.write(conv_result.get("conversation_id", ""))
        st.markdown("**Customer**")
        st.write(md.get("customer_name") or "—")
        st.markdown("**Phone**")
        st.write(md.get("customer_phone") or "—")
    with cols[1]:
        st.markdown("**Customer journey ID**")
        st.write(conv_result.get("conversation_id", ""))
        st.markdown("**Source conversations**")
        st.write(md.get("source_conversation_count") or "—")
        st.markdown("**Source conversation IDs**")
        st.write(md.get("source_conversation_ids") or "—")
    with cols[2]:
        st.markdown("**Started**")
        st.write(md.get("conversation_start_date") or "—")
        st.markdown("**Ended**")
        st.write(md.get("conversation_end_date") or "—")

    objective_cols = st.columns([1, 2])
    with objective_cols[0]:
        st.markdown("**Customer goal type**")
        st.write(humanize_label(pj.get("customer_objective_type")) or "—")
    with objective_cols[1]:
        st.markdown("**Customer primary objective**")
        st.write(pj.get("customer_primary_objective") or "—")

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

    classification_reason = pj.get("classification_reason")
    if classification_reason:
        st.markdown("### Decision Reasoning")
        st.markdown("**Classification reason**")
        st.write(classification_reason)

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

    rec = pj.get("recommended_actions") or []
    if rec:
        st.markdown("### Recommended Next Steps")
        for r in rec:
            st.write(f"- {r}")

    if pj.get("manual_review_required"):
        reason = pj.get("manual_review_reason") or "This conversation needs a closer human check."
        st.warning(f"Human review recommended: {reason}")

    with st.expander("Journey run details and JSON", expanded=False):
        visible_cm = {
            k: v for k, v in cm.items()
            if k not in {"agent_messages", "agent_messages_evaluated"}
        }
        detail_cols = st.columns(3)
        with detail_cols[0]:
            st.markdown("**Computed totals**")
            st.json(visible_cm, expanded=False)
        with detail_cols[1]:
            st.markdown("**Journey metadata**")
            st.json(md, expanded=False)
        with detail_cols[2]:
            st.markdown("**Run status**")
            st.json(
                {
                    "parse_status": conv_result.get("parse_status"),
                    "error_message": conv_result.get("error_message"),
                    "evaluation_target_role": conv_result.get("evaluation_target_role"),
                },
                expanded=False,
            )
        st.markdown("**Conversation-level parsed JSON**")
        st.json(pj, expanded=False)
        with st.expander("Full journey result object", expanded=False):
            st.json(conv_result, expanded=False)


def conversation_filters(
    conv_df: pd.DataFrame,
    key_prefix: str = "conv_filters",
    include_journey_starter: bool = False,
) -> dict:
    """Render filter widgets and return the active filter values."""
    if conv_df.empty:
        return {}

    sel_journey_starter: list[str] = []
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            handled = sorted([h for h in conv_df.get("handled_status", pd.Series(dtype=str)).dropna().unique()])
            sel_handled = st.multiselect(
                "Outcome",
                handled,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_handled",
            )
            experiences = sorted([s for s in conv_df.get("customer_experience", pd.Series(dtype=str)).dropna().unique()])
            sel_experience = st.multiselect(
                "Customer experience",
                experiences,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_experience",
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
            frustration_origins = sorted([o for o in conv_df.get("frustration_origin", pd.Series(dtype=str)).dropna().unique()])
            sel_frustration_origin = st.multiselect(
                "Frustration origin",
                frustration_origins,
                default=[],
                format_func=humanize_label,
                key=f"{key_prefix}_frustration_origin",
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
            if include_journey_starter:
                starters = [
                    value
                    for value in (
                        conv_df.get("journey_starter", pd.Series(dtype=str))
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .unique()
                    )
                    if value
                ]
                starter_priority = {
                    "consumer": 0,
                    "bot": 1,
                    "system": 2,
                    "agent": 3,
                    "unknown": 99,
                }
                starters = sorted(
                    starters,
                    key=lambda value: (starter_priority.get(value, 50), value),
                )
                sel_journey_starter = st.multiselect(
                    "Journey started by",
                    starters,
                    default=[],
                    format_func=humanize_label,
                    key=f"{key_prefix}_journey_starter",
                    help="The sender type of the first message in the appended customer journey.",
                )
            mr_options = ["Any", "Only manual review", "Only no manual review"]
            sel_mr = st.selectbox("Human review", mr_options, index=0, key=f"{key_prefix}_manual_review")
            show_broadcast_only = st.checkbox(
                "Only broadcast-only issue journeys",
                value=False,
                key=f"{key_prefix}_show_broadcast_only_red",
                help=(
                    "When selected, show only journeys where the detected red issue "
                    "came exclusively from a system/broadcast message."
                ),
            )
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
        "handled_status": sel_handled,
        "customer_experience": sel_experience,
        "unhandled_resolution_subtype": sel_subtype,
        "max_frustration_level": sel_frustration,
        "frustration_origin": sel_frustration_origin,
        "main_issue_origin": sel_origin,
        "main_issue_type": sel_issue_type,
        "journey_starter": sel_journey_starter,
        "manual_review": sel_mr,
        "show_broadcast_only_red": show_broadcast_only,
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

    in_filter("handled_status", "handled_status")
    in_filter("customer_experience", "customer_experience")
    in_filter("unhandled_resolution_subtype", "unhandled_resolution_subtype")
    in_filter("max_frustration_level", "max_frustration_level")
    in_filter("frustration_origin", "frustration_origin")
    in_filter("main_issue_origin", "main_issue_origin")
    in_filter("main_issue_type", "main_issue_type")
    in_filter("journey_starter", "journey_starter")
    mr = filters.get("manual_review")
    manual_review_series = None
    if "manual_review_required" in conv_df.columns:
        manual_review_series = conv_df["manual_review_required"].map(
            lambda value: str(value if value is not None else False).strip().lower() in {"true", "1", "yes", "y"}
        )
    if mr == "Only manual review" and "manual_review_required" in conv_df.columns:
        mask &= manual_review_series
    elif mr == "Only no manual review" and "manual_review_required" in conv_df.columns:
        mask &= ~manual_review_series

    show_broadcast_only = bool(filters.get("show_broadcast_only_red"))
    if "broadcast_only_red_issue" in conv_df.columns:
        broadcast_only_series = conv_df["broadcast_only_red_issue"].map(
            lambda value: str(value if value is not None else False).strip().lower() in {"true", "1", "yes", "y"}
        )
        mask &= broadcast_only_series if show_broadcast_only else ~broadcast_only_series

    dr = filters.get("date_range")
    if dr and "conversation_start_date" in conv_df.columns:
        try:
            start, end = dr
            parsed = pd.to_datetime(conv_df["conversation_start_date"], errors="coerce")
            mask &= (parsed.dt.date >= start) & (parsed.dt.date <= end)
        except Exception:
            pass

    return conv_df[mask].copy()
