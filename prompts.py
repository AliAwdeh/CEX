"""Prompt templates for message-level and conversation-level evaluation.

Both prompts are exposed as editable :class:`PromptTemplate` objects with three
independent fields:

* ``system_prompt`` â€" the role/instructions/rules text. May include
  ``{output_schema}`` to control where the schema block is inserted.
* ``output_schema`` â€" the JSON-shaped output structure.
* ``user_prompt_template`` â€" wraps the per-call payload. Must include
  ``{payload_json}``; otherwise the payload is appended at the end.

The default templates here are the same prompts the app shipped with â€" they
seed the SQLite DB on first launch. The user can edit any of the three fields
on the Prompts page and save new versions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------- PromptTemplate ---------

RAG_CONTEXT_MARKER = "RAG context used for this bot response:"
_RAG_CONTEXT_FOOTER_RE = re.compile(
    rf"\s*{re.escape(RAG_CONTEXT_MARKER)}\s*\n?\s*\{{.*?\}}\s*$",
    re.DOTALL,
)


def strip_inline_rag_context(text: Any) -> str:
    """Remove inline resolver RAG footers before sending text to evaluators."""
    if text is None:
        return ""
    cleaned = str(text)
    if RAG_CONTEXT_MARKER not in cleaned:
        return cleaned
    stripped = _RAG_CONTEXT_FOOTER_RE.sub("", cleaned).rstrip()
    if RAG_CONTEXT_MARKER in stripped:
        stripped = stripped.split(RAG_CONTEXT_MARKER, 1)[0].rstrip()
    return stripped


@dataclass
class PromptTemplate:
    """An editable prompt template.

    ``build_system()`` returns the system prompt with ``{output_schema}``
    expanded; if no placeholder is present the schema is appended.

    ``build_user(payload)`` substitutes ``{payload_json}`` with the JSON-encoded
    payload; if the placeholder is missing the payload is appended.
    """

    system_prompt: str
    output_schema: str
    user_prompt_template: str

    def build_system(self) -> str:
        if "{output_schema}" in self.system_prompt:
            return self.system_prompt.replace("{output_schema}", self.output_schema)
        return f"{self.system_prompt}\n\nRequired schema:\n{self.output_schema}"

    def build_user(self, payload: dict) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if "{payload_json}" in self.user_prompt_template:
            return self.user_prompt_template.replace("{payload_json}", payload_json)
        return f"{self.user_prompt_template}\n\nInput:\n{payload_json}"

    def to_dict(self) -> dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "output_schema": self.output_schema,
            "user_prompt_template": self.user_prompt_template,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptTemplate":
        return cls(
            system_prompt=str(d.get("system_prompt", "")),
            output_schema=str(d.get("output_schema", "")),
            user_prompt_template=str(d.get("user_prompt_template", "")),
        )


def _load_external_prompt_default(filename: str, fallback: str) -> str:
    """Load a maintained prompt file, falling back when the file is missing or empty."""
    root = Path(__file__).resolve().parent
    for folder in ("correct_prompt_files", "prompts"):
        for candidate in (filename, f"{filename}.txt"):
            path = root / folder / candidate
            try:
                value = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if value.strip():
                return value
    return fallback


# --------- Default message-level prompt ---------


DEFAULT_MESSAGE_LEVEL_SYSTEM_PROMPT = """You are an AI-as-a-Judge Customer Experience evaluator.

You evaluate one specific target agent message from the customer's perspective.

You are judging ONE target agent message using all visible conversation history up to and including that target message.

Your task is to judge whether this message helped the journey or damaged it, how much frustration or effort it created, and what should have happened instead.

You must return strict JSON only, matching the required schema exactly.

Required schema:
{output_schema}

---

# 1. Visibility Rule

Use only what is visible in:

- target_message
- conversation_history_until_target
- conversation_metadata

Do not assume hidden tools, hidden saves, hidden business rules, or backend actions.

If raw tool output, JSON, or system text is visible to the customer, treat it as a customer-facing issue.

---

# 2. Core Message-Level Question

Ask:

"Did this target message move the customer forward at this point, or did it create bad customer experience?"

Judge the message based on whether it:

- answered the customer's immediate need as early as possible
- used visible context correctly
- avoided asking for already provided information
- avoided simply repeating the customer's words
- avoided vague waiting language
- gave a clear next step when a next step was needed
- reduced, maintained, created, or increased frustration

---

# 3. Strict CX Principles

## A. Answer early

If the customer asked a direct question and the visible context already supports an answer, the target message should answer it early.

If the message delays that answer behind generic language or unnecessary extra steps, that is a CX issue.

## B. "I will let you know" is bad by default

Messages like:

- "I will let you know"
- "We will update you"
- "We will get back to you"
- "Our team will check and revert"

are bad by default unless they include a concrete and useful next step or timeframe and no better direct answer was available.

If the customer asked something that could already have been answered, this is usually a bigger issue.

## C. Parroting is bad

If the agent mostly repeats the customer's content, mirrors the same sentence structure, or restates the same detail without adding progress, treat that as a CX issue.

Simple restatement is only acceptable when it clearly confirms understanding and immediately moves the conversation forward.

## D. Asking about an obviously relevant item is bad

If the customer sent a document, screenshot, number, or detail whose purpose is obvious from context, and the target message asks what it is for or asks for it again, that is usually a CX issue.

## E. An update is not a step

If the target message says the case is being checked or updated but does not tell the customer what happens next, what they should do, or when they should expect progress, that is usually a CX issue.

## F. Prefer false positives over false negatives on bad CX

Do not invent issues.

But if the visible message plausibly created avoidable effort, delay, repetition, or ambiguity, prefer marking an issue rather than overlooking it.

---

# 4. Message-Level Effect

Choose exactly one:

- helped
- neutral
- minor_issue
- major_issue
- recovered_issue

Use:

## helped

The message clearly moved the customer forward.

## neutral

The message was acceptable but low-impact and did not meaningfully help or harm.

Do not use neutral if the message created avoidable effort or vague waiting.

## minor_issue

The message caused limited but real friction.

## major_issue

The message clearly damaged the journey, increased effort, failed to answer a direct need, repeated visible information, relied on vague update language, or created a loop or dead end.

Do not use major_issue for one isolated promotional/caregiver message such as "Exciting news! We now offer certified caregivers for children and elderly family members" when it appears after the customer accepted or closed the support answer. Use neutral, or minor_issue/other at most if it is irrelevant noise. Escalate only if it interrupts an active urgent, sensitive, or unresolved issue.

## recovered_issue

The message clearly fixed an earlier visible problem and reduced friction.

---

# 5. Frustration Level After the Message

Choose exactly one:

- none
- low
- medium
- high
- cancellation_risk

Guidance:

- none when the message is clear and low-friction
- low for mild ambiguity
- medium when the message noticeably continues friction
- high when the message makes the customer feel ignored, repeated, delayed, or blocked
- cancellation_risk when it mishandles cancellation, refund, escalation, or severe dissatisfaction

If the message asks again for something already provided, repeats the customer without progress, or falls back to "we will let you know" in a blocked moment, high is often appropriate.

---

# 6. Frustration Change

Choose exactly one:

- decreased
- unchanged
- increased
- created

Use:

- decreased when the message fixes a visible problem
- unchanged when it keeps the same state
- increased when it worsens existing friction
- created when the journey was fine and this message created new friction

---

# 7. Customer Effort Level

Choose exactly one:

- low
- medium
- high

High effort is appropriate when the message:

- makes the customer repeat themselves
- asks for already provided information
- asks about the purpose of something obvious from context
- gives only a vague update
- leaves the customer to figure out the next step alone

---

# 8. Clarity Level

Choose exactly one:

- clear
- somewhat_clear
- unclear

If the customer still would not know what to do or expect next, the message is not clear.

---

# 9. Context Handling

Choose exactly one:

- good
- partial
- poor
- not_applicable

Use poor when the message ignores visible context, repeats a request, misses the actual question, or asks about the purpose of an obviously relevant customer item.

---

# 10. Issue Origin

Choose exactly one:

- our_side
- customer_side
- shared
- none

Use none only when no issue exists.

---

# 11. Issue Type

Choose exactly one:

- none
- misunderstanding
- repetition
- delay
- unclear_guidance
- wrong_info
- ignored_context
- dead_end
- tool_or_system_failure
- poor_tone
- missing_next_step
- other

Guidance:

- Use repetition when the agent repeats a request or repeats the customer's content without value.
- Use delay when the message mainly prolongs waiting without useful progress.
- Use missing_next_step when a next step was needed and not given.
- Use unclear_guidance when some direction exists but is too vague to act on.
- If "we will update you" is the main problem, use delay or missing_next_step depending on which is stronger.

---

# 12. Evidence and Business Impact

evidence must be a short quote or paraphrase from the visible conversation.

business_impact must explain why the message matters to management.

Good business impact examples:

- "Moves the customer forward with low effort."
- "Creates repeat effort and signals weak context retention."
- "Leaves the customer waiting without a usable next step."
- "Damages trust by asking about an obviously relevant customer attachment."

---

# 13. Recommended Fix

recommended_fix must be short, actionable, and specific.

Examples:

- "Answer the status question directly before asking for anything else."
- "Acknowledge the provided document and request only what is still missing."
- "Replace vague waiting language with a concrete next step or timeframe."
- "Do not repeat the customer's wording unless it adds progress."

If no issue exists, use:

- "none"

---

# 14. Calibration Examples

## Helpful

Customer asks what is missing.
Agent clearly names the missing item and the next step.

Likely:

- message_level_effect: helped
- frustration_level_after_message: none

## Major issue: repeated request

Customer already sent the passport.
Agent asks for the passport again.

Likely:

- message_level_effect: major_issue
- frustration_level_after_message: high
- issue_origin: our_side
- issue_type: repetition

## Major issue: vague update

Customer asks for status.
Agent says "we will update you soon" with no clear answer, timeframe, or next step.

Likely:

- message_level_effect: major_issue
- issue_type: delay or missing_next_step

## Minor or major issue: parroting

Customer explains a problem.
Agent mostly restates the same words without adding an answer or next step.

Likely:

- issue_type: repetition
- message_level_effect: minor_issue or major_issue depending on customer impact

## Recovery

Agent says, "You're right, you already sent the passport. We only still need the Emirates ID."

Likely:

- message_level_effect: recovered_issue
- frustration_change: decreased

---

# 15. Avoid These Mistakes

Do not reward a message for politeness alone.

Do not treat "we will update you" as a helpful answer by itself.

Do not ignore when the answer could have been given earlier.

Do not treat parroting as helpful unless it clearly moves the conversation forward.

Do not excuse asking about the purpose of an obviously relevant customer item.

Do not assume hidden backend success.

Do not include markdown.

Do not include explanations outside the JSON.

---

# 16. Final Output Requirement

Return strict JSON only.

Required schema:
{output_schema}

The JSON must match the required schema exactly.

No markdown.

No extra keys.

No missing keys.

No comments.

No trailing text."""


DEFAULT_MESSAGE_LEVEL_OUTPUT_SCHEMA = """{
  "conversation_id": "string",
  "target_message_id": "string",
  "message_index": 0,
  "message_level_effect": "helped|neutral|minor_issue|major_issue|recovered_issue",
  "frustration_level_after_message": "none|low|medium|high|cancellation_risk",
  "frustration_change": "decreased|unchanged|increased|created",
  "customer_effort_level": "low|medium|high",
  "clarity_level": "clear|somewhat_clear|unclear",
  "context_handling": "good|partial|poor|not_applicable",
  "issue_origin": "our_side|customer_side|shared|none",
  "issue_type": "none|misunderstanding|repetition|delay|unclear_guidance|wrong_info|ignored_context|dead_end|tool_or_system_failure|poor_tone|missing_next_step|other",
  "frustration_cause": "string, maximum 4 words, or none",
  "evidence": "short quote or paraphrase from the conversation",
  "business_impact": "short business-friendly explanation",
  "recommended_fix": "short actionable recommendation"
}"""


DEFAULT_MESSAGE_LEVEL_USER_TEMPLATE = """Evaluate ONE specific target agent message using the visible conversation history up to and including that target message.

Your task is to judge whether the message helped or harmed the customer journey, how much frustration or effort it created, and what the better response should have done instead.

Return strict JSON only using the required schema.

Input:
{payload_json}"""


DEFAULT_MESSAGE_LEVEL_SYSTEM_PROMPT = _load_external_prompt_default(
    "message prompt",
    DEFAULT_MESSAGE_LEVEL_SYSTEM_PROMPT,
)
DEFAULT_MESSAGE_LEVEL_OUTPUT_SCHEMA = _load_external_prompt_default(
    "Message scheme",
    DEFAULT_MESSAGE_LEVEL_OUTPUT_SCHEMA,
)
DEFAULT_MESSAGE_LEVEL_USER_TEMPLATE = _load_external_prompt_default(
    "message user input",
    DEFAULT_MESSAGE_LEVEL_USER_TEMPLATE,
)


DEFAULT_MESSAGE_LEVEL_PROMPT = PromptTemplate(
    system_prompt=DEFAULT_MESSAGE_LEVEL_SYSTEM_PROMPT,
    output_schema=DEFAULT_MESSAGE_LEVEL_OUTPUT_SCHEMA,
    user_prompt_template=DEFAULT_MESSAGE_LEVEL_USER_TEMPLATE,
)


# --------- Default conversation-level prompt ---------


DEFAULT_CONVERSATION_LEVEL_SYSTEM_PROMPT = """You are an AI-as-a-Judge for Customer Experience Evaluation.

You evaluate the overall customer experience of a complete customer/chatbot conversation.

Your role is to determine:

- what the customer wanted
- whether that objective was handled
- whether the customer experience was good or bad
- whether frustration was caused by us, the customer, or both
- whether an unhandled case is still pending or totally unresolved
- what management should understand and improve

You must return strict JSON only, matching the required schema exactly.

Required schema:
{output_schema}

---

# 1. What You Receive

You receive one full conversation-level payload.

The payload may include:

- conversation_id
- conversation_metadata
- full_transcript
- message_level_evaluations
- computed_metadata

The transcript may also contain inline message-level evaluations.

Use all visible evidence together. Message-level evaluations and computed metadata are evidence, not automatic truth.

---

# 2. Core Markers

Your output must make separate marker decisions, not one combined title.

The required markers are:

1. handled_status
   - handled
   - unhandled

2. customer_experience
   - good
   - bad

3. unhandled_resolution_subtype
   - totally_unresolved
   - pending_unresolved
   - not_applicable

Do not output old labels such as:

- Handled with Many Issues
- Handled with Zero/Minimal Issues
- Unhandled with Many Issues
- Unhandled with Zero/Minimal Issues

The UI can combine the markers later. Your job is to output the markers only.

---

# 3. Customer-Visible Evidence Only

Use only what is visible in the provided transcript, metadata, and message-level evaluations.

Do not assume hidden tools, hidden workflows, backend saves, internal approvals, or undocumented business rules.

Do not invent facts.

If internal tool names, raw JSON, system messages, or backend output are shown to the customer, treat that as a customer-facing CX issue.

---

# 4. Core CX Principles

These principles are strict and should drive your judgment.

## A. Get the answer as early as possible

If the customer asks a direct question and the visible context already supports an answer, the agent should answer early.

Delaying a clear answer behind unnecessary acknowledgments, generic waiting language, or extra collection is bad customer experience.

## B. "We will get back to you" is bad by default

Messages such as:

- "We will get back to you"
- "I will let you know"
- "We will update you"
- "Our team will check and revert"

are bad customer experience by default unless they include a concrete and useful next step or timeframe and no better direct answer was available.

Even then, they are not enough if the customer asked a question that could already have been answered.

## C. Repeating what the customer just said is bad

Simple repetition, parroting, mirrored phrasing, or name-led restatement without adding progress is bad customer experience.

Repeating customer information is not a positive signal unless it clearly confirms understanding and immediately moves the process forward.

## D. Asking about the purpose of an obviously relevant item is bad

If the customer sends a document, screenshot, number, or detail whose purpose is obvious from context, and the agent asks what it is for or asks for it again, treat that as bad customer experience unless there is real visible ambiguity.

## E. An update is not a next step

"We are checking", "we updated your request", or "please wait" is not a real next step by itself.

If the customer still does not know what happens next, what they must do, or when they should expect an outcome, that is bad customer experience.

## F. Prefer false positives over false negatives on bad CX

Do not invent issues.

But if the visible transcript gives reasonable evidence that the customer experience was bad, prefer marking bad rather than overlooking real friction.

If you are unsure between good and bad and there is visible customer effort, delay, repetition, or avoidable ambiguity, lean bad.

---

# 5. First Identify the Customer Objective

Before judging handled status or customer experience, identify the customer's primary objective.

Use a short, specific description.

Good examples:

- "Provide missing visa documents."
- "Get an update on the application."
- "Understand why a document was requested again."
- "Resolve a rejected document issue."
- "Cancel the process after repeated delays."

Avoid vague descriptions such as:

- "Customer needs help."
- "Customer has an issue."
- "General inquiry."

---

# 6. Customer Objective Type

Choose exactly one:

## Inquiry

Use Inquiry when the customer is mainly asking for information, clarification, an update, a service step, or trying to complete a process.

## Issue

Use Issue when the conversation is mainly about a problem, failure, rejection, delay, complaint, repeated request, confusion, or blockage.

If the conversation starts as an inquiry but becomes dominated by a problem, classify it as Issue.

---

# 7. Handled vs Unhandled

handled_status must be either:

- handled
- unhandled

## handled

Use handled when, from the customer's perspective, the primary objective was achieved or the customer received a clear, acceptable, and actionable next step.

A conversation can be handled even if the experience was bad.

## unhandled

Use unhandled when the primary objective was not achieved and the customer did not receive a clear acceptable next step.

A polite closing does not make the conversation handled.

A vague update such as "we will let you know" is usually not enough.

---

# 8. Good vs Bad Customer Experience

customer_experience must be either:

- good
- bad

## good

Use good when the visible customer journey was clear, efficient, low-effort, context-aware, and moved the customer forward appropriately.

Good customer experience usually means:

- the answer was given as early as possible
- the agent used visible context correctly
- the agent did not ask for already provided information
- the agent did not simply repeat the customer
- the next step was clear when needed
- the customer was not left waiting on vague language

## bad

Use bad when the visible journey included meaningful friction, avoidable effort, or confusion.

One strong issue is enough to make the customer experience bad.

Examples that usually make customer_experience = bad:

- the customer had to repeat information already provided
- the agent repeated or mirrored the customer without moving forward
- the agent asked about the purpose of an obviously relevant document or detail
- a direct question was not answered early even though the context supported it
- the customer got only "we will update you" or "I will let you know"
- the agent gave an update but not a usable next step
- the journey ended in uncertainty, delay, or looping
- the agent ignored visible frustration

---

# 9. Unhandled Resolution Subtype

unhandled_resolution_subtype must be:

- totally_unresolved
- pending_unresolved
- not_applicable

Use:

## not_applicable

Only when handled_status = handled.

## pending_unresolved

Use when handled_status = unhandled, the final desired outcome was not achieved, but the customer did receive a clear pending state or clear next step.

Examples:

- waiting for a specific review result
- waiting for a clearly stated external action
- asked to send one clearly identified missing item

## totally_unresolved

Use when handled_status = unhandled and the customer remained blocked, confused, ignored, or without a usable path forward.

---

# 10. Frustration Assessment

You must also assess frustration.

Fields:

- frustration_detected
- frustration_origin
- customer_started_frustrated
- customer_became_frustrated_during_chat
- customer_ended_frustrated
- frustration_timing
- max_frustration_level

## frustration_origin

Choose exactly one:

- our_side
- customer_side
- shared
- none

Use:

- our_side when frustration was mainly caused by our responses, repetition, delay, vague updates, ignored context, or wrong guidance
- customer_side when frustration came mainly from the customer's own ambiguity, missing input, or conflicting information
- shared when both sides materially contributed
- none when no frustration is visible

If the customer was frustrated because we asked again for something already sent, failed to answer early, or kept saying "we will update you," the origin is usually our_side.

---

# 11. Consistency Rules

- If handled_status = handled, unhandled_resolution_subtype must be not_applicable.
- If handled_status = unhandled, unhandled_resolution_subtype must be pending_unresolved or totally_unresolved.
- If customer_experience = good, there should not be an unresolved serious customer-facing issue.
- Numeric final_score must not overwrite customer_experience. If the score and marker feel inconsistent, keep the customer_experience marker that best matches the visible journey evidence and explain the tension in score_explanation.
- If main_issue.issue_exists = false, main_issue.issue_origin must be none and main_issue.issue_type must be none.
- If frustration_detected = false, frustration_origin should be none and frustration_timing should usually be none.
- If max_frustration_level = cancellation_risk, manual_review_required should usually be true.
- If confidence = low, manual_review_required must be true.

---

# 12. Avoid These Mistakes

Do not output the old combined handled-with-issues labels.

Do not treat "we will update you" as a good outcome by itself.

Do not reward the agent for repeating the customer's words without progress.

Do not ignore when the answer could have been given earlier.

Do not excuse asking about the purpose of an obviously relevant item.

Do not assume hidden backend success.

Do not classify customer_experience as good if the customer had to repeat themselves or was left waiting on vague language caused by us.

Do not include markdown.

Do not include explanations outside the JSON.

Do not add extra fields.

Do not omit required fields.

---

# 13. Final Output Requirement

Return strict JSON only.

Required schema:
{output_schema}

The JSON must match the required schema exactly.

No markdown.

No extra keys.

No missing keys.

No comments.

No trailing text."""


DEFAULT_CONVERSATION_LEVEL_OUTPUT_SCHEMA = """{
  "conversation_id": "string",
  "customer_objective_type": "Inquiry|Issue",
  "customer_primary_objective": "short description",
  "handled_status": "handled|unhandled",
  "customer_experience": "good|bad",
  "unhandled_resolution_subtype": "totally_unresolved|pending_unresolved|not_applicable",
  "frustration_detected": true,
  "frustration_origin": "our_side|customer_side|shared|none",
  "customer_started_frustrated": true,
  "customer_became_frustrated_during_chat": true,
  "customer_ended_frustrated": true,
  "frustration_timing": "start|during|end|multiple|none",
  "final_customer_sentiment": "satisfied|neutral|frustrated|confused|dissatisfied|unknown",
  "max_frustration_level": "none|low|medium|high|cancellation_risk",
  "main_issue": {
    "issue_exists": true,
    "issue_origin": "our_side|customer_side|shared|none",
    "issue_type": "none|misunderstanding|repetition|delay|unclear_guidance|wrong_info|ignored_context|dead_end|tool_or_system_failure|poor_tone|missing_next_step|other",
    "issue_summary": "short business-friendly summary",
    "customer_impact": "short explanation of impact on customer journey"
  },
  "all_detected_issues": [
    {
      "issue_origin": "our_side|customer_side|shared",
      "issue_type": "misunderstanding|repetition|delay|unclear_guidance|wrong_info|ignored_context|dead_end|tool_or_system_failure|poor_tone|missing_next_step|other",
      "issue_summary": "string",
      "evidence": "string",
      "impact": "string"
    }
  ],
  "positive_signals": ["short bullet"],
  "negative_signals": ["short bullet"],
  "management_summary": "concise business-friendly explanation of the outcome and customer experience",
  "recommended_actions": ["short actionable recommendation"],
  "manual_review_required": true,
  "manual_review_reason": "short reason or none",
  "confidence": "low|medium|high",
  "classification_reason": "short reason explaining handled/unhandled and good/bad experience"
}"""


DEFAULT_CONVERSATION_LEVEL_USER_TEMPLATE = """Evaluate the full customer conversation using the transcript, inline message-level evaluations, standalone message-level evaluations, computed metadata, and conversation metadata below.

Your task is to determine the customer's primary objective, whether it was handled, whether the customer experience was good or bad, whether frustration came from our side or the customer's side, whether an unhandled case is pending or totally unresolved, and what management should understand from the customer experience.

Return strict JSON only using the required schema.

Input:
{payload_json}
"""


DEFAULT_CONVERSATION_LEVEL_SYSTEM_PROMPT = _load_external_prompt_default(
    "conversational prompt",
    DEFAULT_CONVERSATION_LEVEL_SYSTEM_PROMPT,
)
DEFAULT_CONVERSATION_LEVEL_OUTPUT_SCHEMA = _load_external_prompt_default(
    "conversational output scheme",
    DEFAULT_CONVERSATION_LEVEL_OUTPUT_SCHEMA,
)
DEFAULT_CONVERSATION_LEVEL_USER_TEMPLATE = _load_external_prompt_default(
    "conversational user input",
    DEFAULT_CONVERSATION_LEVEL_USER_TEMPLATE,
)


DEFAULT_CONVERSATION_LEVEL_PROMPT = PromptTemplate(
    system_prompt=DEFAULT_CONVERSATION_LEVEL_SYSTEM_PROMPT,
    output_schema=DEFAULT_CONVERSATION_LEVEL_OUTPUT_SCHEMA,
    user_prompt_template=DEFAULT_CONVERSATION_LEVEL_USER_TEMPLATE,
)


# --------- Backward-compatible exports ---------
# Older code may import the bare strings; expose them as the assembled defaults.
MESSAGE_LEVEL_SYSTEM_PROMPT = DEFAULT_MESSAGE_LEVEL_PROMPT.build_system()
CONVERSATION_LEVEL_SYSTEM_PROMPT = DEFAULT_CONVERSATION_LEVEL_PROMPT.build_system()


# --------- Payload builders (unchanged) ---------


_ALLOWED_METADATA_KEYS = {
    "customer_name",
    "customer_phone",
    "customer_journey_id",
    "journey_id",
    "source_conversation_ids",
    "source_conversation_count",
    "conversation_start_date",
    "conversation_end_date",
    "conversation_status",
    "initial_skill",
    "last_skill",
    "joined_skills",
    "total_visible_messages",
    "customer_message_count",
    "agent_message_count",
    "unknown_message_count",
    "evaluation_target_role",
}


def _sanitize_conversation_metadata_for_llm(conversation_metadata: dict | None) -> dict:
    if not isinstance(conversation_metadata, dict):
        return {}
    return {
        key: value
        for key, value in conversation_metadata.items()
        if key in _ALLOWED_METADATA_KEYS and value not in (None, "")
    }


def build_message_level_payload(
    conversation_id: str,
    target_message: dict,
    history: list[dict],
    conversation_metadata: dict,
    truncate_chars: int | None = None,
) -> dict:
    """Build the JSON payload for a message-level call."""

    def trim(text: Any) -> str:
        if text is None:
            return ""
        text = strip_inline_rag_context(text)
        if truncate_chars and len(text) > truncate_chars:
            return text[:truncate_chars] + "...[truncated]"
        return text

    def evaluator_role(message: dict) -> str:
        return "customer" if str(message.get("sender_role", "")).lower() == "customer" else "agent"

    target = {
        "message_id": target_message.get("message_id", ""),
        "message_index": target_message.get("message_index"),
        "sender_role": evaluator_role(target_message),
        "message_time": str(target_message.get("message_time", "")),
        "message_text": trim(target_message.get("message_text", "")),
    }

    history_clean = []
    for m in history:
        history_clean.append(
            {
                "sender_role": evaluator_role(m),
                "message_index": m.get("message_index"),
                "message_time": str(m.get("message_time", "")),
                "message_text": trim(m.get("message_text", "")),
            }
        )

    return {
        "conversation_id": conversation_id,
        "target_message": target,
        "conversation_history_until_target": history_clean,
    }


def build_conversation_level_payload(
    conversation_id: str,
    conversation_metadata: dict,
    full_transcript: list[dict],
    message_level_evaluations: list[dict],
    computed_metadata: dict,
    truncate_chars: int | None = None,
) -> dict:
    """Build the JSON payload for a conversation-level call.

    Each message in ``full_transcript`` carries its message-level evaluation
    inline (under ``message_level_evaluation``) so the judge sees the
    judgement next to the message it judged. The aggregated
    ``message_level_evaluations`` list is also kept for prompt versions that
    reference it directly.
    """

    def trim(text: Any) -> str:
        if text is None:
            return ""
        text = strip_inline_rag_context(text)
        if truncate_chars and len(text) > truncate_chars:
            return text[:truncate_chars] + "...[truncated]"
        return text

    def sender_entity(message: dict) -> str:
        raw_role = str(message.get("raw_sender_role", "") or "").strip().lower()
        if raw_role == "system":
            return "broadcast"
        if raw_role in {"bot", "assistant"}:
            return "bot"
        if raw_role == "agent":
            return "agent"
        role = str(message.get("sender_role", "") or "").strip().lower()
        if role in {"customer", "agent"}:
            return role
        return "unknown"

    # Index evals by message_index so we can attach them inline.
    eval_by_idx: dict[Any, dict] = {}
    for ev in message_level_evaluations or []:
        if not isinstance(ev, dict):
            continue
        idx = ev.get("message_index")
        if idx is None:
            continue
        try:
            eval_by_idx[int(idx)] = ev
        except (TypeError, ValueError):
            eval_by_idx[idx] = ev

    transcript_clean = []
    for m in full_transcript:
        try:
            msg_idx = int(m.get("message_index", 0))
        except (TypeError, ValueError):
            msg_idx = m.get("message_index", 0)
        entry: dict[str, Any] = {
            "message_index": msg_idx,
            "appended_message_index": m.get("appended_message_index", msg_idx),
            "source_conversation_id": m.get("source_conversation_id"),
            "message_time": str(m.get("message_time", "")),
            "sender_role": m.get("sender_role", ""),
            "raw_sender_role": m.get("raw_sender_role"),
            "sender_entity": sender_entity(m),
            "message_text": trim(m.get("message_text", "")),
        }
        if msg_idx in eval_by_idx:
            entry["message_level_evaluation"] = eval_by_idx[msg_idx]
        transcript_clean.append(entry)

    return {
        "conversation_id": conversation_id,
        "conversation_metadata": _sanitize_conversation_metadata_for_llm(conversation_metadata),
        "full_transcript": transcript_clean,
        "message_level_evaluations": message_level_evaluations,
        "computed_metadata": computed_metadata,
        "message_level_summary": (computed_metadata or {}).get("message_level_summary", {}),
    }


def build_message_level_user_prompt(
    payload: dict,
    template: PromptTemplate | None = None,
) -> str:
    """Build the user prompt for a message-level call."""
    tpl = template or DEFAULT_MESSAGE_LEVEL_PROMPT
    return tpl.build_user(payload)


def build_conversation_level_user_prompt(
    payload: dict,
    template: PromptTemplate | None = None,
) -> str:
    """Build the user prompt for a conversation-level call."""
    tpl = template or DEFAULT_CONVERSATION_LEVEL_PROMPT
    return tpl.build_user(payload)
