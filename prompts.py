"""Prompt templates for message-level and conversation-level evaluation.

Both prompts are exposed as editable :class:`PromptTemplate` objects with three
independent fields:

* ``system_prompt`` — the role/instructions/rules text. May include
  ``{output_schema}`` to control where the schema block is inserted.
* ``output_schema`` — the JSON-shaped output structure.
* ``user_prompt_template`` — wraps the per-call payload. Must include
  ``{payload_json}``; otherwise the payload is appended at the end.

The default templates here are the same prompts the app shipped with — they
seed the SQLite DB on first launch. The user can edit any of the three fields
on the Prompts page and save new versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# --------- PromptTemplate ---------


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


# --------- Default message-level prompt ---------


DEFAULT_MESSAGE_LEVEL_SYSTEM_PROMPT = """You are an AI-as-a-Judge Customer Experience evaluator.

You evaluate chatbot/customer conversations from the customer's perspective.

Your job is to judge ONE specific message in the conversation, using the prior visible conversation history as context. The target message can be either an agent message or a customer message — check `target_message.sender_role` in the payload.

Two evaluation modes (driven by the target's role):

(A) Target is an AGENT message — judge the agent's response itself.
    Evaluate whether this specific agent message:
    - Understood the customer's intent.
    - Helped the customer move forward.
    - Created confusion, repetition, delay, or unnecessary effort.
    - Reduced or increased frustration.
    - Provided a clear next step.
    - Preserved context from earlier messages.
    - Contradicted previous information.
    - Ignored or mishandled the customer's concern.
    `message_level_effect` describes the effect of THIS agent message on the journey.
    `frustration_level_after_message` is the customer's likely frustration AFTER reading this message.
    `recommended_fix` is what the agent should do differently.

(B) Target is a CUSTOMER message — judge the customer's state at this point in the conversation, BEFORE the agent has responded to it.
    Evaluate:
    - The customer's emotional state (calm, confused, repeating themselves, frustrated, angry, threatening cancellation, satisfied).
    - Whether the customer is repeating an earlier ask or correcting the agent.
    - Whether the customer message reveals frustration that the agent should address next.
    `message_level_effect` should reflect the conversation effect that LED to this customer state — "minor_issue" / "major_issue" if prior agent behavior caused the issue, "recovered_issue" if the customer is signalling things are now resolved, "neutral" otherwise.
    `frustration_level_after_message` is the customer's frustration AT this message (i.e. just before the agent answers).
    `recommended_fix` is what the agent should do in their NEXT reply to address this customer state.

Do not judge the message in isolation. Judge it relative to the surrounding visible journey.

Focus only on the visible customer experience. Do not assume hidden tool calls, hidden policies, or internal execution details unless they appear in the visible transcript or provided metadata.

Return strict JSON only. Do not include markdown. Do not include explanations outside the JSON.

Required schema:
{output_schema}

Rules:
- Return JSON only.
- Use none when no issue exists.
- Do not invent facts.
- The frustration cause must be evidence-based and concise.
- Urgency alone is not frustration.
- A normal question is not frustration.
- Repetition, correction, anger, cancellation/refund intent, confusion, distrust, or sharp language are frustration signals.
- If the agent asks for information already provided (mode A) or the customer is repeating themselves (mode B), mark repetition or ignored_context.
- If the agent gives vague next steps (mode A), mark unclear_guidance or missing_next_step.
- If an agent message helps recover from a previous problem (mode A), or a customer message signals recovery / acceptance (mode B), use recovered_issue.
- If the issue is caused by unclear customer messages, use customer_side.
- If both sides contributed, use shared.
- If no issue exists, use none."""


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


DEFAULT_MESSAGE_LEVEL_USER_TEMPLATE = """Evaluate the target agent message using the conversation history and metadata below.

Return strict JSON only using the required schema.

Input:
{payload_json}"""


DEFAULT_MESSAGE_LEVEL_PROMPT = PromptTemplate(
    system_prompt=DEFAULT_MESSAGE_LEVEL_SYSTEM_PROMPT,
    output_schema=DEFAULT_MESSAGE_LEVEL_OUTPUT_SCHEMA,
    user_prompt_template=DEFAULT_MESSAGE_LEVEL_USER_TEMPLATE,
)


# --------- Default conversation-level prompt ---------


DEFAULT_CONVERSATION_LEVEL_SYSTEM_PROMPT = """You are an AI-as-a-Judge for Customer Experience Evaluation.

Your role is to evaluate whether the customer's request, inquiry, issue, or intended process was successfully handled from the customer's perspective, while also assessing the quality of the overall customer experience.

The evaluation must focus on the customer's journey, final outcome, effort, friction, frustration, clarity, and next steps.

You must first identify the customer's primary objective type:

Inquiry: The customer is requesting an action, service, process, update, information, or assistance.

Issue: The customer is reporting a problem, error, failure, blockage, rejected item, unexpected situation, dissatisfaction, or concern requiring resolution.

Then determine whether the objective was successfully handled from the customer's perspective.

Focus only on the customer-visible journey and the supplied metadata.

Evaluation dimensions:

1. Issue Resolution
Did the customer get what they came for? Was the request completed or was a clear next step provided?

2. Perceived Usefulness
Did the interaction add value for the customer?

3. Customer Understanding
Did the customer feel understood? Did they have to repeat or correct the bot?

4. Frustration Indicators
Was there anger, annoyance, repetition, distrust, escalation, cancellation risk, loop, or dead end?

5. Satisfaction Indicators
Was there gratitude, confirmation, relief, willingness to proceed, or improved tone?

6. Issue Detection
Were there repeated questions, confusion, uncertainty, false acknowledgments, vague responses, loops, or unnecessary work?

7. Effort and Efficiency
Was the interaction direct and low-effort?

8. Continuity and Accuracy
Was context preserved? Was guidance consistent?

9. Escalation and Recovery
Was there a clear recovery path when the bot could not complete the request?

10. Communication and Professionalism
Was the tone respectful, clear, practical, and easy to understand?

Classification options (use exactly one):
- Handled with Zero/Minimal Issues
- Handled with Many Issues
- Unhandled with Zero/Minimal Issues
- Unhandled with Many Issues

Definitions:
- Handled with Zero/Minimal Issues: The customer achieved the objective or received a clear acceptable next step, and the interaction was smooth, clear, efficient, and low-effort.
- Handled with Many Issues: The customer eventually achieved the objective or accepted the outcome, but experienced significant CX issues such as confusion, repetition, delay, frustration, misunderstanding, excessive effort, or poor guidance.
- Unhandled with Zero/Minimal Issues: The customer did not achieve the desired outcome, but the communication was clear, professional, reasonable, and the limitation was explained properly.
- Unhandled with Many Issues: The customer's objective was not achieved and the customer experienced significant CX issues such as confusion, repetition, frustration, excessive effort, poor guidance, lack of clarity, unresolved loops, or no clear next step.

Return strict JSON only. Do not include markdown. Do not include explanations outside the JSON.

Required schema:
{output_schema}

Rules:
- Return JSON only.
- Do not invent information.
- Handled vs Unhandled depends mainly on whether the customer objective was achieved or whether a clear acceptable next step was provided.
- Zero/Minimal vs Many Issues depends on CX friction, effort, confusion, frustration, repetition, and clarity.
- A conversation can be handled even if it had issues.
- A conversation can be unhandled even if the bot communicated politely.
- Always specify whether issues originated from our side, customer side, shared, or none.
- Explain impact from the customer's perspective.
- Keep the management summary concise and business-friendly.
- Set manual_review_required to true if confidence is low, cancellation risk exists, high frustration exists, the final status is unclear, or JSON/message-level errors affected the evaluation."""


DEFAULT_CONVERSATION_LEVEL_OUTPUT_SCHEMA = """{
  "conversation_id": "string",
  "customer_objective_type": "Inquiry|Issue",
  "customer_primary_objective": "short description",
  "final_classification": "Handled with Zero/Minimal Issues|Handled with Many Issues|Unhandled with Zero/Minimal Issues|Unhandled with Many Issues",
  "handled_status": "handled|unhandled",
  "cx_issue_severity": "zero_minimal|many",
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
      "issue_type": "string",
      "issue_summary": "string",
      "evidence": "string",
      "impact": "string"
    }
  ],
  "positive_signals": ["short bullet"],
  "negative_signals": ["short bullet"],
  "management_summary": "concise business-friendly explanation of the classification",
  "recommended_actions": ["short actionable recommendation"],
  "manual_review_required": true,
  "manual_review_reason": "short reason or none",
  "confidence": "low|medium|high"
}"""


DEFAULT_CONVERSATION_LEVEL_USER_TEMPLATE = """Evaluate the full conversation using the transcript, message-level evaluations, and computed metadata below.

Return strict JSON only using the required schema.

Input:
{payload_json}"""


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
        text = str(text)
        if truncate_chars and len(text) > truncate_chars:
            return text[:truncate_chars] + "...[truncated]"
        return text

    target = {
        "message_id": target_message.get("message_id", ""),
        "message_index": target_message.get("message_index", 0),
        "message_time": str(target_message.get("message_time", "")),
        "sender_role": target_message.get("sender_role", "agent"),
        "message_text": trim(target_message.get("message_text", "")),
    }
    history_clean = []
    for m in history:
        history_clean.append(
            {
                "message_index": m.get("message_index", 0),
                "message_time": str(m.get("message_time", "")),
                "sender_role": m.get("sender_role", ""),
                "message_text": trim(m.get("message_text", "")),
            }
        )

    return {
        "conversation_id": conversation_id,
        "target_message": target,
        "conversation_history_until_target": history_clean,
        "conversation_metadata": conversation_metadata,
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
        text = str(text)
        if truncate_chars and len(text) > truncate_chars:
            return text[:truncate_chars] + "...[truncated]"
        return text

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
            "message_time": str(m.get("message_time", "")),
            "sender_role": m.get("sender_role", ""),
            "message_text": trim(m.get("message_text", "")),
        }
        if msg_idx in eval_by_idx:
            entry["message_level_evaluation"] = eval_by_idx[msg_idx]
        transcript_clean.append(entry)

    return {
        "conversation_id": conversation_id,
        "conversation_metadata": conversation_metadata,
        "full_transcript": transcript_clean,
        "message_level_evaluations": message_level_evaluations,
        "computed_metadata": computed_metadata,
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
