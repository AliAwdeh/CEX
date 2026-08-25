"""Evaluation orchestration: message-level and conversation-level runs.

Includes robust JSON extraction and schema validation, plus a single entry point
``run_evaluation`` that drives the full pipeline with progress callbacks and
graceful per-conversation error handling.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from api_client import APIConfig, chat_completion
from prompts import (
    DEFAULT_CONVERSATION_LEVEL_PROMPT,
    DEFAULT_MESSAGE_LEVEL_PROMPT,
    PromptTemplate,
    build_conversation_level_payload,
    build_message_level_payload,
)
from aggregation import compute_metadata
from data_loader import (
    LEGACY_MESSAGE_ORDER_COLUMN,
    JOURNEY_ID_COLUMN,
    MESSAGE_ORDER_COLUMN,
    TICKET_JOURNEY_ID_COLUMN,
    conversation_metadata_from_group,
    get_conversation_groups,
    message_records_from_group,
    strip_inline_rag_context,
)


# ---------- JSON robustness ----------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


DEFAULT_TICKET_SEGMENTATION_SYSTEM_PROMPT = """You split a complete customer/contract conversation timeline into ticket-style customer journeys.

Return strict JSON only.

A ticket is one evaluable customer thread. A thread can be:
- a concrete issue, request, complaint, process action, or blocked outcome
- a grouped informational inquiry ticket that may contain several standalone questions

Each ticket must have exactly one ticket_category:
- inquiry: the customer asks for information, explanation, price, timing, policy, eligibility, location, status, or a next step, and does not ask the company to perform an action.
- request: the customer asks the company to check, send, provide, book, arrange, change, cancel, renew, process, confirm, transfer, deliver, give a copy/document/link, or otherwise do something, without primarily complaining about a failure. This includes indirect wording such as "is there any possibility to get a copy of the contract" or "can I get a copy".
- issue: the customer reports a problem, complaint, blocker, failed/incorrect action, payment/refund dispute, contradiction, delay, or bad experience.
- Normal process/action tickets such as maid visa request, booking request, document request, renewal request, cancellation request, refund request, or delivery request are ticket_category=request unless the customer is primarily reporting a failure, complaint, dispute, delay, blocker, or bad experience.
- Status/readiness/procedure questions such as "can you tell/let me know if the contract is ready", "is it ready", "any update", "what is the status", or "what is the procedure to change maid" are ticket_category=inquiry unless the customer asks the company to perform, process, change, expedite, or check something operationally, or reports a problem.
- Do not classify a normal inquiry/request as issue only because the agent mentions an internal limitation or uses words like "problem" or "issue". The customer must be reporting or experiencing that problem as the main objective.

Category decision checklist:
1. Identify the customer's main objective, not the agent's reply wording and not the ticket_type name.
2. ticket_category describes the ticket's main objective/lifecycle, not whether the ticket contains problems. A request can contain issues, rejected steps, delays, missing documents, fees, or complaints and still remain ticket_category=request.
3. First material customer objective category anchor rule: inside one ticket, the ticket_category is anchored by the first material customer objective that opened that ticket. If the ticket starts as a request/action/lifecycle, keep ticket_category=request even if later messages contain issues, complaints, blockers, delays, status questions, proof requests, or escalations about completing that request. If the ticket starts as an issue/problem/dispute, keep ticket_category=issue even if later messages ask the company to check, call, escalate, send, refund, provide, or perform a follow-up action to resolve that issue. Only change category or split when the later customer objective is materially separate from the original objective.
4. Embedded issues, requests, and inquiries are not child tickets. If a request ticket contains problems, describe the embedded problems in customer_objective and segmentation_reason, and put only true informational questions in inquiries. If an issue ticket contains action asks, describe those embedded requests/actions in customer_objective and segmentation_reason, and put only true informational questions in inquiries. Do not invent extra fields outside the schema.
5. Choose issue only when the customer's main objective is to report/fix a standalone problem, complaint, blocker, failed/incorrect action, dispute, contradiction, or bad experience, and there is no broader operational request/lifecycle that the problem belongs to.
6. Choose request when the customer wants the company to do or complete an operational action/lifecycle: initiate, check operationally, send, provide, book, arrange, change, cancel, renew, process, confirm, transfer, deliver, upload, replace, collect, refund, escalate, provide a copy/document/link, or complete a visa/residency process.
7. Choose inquiry when the customer wants information only: explanation, price, policy, eligibility, location, timing, next step, readiness, status, whether something is done/issued/approved/active, or why/how something works.
Future renewal eligibility/timing wording such as "can we renew in February?", "am I right that I can renew?", or "when shall we process it, January 15 onward?" is inquiry when the customer is planning and will contact the company later. Operational words inside a question do not make it a request unless the customer asks the company to start, submit, or perform the renewal.
8. Request beats issue when the issue is a stage inside the requested process. Do not classify a visa/residency request as issue just because it contains passport rejection, photo rejection, missing documents, government rejection, late fees/fines, delay, or customer frustration. Keep ticket_category=request and reflect the problem in ticket_type, status, inquiries, and segmentation_reason.
9. Request beats inquiry only when the customer asks for an action to be performed or asks to receive something from the company. "Can you tell me", "let me know", "what is the status", "is it ready", and "any update" are inquiry, even if the agent replies "I will check". But "can I get a copy", "is there any possibility to get a copy", "can you send/provide the contract", and similar document-copy wording are request.
10. Issue beats request/inquiry only when the customer's main objective is the problem itself, not completion of an existing process. Do not infer issue from agent-only words, internal limitations, or a normal future step.

Category examples:
- inquiry: "Can you tell us if the maid contract is ready?", "Is the visa issued?", "Any update?", "How much is the insurance?", "Which card is linked?", "What is the procedure to change maid?", "Can you please let me know if you provide uniforms to the maids?"
- request: "Please send the contract", "Is there any possibility to get a copy of the signed contract?", "Can I get a copy of the contract?", "Can you check which card you are using for auto deduction?", "Please book the medical test", "Cancel my renewal", "Process the refund", "Start the maid visa", "The passport/photo was rejected during visa processing, please upload again", "Please share the contract, insurance network list, and update the sponsor name", "Please order/provide uniforms for the maid"
- issue: "The payment failed", "You charged me twice", "The app is difficult and adds fees", "This is unfair", "The refund was wrongly rejected", "My maid didn't receive her salary yet", "Why not in Du Pay? You said this last month", "I think the AED 1,500 advanced salary is a duplicate", "I already paid one month advance salary in July 2024" when the customer's main objective is the complaint/dispute itself rather than completion of a broader request lifecycle.

Core rules:
- Use only visible messages.
- Input is grouped as source_conversation_blocks. Treat each source_conversation_id as a bracket/header for the messages inside that block; individual message objects intentionally do not repeat source_conversation_id.
- message_index is the continuous order across the full customer journey and does not reset when a new source_conversation_id block starts. Use these exact message_index values in start_message_index, end_message_index, included_message_indexes, and inquiry message_indexes.
- No overlapping ticket indexes: each material message_index should belong to one ticket only. Do not output a broad lifecycle ticket and also child tickets whose included_message_indexes sit inside or mostly overlap that lifecycle. If a message is part of a broader active lifecycle, keep it inside that lifecycle ticket and represent questions as inquiries inside that ticket rather than as separate child tickets.
- If input contains segmentation_context with segmentation_mode=cumulative_source_conversation, this is one pass in a conversation-by-conversation cumulative sequence. The payload contains only the current source_conversation_id block; earlier source conversations are represented only by previous_cumulative_ticket_output. Use that previous ticket summary as the existing ticket map, then revise it using the current source_conversation_blocks. Return the complete ticket list for all processed source conversations, not only the current source conversation. Preserve earlier ticket IDs/categories/statuses unless the current source conversation proves they should be merged, split, linked, reopened, or updated.
- One contract/customer timeline can contain multiple tickets.
- Always inspect previous source_conversation_id blocks and the latest ticket state before opening a new ticket. Open/pending tickets may resume non-contiguously after another objective. For a resolved ticket, reopen the original only when the immediately next substantive source conversation returns to that subject and proves it unresolved. If a different-subject source conversation intervenes before the resolved subject returns, create a new ticket from the returning messages and set previous_ticket_id to the original resolved ticket. Greetings, routing, courtesy, and isolated promotions are not substantive intervening subjects.
- Same service-flow anti-micro-ticket rule: do not create a new ticket just because the same objective appears in a new source_conversation_id, uses slightly different wording, changes from inquiry to complaint/escalation, asks for proof/status/timing/callback, repeats an unanswered question, or gets a partial update. For active visa/residency/EID, employment onboarding, salary/payroll/payment-route, refund, document/admin, medical/insurance, card, delivery, or cancellation flows, merge all stages, reminders, status checks, delay complaints, proof requests, screenshots, escalation requests, callback promises, agent follow-ups, and confirmations into one ticket when they serve the same customer objective. A new ticket requires a materially new customer objective, a separate lifecycle, a separate payment/refund/item, or a post-completion problem that is no longer needed to complete the original flow. Same broad category alone is not enough to merge unrelated objectives, but same category/stage inside one active flow is not enough to split.
- Mandatory final merge audit: before returning JSON, scan your proposed tickets in order. If any adjacent or near-adjacent tickets share the same service object (same visa process, same worker salary/month, same refund/payment, same document/admin request, same medical/insurance/card/delivery/cancellation flow) and the later ticket is only an update, status/timing question, proof/screenshot/statement, repeated ask, escalation, complaint, callback request, agent follow-up, broadcast, or confirmation, merge them. Do not leave two tickets where the segmentation_reason would be "same topic but new source_conversation_id"; source_conversation_id alone is never a reason to split. After the audit, each remaining ticket must pass this test: "Would the customer consider this a separate thing they needed solved, not just another message about the previous thing?" If the answer is no, merge it.
- Visa/residency lifecycle rule: one visa application or visa renewal is one ticket from start to finish. Do not split its stages into separate tickets. Starting the visa, renewing it, passport processing, passport upload, missing documents, rejected documents/photos, government rejection, medical/EID steps, late fees/fines, status updates, and final approval/rejection all belong to the same visa/residency ticket when they are part of the same process.
- Visa application-number and proof escalation rule: while a visa/residency change-of-status process is active, later source_conversation_id blocks asking for an application number, ICP/request/file/UID/transaction number, submission receipt, screenshot/proof, online tracking, change-of-status status, overstay/fine/government-fee handling, supervisor call/escalation, delay complaint, complaint/report threat, cancellation, or refund because the visa action is not done are still the same visa/residency lifecycle ticket. Do not split each source_conversation_id or each demand for proof/application number into separate "other" tickets. Keep them with the earlier visa-processing ticket until the main visa/cancellation/refund lifecycle is visibly completed or the main refund is confirmed/received. Bad split: one ticket for "start visa processing" and later tickets for "provide application number", "what is the update today", "same document/transaction number", "ICP/request number/screenshot", "wrong/cancelled transaction number", and "cancel/refund because nothing happened" for the same maid change-of-status process. Correct split: one visa_processing request ticket through the main cancellation/refund confirmation, then only a later separate AED 400/overstay refund not-received objective becomes a new issue ticket.
- Employment visa onboarding lifecycle rule: for one employment visa/residency onboarding process, document/photo/passport collection, e-visa or entry permit, change of status, medical fitness, Emirates ID/EID, government rejections, address/location for EID delivery, WPS setup, salary deduction setup, salary card/ATM card readiness, Al Ansari pickup/branch transfer, contract-copy/admin documents, and insurance card/policy details are stages or follow-ups in the same lifecycle unless the customer switches to a clearly separate post-completion payroll/payment problem or unrelated service request. Do not split each upload, rejection, status check, ATM, WPS, or insurance step into micro-tickets.
- NOC source conversation boundary rule: NOC/no-objection-certificate requests belong inside the visa/residency lifecycle only when the same source_conversation_id is visibly part of the active visa processing conversation, or when a later source_conversation_id continues an unresolved active visa processing lifecycle. If visa processing has ended/resolved and a later new source_conversation_id asks for NOC/no-objection certificate/no-objection letter, create a separate NOC request ticket. Do not merge NOC requests across different source_conversation_id values unless the later source is itself continuing the active visa/residency process. Repeated NOC messages inside one source_conversation_id stay in one NOC ticket.
- Visa/residency category rule: when a visa/residency lifecycle ticket contains issues, keep ticket_category=request. The issues are embedded stages/problems inside the request. Do not output ticket_category=issue for the whole visa/residency lifecycle unless there is no visible operational request/lifecycle and the only customer objective is a standalone complaint/dispute.
- Visa/residency cancellation and refund follow-up rule: if a visa/residency lifecycle leads to cancellation and the customer is still asking for proof, cancellation, main refund approval, refund confirmation, ICP/change-status/application-number proof, or status of the cancellation/refund process, keep those follow-ups in the same visa/residency lifecycle ticket. Do not split repeated source_conversation_id blocks just because the customer asks for an update, proof, call, application number, or refund confirmation while the main visa/cancellation/refund lifecycle is still open.
- Residual overstay refund split rule: after the main visa/cancellation refund is visibly confirmed, credited, or received, a later customer objective about a separate AED 400/overstay fine refund not received, bank proof, missing bank credit, or "where is the 400" is a new issue ticket. Do not keep repeated AED 400 bank/proof follow-ups inside the already-handled main visa lifecycle; merge those AED 400 follow-ups together into one residual overstay_refund_issue ticket.
- EOS/renewal responsibility disputes are issues, not visa requests, when the customer's main objective is to challenge a policy, entitlement, responsibility, misrepresentation, loophole, penalty, or who must pay. Example: "That's not how you sold it to me", "your loophole to avoid paying", "subvert your responsibilities", or "why do I have to cancel?" should be ticket_category=issue even if a visa-renewal broadcast frames the conversation.
- Salary/payment-route problems are issues, not requests, when the customer reports salary not received, salary missing/late, salary sent through the wrong channel, or a repeated Ansari vs Du Pay routing problem. Example: "My maid didn't receive her salary yet" plus "Why not in Du Pay?", "You said this last month", or "She's been registered for 3 months" should be ticket_category=issue.
- Salary missing/payment-route escalation rule: when one salary/payroll payment is missing, deducted but not visible, not received, routed to the wrong wallet/card, or disputed between Du Pay and Al Ansari, all follow-ups about that same salary/payment stay in one salary/payment-route issue ticket. Merge later source_conversation_id blocks that ask the company to contact/follow up with Du Pay, ask whether the customer can email/call Du Pay, ask when the salary should appear, mention card request/transaction history/wallet balance, complain about having to contact Du Pay, request manager callback/escalation/complaint, or receive a salary statement/proof broadcast for the same worker/payment. Do not split those into general_inquiry, other, callback, complaint, or broadcast-only tickets. Only create a new salary ticket when the later source is a different salary month/payment, a different worker, a separate advanced-salary/duplicate-billing dispute, or a clearly new post-resolution payroll problem.
- Salary destination change request rule: when the customer asks to switch/change/transfer a worker's salary destination from Du Pay/Al Ansari to a bank account, ENBD/Emirates NBD account, IBAN, or other payout account, classify it as ticket_category=request, not inquiry/status and not issue unless the main objective is a standalone failed/missing salary complaint. Merge all later source_conversation_id blocks that provide bank screenshots, IBAN, account holder name, bank name, account number, corrected screenshots, "ok here it is", "details do not fit in one screenshot", agent requests for one clear screenshot, and final submission/confirmation into the same salary_payment_destination_change request ticket. Bad split: Ticket 1 = "can you transfer salary to ENBD instead of Du Pay" and Ticket 2 = "ok here it is" / bank screenshot / request submitted. This must be one request ticket.
- Salary status follow-up rule: after employment visa/onboarding is visibly complete, later questions like "is June salary transferred?", "what time will salary transfer be done?", or "we will wait 24 to 48 hours" are one salary_status inquiry unless the customer reports salary missing, late, not received, wrong route, or a payment dispute. Merge those repeated salary status follow-ups together; do not create separate two-message request/inquiry tickets for each source_conversation_id.
- Advanced salary / duplicate billing disputes are issues, not requests, when the customer contests AED 1,500/AED 1,668 charges, says an advance salary was already paid, asks whether the old payment was lost, says a charge is duplicate, says they paid 25/26 salaries in 2 years, or asks for human escalation because the billing explanation does not answer the charge.
- Category uses only visible evidence. A customer-forwarded official notice such as "Dear Customer, your application ... was approved/rejected" is context, not the customer's objective. If the first actual support objective reports missing files, an incorrect message, a rejection, or an agreed visa/change-status action that was not performed, classify the lifecycle as issue even when the customer later asks the company to submit the application and send proof. If a visible earlier customer objective already requested starting that same visa process, keep the lifecycle as request.
- Objective naming rule: never include customer, maid, worker, helper, or agent personal names in ticket_type or customer_objective. Use neutral role wording such as "the maid", "the worker", or "the customer" only when needed. Do not copy an emotional opener such as "this is totally unacceptable" as the objective; summarize the concrete outcome sought.
- If the first visible source conversation is already about a visa/passport/photo/government rejection and no earlier visa ticket is visible, create a new issue ticket for that visible problem. If later cumulative passes reveal an earlier visa/residency request ticket, move/merge that rejection into the existing visa/residency ticket instead of keeping a separate ticket and preserve the earlier request anchor.
- If the immediately next substantive source conversation resumes a previously resolved visa/residency subject and proves it unresolved, reopen/update the same ticket. If a different-subject source conversation intervenes first, create a new linked visa/residency ticket with previous_ticket_id pointing to the resolved predecessor. A visa/residency ticket that was still pending may resume the same original ticket non-contiguously after an interruption.
- Include the greeting/setup messages that immediately precede the customer's first real request in the same ticket. If the customer says "Hi" and the assistant replies before the customer states the objective, keep those greeting messages inside that ticket.
- Do not create a standalone ticket for greeting/setup/routing exchanges such as customer "Hi" followed by a company message saying this number cannot receive/review messages and directing the customer to a support WhatsApp number. Attach those messages to the next real customer ticket when one exists, or to the previous active ticket if the setup appears after it.
- If the customer asks a real question/action in one source_conversation_id and the only company reply is a routing-only message saying this number cannot receive/review messages or directing them to another support number, do not treat that reply as a resolution. If the next source_conversation_id repeats or continues the same customer objective and gets the real answer/action, merge both source conversations into one ticket and use the final outcome from the later source conversation.
- The word "inquiry" only counts when it is the customer's inquiry. Do not create an inquiry ticket or inquiry-array item because the assistant/company says "your inquiry", "continue with your inquiry", "support inquiry", or similar routing wording.
- Do not exclude a broadcast/system message only because it is a broadcast. Include broadcasts that trigger, frame, confirm, remind about, warn about, follow up on, or are referenced by the customer in relation to the ticket.
- If the customer asks about, reacts to, complains about, confirms, or follows instructions from a broadcast, include that broadcast and the customer interaction in the same ticket.
- An isolated promotional service or marketing message with no customer response must not create a ticket; attach it to the nearest current ticket as contextual noise without changing that ticket's objective, category, status, or inquiries. If the customer meaningfully engages by requesting details, checking price/availability, selecting an option, booking, or pursuing the promoted service, create a separate ticket containing the promotion and that full customer flow.
- Do not omit substantive customer, agent, bot, or broadcast messages. If a message is not clearly isolated unrelated noise, assign it to the closest relevant ticket or create a separate ticket for it.
- Exclude only truly isolated unrelated operational noise that has no relationship to any customer objective and is not needed to understand the ticket.
- Standalone non-service broadcast exclusion rule: a broadcast/system message with no customer objective and no customer response about it must not become a ticket. Isolated promotions attach to the nearest current ticket under the promotional-message rule above. Birthday/holiday/greeting broadcasts such as "maidbirthday is tomorrow" or "wish her a happy birthday" are not support tickets and must be excluded, not labeled as uniform, salary, visa, inquiry, request, or other. Do not infer the ticket_type from later unrelated customer messages.
- Unrelated service/noise rule: home gas/Gasul delivery chatter, birthday wishes, courtesy acknowledgments, or other non-contract events must not open their own ticket and must not cause a split inside an employment visa lifecycle. Exclude them when they are clearly unrelated to the company's service; otherwise attach only the directly relevant acknowledgment to the closest active ticket.
- Uniform availability questions are inquiries: "do you provide uniforms", "can you let me know if uniforms are provided", or "are uniforms included/available" asks for policy/availability information and should be ticket_category=inquiry, ticket_type=uniform_inquiry. Classify as request only if the customer asks the company to actually order, send, buy, or provide uniforms as an action.
- Do not create a standalone ticket for courtesy-only closeout messages such as "thank you", "thanks", "you are welcome", "I am here to help", or "feel free to reach out". Append those messages to the previous related ticket and do not add a new inquiry for them.
- If a new source_conversation_id starts only with an acknowledgement or confirmation such as "yes please", "sure", or "ok please", treat it as a continuation of the previous active ticket unless the customer then states a clearly new objective.
- If a new source_conversation_id starts with a new question or new check/action request after an earlier ticket that already mixed an issue/request with informational inquiries, create a new ticket. Do not append it to the earlier ticket only because the broad topic is similar. Link it with previous_ticket_id only when useful.
- After a medical, insurance, clinic, payment, visa, document, or delivery thread has been answered/closed, a later source_conversation_id asking a different objective such as "What is the procedure to change maid?" is a new ticket, not a continuation. Generic shared domain words such as maid/customer/contract do not make two objectives the same ticket.
- Do not merge an initial visa-renewal information thread with a later advanced-salary/duplicate-billing dispute only because both mention renewal or July charges. Example: messages asking visa expiry, renewal process, renewal cost, cheaper option, breakdown, flight, and labor-card expiry are one inquiry ticket; a later AED 1,500 advanced salary reminder followed by "I don't understand", "I already paid advance salary in July 2024", "is that lost?", "duplicate", or "26th payment" is a separate issue ticket.

Issue/request grouping:
- Create a separate ticket for each distinct issue, complaint, process action, cancellation request, refund request, payment problem, document problem, or operational blocker. Exception: visa/residency/EID process problems are stages inside the same visa/residency lifecycle ticket when they belong to the same process.
- If the customer raises the same underlying issue multiple times, append the later messages to the same issue ticket until it is resolved, clearly pending, abandoned, or replaced by a different issue.
- Do not split a how-to question from the customer's follow-up complaint, fee question, policy question, fairness objection, comparison, or "why is this harder" reaction when all messages concern the same product, service, process, app flow, provider change, policy, document flow, payment flow, booking flow, delivery flow, or operational task. Treat those follow-ups as clarifications/escalation inside one ticket.
- Keep steps, limitations, charges, alternatives, provider comparisons, and complaints about the same flow in one ticket unless the customer switches to a different unrelated objective.
- If two issues are different, keep them as separate tickets even if they happen close together.
- If an informational question is only a clarification inside an active issue, include it in that issue ticket and list it in that ticket's inquiries array.
- For request tickets with multiple action asks in the same document/admin flow, keep them as one request ticket. Example: asking for a contract copy, insurance network list, sponsor-name correction, and contract-field change in one active thread is one request ticket. Put the material action asks in customer_objective and segmentation_reason; do not split each action ask into separate tickets.
- Bad split: Ticket 1 = "how do I do this process?" and Ticket 2 = "why does this process have extra steps/fees or why is the new process worse?" when both discuss the same process. This must be one ticket with multiple inquiries.
- Bad split: Ticket 1 = "complete employment visa process" covering #1-#272, Ticket 2 = "passport status" covering #84-#87, Ticket 3 = "medical/EID timing" covering #203-#208, Ticket 4 = "ATM/insurance readiness" covering #238-#272, plus salary micro-tickets covering #274-#281. This is invalid because Tickets 2-4 overlap the lifecycle and the salary follow-ups should be one later salary_status inquiry.
- Bad split: Ticket 1 = salary transfer/status, Ticket 2 = "what time will it be transferred?", Ticket 3 = "we will wait 24 to 48 hours", Ticket 4 = "please check again" for the same payroll period or same missing salary. This must be one salary_status inquiry or one salary/payment-route issue depending on whether the customer is only checking status or reporting a missing/wrong payment.
- Bad split: Ticket 1 = "missing June salary deducted but not in Du Pay", Ticket 2 = "kindly ask Du Pay Support / when should salary appear", Ticket 3 = "follow up with Du / complaint / manager callback", Ticket 4 = salary-statement broadcast. If all messages concern the same worker and same missing June salary, this must be one salary/payment-route issue ticket, not four tickets.

Inquiry grouping:
- A standalone inquiry is a question asking for information, explanation, timing, status, eligibility, price, policy, location, required document, or next step, without itself being a complaint or operational blocker.
- A standalone inquiry must originate from a customer message. Company-side wording that mentions "inquiry" is not itself an inquiry and must not become a separate ticket or an entry in the inquiries array.
- If the customer asks multiple standalone inquiries about different matters, group them into one inquiry ticket and list every question separately in the inquiries array.
- A general_inquiry ticket may span any number of source_conversation_id blocks. Continue the same ticket when the later block answers an agent question, clarifies, follows up on, repeats, resolves, or materially updates the same customer objective. Create a new inquiry ticket only when the customer introduces a materially different informational objective; the source_conversation_id boundary itself is never evidence of a new ticket.
- If an earlier inquiry was already resolved and the customer later asks a materially different informational topic, create a new general_inquiry ticket and set previous_ticket_id to the earlier related ticket when relevant. A short contextual answer such as "one-time", "in UAE", "yes", a name, a date, or a supplied document continues the preceding exchange even when it starts a new source_conversation_id.
- If the sequence is issue -> inquiry -> issue -> inquiry, keep each distinct issue as its own ticket unless it is the same underlying issue, and group the standalone inquiries together in one inquiry ticket. Preserve the visible message indexes for each inquiry.
- If a sequence looks like inquiry -> complaint -> fee/policy follow-up but all parts are about the same underlying flow, do not use the inquiry-ticket rule. Keep it as one issue/request ticket and list each question in inquiries.
- For every inquiry ticket, the inquiries array must track each inquiry separately, including whether that specific inquiry was resolved, pending_unresolved, or totally_unresolved.
- For issue tickets that contain clarification questions, include those questions in the inquiries array too.
- For request tickets, the inquiries array should contain only informational questions/clarifications, not action asks. "Can you share/send/update/change/check/provide..." is usually an action ask that belongs in the objective/reason, while "Is it okay that details are blank?" belongs in inquiries.

Resolved vs future follow-up logic:
- For issue/request tickets, if the issue is not resolved and later source conversations continue or follow up on the same underlying issue, append those later messages to the same issue ticket.
- For issue/request tickets, use the final outcome after the latest related source conversation as the ticket status. Example: refund pending in source conversation 1, then refund rejected in source conversation 2 = one refund ticket with status totally_unresolved; if source conversation 2 confirms refund accepted = one refund ticket with status resolved.
- For issue/request tickets, if the issue was resolved and later the customer raises a new separate request in the same category, create a new ticket with the same ticket_type and set previous_ticket_id to the earlier resolved ticket id.
- For inquiry tickets, source_conversation_id does not control ticket boundaries. Group messages by the customer's objective, merge related continuations across source blocks, and use the final visible outcome as the ticket status.
- Set should_append_future_conversations to true only for tickets whose final status is pending_unresolved. Resolved and totally_unresolved tickets are closed.

Status rules:
- If a ticket is still waiting for a retry, review, delivery, refund, government step, customer document, bank action, internal action, or future confirmation, status is pending_unresolved.
- If no usable current state or path remains, status is totally_unresolved.
- If the customer objective was answered/completed/accepted, status is resolved.
- For request tickets with multiple action asks, status is resolved only when every material requested action is visibly completed, delivered, accepted, or no longer needed. If the agent only promises to share/check/update later, status is pending_unresolved even if an embedded informational question was answered.
- Direct-debit stop/recall rule: successful card payment and changing future payments to card do not prove that an already-sent current bank instruction was recalled. Resolve only when the company explicitly confirms that the current direct debit was recalled, cancelled, stopped, withdrawn, or is no longer active. "Forms will be deleted", "payment received by card", or a later thanks about future card setup is insufficient; otherwise keep pending_unresolved.
- A routing-only company reply that says the number cannot receive/review messages or sends the customer to a different support number is not a real answer/resolution to the customer's objective. Mark it pending_unresolved unless it is merged with a later source conversation that actually answers or completes the same objective.
- For salary/payment-route issues, a workaround such as "collect it from Ansari this month" resolves only the immediate collection path, not the underlying route issue. If the answer says the next salary/payroll should go to Du Pay, the salary route can switch on the following payroll cycle, or the team will reach out if anything changes, set status=pending_unresolved and should_append_future_conversations=true.
- For grouped inquiry tickets, the ticket status is resolved only if all listed inquiries are resolved. If any inquiry is pending_unresolved, the ticket status is pending_unresolved. If any inquiry is totally_unresolved and no usable path remains for it, the ticket status is totally_unresolved unless another inquiry is still pending with a clear path.
- For grouped inquiry tickets, should_append_future_conversations must be true only when the final ticket status is pending_unresolved.

Ticket type should be short snake_case, such as payment_issue, refund_request, document_request, visa_status, eid_delivery, contract_question, cancellation_request, booking_request, complaint, general_inquiry, other.

Required JSON shape:
{
  "tickets": [
    {
      "ticket_id": "ticket_1",
      "ticket_category": "issue|request|inquiry",
      "request_origin": "company|customer",
      "ticket_type": "short_snake_case",
      "customer_objective": "short description",
      "start_message_index": 1,
      "end_message_index": 5,
      "included_message_indexes": [1, 2, 3, 4, 5],
      "status": "resolved|pending_unresolved|totally_unresolved",
      "should_append_future_conversations": true,
      "previous_ticket_id": "",
      "inquiries": [
        {
          "inquiry_id": "inquiry_1",
          "question": "short customer question",
          "message_indexes": [3],
          "status": "resolved|pending_unresolved|totally_unresolved",
          "answer_summary": "short answer or current state",
          "unresolved_reason": "short reason, or none"
        }
      ],
            "conversation_summaries": [
                {
                    "source_conversation_id": "exact source id",
                    "message_indexes": [1, 2, 3],
                    "customer_intent": "short intent in this source conversation",
                    "outcome": "short outcome or latest state",
                    "status": "resolved|pending_unresolved|totally_unresolved",
                    "ticket_signals": ["short_snake_case_signal"]
                }
            ],
      "segmentation_reason": "short reason"
    }
  ]
}

Use an empty string for previous_ticket_id when there is no predecessor. Use an empty inquiries array only when the ticket contains no informational question.

Do not include markdown, comments, or extra top-level keys."""


def _load_ticket_segmentation_system_prompt() -> str:
    root = Path(__file__).resolve().parent / "correct_prompt_files"
    for filename in ("second ticket segmenation prompt.txt", "ticket segmentation prompt.txt"):
        path = root / filename
        try:
            value = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if value.strip():
            return value
    return DEFAULT_TICKET_SEGMENTATION_SYSTEM_PROMPT


TICKET_SEGMENTATION_SYSTEM_PROMPT = _load_ticket_segmentation_system_prompt()


DEFAULT_TICKET_SEGMENTATION_OUTPUT_SCHEMA = """{
  "tickets": [
    {
      "ticket_id": "ticket_1",
      "ticket_category": "issue|request|inquiry",
      "ticket_type": "short_snake_case",
      "customer_objective": "short description",
      "start_message_index": 1,
      "end_message_index": 5,
      "included_message_indexes": [1, 2, 3, 4, 5],
      "status": "resolved|pending_unresolved|totally_unresolved",
      "should_append_future_conversations": true,
      "previous_ticket_id": "",
      "inquiries": [
        {
          "inquiry_id": "inquiry_1",
          "question": "short customer question",
          "message_indexes": [3],
          "status": "resolved|pending_unresolved|totally_unresolved",
          "answer_summary": "short answer or current state",
          "unresolved_reason": "short reason, or none"
        }
      ],
            "conversation_summaries": [
                {
                    "source_conversation_id": "exact source id",
                    "message_indexes": [1, 2, 3],
                    "customer_intent": "short intent in this source conversation",
                    "outcome": "short outcome or latest state",
                    "status": "resolved|pending_unresolved|totally_unresolved",
                    "ticket_signals": ["short_snake_case_signal"]
                }
            ],
      "segmentation_reason": "short reason"
    }
  ]
}"""


DEFAULT_TICKET_SEGMENTATION_USER_TEMPLATE = """Split this complete customer/contract timeline into ticket-style journeys. The input JSON groups messages under source_conversation_blocks.

Return strict JSON only using the required schema.

Input:
{payload_json}"""


def _load_ticket_prompt_file(filename: str, fallback: str) -> str:
    path = Path(__file__).resolve().parent / "correct_prompt_files" / filename
    try:
        value = path.read_text(encoding="utf-8")
    except OSError:
        return fallback
    return value if value.strip() else fallback


def _default_ticket_segmentation_prompt() -> PromptTemplate:
    return PromptTemplate(
        system_prompt=_load_ticket_segmentation_system_prompt(),
        output_schema=_load_ticket_prompt_file(
            "ticket segmentation scheme.txt",
            DEFAULT_TICKET_SEGMENTATION_OUTPUT_SCHEMA,
        ),
        user_prompt_template=_load_ticket_prompt_file(
            "ticket segmentation user input.txt",
            DEFAULT_TICKET_SEGMENTATION_USER_TEMPLATE,
        ),
    )


TICKET_SEGMENTATION_MODE_SINGLE_PASS = "single_pass"
TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE = "cumulative_source_conversation"
TICKET_SEGMENTATION_MODE_DEFAULT = TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE
TICKET_SEGMENTATION_MODES = {
    TICKET_SEGMENTATION_MODE_SINGLE_PASS,
    TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE,
}


def clean_ticket_segmentation_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in TICKET_SEGMENTATION_MODES else TICKET_SEGMENTATION_MODE_DEFAULT


def extract_json_object(text: str) -> dict:
    """Best-effort extraction of a single JSON object from a model response."""
    if not text:
        raise ValueError("Empty model response")
    text = text.strip()

    # Plain JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Code fence
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Greedy first { ... } block
    m = _FIRST_OBJ_RE.search(text)
    if m:
        candidate = m.group(0).strip()
        # Try progressively trimming from the right end if there is trailing junk.
        for end in range(len(candidate), 0, -1):
            try:
                obj = json.loads(candidate[:end])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    raise ValueError("Could not extract a JSON object from the model response")


# ---------- Schema validators / normalizers ----------

_ML_ENUMS = {
    "message_level_effect": {"helped", "neutral", "minor_issue", "major_issue", "recovered_issue"},
    "frustration_level_after_message": {"none", "low", "medium", "high", "cancellation_risk"},
    "frustration_change": {"decreased", "unchanged", "increased", "created"},
    "customer_effort_level": {"low", "medium", "high"},
    "clarity_level": {"clear", "somewhat_clear", "unclear"},
    "context_handling": {"good", "partial", "poor", "not_applicable"},
    "issue_origin": {"our_side", "customer_side", "shared", "none"},
    "issue_type": {
        "none",
        "misunderstanding",
        "repetition",
        "delay",
        "unclear_guidance",
        "wrong_info",
        "ignored_context",
        "dead_end",
        "tool_or_system_failure",
        "poor_tone",
        "missing_next_step",
        "other",
    },
}

_ML_DEFAULTS = {
    "message_level_effect": "neutral",
    "frustration_level_after_message": "none",
    "frustration_change": "unchanged",
    "customer_effort_level": "low",
    "clarity_level": "clear",
    "context_handling": "not_applicable",
    "issue_origin": "none",
    "issue_type": "none",
    "frustration_cause": "none",
    "evidence": "",
    "business_impact": "",
    "recommended_fix": "",
}
_MISSING_CONTRADICTION_DEBUG_MESSAGE = (
    "Contradiction flagged by the message-level evaluator, but no debug reason was returned."
)

_FORBIDDEN_ID_FIELDS = {
    "conversation_id",
    "thread_id",
    "run_id",
    "customer_id",
    "customer_phone",
    "phone_number",
    "target_message_id",
}

_CAREGIVER_PROMO_RE = re.compile(
    r"\b(certified\s+caregivers?|caregivers?\s+for\s+children|elderly\s+family|view\s+profiles|perfect\s+match)\b",
    re.IGNORECASE,
)
_CUSTOMER_CLOSING_RE = re.compile(
    r"\b(thanks?|thank\s+you|got\s+it|great|ok(?:ay)?|will\s+do|perfect|noted|appreciate)\b",
    re.IGNORECASE,
)
_ACTIVE_SENSITIVE_RE = re.compile(
    r"\b(cancel|refund|payment\s+fail|paid\s+already|overstay|abscond|expired|expiry|urgent|asap|"
    r"complain|complaint|escalat|legal|police|fine|blocked|not\s+working|issue\s+not\s+resolved)\b",
    re.IGNORECASE,
)
_AGENT_SELF_ADMISSION_RE = re.compile(
    r"\b("
    r"i\s+(?:am|was|were)?\s*(?:wrong|mistaken|at\s+fault)|"
    r"my\s+(?:mistake|fault|error)|"
    r"i\s+(?:gave|provided|shared|sent|told|said)\s+(?:you\s+)?(?:the\s+)?(?:wrong|incorrect|inaccurate)|"
    r"(?:we|our\s+team)\s+(?:made|caused)\s+(?:a\s+)?(?:mistake|error)|"
    r"(?:we|our\s+team)\s+(?:gave|provided|shared|sent|told|said)\s+(?:you\s+)?(?:the\s+)?(?:wrong|incorrect|inaccurate)|"
    r"(?:previous\s+agent|the\s+agent|our\s+agent).{0,80}(?:wrong|incorrect|mistake|error)"
    r")\b",
    re.IGNORECASE,
)
_AUTOMATION_MISTAKE_RE = re.compile(
    r"\b(?:automated|automatic|system|bot|broadcast|notification|sms|email|template|auto[-\s]?message)"
    r".{0,80}\b(?:wrong|incorrect|mistake|error)\b|"
    r"\b(?:wrong|incorrect|mistake|error)\b.{0,80}"
    r"\b(?:automated|automatic|system|bot|broadcast|notification|sms|email|template|auto[-\s]?message)\b",
    re.IGNORECASE,
)


def _last_customer_text(history_records: list[dict]) -> str:
    for record in reversed(history_records or []):
        role = str(record.get("sender_role") or record.get("raw_sender_role") or "").strip().lower()
        if role == "customer":
            return str(record.get("message_text") or "")
    return ""


def _is_low_risk_caregiver_promo(target_text: str, history_records: list[dict]) -> bool:
    if not _CAREGIVER_PROMO_RE.search(target_text or ""):
        return False

    last_customer = _last_customer_text(history_records)
    if _CUSTOMER_CLOSING_RE.search(last_customer):
        return True

    recent_customer_text = "\n".join(
        str(record.get("message_text") or "")
        for record in (history_records or [])[-8:]
        if str(record.get("sender_role") or "").strip().lower() == "customer"
    )
    return not _ACTIVE_SENSITIVE_RE.search(recent_customer_text)


def _downgrade_low_risk_caregiver_promo(result: dict, target_text: str, history_records: list[dict]) -> dict:
    if not _is_low_risk_caregiver_promo(target_text, history_records):
        return result
    last_customer = _last_customer_text(history_records)
    post_resolution = (not last_customer.strip()) or bool(_CUSTOMER_CLOSING_RE.search(last_customer))
    if result.get("message_level_effect") == "major_issue":
        result["message_level_effect"] = "neutral" if post_resolution else "minor_issue"
    if result.get("frustration_level_after_message") in {"medium", "high", "cancellation_risk"}:
        result["frustration_level_after_message"] = "none" if result.get("message_level_effect") == "neutral" else "low"
    if result.get("frustration_change") in {"created", "increased"}:
        result["frustration_change"] = "unchanged" if result.get("message_level_effect") == "neutral" else "created"
    if result.get("customer_effort_level") == "high":
        result["customer_effort_level"] = "low"
    if result.get("context_handling") == "poor":
        result["context_handling"] = "not_applicable" if result.get("message_level_effect") == "neutral" else "partial"
    if result.get("message_level_effect") == "neutral":
        result["issue_origin"] = "none"
        result["issue_type"] = "none"
        result["frustration_cause"] = "none"
        result["business_impact"] = "none"
        result["recommended_fix"] = "none"
    elif result.get("message_level_effect") == "minor_issue":
        result["issue_origin"] = "our_side"
        if result.get("issue_type") in {"ignored_context", "poor_tone", "dead_end", "missing_next_step", "delay", "wrong_info"}:
            result["issue_type"] = "other"
        result["frustration_cause"] = result.get("frustration_cause") or "irrelevant promotional message"
        result["business_impact"] = result.get("business_impact") or "Minor promotional noise in the support thread"
        result["recommended_fix"] = result.get("recommended_fix") or "Send promotional messages outside active support threads"
    elif result.get("issue_type") in {"ignored_context", "poor_tone", "dead_end", "missing_next_step", "delay"}:
        result["issue_type"] = "other"
    return result


def validate_message_level_result(data: dict) -> dict:
    """Coerce a parsed message-level JSON object into the strict schema shape.

    Well-known fields are normalized to the dashboard's expected enums. Any
    additional fields produced by a custom schema are preserved so that
    downstream consumers (Debug tab, JSON export) can still see them.
    """
    if not isinstance(data, dict):
        raise ValueError("Message-level result is not a JSON object")

    out: dict[str, Any] = {}
    try:
        out["message_index"] = int(data.get("message_index") or 0)
    except (TypeError, ValueError):
        out["message_index"] = 0

    for field_name, allowed in _ML_ENUMS.items():
        val = str(data.get(field_name, "") or "").strip().lower().replace(" ", "_")
        if val not in allowed:
            val = _ML_DEFAULTS[field_name]
        out[field_name] = val

    for field_name in ("frustration_cause", "evidence", "business_impact", "recommended_fix"):
        out[field_name] = str(data.get(field_name) or _ML_DEFAULTS[field_name]) or _ML_DEFAULTS[field_name]

    # Preserve any fields the user's custom schema produced.
    for k, v in data.items():
        if k not in out and k not in _FORBIDDEN_ID_FIELDS:
            out[k] = v

    return out


_CL_CLASSIFICATION_RULES = {
    "Handled with Minimal Issues": ("handled", "zero_minimal", False, False),
    "Handled with Many Issues": ("handled", "many", False, False),
    "Handled with Minimal Caused Issues": ("handled", "zero_minimal", False, True),
    "Handled with Many Caused Issues": ("handled", "many", False, True),
    "Handled with Minimal Issues and Frustration": ("handled", "zero_minimal", True, False),
    "Handled with Many Issues and Frustration": ("handled", "many", True, False),
    "Handled with Minimal Caused Issues and Frustration": ("handled", "zero_minimal", True, True),
    "Handled with Many Caused Issues and Frustration": ("handled", "many", True, True),
    "Not Handled with Minimal Issues": ("unhandled", "zero_minimal", False, False),
    "Not Handled with Many Issues": ("unhandled", "many", False, False),
    "Not Handled with Minimal Caused Issues": ("unhandled", "zero_minimal", False, True),
    "Not Handled with Many Caused Issues": ("unhandled", "many", False, True),
    "Not Handled with Minimal Issues and Frustration": ("unhandled", "zero_minimal", True, False),
    "Not Handled with Many Issues and Frustration": ("unhandled", "many", True, False),
    "Not Handled with Minimal Caused Issues and Frustration": ("unhandled", "zero_minimal", True, True),
    "Not Handled with Many Caused Issues and Frustration": ("unhandled", "many", True, True),
}

_OLD_CL_CLASSIFICATION_RULES = {
    "Handled with Zero/Minimal Issues": ("handled", "zero_minimal", "not_applicable"),
    "Handled with Many Issues": ("handled", "many", "not_applicable"),
    "Unhandled with Zero/Minimal Issues - Totally Definitive Unresolved": (
        "unhandled",
        "zero_minimal",
        "totally_unresolved",
    ),
    "Unhandled with Zero/Minimal Issues - Pending Unresolved": (
        "unhandled",
        "zero_minimal",
        "pending_unresolved",
    ),
    "Unhandled with Many Issues - Totally Definitive Unresolved": (
        "unhandled",
        "many",
        "totally_unresolved",
    ),
    "Unhandled with Many Issues - Pending Unresolved": (
        "unhandled",
        "many",
        "pending_unresolved",
    ),
}

_UNHANDLED_SUBTYPES = {
    "not_applicable",
    "totally_unresolved",
    "pending_unresolved",
}

_CUSTOMER_EXPERIENCES = {"good", "bad"}
_FRUSTRATION_ORIGINS = {"our_side", "customer_side", "shared", "none"}
_MAIN_ISSUE_ORIGINS = {"our_side", "customer_side", "shared", "third_party", "unclear", "none"}
_TOP_LEVEL_MAIN_ISSUE_ORIGINS = {"our_side", "customer_side", "shared", "third_party", "unclear", "none"}
_FRUSTRATION_TIMINGS = {"start", "during", "end", "multiple", "none"}
_CL_ISSUE_TYPES = {
    "none",
    "misunderstanding",
    "repetition",
    "delay",
    "unclear_guidance",
    "wrong_info",
    "ignored_context",
    "dead_end",
    "tool_or_system_failure",
    "poor_tone",
    "missing_next_step",
    "other",
}


def _normalize_bool_flag(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n", "", "none", "null"}:
        return False
    return default


def _normalize_message_entity(record: dict | None) -> str:
    if not isinstance(record, dict):
        return "unknown"
    raw_role = str(record.get("raw_sender_role") or "").strip().lower()
    if raw_role == "system":
        return "broadcast"
    if raw_role in {"bot", "assistant"}:
        return "bot"
    if raw_role == "agent":
        return "agent"
    role = str(record.get("sender_role") or "").strip().lower()
    if role in {"system", "broadcast"}:
        return "broadcast"
    if role in {"bot", "assistant"}:
        return "bot"
    if role in {"customer", "agent"}:
        return role
    return "unknown"


def _normalize_message_result_flags(result: dict) -> dict:
    parsed = result.get("parsed_json") or {}
    if not isinstance(parsed, dict):
        return result
    if "contradiction" not in parsed:
        parsed["contradiction"] = False
    else:
        parsed["contradiction"] = _normalize_bool_flag(parsed.get("contradiction"), default=False)
    first_id = str(parsed.get("first_contradiction_message_id") or "none").strip()
    parsed["first_contradiction_message_id"] = first_id or "none"
    debug_msg = str(parsed.get("contradiction_debug_message") or "").strip()
    if parsed["contradiction"]:
        parsed["contradiction_debug_message"] = (
            debug_msg
            or _MISSING_CONTRADICTION_DEBUG_MESSAGE
        )
    else:
        parsed["contradiction_debug_message"] = debug_msg or "none"
    result["parsed_json"] = parsed
    result["evaluation_output"] = parsed
    return result


def _set_contradiction_debug_if_missing(parsed: dict, message: str) -> None:
    current = str(parsed.get("contradiction_debug_message") or "").strip()
    if (
        not current
        or current.lower() == "none"
        or current == _MISSING_CONTRADICTION_DEBUG_MESSAGE
    ):
        parsed["contradiction_debug_message"] = message


def _agent_self_admitted_issue(record: dict | None) -> bool:
    text = str((record or {}).get("message_text") or "")
    if not text.strip():
        return False
    if not _AGENT_SELF_ADMISSION_RE.search(text):
        return False
    return not bool(_AUTOMATION_MISTAKE_RE.search(text))


def _clear_message_issue(
    parsed: dict,
    *,
    reason: str,
    contradiction_debug_message: str | None = None,
) -> None:
    parsed["message_level_effect"] = "helped"
    parsed["frustration_level_after_message"] = "none"
    parsed["frustration_change"] = "decreased"
    parsed["customer_effort_level"] = "low"
    parsed["clarity_level"] = "clear"
    parsed["context_handling"] = "good"
    parsed["issue_origin"] = "none"
    parsed["issue_type"] = "none"
    parsed["frustration_cause"] = "none"
    parsed["business_impact"] = reason
    parsed["recommended_fix"] = "none"
    parsed["contradiction_agent_issue_suppressed"] = True
    parsed["contradiction_debug_message"] = (
        contradiction_debug_message
        or "Human agent conflicted with an earlier bot/system/broadcast message, so the contradiction penalty was transferred to the automation source."
    )


def _mark_contradiction_source_issue(parsed: dict, *, target_id: str) -> None:
    parsed["conversation_id"] = parsed.get("conversation_id", "")
    parsed["target_message_id"] = parsed.get("target_message_id", target_id)
    parsed["message_level_effect"] = "major_issue"
    parsed["frustration_level_after_message"] = "high"
    parsed["frustration_change"] = "created"
    parsed["customer_effort_level"] = "high"
    parsed["clarity_level"] = "unclear"
    parsed["context_handling"] = "poor"
    parsed["issue_origin"] = "our_side"
    parsed["issue_type"] = "wrong_info"
    parsed["frustration_cause"] = "contradictory status"
    parsed["business_impact"] = "Original bot/system message created a contradictory customer-visible state later corrected by a human agent."
    parsed["recommended_fix"] = "Fix the original bot/system message so later human agents do not need to contradict it."
    parsed["contradiction"] = True
    parsed["first_contradiction_message_id"] = target_id
    parsed["contradiction_source_penalized"] = True
    parsed["contradiction_debug_message"] = (
        "This bot/system/broadcast message was identified as the automation source contradicted by a later human agent."
    )


def _build_contradiction_source_result(
    *,
    triggering_result: dict,
    source_record: dict,
    source_message_id: str,
) -> dict:
    conversation_id = str(
        triggering_result.get("conversation_id")
        or triggering_result.get("thread_id")
        or ""
    )
    message_index = source_record.get("message_index")
    parsed = {
        "conversation_id": conversation_id,
        "target_message_id": source_message_id,
        "message_index": message_index or 0,
        "message_level_effect": "major_issue",
        "frustration_level_after_message": "high",
        "frustration_change": "created",
        "customer_effort_level": "high",
        "clarity_level": "unclear",
        "context_handling": "poor",
        "issue_origin": "our_side",
        "issue_type": "wrong_info",
        "contradiction": True,
        "first_contradiction_message_id": source_message_id,
        "contradiction_debug_message": "This earlier bot/system/broadcast message was identified as the first source of the contradiction.",
        "frustration_cause": "contradictory status",
        "evidence": "Earlier bot/system message created a contradictory customer-visible state.",
        "business_impact": "Original bot/system message created a contradictory customer-visible state later corrected by a human agent.",
        "recommended_fix": "Fix the original bot/system message so later human agents do not need to contradict it.",
        "contradiction_source_penalized": True,
        "generated_from_contradiction_transfer": True,
    }
    return {
        "thread_id": conversation_id,
        "conversation_id": conversation_id,
        "target_message_id": source_message_id,
        "message_index": message_index,
        "appended_message_index": source_record.get("appended_message_index", message_index),
        "source_conversation_id": source_record.get("source_conversation_id"),
        "message_time": source_record.get("message_time", ""),
        "target_message_text": strip_inline_rag_context(source_record.get("message_text", "")),
        "input_history": None,
        "raw_model_response": None,
        "parsed_json": parsed,
        "evaluation_output": parsed,
        "parse_status": "ok",
        "error_message": None,
        "debug": {
            "generated_from_contradiction_transfer": True,
            "triggering_message_id": triggering_result.get("target_message_id"),
            "triggering_message_index": triggering_result.get("message_index"),
        },
    }


def _message_result_sort_key(result: dict) -> tuple[int, str]:
    idx = result.get("message_index")
    try:
        return (int(idx), str(result.get("target_message_id") or ""))
    except (TypeError, ValueError):
        return (10**12, str(result.get("target_message_id") or ""))


def apply_contradiction_source_suppression(
    message_results: list[dict],
    message_records: list[dict],
) -> list[dict]:
    """Move contradiction blame from later human-agent messages to the first bot/system source."""
    records_by_id: dict[str, dict] = {}
    records_by_idx: dict[Any, dict] = {}
    for record in message_records or []:
        message_id = str(record.get("message_id") or "").strip()
        if message_id:
            records_by_id[message_id] = record
        idx = record.get("message_index")
        if idx is not None:
            records_by_idx[idx] = record
            records_by_idx[str(idx)] = record
            try:
                records_by_idx[int(idx)] = record
            except (TypeError, ValueError):
                pass

    results_by_id: dict[str, dict] = {}
    for result in message_results or []:
        _normalize_message_result_flags(result)
        message_id = str(result.get("target_message_id") or "").strip()
        if message_id:
            results_by_id[message_id] = result

    source_results_to_add: list[dict] = []
    for result in list(message_results or []):
        if result.get("parse_status") != "ok":
            continue
        parsed = result.get("parsed_json") or {}
        if not parsed.get("contradiction"):
            continue
        first_id = str(parsed.get("first_contradiction_message_id") or "none").strip()
        if not first_id or first_id == "none":
            continue
        current_record = records_by_id.get(str(result.get("target_message_id") or ""))
        if current_record is None:
            current_record = records_by_idx.get(result.get("message_index"))
        first_record = records_by_id.get(first_id)
        if first_record is None:
            try:
                first_record = records_by_idx.get(int(first_id))
            except (TypeError, ValueError):
                first_record = records_by_idx.get(first_id)
        source_message_id = str(
            (first_record or {}).get("message_id")
            or first_id
        ).strip()
        current_entity = _normalize_message_entity(current_record)
        first_entity = _normalize_message_entity(first_record)
        if current_entity != "agent" or first_entity not in {"bot", "broadcast"}:
            continue

        if _agent_self_admitted_issue(current_record):
            parsed["contradiction_transfer_skipped"] = True
            _set_contradiction_debug_if_missing(
                parsed,
                "Transfer skipped because the human agent appears to admit an agent-side mistake, not merely correct bot/system/broadcast automation.",
            )
            result["parsed_json"] = parsed
            result["evaluation_output"] = parsed
            continue

        parsed["first_contradiction_message_id"] = source_message_id

        source_result = results_by_id.get(source_message_id) or results_by_id.get(first_id)
        if source_result:
            source_parsed = source_result.get("parsed_json") or {}
            if not isinstance(source_parsed, dict):
                source_parsed = {}
            source_parsed["conversation_id"] = (
                source_parsed.get("conversation_id")
                or source_result.get("conversation_id")
                or source_result.get("thread_id")
                or ""
            )
            source_parsed["target_message_id"] = (
                source_parsed.get("target_message_id")
                or source_result.get("target_message_id")
                or source_message_id
            )
            source_parsed["message_index"] = (
                source_parsed.get("message_index")
                if source_parsed.get("message_index") is not None
                else source_result.get("message_index")
            )
            _mark_contradiction_source_issue(source_parsed, target_id=source_message_id)
            source_result["parsed_json"] = source_parsed
            source_result["evaluation_output"] = source_parsed
            source_result["parse_status"] = "ok"
            source_result["error_message"] = None
        elif first_record:
            source_result = _build_contradiction_source_result(
                triggering_result=result,
                source_record=first_record,
                source_message_id=source_message_id,
            )
            results_by_id[source_message_id] = source_result
            source_results_to_add.append(source_result)

        _clear_message_issue(
            parsed,
            reason="Human agent message surfaced or corrected a contradiction created by an earlier bot/system message.",
            contradiction_debug_message=(
                f"Human agent was treated as source of truth against earlier automation message {source_message_id}; "
                "the agent issue was suppressed and the automation source was penalized."
            ),
        )
        result["parsed_json"] = parsed
        result["evaluation_output"] = parsed

    if source_results_to_add:
        message_results.extend(source_results_to_add)
        message_results.sort(key=_message_result_sort_key)

    return message_results


def _normalize_issue_origin(value: Any, *, allow_third_party: bool) -> str:
    origin = str(value or "").strip().lower().replace(" ", "_")
    if origin not in _MAIN_ISSUE_ORIGINS:
        return "none"
    if origin == "third_party" and not allow_third_party:
        return "unclear"
    return origin


def _normalize_sami_origin(value: Any) -> str:
    """Normalize an origin to the simplified Sami schema."""
    origin = str(value or "").strip().lower().replace(" ", "_")
    return origin if origin in _FRUSTRATION_ORIGINS else "none"


def _normalize_culprits(value: Any) -> list[str]:
    allowed = {"agent", "bot", "broadcast", "customer"}
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        culprit = str(item or "").strip().lower().replace(" ", "_").replace("-", "_")
        if culprit in allowed and culprit not in seen:
            out.append(culprit)
            seen.add(culprit)
    return out


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _normalize_unhandled_subtype(value: Any) -> str:
    subtype = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if subtype in {"n/a", "na", "none", "not_applicable"}:
        return "not_applicable"
    if subtype in {
        "totally_unresolved",
        "totally_definitive_unresolved",
        "definitive_unresolved",
        "definitive",
        "totally",
    }:
        return "totally_unresolved"
    if subtype in {"pending_unresolved", "pending"}:
        return "pending_unresolved"
    return ""


def _normalize_frustration_timing(value: Any) -> str:
    timing = str(value or "").strip().lower().replace(" ", "_")
    return timing if timing in _FRUSTRATION_TIMINGS else ""


def _infer_frustration_timing(started: bool, became: bool, ended: bool) -> str:
    active = [started, became, ended]
    if not any(active):
        return "none"
    if sum(1 for flag in active if flag) > 1:
        return "multiple"
    if started:
        return "start"
    if became:
        return "during"
    return "end"


def _classification_from_parts(
    handled_status: str,
    severity: str,
    frustration_detected: bool,
    main_issue_origin: str,
) -> str:
    if handled_status == "handled":
        if main_issue_origin == "our_side":
            if frustration_detected:
                return (
                    "Handled with Many Caused Issues and Frustration"
                    if severity == "many"
                    else "Handled with Minimal Caused Issues and Frustration"
                )
            return "Handled with Many Caused Issues" if severity == "many" else "Handled with Minimal Caused Issues"
        if not frustration_detected:
            return "Handled with Many Issues" if severity == "many" else "Handled with Minimal Issues"
        return (
            "Handled with Many Issues and Frustration"
            if severity == "many"
            else "Handled with Minimal Issues and Frustration"
        )
    if main_issue_origin == "our_side":
        if frustration_detected:
            return (
                "Not Handled with Many Caused Issues and Frustration"
                if severity == "many"
                else "Not Handled with Minimal Caused Issues and Frustration"
            )
        return "Not Handled with Many Caused Issues" if severity == "many" else "Not Handled with Minimal Caused Issues"
    if not frustration_detected:
        return "Not Handled with Many Issues" if severity == "many" else "Not Handled with Minimal Issues"
    return (
        "Not Handled with Many Issues and Frustration"
        if severity == "many"
        else "Not Handled with Minimal Issues and Frustration"
    )


def _score_number(value: Any, minimum: float, maximum: float) -> int | float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = 0.0
    num = float(max(minimum, min(maximum, num)))
    return int(num) if num.is_integer() else num


def _rating_from_score(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    if value > 10:
        value = value / 10.0
    if value >= 9:
        return "Excellent"
    if value >= 7.5:
        return "Good"
    if value >= 6:
        return "Fair"
    if value >= 4:
        return "Poor"
    return "Critical"


def _normalize_conversation_score(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}

    is_new_scale = any(
        key in value
        for key in ("ai_judgment_score", "message_signal_score", "final_score_100")
    )
    if is_new_scale:
        resolution = _score_number(value.get("resolution_score"), 0, 10)
        context = _score_number(value.get("context_understanding_score"), 0, 10)
        effort = _score_number(value.get("customer_effort_score"), 0, 10)
        frustration_risk = _score_number(
            value.get("trust_frustration_risk_score", value.get("frustration_risk_score")),
            0,
            10,
        )
        ai_judgment = _score_number(
            value.get(
                "ai_judgment_score",
                (float(resolution) + float(context) + float(effort) + float(frustration_risk)) / 4.0,
            ),
            0,
            10,
        )
        message_signal = _score_number(value.get("message_signal_score"), 0, 10)
        raw_total = _score_number(
            value.get("raw_total_score", (float(ai_judgment) + float(message_signal)) / 2.0),
            0,
            10,
        )
        final_score = _score_number(value.get("final_score", raw_total), 0, 10)
        final_score_100 = _score_number(value.get("final_score_100", float(final_score) * 10.0), 0, 100)
        display_max = 10
    else:
        resolution = _score_number(value.get("resolution_score"), 0, 20)
        context = _score_number(value.get("context_understanding_score"), 0, 20)
        effort = _score_number(value.get("customer_effort_score"), 0, 20)
        frustration_risk = _score_number(
            value.get("trust_frustration_risk_score", value.get("frustration_risk_score")),
            0,
            40,
        )
        ai_judgment = _score_number(
            value.get(
                "ai_judgment_score",
                (float(resolution) + float(context) + float(effort) + float(frustration_risk)) / 10.0,
            ),
            0,
            10,
        )
        message_signal = _score_number(value.get("message_signal_score", ai_judgment), 0, 10)
        raw_total = _score_number(
            value.get("raw_total_score", float(resolution) + float(context) + float(effort) + float(frustration_risk)),
            0,
            100,
        )
        final_score = _score_number(value.get("final_score", raw_total / 10.0), 0, 10)
        final_score_100 = _score_number(value.get("final_score_100", raw_total), 0, 100)
        display_max = 10

    rating = str(value.get("score_rating") or "").strip()
    if rating not in {"Excellent", "Good", "Fair", "Poor", "Critical"}:
        rating = _rating_from_score(final_score)

    return {
        "resolution_score": resolution,
        "context_understanding_score": context,
        "customer_effort_score": effort,
        "trust_frustration_risk_score": frustration_risk,
        "ai_judgment_score": ai_judgment,
        "message_signal_score": message_signal,
        "raw_total_score": raw_total,
        "final_score": final_score,
        "final_score_100": final_score_100,
        "display_max": display_max,
        "score_version": "v2" if is_new_scale else "legacy",
        "score_rating": rating,
        "score_explanation": str(value.get("score_explanation", "") or ""),
    }


def _apply_message_signal_to_conversation_score(parsed_json: dict, computed_metadata: dict | None) -> None:
    if not isinstance(parsed_json, dict):
        return
    existing = parsed_json.get("conversation_score") or {}
    score = _normalize_conversation_score(existing)
    if not score:
        score = _normalize_conversation_score(
            {
                "resolution_score": 0,
                "context_understanding_score": 0,
                "customer_effort_score": 0,
                "trust_frustration_risk_score": 0,
                "ai_judgment_score": 0,
                "message_signal_score": (computed_metadata or {}).get("message_signal_score", 0),
                "raw_total_score": 0,
                "final_score": 0,
                "final_score_100": 0,
                "score_explanation": "",
            }
        )
    ai_judgment = _score_number(
        score.get(
            "ai_judgment_score",
            (
                float(score.get("resolution_score", 0))
                + float(score.get("context_understanding_score", 0))
                + float(score.get("customer_effort_score", 0))
                + float(score.get("trust_frustration_risk_score", 0))
            )
            / 4.0,
        ),
        0,
        10,
    )
    message_signal = _score_number(
        (computed_metadata or {}).get("message_signal_score", score.get("message_signal_score", 0)),
        0,
        10,
    )
    raw_total = _score_number((float(ai_judgment) + float(message_signal)) / 2.0, 0, 10)
    final_score = raw_total
    # TEMPORARY GUARD — remove later:
    # If the message-level signal falls below 2/5 of the 0-10 scale, hard-zero
    # the final conversation score regardless of the model's conversation-level
    # judgment. 2/5 on a 0-10 scale is 4.0.
    if float(message_signal) < 4.0:
        raw_total = 0
        final_score = 0
        existing_explanation = str(score.get("score_explanation") or "").strip()
        guard_note = (
            "Temporary scoring guard applied: message-level signal was below "
            "2/5, so the final score was forced to 0."
        )
        score["score_explanation"] = (
            f"{existing_explanation} {guard_note}".strip()
            if existing_explanation
            else guard_note
        )
    score["ai_judgment_score"] = ai_judgment
    score["message_signal_score"] = message_signal
    score["raw_total_score"] = raw_total
    score["final_score"] = final_score
    score["final_score_100"] = _score_number(float(final_score) * 10.0, 0, 100)
    score["display_max"] = 10
    score["score_rating"] = _rating_from_score(final_score)
    parsed_json["conversation_score"] = score


def validate_conversation_level_result(data: dict) -> dict:
    """Coerce a parsed conversation-level JSON object into the Sami schema shape.

    Older runs may still contain the previous final_classification/cx_issue_severity
    fields. Those values are used only to infer the new markers when the new
    fields are absent; extra old fields are preserved for debug/export visibility.
    """
    if not isinstance(data, dict):
        raise ValueError("Conversation-level result is not a JSON object")

    out: dict[str, Any] = {}

    objective_type = str(data.get("customer_objective_type", "") or "").strip()
    if objective_type not in {"Inquiry", "Issue"}:
        objective_type = "Inquiry"
    out["customer_objective_type"] = objective_type
    out["customer_primary_objective"] = str(data.get("customer_primary_objective", "") or "")

    # Backward-compatible inference from old classification fields.
    old_classification = str(data.get("final_classification", "") or "").strip()
    old_severity = str(data.get("cx_issue_severity", "") or "").strip().lower().replace(" ", "_")
    handled_status = str(data.get("handled_status", "") or "").strip().lower()
    subtype = _normalize_unhandled_subtype(data.get("unhandled_resolution_subtype"))

    if old_classification in _CL_CLASSIFICATION_RULES:
        old_handled, old_severity_from_class, old_frustration, old_caused_by_us = _CL_CLASSIFICATION_RULES[old_classification]
        if handled_status not in {"handled", "unhandled"}:
            handled_status = old_handled
        if old_severity not in {"zero_minimal", "many"}:
            old_severity = old_severity_from_class
    elif old_classification in _OLD_CL_CLASSIFICATION_RULES:
        old_handled, old_severity_from_class, old_subtype = _OLD_CL_CLASSIFICATION_RULES[old_classification]
        if handled_status not in {"handled", "unhandled"}:
            handled_status = old_handled
        if old_severity not in {"zero_minimal", "many"}:
            old_severity = old_severity_from_class
        if subtype not in _UNHANDLED_SUBTYPES:
            subtype = old_subtype
    elif old_classification:
        if handled_status not in {"handled", "unhandled"}:
            handled_status = "handled" if old_classification.startswith("Handled") else "unhandled"
        if old_severity not in {"zero_minimal", "many"}:
            old_severity = "many" if "Many" in old_classification else "zero_minimal"

    if handled_status not in {"handled", "unhandled"}:
        handled_status = "handled"
    if handled_status == "handled":
        subtype = "not_applicable"
    elif subtype == "not_applicable" or subtype not in _UNHANDLED_SUBTYPES:
        subtype = "totally_unresolved"

    old_classification_l = old_classification.lower()
    legacy_bad_experience = old_severity == "many" or any(
        marker in old_classification_l for marker in ("many", "caused", "frustration")
    )
    customer_experience = str(data.get("customer_experience", "") or "").strip().lower()
    if customer_experience not in _CUSTOMER_EXPERIENCES or (
        customer_experience == "good" and legacy_bad_experience
    ):
        customer_experience = "bad" if legacy_bad_experience else "good"

    main = data.get("main_issue") or {}
    if not isinstance(main, dict):
        main = {}
    issue_type = str(main.get("issue_type", "none") or "none").strip().lower()
    if issue_type not in _CL_ISSUE_TYPES:
        issue_type = "other" if issue_type else "none"
    main_out = {
        "issue_exists": _normalize_bool_flag(main.get("issue_exists", False)),
        "issue_origin": _normalize_sami_origin(main.get("issue_origin", "none")),
        "issue_type": issue_type,
        "issue_summary": str(main.get("issue_summary", "") or ""),
        "customer_impact": str(main.get("customer_impact", "") or ""),
    }

    frustration_detected = _normalize_bool_flag(data.get("frustration_detected"), default=False)
    customer_started_frustrated = _normalize_bool_flag(data.get("customer_started_frustrated"), default=False)
    customer_became_frustrated_during_chat = _normalize_bool_flag(
        data.get("customer_became_frustrated_during_chat"),
        default=False,
    )
    customer_ended_frustrated = _normalize_bool_flag(data.get("customer_ended_frustrated"), default=False)
    frustration_timing = _normalize_frustration_timing(data.get("frustration_timing"))

    if old_classification in _CL_CLASSIFICATION_RULES and not frustration_detected:
        frustration_detected = _CL_CLASSIFICATION_RULES[old_classification][2]
    elif old_classification and "Frustration" in old_classification and not frustration_detected:
        frustration_detected = True

    frustration_origin = _normalize_sami_origin(data.get("frustration_origin", "none"))
    if frustration_detected and frustration_origin == "none":
        old_origin = _normalize_sami_origin(data.get("main_issue_origin", main_out["issue_origin"]))
        if old_origin != "none":
            frustration_origin = old_origin
        elif old_classification in _CL_CLASSIFICATION_RULES and _CL_CLASSIFICATION_RULES[old_classification][3]:
            frustration_origin = "our_side"

    if not frustration_detected:
        customer_started_frustrated = False
        customer_became_frustrated_during_chat = False
        customer_ended_frustrated = False
        frustration_timing = "none"
        frustration_origin = "none"
    else:
        if frustration_timing:
            if frustration_timing == "start":
                customer_started_frustrated = True
            elif frustration_timing == "during":
                customer_became_frustrated_during_chat = True
            elif frustration_timing == "end":
                customer_ended_frustrated = True
            elif frustration_timing == "multiple":
                customer_started_frustrated = True
                customer_became_frustrated_during_chat = True
                customer_ended_frustrated = True
        elif not any(
            [
                customer_started_frustrated,
                customer_became_frustrated_during_chat,
                customer_ended_frustrated,
            ]
        ):
            customer_became_frustrated_during_chat = True
        frustration_timing = _infer_frustration_timing(
            customer_started_frustrated,
            customer_became_frustrated_during_chat,
            customer_ended_frustrated,
        )

    out["handled_status"] = handled_status
    out["customer_experience"] = customer_experience
    out["unhandled_resolution_subtype"] = subtype
    out["frustration_detected"] = frustration_detected
    out["frustration_origin"] = frustration_origin
    out["customer_started_frustrated"] = customer_started_frustrated
    out["customer_became_frustrated_during_chat"] = customer_became_frustrated_during_chat
    out["customer_ended_frustrated"] = customer_ended_frustrated
    out["frustration_timing"] = frustration_timing

    sentiment = str(data.get("final_customer_sentiment", "") or "").strip().lower()
    if sentiment not in {"satisfied", "neutral", "frustrated", "confused", "dissatisfied", "unknown"}:
        sentiment = "unknown"
    out["final_customer_sentiment"] = sentiment

    max_fl = str(data.get("max_frustration_level", "") or "").strip().lower()
    if max_fl not in {"none", "low", "medium", "high", "cancellation_risk"}:
        max_fl = "none"
    out["max_frustration_level"] = max_fl

    conversation_score = _normalize_conversation_score(data.get("conversation_score"))
    if conversation_score:
        out["conversation_score"] = conversation_score

    if not main_out["issue_exists"]:
        main_out["issue_origin"] = "none"
        main_out["issue_type"] = "none"
        main_out["issue_summary"] = "none"
        main_out["customer_impact"] = "none"
    elif main_out["issue_origin"] == "none":
        main_out["issue_origin"] = _normalize_sami_origin(frustration_origin)
    out["main_issue"] = main_out

    detected = data.get("all_detected_issues") or []
    if not isinstance(detected, list):
        detected = []
    out["all_detected_issues"] = [
        {
            "issue_origin": _normalize_sami_origin(d.get("issue_origin", "")),
            "issue_type": (
                str(d.get("issue_type", "") or "").strip().lower()
                if str(d.get("issue_type", "") or "").strip().lower() in _CL_ISSUE_TYPES
                else "other"
            ),
            "issue_summary": str(d.get("issue_summary", "") or ""),
            "evidence": str(d.get("evidence", "") or ""),
            "impact": str(d.get("impact", "") or ""),
        }
        for d in detected
        if isinstance(d, dict)
    ]

    out["positive_signals"] = [str(x) for x in (data.get("positive_signals") or []) if x]
    out["negative_signals"] = [str(x) for x in (data.get("negative_signals") or []) if x]
    management_summary = str(data.get("management_summary", "") or "").strip()
    classification_reason = str(data.get("classification_reason", "") or "").strip()
    if classification_reason.lower() in {"", "none", "n/a", "na"}:
        classification_reason = management_summary or (
            f"Markers selected from handled_status={handled_status}, "
            f"customer_experience={customer_experience}, frustration_detected={frustration_detected}, "
            f"and frustration_origin={frustration_origin}."
        )
    out["classification_reason"] = classification_reason
    out["management_summary"] = management_summary
    out["culprits"] = _normalize_culprits(data.get("culprits"))
    out["culprit_agent_names"] = (
        _normalize_string_list(data.get("culprit_agent_names"))
        if "agent" in out["culprits"]
        else []
    )
    out["culprit_reason"] = str(data.get("culprit_reason", "") or "")
    out["recommended_actions"] = [str(x) for x in (data.get("recommended_actions") or []) if x]
    out["manual_review_required"] = _normalize_bool_flag(data.get("manual_review_required"), default=False)
    out["manual_review_reason"] = str(data.get("manual_review_reason", "") or "")
    if not out["manual_review_required"] and not out["manual_review_reason"].strip():
        out["manual_review_reason"] = "none"
    confidence = str(data.get("confidence", "") or "").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    out["confidence"] = confidence

    # Preserve any extra fields a custom schema may have introduced.
    for k, v in data.items():
        if k not in out and k not in _FORBIDDEN_ID_FIELDS:
            out[k] = v

    return out


# ---------- Run orchestration ----------


@dataclass
class RunConfig:
    api: APIConfig = field(default_factory=APIConfig)
    # Conversation-level calls may use a different model while reusing the
    # same endpoint, API key, and generation settings.
    conversation_api: Optional[APIConfig] = None
    # Ticket segmentation can use its own model while reusing the same endpoint
    # and API key.
    ticket_api: Optional[APIConfig] = None
    max_conversations: Optional[int] = None
    max_agent_messages_per_conv: Optional[int] = None
    truncate_messages: bool = False
    max_chars_per_message: int = 1500
    include_unknown_in_history: bool = True
    stop_on_error: bool = False
    save_raw_responses: bool = True
    # Which messages the judge evaluates as targets:
    #   "agent"    — judge the agent's response to a (possibly frustrated) customer message
    #   "customer" — judge the customer's state / frustration before the agent answers
    message_target_role: str = "agent"
    # Explicit set of IDs to run on. When non-None, takes
    # precedence over ``max_conversations`` — used by the random sampler.
    selected_conversation_ids: Optional[list[str]] = None
    enable_ticket_segmentation: bool = False
    ticket_segmentation_mode: str = TICKET_SEGMENTATION_MODE_DEFAULT
    # Editable prompts (defaults to the in-memory defaults; the app loads
    # the active prompts from the DB before each run).
    message_prompt: PromptTemplate = field(default_factory=lambda: DEFAULT_MESSAGE_LEVEL_PROMPT)
    conversation_prompt: PromptTemplate = field(default_factory=lambda: DEFAULT_CONVERSATION_LEVEL_PROMPT)
    ticket_prompt: PromptTemplate = field(default_factory=_default_ticket_segmentation_prompt)

    def conversation_api_config(self) -> APIConfig:
        return self.conversation_api or self.api

    def ticket_api_config(self) -> APIConfig:
        return self.ticket_api or self.conversation_api_config()


@dataclass
class RunResults:
    conversation_results: list[dict] = field(default_factory=list)
    message_level_results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: Optional[float] = None


def _eval_message_level(
    client,
    api: APIConfig,
    conversation_id: str,
    target_record: dict,
    history_records: list[dict],
    conversation_metadata: dict,
    save_raw: bool,
    truncate_chars: Optional[int],
    prompt: PromptTemplate,
) -> dict:
    """Run one message-level evaluation. Always returns a record (success or failure)."""
    payload = build_message_level_payload(
        conversation_id=conversation_id,
        target_message=target_record,
        history=history_records,
        conversation_metadata=conversation_metadata,
        truncate_chars=truncate_chars,
    )
    system_prompt = prompt.build_system()
    user_prompt = prompt.build_user(payload)

    record: dict[str, Any] = {
        "thread_id": conversation_id,
        "conversation_id": conversation_id,
        "target_message_id": target_record.get("message_id", ""),
        "message_index": target_record.get("message_index"),
        "appended_message_index": target_record.get("appended_message_index", target_record.get("message_index")),
        "source_conversation_id": target_record.get("source_conversation_id"),
        "message_time": target_record.get("message_time", ""),
        "target_message_text": strip_inline_rag_context(target_record.get("message_text", "")),
        "input_history": history_records if save_raw else None,
        "raw_model_response": None,
        "parsed_json": None,
        "evaluation_output": None,
        "parse_status": "ok",
        "error_message": None,
        "debug": None,
    }

    max_evaluation_attempts = max(1, int(api.retries) + 1)
    attempt_history: list[dict[str, Any]] = []
    final_debug: dict[str, Any] = {}

    for evaluation_attempt in range(1, max_evaluation_attempts + 1):
        record["raw_model_response"] = None
        record["parsed_json"] = None
        record["evaluation_output"] = None
        record["parse_status"] = "ok"
        record["error_message"] = None
        raw = ""
        call_debug: dict[str, Any] = {}
        try:
            raw, call_debug = chat_completion(
                client,
                api,
                system_prompt,
                user_prompt,
                context=f"message_level:{conversation_id}#{target_record.get('message_index')}",
            )
            if save_raw:
                record["raw_model_response"] = raw
            try:
                obj = extract_json_object(raw)
                validated = validate_message_level_result(obj)
                validated = _downgrade_low_risk_caregiver_promo(
                    validated,
                    str(target_record.get("message_text") or ""),
                    history_records,
                )
                if not validated.get("message_index") and record["message_index"] is not None:
                    try:
                        validated["message_index"] = int(record["message_index"])
                    except (TypeError, ValueError):
                        pass
                record["parsed_json"] = validated
                record["evaluation_output"] = validated
            except Exception as je:
                record["parse_status"] = "failed"
                record["error_message"] = f"JSON parse failed: {je}"
        except Exception as e:
            record["parse_status"] = "api_error"
            record["error_message"] = f"API call failed: {e}"

        final_debug = dict(call_debug or {})
        attempt_history.append(
            {
                "evaluation_attempt": evaluation_attempt,
                "status": record["parse_status"],
                "error": record["error_message"],
                "api_attempts": call_debug.get("attempts"),
                "usage": call_debug.get("usage"),
            }
        )
        if record["parse_status"] == "ok":
            break
        # API exceptions already use APIConfig.retries inside chat_completion.
        # Only response/JSON failures need a fresh evaluation request here.
        if record["parse_status"] == "api_error":
            break

    automatic_reruns = max(0, len(attempt_history) - 1)
    recovered_after_rerun = (
        automatic_reruns > 0 and record.get("parse_status") == "ok"
    )
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    has_usage = False
    for attempt in attempt_history:
        usage = attempt.get("usage") or {}
        if not usage:
            continue
        has_usage = True
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                total_usage[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass
        details = usage.get("completion_tokens_details") or {}
        try:
            total_usage["completion_tokens_details"]["reasoning_tokens"] += int(
                details.get("reasoning_tokens") or 0
            )
        except (TypeError, ValueError):
            pass

    final_debug.update(
        {
            "evaluation_attempts": len(attempt_history),
            "automatic_reruns": automatic_reruns,
            "recovered_after_rerun": recovered_after_rerun,
            "evaluation_attempt_history": attempt_history,
        }
    )
    if has_usage:
        final_debug["total_usage_including_reruns"] = total_usage
    record["debug"] = final_debug
    record["automatic_reruns"] = automatic_reruns
    record["recovered_after_rerun"] = recovered_after_rerun
    record["rerun_errors"] = [
        attempt["error"] for attempt in attempt_history if attempt.get("error")
    ]
    _normalize_message_result_flags(record)

    return record


def _eval_conversation_level(
    client,
    api: APIConfig,
    conversation_id: str,
    conversation_metadata: dict,
    full_transcript: list[dict],
    message_level_evaluations: list[dict],
    computed_metadata: dict,
    save_raw: bool,
    truncate_chars: Optional[int],
    prompt: PromptTemplate,
) -> dict:
    """Run one conversation-level evaluation. Always returns a record."""
    payload = build_conversation_level_payload(
        conversation_id=conversation_id,
        conversation_metadata=conversation_metadata,
        full_transcript=full_transcript,
        message_level_evaluations=[
            e["parsed_json"] for e in message_level_evaluations if e.get("parsed_json")
        ],
        computed_metadata=computed_metadata,
        truncate_chars=truncate_chars,
    )
    system_prompt = prompt.build_system()
    user_prompt = prompt.build_user(payload)

    record: dict[str, Any] = {
        "thread_id": conversation_id,
        "conversation_id": conversation_id,
        "run_id": None,
        "raw_model_response": None,
        "parsed_json": None,
        "evaluation_output": None,
        "parse_status": "ok",
        "error_message": None,
        "debug": None,
    }

    max_evaluation_attempts = max(1, int(api.retries) + 1)
    attempt_history: list[dict[str, Any]] = []
    final_debug: dict[str, Any] = {}

    for evaluation_attempt in range(1, max_evaluation_attempts + 1):
        record["raw_model_response"] = None
        record["parsed_json"] = None
        record["evaluation_output"] = None
        record["parse_status"] = "ok"
        record["error_message"] = None
        raw = ""
        call_debug: dict[str, Any] = {}
        try:
            raw, call_debug = chat_completion(
                client,
                api,
                system_prompt,
                user_prompt,
                context=f"conversation_level:{conversation_id}",
            )
            if save_raw:
                record["raw_model_response"] = raw
            try:
                obj = extract_json_object(raw)
                validated = validate_conversation_level_result(obj)
                record["parsed_json"] = validated
                record["evaluation_output"] = validated
            except Exception as je:
                record["parse_status"] = "failed"
                record["error_message"] = f"JSON parse failed: {je}"
        except Exception as e:
            record["parse_status"] = "api_error"
            record["error_message"] = f"API call failed: {e}"

        final_debug = dict(call_debug or {})
        attempt_history.append(
            {
                "evaluation_attempt": evaluation_attempt,
                "status": record["parse_status"],
                "error": record["error_message"],
                "api_attempts": call_debug.get("attempts"),
                "usage": call_debug.get("usage"),
            }
        )
        if record["parse_status"] == "ok":
            break
        # API exceptions already use APIConfig.retries inside chat_completion.
        # Only response/JSON failures need a fresh evaluation request here.
        if record["parse_status"] == "api_error":
            break

    automatic_reruns = max(0, len(attempt_history) - 1)
    recovered_after_rerun = (
        automatic_reruns > 0 and record.get("parse_status") == "ok"
    )
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    has_usage = False
    for attempt in attempt_history:
        usage = attempt.get("usage") or {}
        if not usage:
            continue
        has_usage = True
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                total_usage[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass
        details = usage.get("completion_tokens_details") or {}
        try:
            total_usage["completion_tokens_details"]["reasoning_tokens"] += int(
                details.get("reasoning_tokens") or 0
            )
        except (TypeError, ValueError):
            pass

    final_debug.update(
        {
            "evaluation_attempts": len(attempt_history),
            "automatic_reruns": automatic_reruns,
            "recovered_after_rerun": recovered_after_rerun,
            "evaluation_attempt_history": attempt_history,
        }
    )
    if has_usage:
        final_debug["total_usage_including_reruns"] = total_usage
    record["debug"] = final_debug
    record["automatic_reruns"] = automatic_reruns
    record["recovered_after_rerun"] = recovered_after_rerun
    record["rerun_errors"] = [
        attempt["error"] for attempt in attempt_history if attempt.get("error")
    ]

    return record


def _ticket_source_record_blocks(records: list[dict]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None
    for record in records:
        source_conversation_id = str(record.get("source_conversation_id") or "unknown").strip() or "unknown"
        if current_block is None or current_block["source_conversation_id"] != source_conversation_id:
            current_block = {
                "source_conversation_id": source_conversation_id,
                "records": [],
            }
            blocks.append(current_block)
        current_block["records"].append(record)
    return blocks


def ticket_segmentation_call_count_for_records(
    records: list[dict],
    segmentation_mode: Any = TICKET_SEGMENTATION_MODE_DEFAULT,
) -> int:
    mode = clean_ticket_segmentation_mode(segmentation_mode)
    if mode == TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE:
        return max(1, len(_ticket_source_record_blocks(records)))
    return 1


def _ticket_segmentation_payload(
    conversation_id: str,
    records: list[dict],
    conversation_metadata: dict,
    truncate_chars: Optional[int],
    segmentation_context: dict | None = None,
) -> dict:
    source_conversation_blocks: list[dict] = []
    for block in _ticket_source_record_blocks(records):
        messages: list[dict] = []
        for record in block.get("records") or []:
            text = str(record.get("message_text") or "")
            if truncate_chars and len(text) > truncate_chars:
                text = text[:truncate_chars] + "..."
            messages.append(
                {
                    "message_index": record.get("message_index"),
                    "time": record.get("message_time"),
                    "role": record.get("sender_role"),
                    "text": text,
                }
            )
        source_conversation_blocks.append(
            {
                "source_conversation_id": block.get("source_conversation_id") or "unknown",
                "messages": messages,
            }
        )
    payload = {
        "conversation_id": conversation_id,
        "conversation_metadata": conversation_metadata,
        "input_format": "Messages are grouped under source_conversation_blocks. The source_conversation_id is the block header for the messages inside it.",
        "message_index_rule": "message_index is continuous across the full customer journey and does not reset inside source_conversation_blocks.",
        "source_conversation_blocks": source_conversation_blocks,
    }
    if segmentation_context:
        payload["segmentation_context"] = segmentation_context
    return payload


class TicketSegmentationError(RuntimeError):
    """Ticket segmentation could not produce a real result.

    Raised instead of inventing a ticket. A previous version of this module
    answered every failure -- connection errors, context-window overflow, an
    unparseable response -- with a synthetic ticket_category="inquiry",
    ticket_type="general_inquiry", customer_objective="Original unsplit
    customer journey" ticket. That is indistinguishable in the UI from a real
    classification, so a dead API connection looked like a judgement the model
    had made, and 51 of them accumulated in the logs unnoticed. A failure must
    surface as a failure.
    """


_TICKET_STATUSES = {"resolved", "pending_unresolved", "totally_unresolved"}
_TICKET_CATEGORIES = {"issue", "request", "inquiry"}
_TICKET_REQUEST_ORIGINS = {"company", "customer"}


def _clean_ticket_status(value: Any, default: str = "pending_unresolved") -> str:
    status = str(value or "").strip().lower()
    return status if status in _TICKET_STATUSES else default


def _clean_ticket_category(value: Any, default: str = "") -> str:
    category = str(value or "").strip().lower()
    return category if category in _TICKET_CATEGORIES else default


def _clean_ticket_request_origin(value: Any, default: str = "customer") -> str:
    """Which side opened/triggered the ticket's underlying thread.

    Independent of ticket_category: a company-sent reminder or broadcast that
    the customer later disputes is request_origin=company with
    ticket_category=issue, since the category still reflects the customer's
    own objective (the dispute), not who spoke first.
    """
    origin = str(value or "").strip().lower()
    return origin if origin in _TICKET_REQUEST_ORIGINS else default


def _normalize_ticket_inquiries(ticket: dict, valid_indexes: set[int]) -> list[dict]:
    raw_inquiries = ticket.get("inquiries")
    if not isinstance(raw_inquiries, list):
        return []

    normalized: list[dict] = []
    for idx, inquiry in enumerate(raw_inquiries, start=1):
        if not isinstance(inquiry, dict):
            continue
        message_indexes: list[int] = []
        raw_indexes = inquiry.get("message_indexes")
        if isinstance(raw_indexes, list):
            for value in raw_indexes:
                try:
                    message_index = int(value)
                except (TypeError, ValueError):
                    continue
                if message_index in valid_indexes and message_index not in message_indexes:
                    message_indexes.append(message_index)

        status = _clean_ticket_status(inquiry.get("status"))

        question = str(inquiry.get("question") or "").strip()
        if not question and not message_indexes:
            continue

        normalized.append(
            {
                "inquiry_id": str(inquiry.get("inquiry_id") or f"inquiry_{idx}").strip() or f"inquiry_{idx}",
                "question": question,
                "message_indexes": sorted(message_indexes),
                "status": status,
                "answer_summary": str(inquiry.get("answer_summary") or "").strip(),
                "unresolved_reason": str(inquiry.get("unresolved_reason") or "").strip() or "none",
            }
        )
    return normalized


_TICKET_WORKER_NAME_CONTEXT_RE = re.compile(
    r"\b([A-Z][A-Za-z'-]{1,}(?:\s+[A-Z][A-Za-z'-]{1,}){0,2})"
    r"(?=(?:['\u2019]s)?\s+(?:visa|residency|medical|emirates\s+id|eid|passport|salary|contract|onboarding)\b)"
)
_TICKET_ROLE_NAME_RE = re.compile(
    r"\b(?:maid|nanny|worker|helper)\s+"
    r"([A-Z][A-Za-z'-]{1,}(?:\s+[A-Z][A-Za-z'-]{1,}){0,2})"
    r"(?=\s*(?:$|[.,;:!?)]|(?:['\u2019]s)?\s+(?:visa|residency|medical|emirates\s+id|eid|passport|salary|contract|onboarding)\b))"
)
_TICKET_NAME_STOPWORDS = {
    "customer",
    "dear",
    "domestic",
    "eid",
    "emirates",
    "gdrfa",
    "helper",
    "icp",
    "maid",
    "medical",
    "nanny",
    "residency",
    "uae",
    "visa",
    "worker",
}


def _ticket_worker_name_candidates(ticket: dict, records_by_idx: dict[int, dict]) -> list[str]:
    texts = [str(ticket.get("customer_objective") or "")]
    for record in _ticket_records(ticket, records_by_idx):
        texts.append(str(record.get("message_text") or ""))
    candidates: set[str] = set()
    for text in texts:
        for pattern in (_TICKET_WORKER_NAME_CONTEXT_RE, _TICKET_ROLE_NAME_RE):
            for match in pattern.finditer(text):
                candidate = re.sub(r"['\u2019]s$", "", match.group(1).strip(), flags=re.IGNORECASE)
                words = [word.lower() for word in re.findall(r"[A-Za-z]+", candidate)]
                if words and not all(word in _TICKET_NAME_STOPWORDS for word in words):
                    candidates.add(candidate)
    return sorted(candidates, key=len, reverse=True)


_TICKET_ROLE_TOKEN_RUN_RE = re.compile(
    r"\b(?:the\s+)?(?:maid|nanny|worker|helper)"
    r"(?:\s+(?:the\s+)?(?:maid|nanny|worker|helper))+"
    r"(['\u2019]s)?",
    flags=re.IGNORECASE,
)


def _collapse_role_token_runs(text: str) -> str:
    """Collapse ``the maid maid's`` / ``maid the maid`` runs into one role token.

    These runs come from stacked name substitution here and from inconsistently
    anonymized source transcripts, so normalize them regardless of origin.
    """
    return _TICKET_ROLE_TOKEN_RUN_RE.sub(
        lambda match: "the maid's" if match.group(1) else "the maid",
        text,
    )


def _sanitize_ticket_customer_objective(ticket: dict, records_by_idx: dict[int, dict]) -> str:
    objective = re.sub(r"\s+", " ", str(ticket.get("customer_objective") or "")).strip()
    for name in _ticket_worker_name_candidates(ticket, records_by_idx):
        escaped = re.escape(name)
        objective = re.sub(
            rf"\s+for\s+(?:the\s+)?(?:maid|nanny|worker|helper)\s+{escaped}\b",
            "",
            objective,
            flags=re.IGNORECASE,
        )
        objective = re.sub(rf"\b{escaped}['\u2019]s\b", "the maid's", objective, flags=re.IGNORECASE)
        objective = re.sub(rf"\b{escaped}\b", "the maid", objective, flags=re.IGNORECASE)
    objective = _collapse_role_token_runs(objective)
    return re.sub(r"\s+", " ", objective).strip(" ,.;:-")


def _sanitize_ticket_type_names(ticket: dict, records_by_idx: dict[int, dict]) -> str:
    ticket_type = re.sub(r"[^a-z0-9_]+", "_", str(ticket.get("ticket_type") or "other").strip().lower())
    for name in _ticket_worker_name_candidates(ticket, records_by_idx):
        name_token = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if name_token:
            ticket_type = re.sub(rf"(?:^|_){re.escape(name_token)}(?=_|$)", "_", ticket_type)
    return re.sub(r"_+", "_", ticket_type).strip("_") or "other"


def _ticket_records(ticket: dict, records_by_idx: dict[int, dict]) -> list[dict]:
    records: list[dict] = []
    for value in ticket.get("included_message_indexes") or []:
        try:
            record = records_by_idx.get(int(value))
        except (TypeError, ValueError):
            record = None
        if record:
            records.append(record)
    return records


def _records_by_message_index(records: list[dict]) -> dict[int, dict]:
    return {
        int(record["message_index"]): record
        for record in records
        if record.get("message_index") is not None
    }


def _ticket_index_list(ticket: dict) -> list[int]:
    indexes: list[int] = []
    for value in ticket.get("included_message_indexes") or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in indexes:
            indexes.append(index)
    return sorted(indexes)


def _ticket_append_future_for_status(status: str) -> bool:
    return _clean_ticket_status(status) == "pending_unresolved"


def _renumber_tickets(tickets: list[dict]) -> list[dict]:
    id_map: dict[str, str] = {}
    for idx, ticket in enumerate(tickets, start=1):
        old_id = str(ticket.get("ticket_id") or "").strip()
        new_id = f"ticket_{idx}"
        if old_id:
            id_map[old_id] = new_id
        ticket["ticket_id"] = new_id
    valid_ticket_ids = {str(ticket.get("ticket_id") or "") for ticket in tickets}
    for ticket in tickets:
        previous_id = str(ticket.get("previous_ticket_id") or "").strip()
        if previous_id in id_map:
            ticket["previous_ticket_id"] = id_map[previous_id]
            previous_id = str(ticket.get("previous_ticket_id") or "").strip()
        if previous_id == str(ticket.get("ticket_id") or "") or (
            previous_id and previous_id not in valid_ticket_ids
        ):
            ticket["previous_ticket_id"] = ""
    return tickets


def _copy_inquiry_for_indexes(inquiry: dict, included_indexes: set[int], inquiry_number: int) -> dict | None:
    message_indexes: list[int] = []
    for value in inquiry.get("message_indexes") or []:
        try:
            message_index = int(value)
        except (TypeError, ValueError):
            continue
        if message_index in included_indexes and message_index not in message_indexes:
            message_indexes.append(message_index)
    if not message_indexes:
        return None
    return {
        **inquiry,
        "inquiry_id": f"inquiry_{inquiry_number}",
        "message_indexes": sorted(message_indexes),
        "status": _clean_ticket_status(inquiry.get("status")),
    }


def _filter_inquiries_for_indexes(ticket: dict, included_indexes: list[int]) -> list[dict]:
    included_set = {int(value) for value in included_indexes}
    filtered: list[dict] = []
    for inquiry in ticket.get("inquiries") or []:
        if not isinstance(inquiry, dict):
            continue
        copied = _copy_inquiry_for_indexes(inquiry, included_set, len(filtered) + 1)
        if copied is not None:
            filtered.append(copied)
    return filtered


def _dedupe_ticket_conversation_summaries(summaries: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    by_source: dict[str, dict] = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        source_id = str(summary.get("source_conversation_id") or "").strip()
        if not source_id:
            continue
        indexes = sorted(
            {
                int(value)
                for value in summary.get("message_indexes") or []
                if str(value).strip().lstrip("-").isdigit()
            }
        )
        if not indexes:
            continue
        cleaned = {
            "source_conversation_id": source_id,
            "message_indexes": indexes,
            "customer_intent": str(summary.get("customer_intent") or "").strip(),
            "outcome": str(summary.get("outcome") or "").strip(),
            "status": _clean_ticket_status(summary.get("status")),
            "ticket_signals": list(
                dict.fromkeys(
                    str(value).strip().lower()
                    for value in summary.get("ticket_signals") or []
                    if str(value).strip()
                )
            ),
        }
        existing = by_source.get(source_id)
        if existing is None:
            by_source[source_id] = cleaned
            deduped.append(cleaned)
            continue
        existing["message_indexes"] = sorted(set(existing["message_indexes"]) | set(indexes))
        if cleaned["customer_intent"]:
            existing["customer_intent"] = cleaned["customer_intent"]
        if cleaned["outcome"]:
            existing["outcome"] = cleaned["outcome"]
        existing["status"] = cleaned["status"]
        existing["ticket_signals"] = list(
            dict.fromkeys([*existing.get("ticket_signals", []), *cleaned["ticket_signals"]])
        )
    return deduped


def _normalize_ticket_conversation_summaries(
    ticket: dict,
    included_indexes: list[int],
    records_by_idx: dict[int, dict],
) -> list[dict]:
    included_set = {int(value) for value in included_indexes}
    normalized: list[dict] = []
    for summary in ticket.get("conversation_summaries") or []:
        if not isinstance(summary, dict):
            continue
        grouped_indexes: dict[str, list[int]] = {}
        for value in summary.get("message_indexes") or []:
            try:
                message_index = int(value)
            except (TypeError, ValueError):
                continue
            if message_index not in included_set:
                continue
            record = records_by_idx.get(message_index) or {}
            source_id = str(record.get("source_conversation_id") or "").strip()
            if not source_id:
                source_id = str(summary.get("source_conversation_id") or "").strip()
            if source_id:
                grouped_indexes.setdefault(source_id, []).append(message_index)
        for source_id, message_indexes in grouped_indexes.items():
            normalized.append(
                {
                    **summary,
                    "source_conversation_id": source_id,
                    "message_indexes": sorted(set(message_indexes)),
                }
            )
    return _dedupe_ticket_conversation_summaries(normalized)


def _filter_conversation_summaries_for_indexes(ticket: dict, included_indexes: list[int]) -> list[dict]:
    included_set = {int(value) for value in included_indexes}
    filtered: list[dict] = []
    for summary in ticket.get("conversation_summaries") or []:
        if not isinstance(summary, dict):
            continue
        message_indexes = sorted(
            {
                int(value)
                for value in summary.get("message_indexes") or []
                if str(value).strip().lstrip("-").isdigit() and int(value) in included_set
            }
        )
        if message_indexes:
            filtered.append({**summary, "message_indexes": message_indexes})
    return _dedupe_ticket_conversation_summaries(filtered)


def _normalize_ticket_segments(obj: dict, records: list[dict]) -> list[dict]:
    """Normalize raw LLM ticket JSON into the canonical ticket list.

    ``records`` scopes index validity (``valid_indexes``) and lets a
    message_index be resolved back to its source record; it must stay the full
    cumulative history in cumulative-segmentation mode so carried-forward
    tickets referencing older indexes remain resolvable.
    """
    valid_indexes = {
        int(record["message_index"])
        for record in records
        if record.get("message_index") is not None
    }
    if not valid_indexes:
        raise TicketSegmentationError("No usable message indexes were available.")
    records_by_idx = _records_by_message_index(records)

    raw_tickets = obj.get("tickets") if isinstance(obj, dict) else None
    if not isinstance(raw_tickets, list) or not raw_tickets:
        raise TicketSegmentationError("Ticket segmentation returned no tickets.")

    normalized: list[dict] = []
    for idx, ticket in enumerate(raw_tickets, start=1):
        if not isinstance(ticket, dict):
            continue
        included: list[int] = []
        raw_included = ticket.get("included_message_indexes")
        if isinstance(raw_included, list):
            for value in raw_included:
                try:
                    message_index = int(value)
                except (TypeError, ValueError):
                    continue
                if message_index in valid_indexes and message_index not in included:
                    included.append(message_index)

        if not included:
            try:
                start = int(ticket.get("start_message_index"))
                end = int(ticket.get("end_message_index"))
                low, high = min(start, end), max(start, end)
                included = sorted(i for i in valid_indexes if low <= i <= high)
            except (TypeError, ValueError):
                included = []
        if not included:
            continue

        status = _clean_ticket_status(ticket.get("status"))

        ticket_type = re.sub(r"[^a-z0-9_]+", "_", str(ticket.get("ticket_type") or "other").strip().lower())
        ticket_type = re.sub(r"_+", "_", ticket_type).strip("_") or "other"
        included = sorted(included)
        normalized_ticket = {
                "ticket_id": str(ticket.get("ticket_id") or f"ticket_{idx}").strip() or f"ticket_{idx}",
                "ticket_category": _clean_ticket_category(ticket.get("ticket_category")),
                "model_ticket_category": _clean_ticket_category(ticket.get("ticket_category")),
                "request_origin": _clean_ticket_request_origin(ticket.get("request_origin")),
                "ticket_type": ticket_type,
                "customer_objective": str(ticket.get("customer_objective") or "").strip(),
                "start_message_index": min(included),
                "end_message_index": max(included),
                "included_message_indexes": included,
                "status": status,
                "should_append_future_conversations": _ticket_append_future_for_status(status),
                "previous_ticket_id": str(ticket.get("previous_ticket_id") or "").strip(),
                "inquiries": _normalize_ticket_inquiries(ticket, valid_indexes),
                "conversation_summaries": _normalize_ticket_conversation_summaries(
                    ticket,
                    included,
                    records_by_idx,
                ),
                "segmentation_reason": str(ticket.get("segmentation_reason") or "").strip(),
        }
        normalized.append(normalized_ticket)

    if not normalized:
        raise TicketSegmentationError("Ticket segmentation produced only invalid tickets.")

    # Segmentation decisions belong to the prompt, not to this module.
    #
    # This function used to run ~35 heuristic passes here (four splitters and
    # roughly thirty merge calls) that re-decided which messages formed a
    # ticket and overrode the model's ticket_category and status, plus a
    # coverage-ratio check that threw away the model's entire ticket list when
    # it covered less than 75% of the "material" messages. That gave the
    # pipeline two competing sources of truth, and the heuristics silently won.
    #
    # They were keyed on generic vocabulary, so they collapsed correct output.
    # Verified example: the model correctly returned a foot-x-ray insurance
    # ticket and a separate maid-replacement inquiry. Both were classified as
    # "visa lifecycle" tickets -- one because an agent's boilerplate mentioned
    # a "discounted maid visa plan", the other on clinic/medical wording -- and
    # merged into one ticket because a followup regex matched the word "fees"
    # in a price list. The prompt said to keep them apart; the code did not.
    #
    # Everything below is plumbing only: assign sequential ids and order the
    # list. It must not add, remove, merge, split, or re-label a ticket. If
    # segmentation is wrong, fix the prompt.
    # The one exception is redaction, which is a privacy control rather than a
    # segmentation decision: a worker's real name must never reach a ticket_type
    # or customer_objective. These strip names only; they never change which
    # messages belong to a ticket, nor its category or status.
    normalized = _renumber_tickets(
        sorted(normalized, key=lambda ticket: int(ticket.get("start_message_index") or 0))
    )
    for ticket in normalized:
        ticket["customer_objective"] = _sanitize_ticket_customer_objective(ticket, records_by_idx)
        ticket["ticket_type"] = _sanitize_ticket_type_names(ticket, records_by_idx)
        ticket["conversation_summaries"] = _filter_conversation_summaries_for_indexes(
            ticket,
            _ticket_index_list(ticket),
        )
    return normalized


def _ticket_prompt_context(tickets: list[dict], records: list[dict] | None = None) -> dict:
    compact_tickets: list[dict] = []
    keep_keys = (
        "ticket_id",
        "ticket_category",
        "request_origin",
        "ticket_type",
        "customer_objective",
        "start_message_index",
        "end_message_index",
        "included_message_indexes",
        "status",
        "should_append_future_conversations",
        "previous_ticket_id",
        "inquiries",
        "conversation_summaries",
        "segmentation_reason",
    )
    records_by_idx = _records_by_message_index(records or [])
    for ticket in tickets or []:
        if not isinstance(ticket, dict):
            continue
        compact = {key: ticket.get(key) for key in keep_keys if key in ticket}
        source_ids: list[str] = []
        for value in ticket.get("included_message_indexes") or []:
            try:
                record = records_by_idx.get(int(value))
            except (TypeError, ValueError):
                record = None
            source_id = str((record or {}).get("source_conversation_id") or "").strip()
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
        if source_ids:
            compact["source_conversation_ids"] = source_ids
        compact_tickets.append(compact)
    return {"tickets": compact_tickets}


def _eval_ticket_segmentation_once(
    client,
    api: APIConfig,
    conversation_id: str,
    records: list[dict],
    conversation_metadata: dict,
    truncate_chars: Optional[int],
    ticket_prompt: PromptTemplate | None = None,
    segmentation_context: dict | None = None,
    normalization_records: list[dict] | None = None,
) -> tuple[list[dict], dict | None]:
    payload = _ticket_segmentation_payload(
        conversation_id,
        records,
        conversation_metadata,
        truncate_chars,
        segmentation_context,
    )
    tpl = ticket_prompt or _default_ticket_segmentation_prompt()
    call_context = f"ticket_segmentation:{conversation_id}"
    if segmentation_context:
        pass_label = f"{segmentation_context.get('pass_index')}/{segmentation_context.get('total_passes')}"
        source_label = segmentation_context.get("current_source_conversation_id")
        call_context = f"{call_context}:pass{pass_label}:{source_label}"
    raw, debug = chat_completion(
        client,
        api,
        tpl.build_system(),
        tpl.build_user(payload),
        context=call_context,
    )
    obj = extract_json_object(raw)
    return _normalize_ticket_segments(obj, normalization_records or records), {
        "raw_model_response": raw,
        "debug": debug,
    }


def _carry_forward_missing_previous_tickets(
    previous_tickets: list[dict] | None,
    current_tickets: list[dict],
    records: list[dict],
) -> list[dict]:
    """Restore previous-ticket indexes the latest cumulative pass silently dropped.

    Each cumulative pass is asked to re-emit the complete ticket list, but it
    only sees a compact summary of earlier tickets, not their original text —
    so it can drop indexes from an old ticket while otherwise faithfully
    preserving it. The original version of this function only restored a
    previous ticket when it disappeared *entirely*; a ticket that lost only
    some of its indexes looked "handled" and the missing indexes were gone for
    good. This restores exactly the missing remainder as a fragment when the
    loss is partial, and the whole ticket when the loss is total.
    """
    if not previous_tickets:
        return current_tickets
    returned_indexes: set[int] = set()
    for ticket in current_tickets or []:
        for value in ticket.get("included_message_indexes") or []:
            try:
                returned_indexes.add(int(value))
            except (TypeError, ValueError):
                continue

    carried: list[dict] = []
    for ticket in previous_tickets:
        ticket_indexes: set[int] = set()
        for value in ticket.get("included_message_indexes") or []:
            try:
                ticket_indexes.add(int(value))
            except (TypeError, ValueError):
                continue
        if not ticket_indexes:
            continue
        missing_indexes = sorted(ticket_indexes - returned_indexes)
        if not missing_indexes:
            continue
        if len(missing_indexes) == len(ticket_indexes):
            carried.append(dict(ticket))
            continue
        original_id = str(ticket.get("ticket_id") or "").strip()
        fragment = {
            **ticket,
            "ticket_id": f"{original_id}::carried" if original_id else "ticket_carried_fragment",
            "included_message_indexes": missing_indexes,
            "start_message_index": min(missing_indexes),
            "end_message_index": max(missing_indexes),
            "inquiries": _filter_inquiries_for_indexes(ticket, missing_indexes),
            "conversation_summaries": _filter_conversation_summaries_for_indexes(ticket, missing_indexes),
            "previous_ticket_id": str(ticket.get("previous_ticket_id") or "").strip(),
            "segmentation_reason": " ".join(
                part
                for part in [
                    str(ticket.get("segmentation_reason") or "").strip(),
                    "Carried forward: partially dropped from the previous cumulative pass's re-statement.",
                ]
                if part
            ),
        }
        carried.append(fragment)

    if not carried:
        return current_tickets
    return _normalize_ticket_segments({"tickets": [*carried, *current_tickets]}, records)


def _carry_forward_previous_conversation_summaries(
    previous_tickets: list[dict] | None,
    current_tickets: list[dict],
) -> list[dict]:
    if not previous_tickets:
        return current_tickets
    for current in current_tickets:
        current_indexes = set(_ticket_index_list(current))
        if not current_indexes:
            continue
        prior_summaries: list[dict] = []
        for previous in previous_tickets:
            if not (current_indexes & set(_ticket_index_list(previous))):
                continue
            prior_summaries.extend(
                _filter_conversation_summaries_for_indexes(previous, sorted(current_indexes))
            )
        current["conversation_summaries"] = _dedupe_ticket_conversation_summaries(
            [*prior_summaries, *(current.get("conversation_summaries") or [])]
        )
    return current_tickets


def _eval_ticket_segmentation_cumulative(
    client,
    api: APIConfig,
    conversation_id: str,
    records: list[dict],
    conversation_metadata: dict,
    truncate_chars: Optional[int],
    ticket_prompt: PromptTemplate | None = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> tuple[list[dict], dict | None]:
    blocks = _ticket_source_record_blocks(records)
    if not blocks:
        raise TicketSegmentationError("Ticket segmentation received no visible records.")

    prefix_records: list[dict] = []
    previous_tickets: list[dict] | None = None
    last_debug: dict | None = None
    pass_summaries: list[dict[str, Any]] = []
    processed_source_ids: list[str] = []
    pass_errors: list[dict[str, Any]] = []
    total_passes = len(blocks)

    for pass_index, block in enumerate(blocks, start=1):
        if cancel_requested and cancel_requested():
            final_debug = dict(last_debug or {})
            final_debug["segmentation_mode"] = TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE
            final_debug["cumulative_pass_count"] = total_passes
            final_debug["cumulative_passes"] = pass_summaries
            final_debug["cancelled"] = True
            final_debug["cancelled_before_pass"] = pass_index
            final_debug["failed_passes"] = pass_errors
            return previous_tickets or [], final_debug

        previous_prefix_records = list(prefix_records)
        block_records = list(block.get("records") or [])
        prefix_records.extend(block_records)
        source_id = str(block.get("source_conversation_id") or "unknown").strip() or "unknown"
        context: dict[str, Any] = {
            "segmentation_mode": TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE,
            "instruction": (
                "This is a conversation-by-conversation cumulative pass. The payload contains only the "
                "current source_conversation_id block, not the full prior transcript. Use "
                "previous_cumulative_ticket_output as the summary/current ticket map from earlier source "
                "conversations, then append, reopen, update, merge, or create tickets using the current "
                "source conversation. Return the complete ticket list for all processed source conversations, "
                "not only the current source conversation."
            ),
            "pass_index": pass_index,
            "total_passes": total_passes,
            "current_source_conversation_id": source_id,
            "processed_previous_source_conversation_ids": list(processed_source_ids),
            "previous_cumulative_ticket_output": (
                _ticket_prompt_context(previous_tickets, previous_prefix_records)
                if previous_tickets is not None else None
            ),
        }
        try:
            tickets, debug = _eval_ticket_segmentation_once(
                client,
                api,
                conversation_id,
                block_records,
                conversation_metadata,
                truncate_chars,
                ticket_prompt,
                context,
                normalization_records=prefix_records,
            )
        except Exception as exc:
            # This pass produced nothing. Never invent a ticket for the block:
            # a synthetic ticket is indistinguishable from a real one in the UI,
            # so a dead connection reads as a segmentation decision. Keep the
            # tickets earlier passes really produced, record the failure, and
            # leave this block's messages genuinely unsegmented.
            pass_errors.append(
                {
                    "pass_index": pass_index,
                    "source_conversation_id": source_id,
                    "message_indexes": [
                        record.get("message_index") for record in block_records
                    ],
                    "error": str(exc),
                }
            )
            pass_summaries.append(
                {
                    "pass_index": pass_index,
                    "total_passes": total_passes,
                    "current_source_conversation_id": source_id,
                    "current_message_count": len(block_records),
                    "failed": True,
                    "error": str(exc),
                }
            )
            processed_source_ids.append(source_id)
            last_debug = {"error": str(exc)}
            continue
        tickets = _carry_forward_previous_conversation_summaries(previous_tickets, tickets)
        tickets = _carry_forward_missing_previous_tickets(previous_tickets, tickets, prefix_records)
        previous_tickets = tickets
        last_debug = debug
        processed_source_ids.append(source_id)
        pass_summaries.append(
            {
                "pass_index": pass_index,
                "total_passes": total_passes,
                "current_source_conversation_id": source_id,
                "processed_previous_source_conversation_ids": list(processed_source_ids[:-1]),
                "current_message_count": len(block_records),
                "cumulative_message_count": len(prefix_records),
                "ticket_count": len(tickets),
                "raw_response_chars": len(str((debug or {}).get("raw_model_response") or "")),
            }
        )

    final_debug = dict(last_debug or {})
    final_debug["segmentation_mode"] = TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE
    final_debug["cumulative_pass_count"] = total_passes
    final_debug["cumulative_passes"] = pass_summaries
    final_debug["failed_passes"] = pass_errors
    if not previous_tickets:
        raise TicketSegmentationError(
            "Cumulative ticket segmentation produced no tickets"
            + (f"; {len(pass_errors)} of {total_passes} passes failed: {pass_errors[0]['error']}" if pass_errors else ".")
        )
    return previous_tickets, final_debug


def _eval_ticket_segmentation(
    client,
    api: APIConfig,
    conversation_id: str,
    records: list[dict],
    conversation_metadata: dict,
    truncate_chars: Optional[int],
    ticket_prompt: PromptTemplate | None = None,
    segmentation_mode: Any = TICKET_SEGMENTATION_MODE_DEFAULT,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> tuple[list[dict], dict | None]:
    mode = clean_ticket_segmentation_mode(segmentation_mode)
    if mode == TICKET_SEGMENTATION_MODE_CUMULATIVE_SOURCE:
        return _eval_ticket_segmentation_cumulative(
            client,
            api,
            conversation_id,
            records,
            conversation_metadata,
            truncate_chars,
            ticket_prompt,
            cancel_requested=cancel_requested,
        )
    if cancel_requested and cancel_requested():
        return [], {
            "cancelled": True,
            "segmentation_mode": TICKET_SEGMENTATION_MODE_SINGLE_PASS,
        }
    return _eval_ticket_segmentation_once(
        client,
        api,
        conversation_id,
        records,
        conversation_metadata,
        truncate_chars,
        ticket_prompt,
    )


def preview_ticket_segmentation(
    client,
    api: APIConfig,
    conversation_id: str,
    records: list[dict],
    conversation_metadata: dict,
    truncate_chars: Optional[int] = None,
    ticket_prompt: PromptTemplate | None = None,
    segmentation_mode: Any = TICKET_SEGMENTATION_MODE_DEFAULT,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> dict:
    """Run only the ticket segmentation prompt and return normalized preview data."""
    tickets, debug = _eval_ticket_segmentation(
        client,
        api,
        conversation_id,
        records,
        conversation_metadata,
        truncate_chars,
        ticket_prompt,
        segmentation_mode,
        cancel_requested=cancel_requested,
    )
    return {
        "conversation_id": conversation_id,
        "conversation_metadata": conversation_metadata,
        "tickets": tickets,
        "debug": debug,
    }


def _filtered_groups_for_segmentation(
    df: pd.DataFrame,
    config: RunConfig,
) -> list[tuple[str, pd.DataFrame]]:
    groups = get_conversation_groups(df)
    if config.selected_conversation_ids is not None:
        wanted = {str(x) for x in config.selected_conversation_ids}
        return [group for group in groups if str(group[0]) in wanted]
    if config.max_conversations is not None:
        return groups[: config.max_conversations]
    return groups


def segment_dataframe_into_ticket_journeys(
    df: pd.DataFrame,
    client,
    config: RunConfig,
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """Split customer timelines into virtual ticket journeys using AI."""
    if df.empty:
        return df

    groups = _filtered_groups_for_segmentation(df, config)
    if not groups:
        if on_progress:
            on_progress(
                {
                    "phase": "ticket_segmentation_skipped",
                    "reason": "No selected customer timelines were available for ticket splitting.",
                    "total_conversations": 0,
                    "planned_ticket_calls": 0,
                }
            )
        return df.iloc[0:0].copy()

    index_col = MESSAGE_ORDER_COLUMN if MESSAGE_ORDER_COLUMN in df.columns else LEGACY_MESSAGE_ORDER_COLUMN
    out_frames: list[pd.DataFrame] = []
    truncate_chars = config.max_chars_per_message if config.truncate_messages else None
    api = config.ticket_api_config()
    workers = min(max(1, int(getattr(api, "concurrency", 1) or 1)), len(groups))
    segmentation_mode = clean_ticket_segmentation_mode(config.ticket_segmentation_mode)

    segmentation_tasks: list[dict[str, Any]] = []
    for group_index, (parent_id, group) in enumerate(groups, start=1):
        parent_id = str(parent_id)
        segmentation_tasks.append(
            {
                "group_index": group_index,
                "parent_id": parent_id,
                "group": group,
                "records": message_records_from_group(group, parent_id),
                "metadata": conversation_metadata_from_group(group),
            }
        )
    planned_ticket_calls = sum(
        ticket_segmentation_call_count_for_records(task["records"], segmentation_mode)
        for task in segmentation_tasks
    )

    if on_progress:
        on_progress(
            {
                "phase": "ticket_segmentation_start",
                "total_conversations": len(groups),
                "workers": workers,
                "segmentation_mode": segmentation_mode,
                "planned_ticket_calls": planned_ticket_calls,
            }
        )

    def run_segmentation_task(task: dict[str, Any]) -> dict[str, Any]:
        records = task["records"]
        ticket_calls_used = ticket_segmentation_call_count_for_records(records, segmentation_mode)
        if cancel_requested and cancel_requested():
            return {
                "group_index": task["group_index"],
                "parent_id": task["parent_id"],
                "tickets": [],
                "debug": {"cancelled": True},
                "failed_reason": "Ticket segmentation cancelled before this customer timeline started.",
                "ticket_calls_used": 0,
                "cancelled": True,
            }
        try:
            tickets, debug = _eval_ticket_segmentation(
                client,
                api,
                task["parent_id"],
                records,
                task["metadata"],
                truncate_chars,
                config.ticket_prompt,
                segmentation_mode,
                cancel_requested=cancel_requested,
            )
            failed_reason = ""
            # A conversation whose passes partly failed keeps the tickets that
            # really were produced, but must still report as failed.
            failed_passes = (debug or {}).get("failed_passes") or []
            if failed_passes:
                failed_reason = (
                    f"{len(failed_passes)} of {(debug or {}).get('cumulative_pass_count') or '?'} "
                    f"segmentation passes failed: {failed_passes[0].get('error')}"
                )
        except Exception as exc:
            # No ticket is invented here. The conversation contributes no rows
            # and is reported as failed.
            tickets = []
            debug = None
            failed_reason = str(exc)
        return {
            "group_index": task["group_index"],
            "parent_id": task["parent_id"],
            "tickets": tickets,
            "debug": debug,
            "failed_reason": failed_reason,
            "ticket_calls_used": ticket_calls_used,
            "cancelled": bool((debug or {}).get("cancelled")),
        }

    segmentation_results: dict[int, dict[str, Any]] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_task = {
            ex.submit(run_segmentation_task, task): task
            for task in segmentation_tasks
        }
        completed_count = 0
        cancellation_seen = False
        for future in cf.as_completed(future_to_task):
            task = future_to_task[future]
            if future.cancelled():
                continue
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "group_index": task["group_index"],
                    "parent_id": task["parent_id"],
                    "tickets": [],
                    "debug": None,
                    "failed_reason": str(exc),
                    "ticket_calls_used": ticket_segmentation_call_count_for_records(
                        task["records"],
                        segmentation_mode,
                    ),
                }
            segmentation_results[int(result["group_index"])] = result
            if not result.get("cancelled"):
                completed_count += 1
            if on_progress:
                on_progress(
                    {
                        "phase": "ticket_segmentation_done",
                        "conversation_index": result["group_index"],
                        "completed_conversations": completed_count,
                        "total_conversations": len(segmentation_tasks),
                        "conversation_id": result["parent_id"],
                        "tickets_created": len(result["tickets"]),
                        "error": result["failed_reason"],
                        "workers": workers,
                        "segmentation_mode": segmentation_mode,
                        "planned_ticket_calls": planned_ticket_calls,
                        "ticket_calls_used": result.get("ticket_calls_used") or 1,
                    }
                )
            if result.get("cancelled") or (cancel_requested and cancel_requested()):
                cancellation_seen = True
                for pending_future in future_to_task:
                    if pending_future is not future and not pending_future.done():
                        pending_future.cancel()
                if on_progress:
                    on_progress(
                        {
                            "phase": "cancelled",
                            "scope": "ticket_segmentation",
                            "completed_conversations": completed_count,
                            "total_conversations": len(segmentation_tasks),
                        }
                    )

    for task in segmentation_tasks:
        if cancellation_seen and int(task["group_index"]) not in segmentation_results:
            continue
        group = task["group"]
        parent_id = task["parent_id"]
        result = segmentation_results.get(int(task["group_index"]))
        if not result:
            continue
        tickets = result["tickets"]
        debug = result["debug"]
        failed_reason = result["failed_reason"]
        group_index_values = pd.to_numeric(group[index_col], errors="coerce")
        for ticket_number, ticket in enumerate(tickets, start=1):
            included = {int(value) for value in ticket["included_message_indexes"]}
            ticket_rows = group[group_index_values.map(lambda value: pd.notna(value) and int(value) in included)].copy()
            if ticket_rows.empty:
                continue

            ticket_journey_id = f"{parent_id}::ticket_{ticket_number:02d}"
            ticket_rows[TICKET_JOURNEY_ID_COLUMN] = ticket_journey_id
            ticket_rows["PARENT_JOURNEY_ID"] = parent_id
            ticket_rows["TICKET_ID"] = ticket.get("ticket_id") or f"ticket_{ticket_number}"
            ticket_rows["TICKET_CATEGORY"] = ticket.get("ticket_category") or "inquiry"
            ticket_rows["TICKET_MODEL_CATEGORY"] = ticket.get("model_ticket_category") or ""
            ticket_rows["TICKET_REQUEST_ORIGIN"] = ticket.get("request_origin") or "customer"
            ticket_rows["TICKET_CATEGORY_DEBUG_JSON"] = json.dumps(
                ticket.get("ticket_category_debug") or {},
                ensure_ascii=False,
                default=str,
            )
            ticket_rows["TICKET_TYPE"] = ticket.get("ticket_type") or "other"
            ticket_rows["TICKET_STATUS"] = ticket.get("status") or "pending_unresolved"
            ticket_rows["TICKET_OBJECTIVE"] = ticket.get("customer_objective") or ""
            ticket_rows["TICKET_PREVIOUS_TICKET_ID"] = ticket.get("previous_ticket_id") or ""
            ticket_rows["TICKET_INQUIRIES_JSON"] = json.dumps(ticket.get("inquiries") or [], ensure_ascii=False)
            ticket_rows["TICKET_SEGMENTATION_REASON"] = ticket.get("segmentation_reason") or failed_reason
            ticket_rows["TICKET_SHOULD_APPEND_FUTURE"] = bool(ticket.get("should_append_future_conversations"))
            if debug and config.save_raw_responses:
                ticket_rows["TICKET_SEGMENTATION_RAW"] = str(debug.get("raw_model_response") or "")
            out_frames.append(ticket_rows)

    if not out_frames:
        return df.iloc[0:0].copy()
    segmented = pd.concat(out_frames, ignore_index=True)
    return segmented


def _saved_conversations_to_dataframe(existing_conversation_results: list[dict] | None) -> pd.DataFrame:
    """Rebuild CSV-like message rows from saved DB transcripts.

    The ticket splitter already operates on the uploaded CSV shape. Saved DB
    runs keep the same information in transcript dictionaries, so this adapter
    gives DB-backed full runs the same optional ticket-splitting path.
    """
    rows: list[dict[str, Any]] = []
    metadata_to_columns = {
        "conversation_start_date": "CONVERSATION_START_DATE",
        "conversation_end_date": "CONVERSATION_END_DATE",
        "conversation_status": "CONVERSATION_STATUS",
        "initial_skill": "INITIAL_SKILL",
        "last_skill": "LAST_SKILL",
        "joined_skills": "JOINED_SKILLS",
        "conversation_agent_full_name": "CONVERSATION_AGENT_FULL_NAME",
        "conversation_agent_login_name": "CONVERSATION_AGENT_LOGIN_NAME",
        "customer_name": "CUSTOMER_NAME",
        "ticket_journey_id": "TICKET_JOURNEY_ID",
        "parent_journey_id": "PARENT_JOURNEY_ID",
        "ticket_id": "TICKET_ID",
        "ticket_category": "TICKET_CATEGORY",
        "model_ticket_category": "TICKET_MODEL_CATEGORY",
        "request_origin": "TICKET_REQUEST_ORIGIN",
        "ticket_category_debug_json": "TICKET_CATEGORY_DEBUG_JSON",
        "ticket_type": "TICKET_TYPE",
        "ticket_status": "TICKET_STATUS",
        "ticket_objective": "TICKET_OBJECTIVE",
        "ticket_previous_ticket_id": "TICKET_PREVIOUS_TICKET_ID",
        "ticket_inquiries_json": "TICKET_INQUIRIES_JSON",
        "ticket_segmentation_reason": "TICKET_SEGMENTATION_REASON",
        "ticket_should_append_future": "TICKET_SHOULD_APPEND_FUTURE",
        "source_conversation_ids": "CONVERSATION_IDS",
        "source_conversation_count": "SOURCE_CONVERSATION_COUNT",
        "total_visible_messages": "TOTAL_VISIBLE_MESSAGES",
        "customer_message_count": "CUSTOMER_MESSAGE_COUNT",
        "agent_message_count": "AGENT_MESSAGE_COUNT",
    }
    for loaded in existing_conversation_results or []:
        if not isinstance(loaded, dict):
            continue
        parent_id = str(loaded.get("conversation_id") or loaded.get("thread_id") or "").strip()
        if not parent_id:
            continue
        metadata = dict(loaded.get("conversation_metadata") or {})
        records = list(loaded.get("transcript") or [])
        for fallback_index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            msg_index = (
                record.get("appended_message_index")
                if record.get("appended_message_index") is not None
                else record.get("message_index")
            )
            if msg_index is None:
                msg_index = fallback_index
            source_conversation_id = (
                record.get("source_conversation_id")
                or record.get("conversation_id")
                or metadata.get("source_conversation_ids")
                or ""
            )
            row = {
                JOURNEY_ID_COLUMN: parent_id,
                MESSAGE_ORDER_COLUMN: msg_index,
                LEGACY_MESSAGE_ORDER_COLUMN: msg_index,
                "MESSAGE_TIME": record.get("message_time") or "",
                "SENDER_ROLE": str(record.get("sender_role") or "unknown").strip().lower(),
                "RAW_SENDER_ROLE": record.get("raw_sender_role"),
                "MESSAGE_TEXT": record.get("message_text") or "",
                "CONVERSATION_ID": source_conversation_id,
                "MESSAGE_AGENT_FULL_NAME": record.get("agent_full_name"),
                "MESSAGE_SKILL": record.get("message_skill"),
                "HAS_RAG_RETRIEVAL": record.get("has_rag_retrieval"),
                "RAG_RETRIEVAL_COUNT": record.get("rag_retrieval_count"),
                "RAG_RETRIEVALS": record.get("rag_retrievals"),
                "CHUNKS_FETCHED": record.get("chunks_fetched"),
                "CHUNK_JUSTIFICATION": record.get("chunk_justification"),
                "CHUNK_TIME": record.get("chunk_time"),
            }
            for md_key, column in metadata_to_columns.items():
                value = metadata.get(md_key)
                if value not in (None, ""):
                    row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _evaluation_sources(
    df: pd.DataFrame | None,
    existing_conversation_results: Optional[list[dict]] = None,
) -> list[tuple[str, list[dict], dict]]:
    """Return normalized conversation sources from a CSV dataframe or saved DB rows."""
    sources: list[tuple[str, list[dict], dict]] = []
    if df is not None and not df.empty:
        for conversation_id, group in get_conversation_groups(df):
            conversation_id = str(conversation_id)
            sources.append(
                (
                    conversation_id,
                    message_records_from_group(group, conversation_id),
                    conversation_metadata_from_group(group),
                )
            )
        return sources

    for loaded in existing_conversation_results or []:
        if not isinstance(loaded, dict):
            continue
        conversation_id = str(loaded.get("conversation_id") or loaded.get("thread_id") or "")
        if not conversation_id:
            continue
        records = list(loaded.get("transcript") or [])
        if not records:
            continue
        sources.append(
            (
                conversation_id,
                records,
                dict(loaded.get("conversation_metadata") or {}),
            )
        )
    return sources


def run_evaluation(
    df: pd.DataFrame | None,
    client,
    config: RunConfig,
    existing_conversation_results: Optional[list[dict]] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
    on_message_result: Optional[Callable[[dict], None]] = None,
    on_conversation_result: Optional[Callable[[dict], None]] = None,
    on_error: Optional[Callable[[dict], None]] = None,
) -> RunResults:
    """Run the full message-level + conversation-level evaluation pipeline.

    Message-level and conversation-level work share ONE ``ThreadPoolExecutor``
    whose worker count equals ``config.api.concurrency``. Concurrency is scoped
    by evaluation source: at most one message/conversation call is active for a
    given source at a time, while different journeys or generated ticket
    journeys run concurrently. As soon as the last message-level call for a
    source completes, its conversation-level call receives priority for the next
    free slot.

    Optional persistence callbacks (``on_message_result``,
    ``on_conversation_result``, ``on_error``) are invoked on the calling thread
    as each record finishes, so the app can write incrementally to its DB.

    Calls ``on_progress`` with a small status dict at each step. All callbacks
    fire on the calling thread; only the OpenAI API calls run in worker threads.
    """
    results = RunResults(started_at=time.time())
    truncate_chars = config.max_chars_per_message if config.truncate_messages else None

    workers = max(1, int(getattr(config.api, "concurrency", 1) or 1))

    target_role = (config.message_target_role or "agent").strip().lower()
    if target_role not in ("agent", "customer"):
        target_role = "agent"

    segmented_input = False
    source_df = df
    source_existing_conversation_results = existing_conversation_results
    if (
        config.enable_ticket_segmentation
        and df is not None
        and not df.empty
        and existing_conversation_results is None
    ):
        source_df = segment_dataframe_into_ticket_journeys(
            df,
            client,
            config,
            on_progress=on_progress,
            cancel_requested=cancel_requested,
        )
        segmented_input = True
    elif (
        config.enable_ticket_segmentation
        and (df is None or df.empty)
        and existing_conversation_results is not None
    ):
        rebuilt_df = _saved_conversations_to_dataframe(existing_conversation_results)
        if not rebuilt_df.empty:
            source_df = segment_dataframe_into_ticket_journeys(
                rebuilt_df,
                client,
                config,
                on_progress=on_progress,
                cancel_requested=cancel_requested,
            )
            source_existing_conversation_results = None
            segmented_input = True

    if cancel_requested and cancel_requested():
        results.finished_at = time.time()
        if on_progress:
            on_progress(
                {
                    "phase": "cancelled",
                    "scope": "evaluation",
                    "total_conversations": 0,
                }
            )
            on_progress({"phase": "done", "total_conversations": 0})
        return results

    sources = _evaluation_sources(source_df, source_existing_conversation_results)
    # Selection precedence: explicit IDs (random sampler) > max_conversations slice.
    if segmented_input:
        pass
    elif config.selected_conversation_ids is not None:
        wanted = set(str(x) for x in config.selected_conversation_ids)
        sources = [source for source in sources if str(source[0]) in wanted]
    elif config.max_conversations is not None:
        sources = sources[: config.max_conversations]

    total_conversations = len(sources)
    planned_message_calls = 0
    for _, records, _ in sources:
        targets = [r for r in records if r.get("sender_role") == target_role]
        if config.max_agent_messages_per_conv is not None:
            targets = targets[: config.max_agent_messages_per_conv]
        planned_message_calls += len(targets)
    if on_progress:
        on_progress(
            {
                "phase": "start",
                "total_conversations": total_conversations,
                "total_message_calls": int(planned_message_calls),
                "total_conversation_calls": int(total_conversations),
                "total_calls": int(planned_message_calls + total_conversations),
                "workers": workers,
            }
        )

    # ---- Pre-build per-conversation state on the main thread ----------------

    def visible_history_of(records: list[dict], up_to_index: Any) -> list[dict]:
        out = []
        for r in records:
            idx = r["message_index"]
            if idx is None:
                continue
            if idx > up_to_index:
                break
            role = r.get("sender_role", "unknown")
            if role == "unknown" and not config.include_unknown_in_history:
                continue
            out.append(r)
        return out

    conv_state: dict[str, dict[str, Any]] = {}
    conv_order: list[str] = []
    # Keep queued tasks small. Building and retaining one cumulative history
    # list per target message makes long journeys quadratic in memory (a
    # 500-message journey holds hundreds of overlapping transcript copies).
    # Histories are constructed only when a task enters an available worker
    # slot below, so at most ``workers`` temporary history lists exist.
    ready_ml: deque[str] = deque()
    no_target_convs: list[str] = []

    for ci, (conversation_id, records, conversation_metadata) in enumerate(sources, start=1):
        targets = [r for r in records if r.get("sender_role") == target_role]
        if config.max_agent_messages_per_conv is not None:
            targets = targets[: config.max_agent_messages_per_conv]

        state = {
            "conversation_id": conversation_id,
            "conversation_index": ci,
            "records": records,
            "conversation_metadata": conversation_metadata,
            "targets": targets,
            "results_by_idx": {},          # message_index -> message-level record
            "target_queue": deque(targets),
            "started": False,
            "ml_total": len(targets),
            "ml_done": 0,
            "cl_submitted": False,
            "cl_done": False,
        }
        conv_state[conversation_id] = state
        conv_order.append(conversation_id)

        if not targets:
            no_target_convs.append(conversation_id)
            continue

        ready_ml.append(conversation_id)

    # ---- One shared pool drives everything ---------------------------------

    stop_signal = {"flag": False, "reason": None}

    def _emit_conversation_start(conversation_id: str) -> None:
        state = conv_state[conversation_id]
        if state.get("started"):
            return
        state["started"] = True
        if on_progress:
            on_progress(
                {
                    "phase": "conversation_start",
                    "conversation_index": state["conversation_index"],
                    "conversation_id": conversation_id,
                    "agent_messages": state["ml_total"],
                    "target_messages": state["ml_total"],
                    "target_role": target_role,
                    "total_conversations": total_conversations,
                    "workers": workers,
                }
            )

    def _submit_cl(ex: cf.ThreadPoolExecutor, conversation_id: str) -> cf.Future:
        """Build the conversation-level payload and submit it to the pool."""
        _emit_conversation_start(conversation_id)
        state = conv_state[conversation_id]
        message_results_ordered = [
            state["results_by_idx"][t["message_index"]]
            for t in state["targets"]
            if t["message_index"] in state["results_by_idx"]
        ]
        message_results_ordered = apply_contradiction_source_suppression(
            message_results_ordered,
            state["records"],
        )
        computed_md = compute_metadata(message_results_ordered, state["records"])
        computed_md["evaluation_target_role"] = target_role
        computed_md["target_messages_evaluated"] = sum(
            1 for m in message_results_ordered if m.get("parse_status") == "ok"
        )
        full_transcript = (
            state["records"] if config.include_unknown_in_history
            else [r for r in state["records"] if r.get("sender_role") != "unknown"]
        )
        conv_md_for_judge = dict(state["conversation_metadata"])
        conv_md_for_judge["evaluation_target_role"] = target_role

        state["message_results_ordered"] = message_results_ordered
        state["computed_metadata"] = computed_md
        state["full_transcript"] = full_transcript
        state["cl_submitted"] = True

        return ex.submit(
            _eval_conversation_level,
            client=client,
            api=config.conversation_api_config(),
            conversation_id=conversation_id,
            conversation_metadata=conv_md_for_judge,
            full_transcript=full_transcript,
            message_level_evaluations=message_results_ordered,
            computed_metadata=computed_md,
            save_raw=config.save_raw_responses,
            truncate_chars=truncate_chars,
            prompt=config.conversation_prompt,
        )

    def _finalize_cl_record(conversation_id: str, cr: dict) -> dict:
        state = conv_state[conversation_id]
        cr["thread_id"] = conversation_id
        cr["conversation_metadata"] = state["conversation_metadata"]
        cr["computed_metadata"] = state["computed_metadata"]
        cr["transcript"] = state["records"]
        cr["message_level_results"] = state["message_results_ordered"]
        cr["evaluation_target_role"] = target_role
        if cr.get("parse_status") != "ok" and not cr.get("parsed_json"):
            # Inject a stub so the dashboard still has a row for this conversation.
            cr["parsed_json"] = {
                "customer_objective_type": "Inquiry",
                "customer_primary_objective": "",
                "handled_status": "unhandled",
                "customer_experience": "bad",
                "unhandled_resolution_subtype": "totally_unresolved",
                "frustration_detected": True,
                "frustration_origin": "our_side",
                "customer_started_frustrated": False,
                "customer_became_frustrated_during_chat": True,
                "customer_ended_frustrated": False,
                "frustration_timing": "during",
                "final_customer_sentiment": "unknown",
                "max_frustration_level": state["computed_metadata"].get("max_frustration_level", "none"),
                "main_issue": {
                    "issue_exists": True,
                    "issue_origin": "our_side",
                    "issue_type": "other",
                    "issue_summary": "Conversation-level evaluator failed to parse",
                    "customer_impact": "Unable to assess automatically",
                },
                "all_detected_issues": [],
                "positive_signals": [],
                "negative_signals": [],
                "classification_reason": "The conversation-level evaluator failed, so the result is treated as unresolved and high risk for review.",
                "management_summary": "Automatic evaluation could not parse a result for this conversation. Manual review required.",
                "recommended_actions": ["Review this conversation manually."],
                "manual_review_required": True,
                "manual_review_reason": cr.get("error_message") or "Parse failure",
                "confidence": "low",
            }
        _apply_message_signal_to_conversation_score(
            cr.get("parsed_json") or {},
            state.get("computed_metadata") or {},
        )
        cr["evaluation_output"] = cr.get("parsed_json")
        return cr

    fut_info: dict[cf.Future, dict] = {}
    pending: set[cf.Future] = set()
    ready_cl = deque(no_target_convs)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        def fill_worker_slots() -> None:
            """Keep the pool full, prioritizing newly-ready conversation calls."""
            while len(pending) < workers and (ready_cl or ready_ml):
                if ready_cl:
                    conversation_id = ready_cl.popleft()
                    fut = _submit_cl(ex, conversation_id)
                    pending.add(fut)
                    fut_info[fut] = {
                        "type": "cl",
                        "conversation_id": conversation_id,
                    }
                    continue

                conversation_id = ready_ml.popleft()
                _emit_conversation_start(conversation_id)
                target = conv_state[conversation_id]["target_queue"].popleft()
                history = visible_history_of(
                    conv_state[conversation_id]["records"],
                    target["message_index"],
                )
                fut = ex.submit(
                    _eval_message_level,
                    client=client,
                    api=config.api,
                    conversation_id=conversation_id,
                    target_record=target,
                    history_records=history,
                    conversation_metadata=conv_state[conversation_id]["conversation_metadata"],
                    save_raw=config.save_raw_responses,
                    truncate_chars=truncate_chars,
                    prompt=config.message_prompt,
                )
                pending.add(fut)
                fut_info[fut] = {
                    "type": "ml",
                    "conversation_id": conversation_id,
                    "target": target,
                }

        fill_worker_slots()

        # Drain active work. Ready conversation calls are queued before any
        # additional message calls whenever worker slots become available.
        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                info = fut_info.pop(fut)
                conversation_id = info["conversation_id"]
                state = conv_state[conversation_id]

                if info["type"] == "ml":
                    target = info["target"]
                    try:
                        mr = fut.result()
                    except Exception as e:  # noqa: BLE001
                        mr = {
                            "conversation_id": conversation_id,
                            "target_message_id": target.get("message_id", ""),
                            "message_index": target.get("message_index"),
                            "appended_message_index": target.get("appended_message_index", target.get("message_index")),
                            "source_conversation_id": target.get("source_conversation_id"),
                            "message_time": target.get("message_time", ""),
                            "target_message_text": strip_inline_rag_context(target.get("message_text", "")),
                            "input_history": None,
                            "raw_model_response": None,
                            "parsed_json": None,
                            "parse_status": "api_error",
                            "error_message": f"Worker raised: {e}",
                            "debug": None,
                        }
                    state["results_by_idx"][target["message_index"]] = mr
                    state["ml_done"] += 1

                    if on_message_result:
                        try:
                            on_message_result(mr)
                        except Exception:
                            pass

                    if mr.get("parse_status") != "ok":
                        err = {
                            "level": "message",
                            "conversation_id": conversation_id,
                            "message_index": target.get("message_index"),
                            "error": mr.get("error_message"),
                        }
                        results.errors.append(err)
                        if on_error:
                            try:
                                on_error(err)
                            except Exception:
                                pass

                    if on_progress:
                        on_progress(
                            {
                                "phase": "message_done",
                                "conversation_index": state["conversation_index"],
                                "conversation_id": conversation_id,
                                "message_index": target.get("message_index"),
                                "message_in_conversation": state["ml_done"],
                                "total_in_conversation": state["ml_total"],
                                "status": mr.get("parse_status"),
                                "automatic_reruns": int(mr.get("automatic_reruns") or 0),
                                "recovered_after_rerun": bool(
                                    mr.get("recovered_after_rerun")
                                ),
                                "rerun_errors": list(mr.get("rerun_errors") or []),
                            }
                        )

                    if config.stop_on_error and mr.get("parse_status") == "api_error":
                        stop_signal["flag"] = True
                        stop_signal["reason"] = mr.get("error_message")
                    if cancel_requested and cancel_requested():
                        stop_signal["flag"] = True
                        stop_signal["reason"] = stop_signal["reason"] or "cancelled"

                    # Give this conversation's CL priority for the next free slot.
                    if (
                        not stop_signal["flag"]
                        and not state["cl_submitted"]
                        and state["ml_done"] >= state["ml_total"]
                    ):
                        # Mark now so this source cannot be queued twice before
                        # its conversation-level call is submitted.
                        state["cl_submitted"] = True
                        ready_cl.append(conversation_id)
                    elif not stop_signal["flag"] and state["target_queue"]:
                        ready_ml.append(conversation_id)

                elif info["type"] == "cl":
                    try:
                        cr = fut.result()
                    except Exception as e:  # noqa: BLE001
                        cr = {
                            "conversation_id": conversation_id,
                            "raw_model_response": None,
                            "parsed_json": None,
                            "parse_status": "api_error",
                            "error_message": f"Worker raised: {e}",
                            "debug": None,
                        }
                    cr = _finalize_cl_record(conversation_id, cr)
                    state["cl_done"] = True

                    if cr.get("parse_status") != "ok":
                        err = {
                            "level": "conversation",
                            "conversation_id": conversation_id,
                            "error": cr.get("error_message"),
                        }
                        results.errors.append(err)
                        if on_error:
                            try:
                                on_error(err)
                            except Exception:
                                pass
                        if config.stop_on_error and cr.get("parse_status") == "api_error":
                            stop_signal["flag"] = True
                            stop_signal["reason"] = cr.get("error_message")

                    results.conversation_results.append(cr)
                    if on_conversation_result:
                        try:
                            on_conversation_result(cr)
                        except Exception:
                            pass

                    if on_progress:
                        on_progress(
                            {
                                "phase": "conversation_done",
                                "conversation_index": state["conversation_index"],
                                "conversation_id": conversation_id,
                                "total_conversations": total_conversations,
                                "status": cr.get("parse_status"),
                                "automatic_reruns": int(cr.get("automatic_reruns") or 0),
                                "recovered_after_rerun": bool(cr.get("recovered_after_rerun")),
                                "rerun_errors": cr.get("rerun_errors") or [],
                            }
                        )

            if stop_signal["flag"]:
                # Cancel anything that hasn't started yet and drop the rest.
                ready_ml.clear()
                ready_cl.clear()
                for f in list(pending):
                    if not f.done():
                        f.cancel()
                    pending.discard(f)
                if on_progress:
                    on_progress(
                        {
                            "phase": "stopped_on_error",
                            "error": stop_signal["reason"],
                        }
                    )
                break
            fill_worker_slots()

    # Sort outputs by the original conversation order, then by message_index,
    # so the dashboard and exports stay deterministic regardless of completion
    # order in the streaming pool.
    order_by_cid = {cid: i for i, cid in enumerate(conv_order)}
    results.conversation_results.sort(
        key=lambda c: order_by_cid.get(c.get("conversation_id"), 0)
    )

    # Flatten per-conversation ordered message results into the global list.
    results.message_level_results = []
    for cid in conv_order:
        state = conv_state.get(cid, {})
        ordered = state.get("message_results_ordered")
        if ordered is None:
            ordered = [
                state.get("results_by_idx", {})[t["message_index"]]
                for t in state.get("targets", [])
                if t["message_index"] in state.get("results_by_idx", {})
            ]
        results.message_level_results.extend(ordered)

    results.finished_at = time.time()
    if on_progress:
        on_progress({"phase": "done", "total_conversations": total_conversations})
    return results


def run_message_level_repair(
    df: pd.DataFrame | None,
    client,
    config: RunConfig,
    selected_message_indices_by_conversation: dict[str, list[Any]],
    existing_conversation_results: Optional[list[dict]] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
    on_message_result: Optional[Callable[[dict], None]] = None,
    on_error: Optional[Callable[[dict], None]] = None,
) -> RunResults:
    """Rerun only selected message-level rows from an existing run."""
    results = RunResults(started_at=time.time())
    truncate_chars = config.max_chars_per_message if config.truncate_messages else None
    workers = max(1, int(getattr(config.api, "concurrency", 1) or 1))

    target_role = (config.message_target_role or "agent").strip().lower()
    if target_role not in ("agent", "customer"):
        target_role = "agent"

    wanted: dict[str, set[Any]] = {}
    for conversation_id, raw_indices in (selected_message_indices_by_conversation or {}).items():
        normalized: set[Any] = set()
        for idx in raw_indices or []:
            try:
                normalized.add(int(idx))
            except (TypeError, ValueError):
                normalized.add(idx)
        if normalized:
            wanted[str(conversation_id)] = normalized

    sources: list[tuple[str, list[dict], dict]] = []
    if df is not None and not df.empty:
        for conversation_id, group in get_conversation_groups(df):
            conversation_id = str(conversation_id)
            if conversation_id not in wanted:
                continue
            sources.append(
                (
                    conversation_id,
                    message_records_from_group(group, conversation_id),
                    conversation_metadata_from_group(group),
                )
            )
    else:
        for loaded in existing_conversation_results or []:
            conversation_id = str(loaded.get("conversation_id") or loaded.get("thread_id") or "")
            if not conversation_id or conversation_id not in wanted:
                continue
            sources.append(
                (
                    conversation_id,
                    list(loaded.get("transcript") or []),
                    dict(loaded.get("conversation_metadata") or {}),
                )
            )

    total_conversations = len(sources)

    if on_progress:
        on_progress(
            {
                "phase": "start",
                "total_conversations": total_conversations,
                "workers": workers,
            }
        )

    def visible_history_of(records: list[dict], up_to_index: Any) -> list[dict]:
        out = []
        for r in records:
            idx = r["message_index"]
            if idx is None:
                continue
            if idx > up_to_index:
                break
            role = r.get("sender_role", "unknown")
            if role == "unknown" and not config.include_unknown_in_history:
                continue
            out.append(r)
        return out

    tasks: list[tuple[str, int, dict, list[dict], dict]] = []
    conversation_index: dict[str, int] = {}
    found_conversation_ids: set[str] = set()
    for ci, (conversation_id, records, metadata) in enumerate(sources, start=1):
        found_conversation_ids.add(conversation_id)
        conversation_index[conversation_id] = ci
        wanted_indices = wanted.get(conversation_id, set())
        targets: list[dict] = []
        for record in records:
            if record.get("sender_role") != target_role:
                continue
            idx = record.get("message_index")
            try:
                normalized_idx: Any = int(idx)
            except (TypeError, ValueError):
                normalized_idx = idx
            if normalized_idx in wanted_indices:
                targets.append(record)

        missing = set(wanted_indices)
        for target in targets:
            try:
                normalized_idx = int(target.get("message_index"))
            except (TypeError, ValueError):
                normalized_idx = target.get("message_index")
            missing.discard(normalized_idx)
        if missing:
            err = {
                "level": "message",
                "conversation_id": conversation_id,
                "error": (
                    "Message repair skipped missing or non-target message_index "
                    f"values: {sorted(str(x) for x in missing)[:10]}"
                ),
            }
            results.errors.append(err)
            if on_error:
                try:
                    on_error(err)
                except Exception:
                    pass

        if on_progress:
            on_progress(
                {
                    "phase": "conversation_start",
                    "conversation_index": ci,
                    "conversation_id": conversation_id,
                    "agent_messages": len(targets),
                    "target_messages": len(targets),
                    "target_role": target_role,
                    "total_conversations": total_conversations,
                    "workers": workers,
                }
            )

        for target in targets:
            tasks.append(
                (
                    conversation_id,
                    ci,
                    target,
                    records,
                    metadata,
                )
            )

    for conversation_id in sorted(set(wanted) - found_conversation_ids):
        err = {
            "level": "message",
            "conversation_id": conversation_id,
            "error": "Message repair skipped because the saved transcript was not available.",
        }
        results.errors.append(err)
        if on_error:
            try:
                on_error(err)
            except Exception:
                pass

    completed_by_conv: dict[str, int] = {conversation_id: 0 for conversation_id, _, _ in sources}
    total_calls = len(tasks)
    if on_progress:
        on_progress(
            {
                "phase": "message_repair_start",
                "total_message_calls": total_calls,
                "total_conversations": total_conversations,
            }
        )

    fut_info: dict[cf.Future, dict] = {}
    pending: set[cf.Future] = set()

    def evaluate_repair_target(
        conversation_id: str,
        target: dict,
        records: list[dict],
        metadata: dict,
    ) -> dict:
        return _eval_message_level(
            client=client,
            api=config.api,
            conversation_id=conversation_id,
            target_record=target,
            history_records=visible_history_of(records, target["message_index"]),
            conversation_metadata=metadata,
            save_raw=config.save_raw_responses,
            truncate_chars=truncate_chars,
            prompt=config.message_prompt,
        )

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for conversation_id, ci, target, records, metadata in tasks:
            fut = ex.submit(
                evaluate_repair_target,
                conversation_id,
                target,
                records,
                metadata,
            )
            pending.add(fut)
            fut_info[fut] = {
                "conversation_id": conversation_id,
                "conversation_index": ci,
                "target": target,
            }

        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                info = fut_info.pop(fut)
                conversation_id = info["conversation_id"]
                target = info["target"]
                try:
                    mr = fut.result()
                except Exception as e:  # noqa: BLE001
                    mr = {
                        "conversation_id": conversation_id,
                        "target_message_id": target.get("message_id", ""),
                        "message_index": target.get("message_index"),
                        "appended_message_index": target.get("appended_message_index", target.get("message_index")),
                        "source_conversation_id": target.get("source_conversation_id"),
                        "message_time": target.get("message_time", ""),
                        "target_message_text": strip_inline_rag_context(target.get("message_text", "")),
                        "input_history": None,
                        "raw_model_response": None,
                        "parsed_json": None,
                        "parse_status": "api_error",
                        "error_message": f"Worker raised: {e}",
                        "debug": None,
                    }

                completed_by_conv[conversation_id] = completed_by_conv.get(conversation_id, 0) + 1
                results.message_level_results.append(mr)
                if on_message_result:
                    try:
                        on_message_result(mr)
                    except Exception:
                        pass

                if mr.get("parse_status") != "ok":
                    err = {
                        "level": "message",
                        "conversation_id": conversation_id,
                        "message_index": target.get("message_index"),
                        "error": mr.get("error_message"),
                    }
                    results.errors.append(err)
                    if on_error:
                        try:
                            on_error(err)
                        except Exception:
                            pass

                if on_progress:
                    on_progress(
                        {
                            "phase": "message_done",
                            "conversation_index": info["conversation_index"],
                            "conversation_id": conversation_id,
                            "message_index": target.get("message_index"),
                            "message_in_conversation": completed_by_conv[conversation_id],
                            "total_in_conversation": len(wanted.get(conversation_id, set())),
                            "status": mr.get("parse_status"),
                            "automatic_reruns": int(mr.get("automatic_reruns") or 0),
                            "recovered_after_rerun": bool(mr.get("recovered_after_rerun")),
                            "rerun_errors": list(mr.get("rerun_errors") or []),
                        }
                    )

            if cancel_requested and cancel_requested():
                for fut in list(pending):
                    if not fut.done():
                        fut.cancel()
                    pending.discard(fut)
                break

    results.message_level_results.sort(
        key=lambda mr: (
            conversation_index.get(str(mr.get("conversation_id") or mr.get("thread_id") or ""), 0),
            str(mr.get("message_index") if mr.get("message_index") is not None else ""),
        )
    )
    results.finished_at = time.time()
    if on_progress:
        on_progress({"phase": "done", "total_conversations": total_conversations})
    return results


def run_conversation_level_only(
    existing_message_level_results: list[dict],
    client,
    config: RunConfig,
    df: pd.DataFrame | None = None,
    existing_conversation_results: Optional[list[dict]] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
    on_message_result: Optional[Callable[[dict], None]] = None,
    on_conversation_result: Optional[Callable[[dict], None]] = None,
    on_error: Optional[Callable[[dict], None]] = None,
) -> RunResults:
    """Run only the conversation-level layer using existing message-level results."""
    results = RunResults(started_at=time.time())
    truncate_chars = config.max_chars_per_message if config.truncate_messages else None
    workers = max(1, int(getattr(config.api, "concurrency", 1) or 1))

    target_role = (config.message_target_role or "agent").strip().lower()
    if target_role not in ("agent", "customer"):
        target_role = "agent"

    by_conv: dict[str, dict[Any, dict]] = {}
    for mr in existing_message_level_results or []:
        if not isinstance(mr, dict):
            continue
        conversation_id = str(mr.get("conversation_id") or mr.get("thread_id") or "")
        message_index = mr.get("message_index")
        if not conversation_id or message_index is None:
            continue
        try:
            normalized_idx: Any = int(message_index)
        except (TypeError, ValueError):
            normalized_idx = message_index
        by_conv.setdefault(conversation_id, {})[normalized_idx] = dict(mr)

    conv_state: dict[str, dict[str, Any]] = {}
    conv_order: list[str] = []

    wanted_ids = (
        set(str(x) for x in config.selected_conversation_ids)
        if config.selected_conversation_ids is not None
        else None
    )

    if df is not None and not df.empty:
        groups = get_conversation_groups(df)
        if wanted_ids is not None:
            groups = [g for g in groups if str(g[0]) in wanted_ids]
        elif config.max_conversations is not None:
            groups = groups[: config.max_conversations]

        for _, (conversation_id, group) in enumerate(groups, start=1):
            records = message_records_from_group(group, conversation_id)
            conversation_metadata = conversation_metadata_from_group(group)
            targets = [r for r in records if r.get("sender_role") == target_role]
            if config.max_agent_messages_per_conv is not None:
                targets = targets[: config.max_agent_messages_per_conv]

            available = by_conv.get(str(conversation_id), {})
            missing_indices: list[Any] = []
            message_results_ordered: list[dict] = []
            for target in targets:
                idx = target.get("message_index")
                try:
                    normalized_idx: Any = int(idx)
                except (TypeError, ValueError):
                    normalized_idx = idx
                if normalized_idx in available:
                    message_results_ordered.append(dict(available[normalized_idx]))
                else:
                    missing_indices.append(idx)

            if missing_indices:
                err = {
                    "level": "conversation",
                    "conversation_id": conversation_id,
                    "error": (
                        "Conversation-only run skipped because required message-level "
                        f"results are missing for message_index values: {missing_indices[:10]}"
                    ),
                }
                results.errors.append(err)
                if on_error:
                    try:
                        on_error(err)
                    except Exception:
                        pass
                continue

            message_results_ordered = apply_contradiction_source_suppression(
                message_results_ordered,
                records,
            )
            computed_md = compute_metadata(message_results_ordered, records)
            computed_md["evaluation_target_role"] = target_role
            computed_md["target_messages_evaluated"] = sum(
                1 for m in message_results_ordered if m.get("parse_status") == "ok"
            )
            full_transcript = (
                records if config.include_unknown_in_history
                else [r for r in records if r.get("sender_role") != "unknown"]
            )
            conv_md_for_judge = dict(conversation_metadata)
            conv_md_for_judge["evaluation_target_role"] = target_role

            state = {
                "conversation_id": conversation_id,
                "conversation_index": len(conv_order) + 1,
                "records": records,
                "conversation_metadata": conversation_metadata,
                "message_results_ordered": message_results_ordered,
                "computed_metadata": computed_md,
                "full_transcript": full_transcript,
                "conv_md_for_judge": conv_md_for_judge,
            }
            conv_state[conversation_id] = state
            conv_order.append(conversation_id)
    else:
        loaded_conversations = existing_conversation_results or []
        for loaded in loaded_conversations:
            conversation_id = str(loaded.get("conversation_id") or loaded.get("thread_id") or "")
            if not conversation_id:
                continue
            if wanted_ids is not None and conversation_id not in wanted_ids:
                continue
            if wanted_ids is None and config.max_conversations is not None and len(conv_order) >= config.max_conversations:
                break

            records = list(loaded.get("transcript") or [])
            conversation_metadata = dict(loaded.get("conversation_metadata") or {})
            message_results_ordered = list(loaded.get("message_level_results") or [])
            if not message_results_ordered:
                available = by_conv.get(conversation_id, {})
                message_results_ordered = [available[idx] for idx in sorted(available)]
            if config.max_agent_messages_per_conv is not None:
                message_results_ordered = message_results_ordered[: config.max_agent_messages_per_conv]

            message_results_ordered = apply_contradiction_source_suppression(
                message_results_ordered,
                records,
            )
            computed_md = compute_metadata(message_results_ordered, records)
            computed_md["evaluation_target_role"] = target_role
            computed_md["target_messages_evaluated"] = sum(
                1 for m in message_results_ordered if m.get("parse_status") == "ok"
            )
            full_transcript = (
                records if config.include_unknown_in_history
                else [r for r in records if r.get("sender_role") != "unknown"]
            )
            conv_md_for_judge = dict(conversation_metadata)
            conv_md_for_judge["evaluation_target_role"] = target_role

            state = {
                "conversation_id": conversation_id,
                "conversation_index": len(conv_order) + 1,
                "records": records,
                "conversation_metadata": conversation_metadata,
                "message_results_ordered": message_results_ordered,
                "computed_metadata": computed_md,
                "full_transcript": full_transcript,
                "conv_md_for_judge": conv_md_for_judge,
            }
            conv_state[conversation_id] = state
            conv_order.append(conversation_id)

    total_conversations = len(conv_order)
    if on_progress:
        on_progress(
            {
                "phase": "start",
                "total_conversations": total_conversations,
                "workers": workers,
            }
        )

    for conversation_id in conv_order:
        state = conv_state[conversation_id]
        for mr in state["message_results_ordered"]:
            results.message_level_results.append(mr)
            if on_message_result:
                try:
                    on_message_result(dict(mr))
                except Exception:
                    pass

    def _finalize_cl_record(conversation_id: str, cr: dict) -> dict:
        state = conv_state[conversation_id]
        cr["thread_id"] = conversation_id
        cr["conversation_metadata"] = state["conversation_metadata"]
        cr["computed_metadata"] = state["computed_metadata"]
        cr["transcript"] = state["records"]
        cr["message_level_results"] = state["message_results_ordered"]
        cr["evaluation_target_role"] = target_role
        if cr.get("parse_status") != "ok" and not cr.get("parsed_json"):
            cr["parsed_json"] = {
                "customer_objective_type": "Inquiry",
                "customer_primary_objective": "",
                "handled_status": "unhandled",
                "customer_experience": "bad",
                "unhandled_resolution_subtype": "totally_unresolved",
                "frustration_detected": True,
                "frustration_origin": "our_side",
                "customer_started_frustrated": False,
                "customer_became_frustrated_during_chat": True,
                "customer_ended_frustrated": False,
                "frustration_timing": "during",
                "final_customer_sentiment": "unknown",
                "max_frustration_level": state["computed_metadata"].get("max_frustration_level", "none"),
                "main_issue": {
                    "issue_exists": True,
                    "issue_origin": "our_side",
                    "issue_type": "other",
                    "issue_summary": "Conversation-level evaluator failed to parse",
                    "customer_impact": "Unable to assess automatically",
                },
                "all_detected_issues": [],
                "positive_signals": [],
                "negative_signals": [],
                "classification_reason": "The conversation-level evaluator failed, so the result is treated as unresolved and high risk for review.",
                "management_summary": "Automatic evaluation could not parse a result for this conversation. Manual review required.",
                "recommended_actions": ["Review this conversation manually."],
                "manual_review_required": True,
                "manual_review_reason": cr.get("error_message") or "Parse failure",
                "confidence": "low",
            }
        _apply_message_signal_to_conversation_score(
            cr.get("parsed_json") or {},
            state.get("computed_metadata") or {},
        )
        cr["evaluation_output"] = cr.get("parsed_json")
        return cr

    pending: set[cf.Future] = set()
    fut_info: dict[cf.Future, str] = {}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for conversation_id in conv_order:
            state = conv_state[conversation_id]
            if on_progress:
                on_progress(
                    {
                        "phase": "conversation_start",
                        "conversation_index": state["conversation_index"],
                        "conversation_id": conversation_id,
                        "agent_messages": len(state["message_results_ordered"]),
                        "target_messages": len(state["message_results_ordered"]),
                        "target_role": target_role,
                        "total_conversations": total_conversations,
                        "workers": workers,
                    }
                )
            fut = ex.submit(
                _eval_conversation_level,
                client=client,
                api=config.conversation_api_config(),
                conversation_id=conversation_id,
                conversation_metadata=state["conv_md_for_judge"],
                full_transcript=state["full_transcript"],
                message_level_evaluations=state["message_results_ordered"],
                computed_metadata=state["computed_metadata"],
                save_raw=config.save_raw_responses,
                truncate_chars=truncate_chars,
                prompt=config.conversation_prompt,
            )
            pending.add(fut)
            fut_info[fut] = conversation_id

        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                conversation_id = fut_info.pop(fut)
                state = conv_state[conversation_id]
                try:
                    cr = fut.result()
                except Exception as e:  # noqa: BLE001
                    cr = {
                        "conversation_id": conversation_id,
                        "raw_model_response": None,
                        "parsed_json": None,
                        "parse_status": "api_error",
                        "error_message": f"Worker raised: {e}",
                        "debug": None,
                    }
                cr = _finalize_cl_record(conversation_id, cr)

                if cr.get("parse_status") != "ok":
                    err = {
                        "level": "conversation",
                        "conversation_id": conversation_id,
                        "error": cr.get("error_message"),
                    }
                    results.errors.append(err)
                    if on_error:
                        try:
                            on_error(err)
                        except Exception:
                            pass

                results.conversation_results.append(cr)
                if on_conversation_result:
                    try:
                        on_conversation_result(cr)
                    except Exception:
                        pass

                if on_progress:
                    on_progress(
                        {
                            "phase": "conversation_done",
                            "conversation_index": state["conversation_index"],
                            "conversation_id": conversation_id,
                            "total_conversations": total_conversations,
                            "status": cr.get("parse_status"),
                            "automatic_reruns": int(cr.get("automatic_reruns") or 0),
                            "recovered_after_rerun": bool(cr.get("recovered_after_rerun")),
                            "rerun_errors": cr.get("rerun_errors") or [],
                        }
                    )

            if cancel_requested and cancel_requested():
                for fut in list(pending):
                    if not fut.done():
                        fut.cancel()
                    pending.discard(fut)
                break

    order_by_cid = {cid: i for i, cid in enumerate(conv_order)}
    results.conversation_results.sort(
        key=lambda c: order_by_cid.get(c.get("conversation_id"), 0)
    )
    results.finished_at = time.time()
    if on_progress:
        on_progress({"phase": "done", "total_conversations": total_conversations})
    return results
