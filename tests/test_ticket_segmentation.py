"""Regression tests for evaluator.py's ticket-segmentation plumbing.

What is left of that pipeline after the heuristics were removed is plumbing
only: renumbering, carry-forward across cumulative passes, prompt context and
conversation summaries. It must not add, remove, merge, split or re-label a
ticket, so these are the only deterministic behaviours worth pinning here --
anything about *how* a journey should be segmented belongs to the prompt.

Unit tests against those functions directly (no real LLM call, except the one
test that monkeypatches evaluator.chat_completion to simulate a mid-sequence
cumulative-pass failure).
"""

from __future__ import annotations

import json

import pytest

import evaluator as ev
from tests.conftest import make_record, make_ticket


# ---------------------------------------------------------------------------
# Bug: _carry_forward_missing_previous_tickets only restored a previous
# ticket that disappeared entirely, not one that was partially re-stated.
# ---------------------------------------------------------------------------


def test_carry_forward_restores_only_missing_remainder_on_partial_loss():
    records = [make_record(i, "customer", f"message {i} about the visa process") for i in range(1, 6)]
    previous = [
        make_ticket(
            "ticket_1", [1, 2, 3, 4, 5], ticket_type="visa_processing", customer_objective="Complete the visa process"
        )
    ]
    # This pass's model output re-states only 3,4,5 of the same ticket,
    # silently dropping 1,2 -- a partial re-statement, not total disappearance.
    current = [
        make_ticket(
            "ticket_1", [3, 4, 5], ticket_type="visa_processing", customer_objective="Complete the visa process"
        )
    ]

    result = ev._carry_forward_missing_previous_tickets(previous, current, records)

    covered = sorted({idx for t in result for idx in t["included_message_indexes"]})
    assert covered == [1, 2, 3, 4, 5], f"partially-lost indexes were not restored: {result}"


def test_carry_forward_restores_whole_ticket_on_total_loss():
    records = [make_record(i, "customer", f"message {i} about the visa process") for i in range(1, 4)] + [
        make_record(i, "customer", f"unrelated message {i}") for i in range(4, 6)
    ]
    previous = [
        make_ticket(
            "ticket_1", [1, 2, 3], ticket_type="visa_processing", customer_objective="Complete the visa process"
        )
    ]
    current = [
        make_ticket("ticket_2", [4, 5], ticket_type="general_inquiry", customer_objective="Unrelated question")
    ]

    result = ev._carry_forward_missing_previous_tickets(previous, current, records)

    covered = sorted({idx for t in result for idx in t["included_message_indexes"]})
    assert covered == [1, 2, 3, 4, 5], f"totally-dropped ticket was not restored: {result}"


def test_carry_forward_is_noop_when_nothing_missing():
    records = [make_record(i, "customer", f"message {i}") for i in range(1, 4)]
    previous = [make_ticket("ticket_1", [1, 2, 3])]
    current = [make_ticket("ticket_1", [1, 2, 3])]
    result = ev._carry_forward_missing_previous_tickets(previous, current, records)
    assert result == current


def test_renumber_tickets_clears_self_referential_or_dangling_previous_ids():
    tickets = [
        make_ticket("old_1", [1], previous_ticket_id="old_1"),
        make_ticket("old_2", [2], previous_ticket_id="missing_ticket"),
        make_ticket("old_3", [3], previous_ticket_id="old_1"),
    ]

    result = ev._renumber_tickets(tickets)

    assert result[0]["previous_ticket_id"] == ""
    assert result[1]["previous_ticket_id"] == ""
    assert result[2]["previous_ticket_id"] == "ticket_1"


# ---------------------------------------------------------------------------
# Bug: a single cumulative pass's JSON-parse/API failure discarded every
# ticket accumulated from earlier passes in the same sequence.
# ---------------------------------------------------------------------------


def test_cumulative_mode_survives_single_pass_failure(monkeypatch):
    records = [
        make_record(1, "customer", "Please renew my visa.", source="conv_a"),
        make_record(2, "agent", "Sure, we started the renewal.", source="conv_a"),
        make_record(3, "customer", "Any update on the salary?", source="conv_b"),
        make_record(4, "agent", "Checking now.", source="conv_b"),
    ]

    call_count = {"n": 0}

    def fake_chat_completion(client, api, system_prompt, user_prompt, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            payload = {
                "tickets": [
                    {
                        "ticket_id": "ticket_1",
                        "ticket_category": "request",
                        "ticket_type": "visa_processing",
                        "customer_objective": "Renew the visa",
                        "start_message_index": 1,
                        "end_message_index": 2,
                        "included_message_indexes": [1, 2],
                        "status": "pending_unresolved",
                        "should_append_future_conversations": True,
                        "previous_ticket_id": "",
                        "inquiries": [],
                        "segmentation_reason": "visa renewal",
                    }
                ]
            }
            return json.dumps(payload), {}
        raise RuntimeError("simulated API failure on pass 2")

    monkeypatch.setattr(ev, "chat_completion", fake_chat_completion)

    tickets, debug = ev._eval_ticket_segmentation_cumulative(
        client=None,
        api=None,
        conversation_id="conv_test",
        records=records,
        conversation_metadata={},
        truncate_chars=None,
    )

    # Pass 1's visa ticket must survive pass 2's failure, not be wiped out.
    visa_tickets = [t for t in tickets if t.get("ticket_type") == "visa_processing"]
    assert visa_tickets, f"visa ticket from pass 1 was lost after pass 2 failed: {tickets}"
    assert 1 in visa_tickets[0]["included_message_indexes"]
    assert 2 in visa_tickets[0]["included_message_indexes"]

    # Pass 2 failed, so it must contribute NO ticket. Inventing one here used to
    # emit a synthetic "general_inquiry / Original unsplit customer journey"
    # ticket that was indistinguishable from a real classification, which is how
    # dead API connections came to look like segmentation decisions.
    covered = {idx for t in tickets for idx in t["included_message_indexes"]}
    assert 3 not in covered and 4 not in covered, (
        f"failed pass 2 fabricated a ticket for its messages: {tickets}"
    )
    assert not any(t.get("ticket_type") == "general_inquiry" for t in tickets)

    # ...and the failure must be reported rather than silently swallowed.
    failed = debug["failed_passes"]
    assert len(failed) == 1
    assert failed[0]["pass_index"] == 2
    assert failed[0]["source_conversation_id"] == "conv_b"
    assert failed[0]["message_indexes"] == [3, 4]
    assert "simulated API failure on pass 2" in failed[0]["error"]
    assert call_count["n"] == 2


def test_total_cumulative_failure_raises_instead_of_inventing_a_ticket(monkeypatch):
    records = [
        make_record(1, "customer", "Please renew my visa.", source="conv_a"),
        make_record(2, "agent", "Sure, we started the renewal.", source="conv_a"),
    ]

    def always_fails(client, api, system_prompt, user_prompt, **kwargs):
        raise RuntimeError("Connection error.")

    monkeypatch.setattr(ev, "chat_completion", always_fails)

    with pytest.raises(ev.TicketSegmentationError) as excinfo:
        ev._eval_ticket_segmentation_cumulative(
            client=None,
            api=None,
            conversation_id="conv_test",
            records=records,
            conversation_metadata={},
            truncate_chars=None,
        )
    assert "Connection error." in str(excinfo.value)


def test_empty_model_output_raises_instead_of_inventing_a_ticket():
    records = [make_record(1, "customer", "Please renew my visa.", source="conv_a")]

    with pytest.raises(ev.TicketSegmentationError):
        ev._normalize_ticket_segments({"tickets": []}, records)


# ---------------------------------------------------------------------------
# Conversation summaries: what each cumulative pass carries forward about the
# source conversations it has already processed.
# ---------------------------------------------------------------------------


def test_ticket_prompt_context_keeps_structured_conversation_summaries():
    records = [
        make_record(1, "customer", "Where is the salary card?", source="salary_1"),
        make_record(2, "agent", "It is not ready yet.", source="salary_1"),
    ]
    ticket = make_ticket(
        "ticket_1",
        [1, 2],
        ticket_category="inquiry",
        ticket_type="salary",
        customer_objective="Understand salary setup and card readiness",
        status="pending_unresolved",
    )
    ticket["conversation_summaries"] = [
        {
            "source_conversation_id": "salary_1",
            "message_indexes": [1, 2],
            "customer_intent": "Learn when the salary card will be ready",
            "outcome": "The card is not ready yet",
            "status": "pending_unresolved",
            "ticket_signals": ["salary_card", "readiness"],
        }
    ]

    context = ev._ticket_prompt_context([ticket], records)

    assert context["tickets"][0]["conversation_summaries"] == ticket["conversation_summaries"]


def test_conversation_summaries_are_clipped_and_grouped_by_actual_source():
    records = [
        make_record(1, "customer", "Where is the salary card?", source="salary_1"),
        make_record(2, "agent", "It is not ready yet.", source="salary_1"),
        make_record(3, "customer", "Following up on the salary card.", source="salary_2"),
        make_record(4, "agent", "We will update you tomorrow.", source="salary_2"),
    ]
    records_by_idx = ev._records_by_message_index(records)
    ticket = make_ticket(
        "ticket_1",
        [1, 2, 3],
        ticket_category="inquiry",
        ticket_type="salary",
        customer_objective="Learn when the salary card will be ready",
        status="pending_unresolved",
    )
    ticket["conversation_summaries"] = [
        {
            "source_conversation_id": "incorrect_source",
            "message_indexes": [1, 2, 3, 4, 999],
            "customer_intent": "Learn when the salary card will be ready",
            "outcome": "The card remains pending",
            "status": "pending_unresolved",
            "ticket_signals": ["salary_card"],
        }
    ]

    summaries = ev._normalize_ticket_conversation_summaries(ticket, [1, 2, 3], records_by_idx)

    assert [(summary["source_conversation_id"], summary["message_indexes"]) for summary in summaries] == [
        ("salary_1", [1, 2]),
        ("salary_2", [3]),
    ]


def test_cumulative_summary_carry_forward_preserves_prior_source_history():
    previous = make_ticket(
        "ticket_1",
        [1, 2],
        ticket_category="inquiry",
        ticket_type="salary",
        customer_objective="Learn when the salary card will be ready",
        status="pending_unresolved",
    )
    previous["conversation_summaries"] = [
        {
            "source_conversation_id": "salary_1",
            "message_indexes": [1, 2],
            "customer_intent": "Learn when the salary card will be ready",
            "outcome": "The card was not ready",
            "status": "pending_unresolved",
            "ticket_signals": ["salary_card", "readiness"],
        }
    ]
    current = make_ticket(
        "ticket_1",
        [1, 2, 3, 4],
        ticket_category="inquiry",
        ticket_type="salary",
        customer_objective="Learn when the salary card will be ready",
        status="pending_unresolved",
    )
    current["conversation_summaries"] = [
        {
            "source_conversation_id": "salary_2",
            "message_indexes": [3, 4],
            "customer_intent": "Follow up on salary-card readiness",
            "outcome": "The card remains pending",
            "status": "pending_unresolved",
            "ticket_signals": ["salary_card", "follow_up"],
        }
    ]

    carried = ev._carry_forward_previous_conversation_summaries([previous], [current])

    assert [summary["source_conversation_id"] for summary in carried[0]["conversation_summaries"]] == [
        "salary_1",
        "salary_2",
    ]
