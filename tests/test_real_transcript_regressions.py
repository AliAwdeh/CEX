"""Regression tests derived from real (lightly-scrubbed) production transcripts
that exposed ticket-segmentation mistakes.

Most of this file used to assert that post-processing heuristics reshaped the
model's tickets -- merging, splitting and re-labelling them. Those heuristics
are gone: segmentation is the prompt's job now, so a hand-written ticket fed to
_normalize_ticket_segments comes back unchanged and those assertions could only
ever fail. What is left are the two behaviours this module still owns: the
model's own status survives, and a worker's real name never reaches a
ticket_type or customer_objective. The deleted scenarios belong in a
prompt-level eval, not here.
"""

from __future__ import annotations

import evaluator as ev
from tests.conftest import make_record, make_ticket


def test_explicit_current_direct_debit_recall_can_resolve_dispute():
    records = [
        make_record(1, "customer", "Stop and recall the current direct debit instruction.", source="dd"),
        make_record(2, "agent", "The current direct debit instruction has been recalled successfully.", source="dd"),
        make_record(3, "customer", "Great, thank you.", source="dd"),
    ]
    raw = {
        "tickets": [
            make_ticket(
                "ticket_1",
                [1, 2, 3],
                ticket_category="issue",
                model_ticket_category="issue",
                ticket_type="direct_debit_dispute",
                customer_objective="Recall the unwanted current direct debit",
                status="resolved",
            )
        ]
    }

    result = ev._normalize_ticket_segments(raw, records)

    assert len(result) == 1
    assert result[0]["status"] == "resolved"
    assert result[0]["should_append_future_conversations"] is False


def test_worker_name_is_removed_from_generic_visa_objective():
    records = [
        make_record(1, "customer", "What is the update on Ravina Visa?", source="visa"),
        make_record(2, "agent", "Ravina's medical and residency steps are still processing.", source="visa"),
        make_record(3, "customer", "Please complete her visa and Emirates ID process.", source="visa"),
    ]
    raw = {
        "tickets": [
            make_ticket(
                "ticket_1",
                [1, 2, 3],
                ticket_category="request",
                model_ticket_category="request",
                ticket_type="ravina_employment_onboarding",
                customer_objective="Complete employment onboarding and visa/residency process for maid Ravina",
                status="pending_unresolved",
            )
        ]
    }

    result = ev._normalize_ticket_segments(raw, records)

    assert len(result) == 1
    assert result[0]["ticket_type"] == "employment_onboarding"
    assert result[0]["customer_objective"] == "Complete employment onboarding and visa/residency process"
    assert "ravina" not in result[0]["customer_objective"].lower()
