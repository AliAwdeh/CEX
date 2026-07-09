"""Ad-hoc report: for already-evaluated runs, break down whether detected
issues were attributed to the bot or to a human agent.

Reuses the AI judge's own `culprits` classification (restricted to
agent | bot | broadcast | customer) already stored in
conversation_results.parsed_json - no re-evaluation needed.

Usage:
    python bot_vs_agent_breakdown.py                  # all runs
    python bot_vs_agent_breakdown.py --run-id 12 15    # specific runs
    python bot_vs_agent_breakdown.py --out detail.csv  # also dump per-conversation rows
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from db import Database

VALID_CULPRITS = {"agent", "bot", "broadcast", "customer"}


def collect_rows(db: Database, run_ids: list[int] | None) -> list[dict]:
    runs = db.list_runs(limit=1000)
    if run_ids:
        wanted = set(run_ids)
        runs = [r for r in runs if r["id"] in wanted]

    rows = []
    for run in runs:
        data = db.load_run_summary_results(run["id"])
        for cr in data["conversation_results"]:
            pj = cr.get("parsed_json") or {}
            main_issue = pj.get("main_issue") or {}
            if not isinstance(main_issue, dict):
                main_issue = {}
            has_issue = bool(main_issue.get("issue_exists")) or bool(pj.get("all_detected_issues"))

            culprits = pj.get("culprits") or []
            if not isinstance(culprits, list):
                culprits = []
            culprits = [str(c).strip().lower() for c in culprits if c]

            rows.append(
                {
                    "run_id": run["id"],
                    "run_name": run.get("name"),
                    "conversation_id": cr.get("conversation_id"),
                    "has_issue": has_issue,
                    "culprits": culprits,
                    "culprit_reason": pj.get("culprit_reason"),
                    "main_issue_type": main_issue.get("issue_type") or pj.get("main_issue_type"),
                }
            )
    return rows


def summarize(rows: list[dict]) -> pd.DataFrame:
    issue_rows = [r for r in rows if r["has_issue"]]
    counter: Counter = Counter()
    for r in issue_rows:
        culprits = r["culprits"] or ["unspecified"]
        for c in culprits:
            counter[c if c in VALID_CULPRITS else "unspecified"] += 1

    total = len(issue_rows)
    data = [
        {
            "culprit": culprit,
            "count": count,
            "pct_of_issues": round(count / total * 100, 1) if total else 0.0,
        }
        for culprit, count in counter.most_common()
    ]
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=int, nargs="*", default=None, help="Restrict to these run IDs (default: all runs)")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path for per-conversation detail rows")
    parser.add_argument("--db-path", type=str, default=None, help="Path to cx_evaluator.db (default: ./cx_evaluator.db)")
    args = parser.parse_args()

    db = Database(args.db_path) if args.db_path else Database()
    rows = collect_rows(db, args.run_id)

    n_issues = sum(1 for r in rows if r["has_issue"])
    print(f"Evaluated conversations scanned: {len(rows)}")
    print(f"Conversations with a detected issue: {n_issues}")
    print()
    print(summarize(rows).to_string(index=False))

    if args.out:
        detail = pd.DataFrame(
            [{**r, "culprits": " | ".join(r["culprits"])} for r in rows if r["has_issue"]]
        )
        detail.to_csv(args.out, index=False)
        print(f"\nWrote {len(detail)} per-conversation rows to {args.out}")


if __name__ == "__main__":
    main()
