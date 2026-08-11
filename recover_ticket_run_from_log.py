from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_loader import (
    conversation_metadata_from_group,
    get_conversation_groups,
    load_csv,
    message_records_from_group,
    normalize_dataframe,
)
from evaluator import extract_json_object


CONTEXT_RE = re.compile(
    r"^ticket_segmentation:(?P<conversation_id>[^:]+):"
    r"pass(?P<pass_index>\d+)/(?P<total_passes>\d+):(?P<source_id>.*)$"
)


def recover(
    log_path: Path,
    run_date: str,
    output_dir: Path,
    csv_path: Path | None = None,
) -> tuple[Path, Path]:
    journeys: dict[str, dict[str, Any]] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if not str(entry.get("timestamp") or "").startswith(run_date):
                continue
            match = CONTEXT_RE.match(str(entry.get("context") or ""))
            if not match:
                continue

            conversation_id = match.group("conversation_id")
            pass_index = int(match.group("pass_index"))
            total_passes = int(match.group("total_passes"))
            journey = journeys.setdefault(
                conversation_id,
                {
                    "conversation_id": conversation_id,
                    "total_passes": total_passes,
                    "successful_passes": [],
                    "failed_passes": [],
                    "latest_success": None,
                    "conversation_metadata": {},
                    "records_by_index": {},
                },
            )
            journey["total_passes"] = max(journey["total_passes"], total_passes)

            try:
                payload = extract_json_object(str(entry.get("user_prompt") or ""))
            except Exception:
                payload = {}
            metadata = payload.get("conversation_metadata")
            if isinstance(metadata, dict):
                journey["conversation_metadata"].update(metadata)
            for block in payload.get("source_conversation_blocks") or []:
                if not isinstance(block, dict):
                    continue
                source_id = str(block.get("source_conversation_id") or "")
                for message in block.get("messages") or []:
                    if not isinstance(message, dict):
                        continue
                    try:
                        message_index = int(message.get("message_index"))
                    except (TypeError, ValueError):
                        continue
                    journey["records_by_index"][message_index] = {
                        "conversation_id": conversation_id,
                        "source_conversation_id": source_id,
                        "message_index": message_index,
                        "sender_role": str(message.get("role") or "unknown"),
                        "message_text": str(message.get("text") or ""),
                        "message_time": message.get("time"),
                    }

            if entry.get("success"):
                journey["successful_passes"].append(pass_index)
                current = journey.get("latest_success")
                if current is None or pass_index >= current["pass_index"]:
                    response_text = str(entry.get("response_text") or "")
                    parse_error = ""
                    parsed_response = None
                    try:
                        parsed_response = extract_json_object(response_text)
                    except Exception as exc:  # noqa: BLE001
                        parse_error = str(exc)
                    journey["latest_success"] = {
                        "pass_index": pass_index,
                        "source_conversation_id": match.group("source_id"),
                        "timestamp": entry.get("timestamp"),
                        "parsed_response": parsed_response,
                        "parse_error": parse_error,
                        "raw_response": response_text,
                    }
            else:
                journey["failed_passes"].append(pass_index)

    recovered: list[dict[str, Any]] = []
    for journey in journeys.values():
        successful = sorted(set(journey.pop("successful_passes")))
        failed = sorted(set(journey.pop("failed_passes")))
        records_by_index = journey.pop("records_by_index")
        journey["records"] = [records_by_index[index] for index in sorted(records_by_index)]
        total_passes = int(journey["total_passes"])
        missing = [index for index in range(1, total_passes + 1) if index not in successful]
        latest = journey.get("latest_success") or {}
        parsed = latest.get("parsed_response") or {}
        journey.update(
            {
                "successful_passes": successful,
                "failed_passes": failed,
                "missing_passes": missing,
                "highest_successful_pass": max(successful, default=0),
                "complete": not missing,
                "recovered_ticket_count": len(parsed.get("tickets") or []),
            }
        )
        recovered.append(journey)

    matched_csv_journeys = 0
    if csv_path is not None:
        recovered_by_id = {journey["conversation_id"]: journey for journey in recovered}
        dataframe = normalize_dataframe(load_csv(csv_path))
        for conversation_id, group in get_conversation_groups(dataframe):
            journey = recovered_by_id.get(conversation_id)
            if journey is None:
                continue
            journey["conversation_metadata"] = conversation_metadata_from_group(group)
            journey["records"] = message_records_from_group(group, conversation_id)
            matched_csv_journeys += 1

    recovered.sort(key=lambda item: str((item.get("latest_success") or {}).get("timestamp") or ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ticket_run_{run_date}_recovered"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_summary.csv"

    bundle = {
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "source_log": str(log_path.resolve()),
        "source_csv": str(csv_path.resolve()) if csv_path is not None else None,
        "csv_matched_journey_count": matched_csv_journeys,
        "journey_count": len(recovered),
        "complete_journey_count": sum(bool(item["complete"]) for item in recovered),
        "partial_journey_count": sum(not bool(item["complete"]) for item in recovered),
        "note": (
            "Each journey contains the latest successful cumulative model response. "
            "Partial journeys include all ticket state accumulated through highest_successful_pass."
        ),
        "journeys": recovered,
    }
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversation_id",
                "complete",
                "highest_successful_pass",
                "total_passes",
                "recovered_ticket_count",
                "customer_name",
                "failed_passes",
                "missing_passes",
                "latest_timestamp",
            ],
        )
        writer.writeheader()
        for journey in recovered:
            latest = journey.get("latest_success") or {}
            writer.writerow(
                {
                    "conversation_id": journey["conversation_id"],
                    "complete": journey["complete"],
                    "highest_successful_pass": journey["highest_successful_pass"],
                    "total_passes": journey["total_passes"],
                    "recovered_ticket_count": journey["recovered_ticket_count"],
                    "customer_name": (journey.get("conversation_metadata") or {}).get("customer_name") or "",
                    "failed_passes": ",".join(map(str, journey["failed_passes"])),
                    "missing_passes": ",".join(map(str, journey["missing_passes"])),
                    "latest_timestamp": latest.get("timestamp") or "",
                }
            )

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover cumulative ticket results from LLM logs.")
    parser.add_argument("--date", required=True, help="UTC date in YYYY-MM-DD format")
    parser.add_argument("--log", type=Path, default=Path("logs/llm_calls.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("recovered_runs"))
    parser.add_argument("--csv", type=Path, help="Original CSV used to restore full transcripts and customer metadata")
    args = parser.parse_args()
    json_path, csv_path = recover(args.log, args.date, args.output_dir, args.csv)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()