from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

from .config import PipelineSettings, get_settings


def notify_dashboard_run_finished(
    *,
    run_id: str,
    status: str,
    stats: dict[str, Any] | None = None,
    error: str = "",
    settings: PipelineSettings | None = None,
) -> bool:
    settings = settings or get_settings()
    if not settings.dashboard_webhook_url:
        return False

    body = json.dumps(
        {
            "event": "analysis_run_finished",
            "run_id": run_id,
            "status": status,
            "stats": stats or {},
            "error": error,
        },
        default=str,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.dashboard_webhook_secret:
        headers["X-CX-Dashboard-Webhook-Secret"] = settings.dashboard_webhook_secret

    request = urllib.request.Request(
        settings.dashboard_webhook_url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.dashboard_webhook_timeout) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"Dashboard webhook failed for run {run_id}: {exc}", file=sys.stderr, flush=True)
        return False
