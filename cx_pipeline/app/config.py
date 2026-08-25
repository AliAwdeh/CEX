from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, str(default))).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class PipelineSettings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg://localhost:5432/cex_pipeline")
    input_dir: str = os.getenv("INPUT_CSV_DIR", str(PROJECT_DIR / "data" / "inbox"))
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://langcc.maidstech.ai/v1")
    message_model: str = os.getenv("MESSAGE_MODEL", "openai/gpt-5.4-mini")
    conversation_model: str = os.getenv("CONVERSATION_MODEL", "openai/gpt-5.4-mini")
    ticket_model: str = os.getenv("TICKET_MODEL", "openai/gpt-5.4-mini")
    service_tier: str | None = os.getenv("SERVICE_TIER") or None
    message_thinking_effort: str = os.getenv("MESSAGE_THINKING_EFFORT", "medium")
    conversation_thinking_effort: str = os.getenv("CONVERSATION_THINKING_EFFORT", "medium")
    ticket_thinking_effort: str = os.getenv("TICKET_THINKING_EFFORT", "medium")
    max_total_workers: int = _int_env("MAX_TOTAL_WORKERS", 20)
    max_ticketing_workers: int = _int_env("MAX_TICKETING_WORKERS", 4)
    max_message_workers: int = _int_env("MAX_MESSAGE_WORKERS", 12)
    max_ticket_cx_workers: int = _int_env("MAX_TICKET_CX_WORKERS", 4)
    timeout: float = _float_env("OPENAI_TIMEOUT", 600.0)
    retries: int = _int_env("OPENAI_RETRIES", 2)
    temperature: float = _float_env("OPENAI_TEMPERATURE", 0.1)
    top_p: float = _float_env("OPENAI_TOP_P", 1.0)
    message_target_role: str = os.getenv("MESSAGE_TARGET_ROLE", "agent")
    save_raw_responses: bool = _bool_env("SAVE_RAW_RESPONSES", False)
    debug_log_calls: bool = _bool_env("DEBUG_LOG_CALLS", False)
    dashboard_webhook_url: str = os.getenv(
        "DASHBOARD_WEBHOOK_URL",
        "http://127.0.0.1:8090/api/webhooks/analysis-run-finished",
    )
    dashboard_webhook_secret: str = os.getenv("DASHBOARD_WEBHOOK_SECRET", "")
    dashboard_webhook_timeout: float = _float_env("DASHBOARD_WEBHOOK_TIMEOUT", 10.0)


def get_settings() -> PipelineSettings:
    return PipelineSettings()
