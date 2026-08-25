from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from .config import PROJECT_DIR, get_settings


_ENGINE: Engine | None = None


def engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        settings = get_settings()
        _ENGINE = create_engine(settings.database_url, pool_pre_ping=True, future=True)
    return _ENGINE


def init_db() -> None:
    schema_path = PROJECT_DIR / "app" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    with engine().begin() as conn:
        conn.exec_driver_sql(sql)


@contextmanager
def tx() -> Iterable[Connection]:
    with engine().begin() as conn:
        yield conn


def row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(row._mapping if hasattr(row, "_mapping") else row)


def rows_to_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    return [row_to_dict(row) for row in rows]


def fetch_one(conn: Connection, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    row = conn.execute(text(sql), params or {}).mappings().first()
    return dict(row) if row else None


def fetch_all(conn: Connection, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(text(sql), params or {}).mappings().all()]


def execute(conn: Connection, sql: str, params: dict[str, Any] | None = None) -> None:
    conn.execute(text(sql), params or {})


def json_default(value: Any) -> str:
    return str(value)


def as_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=json_default)


def load_sql(name: str) -> str:
    return (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")

