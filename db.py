"""SQLite persistence layer.

Stores prompts, evaluation runs, conversation/message-level results, errors,
and arbitrary app settings.

The schema is created on first connection and the default prompt templates
(from :mod:`prompts`) are seeded if no rows exist for that kind. The DB file
defaults to ``./cx_evaluator.db`` next to the app.

All writes go through the :class:`Database` instance. SQLite is used in
``check_same_thread=False`` mode with an internal lock so the app can call it
from Streamlit callbacks without worrying about thread affinity.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from prompts import (
    DEFAULT_CONVERSATION_LEVEL_PROMPT,
    DEFAULT_MESSAGE_LEVEL_PROMPT,
    PromptTemplate,
)


DEFAULT_DB_PATH = Path("cx_evaluator.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    output_schema TEXT NOT NULL,
    user_prompt_template TEXT NOT NULL,
    is_default INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompts_kind ON prompt_templates(kind);
CREATE INDEX IF NOT EXISTS idx_prompts_active ON prompt_templates(kind, is_active);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    csv_name TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    run_config_json TEXT NOT NULL,
    message_prompt_id INTEGER,
    conversation_prompt_id INTEGER,
    n_conversations INTEGER NOT NULL DEFAULT 0,
    n_message_calls INTEGER NOT NULL DEFAULT 0,
    n_errors INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (message_prompt_id) REFERENCES prompt_templates(id),
    FOREIGN KEY (conversation_prompt_id) REFERENCES prompt_templates(id)
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS conversation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    parse_status TEXT NOT NULL,
    error_message TEXT,
    raw_response TEXT,
    parsed_json TEXT,
    conversation_metadata TEXT,
    computed_metadata TEXT,
    transcript_json TEXT,
    debug_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conv_results_run ON conversation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_conv_results_conv ON conversation_results(conversation_id);

CREATE TABLE IF NOT EXISTS message_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    target_message_id TEXT,
    message_index INTEGER,
    message_time TEXT,
    target_message_text TEXT,
    parse_status TEXT NOT NULL,
    error_message TEXT,
    raw_response TEXT,
    parsed_json TEXT,
    debug_json TEXT,
    input_history_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_msg_results_run ON message_results(run_id);
CREATE INDEX IF NOT EXISTS idx_msg_results_conv ON message_results(run_id, conversation_id);

CREATE TABLE IF NOT EXISTS run_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    level TEXT,
    conversation_id TEXT,
    message_index INTEGER,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
"""


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_load(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


class Database:
    """Thin wrapper around a SQLite file.

    Use one instance per app process. Methods acquire an internal lock so they
    are safe to call from multiple threads / streamlit reruns.
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock:
            self._conn.executescript(SCHEMA)
        self._seed_default_prompts()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # -------- internal --------

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _exec(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def _fetchall(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        return self._exec(sql, params).fetchall()

    def _fetchone(self, sql: str, params: Iterable = ()) -> Optional[sqlite3.Row]:
        return self._exec(sql, params).fetchone()

    # -------- settings (free-form key/value) --------

    def set_setting(self, key: str, value: Any) -> None:
        now = _now_iso()
        self._exec(
            "INSERT INTO settings(key, value, updated_at) VALUES(?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, _json_dump(value), now),
        )

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self._fetchone("SELECT value FROM settings WHERE key=?", (key,))
        if not row:
            return default
        return _json_load(row["value"])

    # -------- prompt templates --------

    def _seed_default_prompts(self) -> None:
        for kind, tpl in (
            ("message_level", DEFAULT_MESSAGE_LEVEL_PROMPT),
            ("conversation_level", DEFAULT_CONVERSATION_LEVEL_PROMPT),
        ):
            existing = self._fetchone(
                "SELECT id FROM prompt_templates WHERE kind=? AND is_default=1",
                (kind,),
            )
            if existing:
                now = _now_iso()
                self._exec(
                    "UPDATE prompt_templates SET system_prompt=?, output_schema=?, "
                    "user_prompt_template=?, updated_at=? WHERE id=?",
                    (
                        tpl.system_prompt,
                        tpl.output_schema,
                        tpl.user_prompt_template,
                        now,
                        int(existing["id"]),
                    ),
                )
                continue
            now = _now_iso()
            with self._tx() as c:
                # Deactivate any existing rows of this kind, then insert default as active.
                c.execute(
                    "UPDATE prompt_templates SET is_active=0 WHERE kind=?",
                    (kind,),
                )
                c.execute(
                    "INSERT INTO prompt_templates"
                    "(kind, name, system_prompt, output_schema, user_prompt_template, is_default, is_active, created_at, updated_at)"
                    " VALUES(?, ?, ?, ?, ?, 1, 1, ?, ?)",
                    (
                        kind,
                        "Default",
                        tpl.system_prompt,
                        tpl.output_schema,
                        tpl.user_prompt_template,
                        now,
                        now,
                    ),
                )

    def list_prompts(self, kind: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT id, kind, name, is_default, is_active, created_at, updated_at "
            "FROM prompt_templates WHERE kind=? ORDER BY is_active DESC, updated_at DESC",
            (kind,),
        )
        return [dict(r) for r in rows]

    def get_prompt(self, prompt_id: int) -> Optional[dict]:
        row = self._fetchone(
            "SELECT * FROM prompt_templates WHERE id=?",
            (int(prompt_id),),
        )
        return dict(row) if row else None

    def get_active_prompt(self, kind: str) -> Optional[dict]:
        row = self._fetchone(
            "SELECT * FROM prompt_templates WHERE kind=? AND is_active=1 LIMIT 1",
            (kind,),
        )
        if not row:
            # Fall back to default
            row = self._fetchone(
                "SELECT * FROM prompt_templates WHERE kind=? AND is_default=1 LIMIT 1",
                (kind,),
            )
        return dict(row) if row else None

    def get_active_prompt_template(self, kind: str) -> PromptTemplate:
        row = self.get_active_prompt(kind)
        if not row:
            # Fall back to in-memory defaults.
            return (
                DEFAULT_MESSAGE_LEVEL_PROMPT
                if kind == "message_level"
                else DEFAULT_CONVERSATION_LEVEL_PROMPT
            )
        return PromptTemplate(
            system_prompt=row["system_prompt"],
            output_schema=row["output_schema"],
            user_prompt_template=row["user_prompt_template"],
        )

    def save_prompt(
        self,
        kind: str,
        name: str,
        system_prompt: str,
        output_schema: str,
        user_prompt_template: str,
        set_active: bool = True,
    ) -> int:
        """Insert a new prompt version. If ``set_active`` is True it becomes
        the active prompt for this kind."""
        now = _now_iso()
        with self._tx() as c:
            if set_active:
                c.execute(
                    "UPDATE prompt_templates SET is_active=0 WHERE kind=?",
                    (kind,),
                )
            cur = c.execute(
                "INSERT INTO prompt_templates"
                "(kind, name, system_prompt, output_schema, user_prompt_template, is_default, is_active, created_at, updated_at)"
                " VALUES(?, ?, ?, ?, ?, 0, ?, ?, ?)",
                (
                    kind,
                    name or f"Custom {now}",
                    system_prompt,
                    output_schema,
                    user_prompt_template,
                    1 if set_active else 0,
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def set_active_prompt(self, prompt_id: int) -> None:
        row = self.get_prompt(prompt_id)
        if not row:
            raise ValueError(f"Prompt {prompt_id} not found")
        kind = row["kind"]
        now = _now_iso()
        with self._tx() as c:
            c.execute(
                "UPDATE prompt_templates SET is_active=0 WHERE kind=?",
                (kind,),
            )
            c.execute(
                "UPDATE prompt_templates SET is_active=1, updated_at=? WHERE id=?",
                (now, prompt_id),
            )

    def delete_prompt(self, prompt_id: int) -> None:
        row = self.get_prompt(prompt_id)
        if not row:
            return
        if row["is_default"]:
            raise ValueError("Cannot delete the default prompt.")
        with self._tx() as c:
            c.execute("DELETE FROM prompt_templates WHERE id=?", (prompt_id,))
            # If we just deleted the active one, fall back to the default.
            remaining = c.execute(
                "SELECT id FROM prompt_templates WHERE kind=? AND is_active=1 LIMIT 1",
                (row["kind"],),
            ).fetchone()
            if not remaining:
                default = c.execute(
                    "SELECT id FROM prompt_templates WHERE kind=? AND is_default=1 LIMIT 1",
                    (row["kind"],),
                ).fetchone()
                if default:
                    c.execute(
                        "UPDATE prompt_templates SET is_active=1 WHERE id=?",
                        (default["id"],),
                    )

    def reset_to_default(self, kind: str) -> None:
        """Make the seeded default active for this kind."""
        default = self._fetchone(
            "SELECT id FROM prompt_templates WHERE kind=? AND is_default=1 LIMIT 1",
            (kind,),
        )
        if not default:
            # Re-seed if somebody deleted the row at the SQL level.
            self._seed_default_prompts()
            return
        self.set_active_prompt(int(default["id"]))

    # -------- runs --------

    def start_run(
        self,
        csv_name: Optional[str],
        run_config: dict,
        message_prompt_id: Optional[int],
        conversation_prompt_id: Optional[int],
        name: Optional[str] = None,
    ) -> int:
        now = _now_iso()
        cur = self._exec(
            "INSERT INTO runs"
            "(name, csv_name, started_at, status, run_config_json, message_prompt_id, conversation_prompt_id)"
            " VALUES(?, ?, ?, 'running', ?, ?, ?)",
            (name, csv_name, now, _json_dump(run_config), message_prompt_id, conversation_prompt_id),
        )
        return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str,
        n_conversations: int,
        n_message_calls: int,
        n_errors: int,
    ) -> None:
        self._exec(
            "UPDATE runs SET finished_at=?, status=?, n_conversations=?, n_message_calls=?, n_errors=? WHERE id=?",
            (_now_iso(), status, int(n_conversations), int(n_message_calls), int(n_errors), int(run_id)),
        )

    def rename_run(self, run_id: int, name: str) -> None:
        self._exec("UPDATE runs SET name=? WHERE id=?", (name, int(run_id)))

    def list_runs(self, limit: int = 200) -> list[dict]:
        rows = self._fetchall(
            "SELECT id, name, csv_name, started_at, finished_at, status, "
            "n_conversations, n_message_calls, n_errors "
            "FROM runs ORDER BY started_at DESC LIMIT ?",
            (int(limit),),
        )
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM runs WHERE id=?", (int(run_id),))
        if not row:
            return None
        d = dict(row)
        d["run_config"] = _json_load(d.pop("run_config_json")) or {}
        return d

    def delete_run(self, run_id: int) -> None:
        # ON DELETE CASCADE handles related rows.
        self._exec("DELETE FROM runs WHERE id=?", (int(run_id),))

    # -------- results --------

    def save_message_result(self, run_id: int, mr: dict) -> int:
        now = _now_iso()
        cur = self._exec(
            "INSERT INTO message_results"
            "(run_id, conversation_id, target_message_id, message_index, message_time, target_message_text,"
            " parse_status, error_message, raw_response, parsed_json, debug_json, input_history_json, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                str(mr.get("thread_id") or mr.get("conversation_id", "")),
                mr.get("target_message_id"),
                int(mr["message_index"]) if mr.get("message_index") is not None else None,
                mr.get("message_time"),
                mr.get("target_message_text"),
                mr.get("parse_status", "ok"),
                mr.get("error_message"),
                mr.get("raw_model_response"),
                _json_dump(mr.get("evaluation_output", mr.get("parsed_json")))
                if mr.get("evaluation_output", mr.get("parsed_json")) is not None else None,
                _json_dump(mr.get("debug")) if mr.get("debug") is not None else None,
                _json_dump(mr.get("input_history")) if mr.get("input_history") is not None else None,
                now,
            ),
        )
        return int(cur.lastrowid)

    def save_conversation_result(self, run_id: int, cr: dict) -> int:
        now = _now_iso()
        cur = self._exec(
            "INSERT INTO conversation_results"
            "(run_id, conversation_id, parse_status, error_message, raw_response, parsed_json,"
            " conversation_metadata, computed_metadata, transcript_json, debug_json, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                str(cr.get("thread_id") or cr.get("conversation_id", "")),
                cr.get("parse_status", "ok"),
                cr.get("error_message"),
                cr.get("raw_model_response"),
                _json_dump(cr.get("evaluation_output", cr.get("parsed_json")))
                if cr.get("evaluation_output", cr.get("parsed_json")) is not None else None,
                _json_dump(cr.get("conversation_metadata")) if cr.get("conversation_metadata") is not None else None,
                _json_dump(cr.get("computed_metadata")) if cr.get("computed_metadata") is not None else None,
                _json_dump(cr.get("transcript")) if cr.get("transcript") is not None else None,
                _json_dump(cr.get("debug")) if cr.get("debug") is not None else None,
                now,
            ),
        )
        return int(cur.lastrowid)

    def save_error(self, run_id: int, err: dict) -> int:
        cur = self._exec(
            "INSERT INTO run_errors(run_id, level, conversation_id, message_index, error, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                err.get("level"),
                err.get("conversation_id"),
                int(err["message_index"]) if err.get("message_index") is not None else None,
                err.get("error"),
                _now_iso(),
            ),
        )
        return int(cur.lastrowid)

    def load_run_results(self, run_id: int) -> dict:
        """Reconstruct the structures the rest of the app uses for a saved run.

        Returns a dict with keys ``conversation_results``, ``message_level_results``,
        ``errors``, ``started_at``, ``finished_at``.
        """
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        conv_rows = self._fetchall(
            "SELECT * FROM conversation_results WHERE run_id=? ORDER BY id ASC",
            (int(run_id),),
        )
        conversation_results: list[dict] = []
        for r in conv_rows:
            d = dict(r)
            conversation_results.append(
                {
                    "thread_id": d["conversation_id"],
                    "conversation_id": d["conversation_id"],
                    "run_id": int(run_id),
                    "parse_status": d["parse_status"],
                    "error_message": d.get("error_message"),
                    "raw_model_response": d.get("raw_response"),
                    "parsed_json": _json_load(d.get("parsed_json")),
                    "evaluation_output": _json_load(d.get("parsed_json")),
                    "conversation_metadata": _json_load(d.get("conversation_metadata")) or {},
                    "computed_metadata": _json_load(d.get("computed_metadata")) or {},
                    "transcript": _json_load(d.get("transcript_json")) or [],
                    "debug": _json_load(d.get("debug_json")),
                    "message_level_results": [],  # filled below
                }
            )

        msg_rows = self._fetchall(
            "SELECT * FROM message_results WHERE run_id=? ORDER BY conversation_id, message_index ASC",
            (int(run_id),),
        )
        message_level_results: list[dict] = []
        by_conv: dict[str, list[dict]] = {}
        for r in msg_rows:
            d = dict(r)
            mr = {
                "thread_id": d["conversation_id"],
                "conversation_id": d["conversation_id"],
                "run_id": int(run_id),
                "target_message_id": d.get("target_message_id"),
                "message_index": d.get("message_index"),
                "message_time": d.get("message_time"),
                "target_message_text": d.get("target_message_text"),
                "parse_status": d.get("parse_status"),
                "error_message": d.get("error_message"),
                "raw_model_response": d.get("raw_response"),
                "parsed_json": _json_load(d.get("parsed_json")),
                "evaluation_output": _json_load(d.get("parsed_json")),
                "debug": _json_load(d.get("debug_json")),
                "input_history": _json_load(d.get("input_history_json")),
            }
            message_level_results.append(mr)
            by_conv.setdefault(mr["conversation_id"], []).append(mr)

        for c in conversation_results:
            c["message_level_results"] = by_conv.get(c["conversation_id"], [])

        err_rows = self._fetchall(
            "SELECT level, conversation_id, message_index, error FROM run_errors WHERE run_id=? ORDER BY id ASC",
            (int(run_id),),
        )
        errors = [dict(r) for r in err_rows]

        # Convert started/finished ISO strings to epoch floats so RunResults.duration math works.
        def _to_epoch(iso: Optional[str]) -> float:
            if not iso:
                return 0.0
            try:
                if iso.endswith("Z"):
                    iso = iso[:-1]
                return datetime.fromisoformat(iso).timestamp()
            except Exception:
                return 0.0

        return {
            "run": run,
            "conversation_results": conversation_results,
            "message_level_results": message_level_results,
            "errors": errors,
            "started_at": _to_epoch(run.get("started_at")),
            "finished_at": _to_epoch(run.get("finished_at")),
        }
