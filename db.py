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

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
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
    source_conversation_id TEXT,
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

CREATE TABLE IF NOT EXISTS reviewer_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reviewer_name TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_by TEXT,
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reviewer_keys_active ON reviewer_keys(is_active);

CREATE TABLE IF NOT EXISTS journey_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    reviewer_key_id INTEGER,
    reviewer_name TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    review_comment TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
    FOREIGN KEY (reviewer_key_id) REFERENCES reviewer_keys(id),
    UNIQUE(run_id, conversation_id, reviewer_name)
);

CREATE INDEX IF NOT EXISTS idx_journey_reviews_lookup ON journey_reviews(run_id, conversation_id);

CREATE TABLE IF NOT EXISTS journey_review_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    reviewer_key_id INTEGER,
    reviewer_name TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    review_comment TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
    FOREIGN KEY (reviewer_key_id) REFERENCES reviewer_keys(id)
);

CREATE INDEX IF NOT EXISTS idx_journey_review_history_lookup ON journey_review_history(run_id, conversation_id, reviewer_name, reviewed_at);
"""


_SECRET_HASH_ALGO = "pbkdf2_sha256"
_SECRET_HASH_ITERATIONS = 260_000
_NOW_LOCK = threading.Lock()
_LAST_NOW: datetime | None = None


def _now_iso() -> str:
    global _LAST_NOW
    with _NOW_LOCK:
        now = datetime.utcnow()
        if _LAST_NOW is not None and now <= _LAST_NOW:
            now = _LAST_NOW + timedelta(microseconds=1)
        _LAST_NOW = now
        return now.isoformat(timespec="microseconds") + "Z"


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_load(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _hash_secret(secret: str, salt_hex: str | None = None) -> str:
    """Return a salted PBKDF2 hash suitable for storing auth secrets."""
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        bytes.fromhex(salt_hex),
        _SECRET_HASH_ITERATIONS,
    ).hex()
    return f"{_SECRET_HASH_ALGO}${_SECRET_HASH_ITERATIONS}${salt_hex}${digest}"


def _verify_secret(secret: str, encoded: str) -> bool:
    try:
        algo, iterations, salt_hex, digest = str(encoded or "").split("$", 3)
        if algo != _SECRET_HASH_ALGO or int(iterations) != _SECRET_HASH_ITERATIONS:
            return False
        candidate = _hash_secret(secret, salt_hex).rsplit("$", 1)[-1]
        return hmac.compare_digest(candidate, digest)
    except Exception:
        return False


def _backfill_conversation_parsed_json(pj: Any) -> None:
    """Fill in new marker fields that are missing from old stored results.

    Old runs stored final_classification/cx_issue_severity instead of the
    separate handled_status / customer_experience markers.  When loading from
    the DB we patch the dict in-place so the UI always has the fields it needs.
    """
    if not isinstance(pj, dict):
        return
    old_class = str(pj.get("final_classification", "") or "")
    old_class_l = old_class.strip().lower()
    old_severity = str(pj.get("cx_issue_severity", "") or "").strip().lower()
    if "handled_status" not in pj or not pj["handled_status"]:
        pj["handled_status"] = (
            "unhandled"
            if old_class_l.startswith(("unhandled", "not handled"))
            else "handled"
        )
    legacy_bad_experience = old_severity == "many" or any(
        marker in old_class_l for marker in ("many", "caused", "frustration")
    )
    customer_experience = str(pj.get("customer_experience", "") or "").strip().lower()
    if customer_experience not in {"good", "bad"} or (
        customer_experience == "good" and legacy_bad_experience
    ):
        pj["customer_experience"] = "bad" if legacy_bad_experience else "good"
    if "unhandled_resolution_subtype" not in pj or not pj["unhandled_resolution_subtype"]:
        pj["unhandled_resolution_subtype"] = (
            "not_applicable"
            if pj["handled_status"] == "handled"
            else "pending_unresolved"
            if "pending" in old_class_l
            else "totally_unresolved"
        )
    if "frustration_origin" not in pj or not pj["frustration_origin"]:
        origin = str(pj.get("main_issue_origin", "") or "").strip().lower()
        origin_map = {
            "our_side": "our_side",
            "customer": "customer_side",
            "customer_side": "customer_side",
            "shared": "shared",
            "none": "none",
        }
        pj["frustration_origin"] = origin_map.get(
            origin,
            "our_side" if pj.get("frustration_detected") and "caused" in old_class_l else "none",
        )


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
        self._ensure_runtime_columns()
        self._seed_default_prompts()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def vacuum(self) -> None:
        """Reclaim disk space freed by deleted rows.

        SQLite's DELETE only marks pages as free for reuse; it never shrinks
        the file on disk. VACUUM rewrites the whole database file compactly,
        which is the only way to actually reclaim that space after deleting
        runs. This requires no other pending transaction on the connection,
        so it's run outside of ``_tx``/autocommit statement batching.
        """
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.execute("VACUUM")

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

    def _ensure_runtime_columns(self) -> None:
        """Apply small additive migrations for existing local SQLite files."""
        with self._lock:
            cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(message_results)").fetchall()
            }
            if "source_conversation_id" not in cols:
                self._conn.execute("ALTER TABLE message_results ADD COLUMN source_conversation_id TEXT")

            review_cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(journey_reviews)").fetchall()
            }
            if "review_comment" not in review_cols:
                self._conn.execute("ALTER TABLE journey_reviews ADD COLUMN review_comment TEXT")

            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS journey_review_history ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "run_id INTEGER NOT NULL, "
                "conversation_id TEXT NOT NULL, "
                "reviewer_key_id INTEGER, "
                "reviewer_name TEXT NOT NULL, "
                "reviewed_at TEXT NOT NULL, "
                "review_comment TEXT, "
                "FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE, "
                "FOREIGN KEY (reviewer_key_id) REFERENCES reviewer_keys(id))"
            )
            history_schema = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='journey_review_history'"
            ).fetchone()
            if history_schema and "UNIQUE(run_id, conversation_id, reviewer_name, reviewed_at)" in str(history_schema["sql"]):
                self._conn.execute(
                    "CREATE TABLE journey_review_history_new ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "run_id INTEGER NOT NULL, "
                    "conversation_id TEXT NOT NULL, "
                    "reviewer_key_id INTEGER, "
                    "reviewer_name TEXT NOT NULL, "
                    "reviewed_at TEXT NOT NULL, "
                    "review_comment TEXT, "
                    "FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE, "
                    "FOREIGN KEY (reviewer_key_id) REFERENCES reviewer_keys(id))"
                )
                self._conn.execute(
                    "INSERT INTO journey_review_history_new"
                    "(id, run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment) "
                    "SELECT id, run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment "
                    "FROM journey_review_history"
                )
                self._conn.execute("DROP TABLE journey_review_history")
                self._conn.execute("ALTER TABLE journey_review_history_new RENAME TO journey_review_history")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_journey_review_history_lookup "
                "ON journey_review_history(run_id, conversation_id, reviewer_name, reviewed_at)"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO journey_review_history"
                "(run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment) "
                "SELECT run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment "
                "FROM journey_reviews"
            )

            self._conn.execute("DELETE FROM settings WHERE key='auth_master_hash'")

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

    # -------- authentication / reviewer tracking --------

    def has_master_key(self) -> bool:
        return False

    def set_master_key(self, master_key: str) -> None:
        return None

    def verify_master_key(self, master_key: str) -> bool:
        return False

    def create_reviewer_key(self, reviewer_name: str, created_by: str = "master") -> dict:
        name = str(reviewer_name or "").strip()
        if not name:
            raise ValueError("Reviewer name is required")
        plain_key = f"rvw_{secrets.token_urlsafe(24)}"
        now = _now_iso()
        cur = self._exec(
            "INSERT INTO reviewer_keys"
            "(reviewer_name, key_hash, key_prefix, is_active, created_by, created_at)"
            " VALUES(?, ?, ?, 1, ?, ?)",
            (name, _hash_secret(plain_key), plain_key[:12], created_by, now),
        )
        return {
            "id": int(cur.lastrowid),
            "reviewer_name": name,
            "reviewer_key": plain_key,
            "key_prefix": plain_key[:12],
            "created_at": now,
        }

    def list_reviewer_keys(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT id, reviewer_name, key_prefix, is_active, created_by, created_at, revoked_at, last_used_at "
            "FROM reviewer_keys ORDER BY is_active DESC, reviewer_name ASC, created_at DESC"
        )
        return [dict(row) for row in rows]

    def verify_reviewer_key(self, reviewer_key: str) -> Optional[dict]:
        key = str(reviewer_key or "").strip()
        if not key:
            return None
        rows = self._fetchall(
            "SELECT id, reviewer_name, key_hash, key_prefix, is_active FROM reviewer_keys WHERE is_active=1"
        )
        for row in rows:
            if _verify_secret(key, row["key_hash"]):
                self._exec(
                    "UPDATE reviewer_keys SET last_used_at=? WHERE id=?",
                    (_now_iso(), int(row["id"])),
                )
                return {
                    "id": int(row["id"]),
                    "reviewer_name": row["reviewer_name"],
                    "key_prefix": row["key_prefix"],
                }
        return None

    def revoke_reviewer_key(self, key_id: int) -> None:
        self._exec(
            "UPDATE reviewer_keys SET is_active=0, revoked_at=? WHERE id=?",
            (_now_iso(), int(key_id)),
        )

    def record_journey_review(
        self,
        run_id: int,
        conversation_id: str,
        reviewer_name: str,
        reviewer_key_id: int | None = None,
        review_comment: str | None = None,
    ) -> None:
        now = _now_iso()
        params = (
            int(run_id),
            str(conversation_id),
            int(reviewer_key_id) if reviewer_key_id is not None else None,
            str(reviewer_name),
            now,
            str(review_comment or "").strip() or None,
        )
        self._exec(
            "INSERT INTO journey_review_history"
            "(run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            params,
        )
        self._exec(
            "INSERT INTO journey_reviews(run_id, conversation_id, reviewer_key_id, reviewer_name, reviewed_at, review_comment) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, conversation_id, reviewer_name) DO UPDATE SET "
            "reviewer_key_id=excluded.reviewer_key_id, reviewed_at=excluded.reviewed_at, "
            "review_comment=excluded.review_comment",
            params,
        )

    def list_journey_reviews(self, run_id: int, conversation_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT id, reviewer_name, reviewed_at, review_comment FROM journey_review_history "
            "WHERE run_id=? AND conversation_id=? ORDER BY reviewed_at ASC, id ASC",
            (int(run_id), str(conversation_id)),
        )
        return [dict(row) for row in rows]

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

    def refresh_default_prompts(self) -> None:
        """Refresh seeded default prompt rows from the current prompt files."""
        self._seed_default_prompts()

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
            "n_conversations, n_message_calls, n_errors, "
            "(SELECT COUNT(*) FROM conversation_results WHERE run_id=runs.id) AS saved_conversations, "
            "(SELECT COUNT(*) FROM message_results WHERE run_id=runs.id) AS saved_message_results, "
            "(SELECT COUNT(*) FROM run_errors WHERE run_id=runs.id) AS saved_errors "
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
            "(run_id, conversation_id, target_message_id, message_index, source_conversation_id,"
            " message_time, target_message_text, parse_status, error_message, raw_response,"
            " parsed_json, debug_json, input_history_json, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(run_id),
                str(mr.get("thread_id") or mr.get("conversation_id", "")),
                mr.get("target_message_id"),
                int(mr["message_index"]) if mr.get("message_index") is not None else None,
                mr.get("source_conversation_id"),
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

    def get_run_result_counts(self, run_id: int) -> dict[str, int]:
        run_id = int(run_id)
        conv = self._fetchone(
            "SELECT COUNT(*) AS n FROM conversation_results WHERE run_id=?",
            (run_id,),
        )
        msg = self._fetchone(
            "SELECT COUNT(*) AS n FROM message_results WHERE run_id=?",
            (run_id,),
        )
        err = self._fetchone(
            "SELECT COUNT(*) AS n FROM run_errors WHERE run_id=?",
            (run_id,),
        )
        return {
            "conversation_results": int(conv["n"] if conv else 0),
            "message_results": int(msg["n"] if msg else 0),
            "run_errors": int(err["n"] if err else 0),
        }

    def get_run_result_counts_bulk(self, run_ids: Iterable[int]) -> dict[int, dict[str, int]]:
        """Batched version of :meth:`get_run_result_counts` for many runs at once.

        Avoids issuing 3 queries per run (N+1) when rendering a list of runs.
        """
        ids = sorted({int(x) for x in run_ids})
        result = {
            rid: {"conversation_results": 0, "message_results": 0, "run_errors": 0}
            for rid in ids
        }
        if not ids:
            return result
        placeholders = ",".join("?" * len(ids))
        for table, key in (
            ("conversation_results", "conversation_results"),
            ("message_results", "message_results"),
            ("run_errors", "run_errors"),
        ):
            rows = self._fetchall(
                f"SELECT run_id, COUNT(*) AS n FROM {table} WHERE run_id IN ({placeholders}) GROUP BY run_id",
                ids,
            )
            for r in rows:
                result[int(r["run_id"])][key] = int(r["n"])
        return result

    def clear_run_results(self, run_id: int) -> None:
        run_id = int(run_id)
        with self._tx() as c:
            c.execute("DELETE FROM message_results WHERE run_id=?", (run_id,))
            c.execute("DELETE FROM conversation_results WHERE run_id=?", (run_id,))
            c.execute("DELETE FROM run_errors WHERE run_id=?", (run_id,))

    def list_run_conversation_ids(self, run_id: int) -> list[str]:
        """Return conversation IDs saved for a run without decoding result blobs.

        Used when callers only need the journey scope (e.g. reusing a previous
        run's selection) and would otherwise pay to JSON-decode every saved
        transcript/response/debug blob via :meth:`load_run_results` just to
        throw the decoded content away.
        """
        rows = self._fetchall(
            "SELECT conversation_id FROM conversation_results WHERE run_id=? ORDER BY id ASC",
            (int(run_id),),
        )
        return [str(r["conversation_id"]) for r in rows]

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
            pj = _json_load(d.get("parsed_json"))
            _backfill_conversation_parsed_json(pj)
            conversation_results.append(
                {
                    "thread_id": d["conversation_id"],
                    "conversation_id": d["conversation_id"],
                    "run_id": int(run_id),
                    "parse_status": d["parse_status"],
                    "error_message": d.get("error_message"),
                    "raw_model_response": d.get("raw_response"),
                    "parsed_json": pj,
                    "evaluation_output": pj,
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
                "appended_message_index": d.get("message_index"),
                "source_conversation_id": d.get("source_conversation_id"),
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
