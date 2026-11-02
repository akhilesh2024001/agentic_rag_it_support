
import json
import sqlite3

from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

from app.core.config import get_settings

settings = get_settings()


def get_database_path() -> Path:
    db_path = Path(settings.audit_db_path)

    # Create parent directory if it does not exist.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return db_path


@contextmanager
def get_connection():
    con = sqlite3.connect(
        get_database_path(),
        timeout=10,
    )

    try:
        yield con
        con.commit()

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()


def init_db() -> None:
    with get_connection() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )
            """
        )


def write_audit(
    question: str,
    source_used: str,
    trace: list[str],
) -> None:

    with get_connection() as con:
        con.execute(
            """
            INSERT INTO query_audit (
                created_at,
                question,
                source_used,
                trace_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                source_used,
                json.dumps(trace),
            ),
        )
