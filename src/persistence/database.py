"""
Database connection manager and SQLite safety configuration for ExamGuard.
"""

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Generator, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "storage/db/examguard.sqlite3"


class DatabaseManager:
    """Thread-safe SQLite database manager with WAL mode and foreign key enforcement."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Resolve relative to repo root
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.db_path = str(repo_root / DEFAULT_DB_PATH)
        else:
            self.db_path = db_path

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply strict SQLite safety PRAGMAs."""
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA busy_timeout = 10000;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.close()

    def get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local SQLite connection with safety PRAGMAs configured."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=10.0,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode by default, explicit BEGIN for transactions
            )
            self._configure_connection(conn)
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Thread-safe transactional execution block with automatic rollback on error."""
        with self._lock:
            conn = self.get_connection()
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            try:
                yield cur
                if conn.in_transaction:
                    conn.execute("COMMIT;")
            except Exception as e:
                if conn.in_transaction:
                    conn.execute("ROLLBACK;")
                logger.error(f"Transaction rolled back due to error: {e}")
                raise
            finally:
                cur.close()

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Thread-safe read-only or single-query cursor."""
        with self._lock:
            conn = self.get_connection()
            cur = conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def check_health(self) -> bool:
        """Lightweight database connectivity check."""
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1;")
                row = cur.fetchone()
                return row is not None and row[0] == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def close(self) -> None:
        """Close thread-local connection if present."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None
