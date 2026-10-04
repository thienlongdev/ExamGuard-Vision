"""
Repository for reviews table operations (append-only review audit history).
"""

import logging
from typing import Optional, List, Any
from src.persistence.database import DatabaseManager
from src.persistence.models import ReviewRecord

logger = logging.getLogger(__name__)


class ReviewRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add_review(self, rev: ReviewRecord) -> None:
        """Append a review decision record (never overwriting prior decisions)."""
        sql = """
        INSERT INTO reviews (
            review_id, event_id, decision, reviewer_id, reviewer_name,
            note, reviewed_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.transaction() as cur:
            cur.execute(
                sql,
                (
                    rev.review_id,
                    rev.event_id,
                    rev.decision,
                    rev.reviewer_id,
                    rev.reviewer_name,
                    rev.note,
                    rev.reviewed_at,
                    rev.created_at,
                ),
            )

    def get_review_history(self, event_id: str) -> List[ReviewRecord]:
        """Fetch all historical reviews for an event in chronological order."""
        sql = "SELECT * FROM reviews WHERE event_id = ? ORDER BY reviewed_at ASC;"
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, (event_id,))
            for row in cur.fetchall():
                results.append(self._row_to_review(row))
        return results

    def get_latest_review(self, event_id: str) -> Optional[ReviewRecord]:
        """Fetch the latest review record for an event."""
        sql = "SELECT * FROM reviews WHERE event_id = ? ORDER BY reviewed_at DESC LIMIT 1;"
        with self.db.cursor() as cur:
            cur.execute(sql, (event_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_review(row)

    def _row_to_review(self, row: Any) -> ReviewRecord:
        return ReviewRecord(
            review_id=row["review_id"],
            event_id=row["event_id"],
            decision=row["decision"],
            reviewer_id=row["reviewer_id"],
            reviewer_name=row["reviewer_name"],
            note=row["note"],
            reviewed_at=row["reviewed_at"],
            created_at=row["created_at"],
        )
