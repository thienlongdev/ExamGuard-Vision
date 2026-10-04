"""
ExamGuard Persistence Package.
"""

from src.persistence.models import (
    ExamSession,
    PersistedEvent,
    EventEvidence,
    ReviewRecord,
    AuditLogEntry,
)
from src.persistence.database import DatabaseManager, DEFAULT_DB_PATH
from src.persistence.migrations import run_migrations, CURRENT_SCHEMA_VERSION
from src.persistence.service import PersistenceService, compute_file_sha256
from src.persistence.backup import (
    create_local_backup,
    verify_backup_directory,
    preview_retention_candidates,
)

__all__ = [
    "ExamSession",
    "PersistedEvent",
    "EventEvidence",
    "ReviewRecord",
    "AuditLogEntry",
    "DatabaseManager",
    "DEFAULT_DB_PATH",
    "run_migrations",
    "CURRENT_SCHEMA_VERSION",
    "PersistenceService",
    "compute_file_sha256",
    "create_local_backup",
    "verify_backup_directory",
    "preview_retention_candidates",
]
