from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
import threading
import time
import uuid


class JobStatus(str, Enum):
    QUEUED = "queued"
    VAD_DIARIZE = "vad_diarize"
    TRANSCRIBING = "transcribing"
    REVIEWING = "reviewing"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    audio_path: str
    glossary_yaml: str
    status: JobStatus
    artifact_dir: str | None
    error: str | None
    created_at: float
    updated_at: float


_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    audio_path TEXT NOT NULL,
    glossary_yaml TEXT NOT NULL,
    status TEXT NOT NULL,
    artifact_dir TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status_created ON jobs(status, created_at);
"""

_COLS = (
    "id", "audio_path", "glossary_yaml", "status",
    "artifact_dir", "error", "created_at", "updated_at",
)


class JobStore:
    """SQLite-backed job queue + state machine for the meeting pipeline.

    Thread-safe via a single lock around mutating operations. Read paths
    are also serialized for simplicity (jobs are coarse-grained — the
    overhead is negligible compared to actual pipeline work).
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._lock = threading.Lock()

    def create(self, *, audio_path: str, glossary_yaml: str) -> str:
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(id,audio_path,glossary_yaml,status,artifact_dir,error,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (jid, audio_path, glossary_yaml, JobStatus.QUEUED.value, None, None, now, now),
            )
            self._conn.commit()
        return jid

    def get(self, jid: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {','.join(_COLS)} FROM jobs WHERE id=?",
                (jid,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def update(self, jid: str, **fields) -> None:
        if not fields:
            return
        if "status" in fields and isinstance(fields["status"], JobStatus):
            fields["status"] = fields["status"].value
        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{k}=?" for k in fields)
        params = list(fields.values()) + [jid]
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {set_clause} WHERE id=?", params)
            self._conn.commit()

    def pop_next_queued(self) -> Job | None:
        """Atomically transition the oldest QUEUED job to VAD_DIARIZE and return it."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE status=? ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            jid = row[0]
            self._conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                (JobStatus.VAD_DIARIZE.value, now, jid),
            )
            self._conn.commit()
            full = self._conn.execute(
                f"SELECT {','.join(_COLS)} FROM jobs WHERE id=?",
                (jid,),
            ).fetchone()
        return self._row_to_job(full)

    @staticmethod
    def _row_to_job(row) -> Job:
        d = dict(zip(_COLS, row))
        return Job(
            id=d["id"],
            audio_path=d["audio_path"],
            glossary_yaml=d["glossary_yaml"],
            status=JobStatus(d["status"]),
            artifact_dir=d["artifact_dir"],
            error=d["error"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )
