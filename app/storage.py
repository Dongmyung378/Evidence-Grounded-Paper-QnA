"""Durable per-paper metadata and idempotent analysis jobs in SQLite."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class QueueFull(Exception):
    """The serial worker's bounded backlog is full."""


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "service.sqlite3"
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, page_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL UNIQUE REFERENCES papers(paper_id),
                    status TEXT NOT NULL, seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    page_count INTEGER, text_page_count INTEGER, chunk_count INTEGER,
                    warning_count INTEGER, error_code TEXT, error_message TEXT
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def register(self, paper_id, filename, size, pages):
        with self.connect() as db:
            db.execute("INSERT INTO papers VALUES (?,?,?,?,?)",
                       (paper_id, filename, size, pages, now()))

    def enqueue(self, paper_id, seed, max_pending=16):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM papers WHERE paper_id=?", (paper_id,)).fetchone():
                raise KeyError(paper_id)
            row = db.execute("SELECT * FROM jobs WHERE paper_id=?", (paper_id,)).fetchone()
            schedule = row is None or row["status"] == "failed"
            if schedule:
                pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
                if pending >= max_pending:
                    raise QueueFull("Analysis queue is full")
            if row is None:
                timestamp = now()
                db.execute(
                    "INSERT INTO jobs(job_id,paper_id,status,seed,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?)",
                    ("job-" + uuid4().hex, paper_id, "queued", seed, timestamp, timestamp),
                )
            elif row["status"] == "failed":
                db.execute(
                    "UPDATE jobs SET status='queued',updated_at=?,error_code=NULL,"
                    "error_message=NULL,page_count=NULL,text_page_count=NULL,"
                    "chunk_count=NULL,warning_count=NULL WHERE paper_id=?",
                    (now(), paper_id),
                )
            row = db.execute("SELECT * FROM jobs WHERE paper_id=?", (paper_id,)).fetchone()
            return dict(row), schedule

    def job(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def recover(self):
        # Called only while this service owns the exclusive runtime-directory lock.
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='queued',updated_at=? WHERE status='running'", (now(),))
            return [row[0] for row in db.execute("SELECT job_id FROM jobs WHERE status='queued'")]

    def claim(self, job_id):
        with self.connect() as db:
            return db.execute(
                "UPDATE jobs SET status='running',updated_at=? WHERE job_id=? AND status='queued'",
                (now(), job_id),
            ).rowcount == 1

    def finish(self, job_id, *, page_count=None, text_page_count=None,
               chunk_count=None, warning_count=None, error_code=None, error_message=None):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,updated_at=?,page_count=?,text_page_count=?,"
                "chunk_count=?,warning_count=?,error_code=?,error_message=? WHERE job_id=?",
                ("failed" if error_code else "completed", now(), page_count,
                 text_page_count, chunk_count, warning_count, error_code, error_message, job_id),
            )
