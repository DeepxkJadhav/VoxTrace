import sqlite3
import json
import os
import time
from typing import Dict, Any, List, Optional

DB_FILE = os.getenv("VOXTRACE_DB_PATH", "voxtrace.db")

class VoxTraceStorage:
    """Thread-safe durable SQLite storage for audio transcription jobs, transcripts, and vector chunks."""
    
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging for high-concurrency read/write
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_jobs (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_name TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    duration_seconds REAL,
                    error_message TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    job_id TEXT PRIMARY KEY,
                    language TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    segments_json TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES audio_jobs(id)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    source_metadata_json TEXT,
                    FOREIGN KEY(job_id) REFERENCES audio_jobs(id)
                );
            """)
            conn.commit()

    def create_job(self, job_id: str, filename: str, session_name: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO audio_jobs (id, filename, status, session_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, filename, "QUEUED", session_name or filename, now)
            )
            conn.commit()
        return {
            "id": job_id,
            "filename": filename,
            "status": "QUEUED",
            "session_name": session_name or filename,
            "created_at": now
        }

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        duration: Optional[float] = None
    ):
        now = time.time()
        with self._get_connection() as conn:
            if status in ["COMPLETED", "FAILED"]:
                conn.execute(
                    "UPDATE audio_jobs SET status = ?, completed_at = ?, duration_seconds = ?, error_message = ? WHERE id = ?",
                    (status, now, duration, error_message, job_id)
                )
            else:
                conn.execute(
                    "UPDATE audio_jobs SET status = ?, error_message = ? WHERE id = ?",
                    (status, error_message, job_id)
                )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM audio_jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            job = dict(row)
            
            # If completed, attach transcript
            t_row = conn.execute("SELECT * FROM transcripts WHERE job_id = ?", (job_id,)).fetchone()
            if t_row:
                job["transcript"] = {
                    "language": t_row["language"],
                    "full_text": t_row["full_text"],
                    "segments": json.loads(t_row["segments_json"])
                }
            return job

    def save_transcript_and_chunks(
        self,
        job_id: str,
        language: str,
        full_text: str,
        segments: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ):
        with self._get_connection() as conn:
            # 1. Insert transcript
            conn.execute(
                "INSERT OR REPLACE INTO transcripts (job_id, language, full_text, segments_json) VALUES (?, ?, ?, ?)",
                (job_id, language, full_text, json.dumps(segments))
            )
            # 2. Insert chunks
            for c in chunks:
                chunk_id = f"{job_id}_{c['chunk_id']}"
                conn.execute(
                    """INSERT OR REPLACE INTO vector_chunks 
                       (id, job_id, chunk_index, chunk_text, embedding_json, source_metadata_json) 
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk_id,
                        job_id,
                        c["chunk_id"],
                        c["text"],
                        json.dumps(c["embedding"]),
                        json.dumps(c.get("source", {}))
                    )
                )
            conn.commit()

    def load_all_indexed_chunks(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM vector_chunks").fetchall()
            chunks = []
            for r in rows:
                chunks.append({
                    "id": r["id"],
                    "job_id": r["job_id"],
                    "chunk_id": r["chunk_index"],
                    "text": r["chunk_text"],
                    "embedding": json.loads(r["embedding_json"]),
                    "source": json.loads(r["source_metadata_json"]) if r["source_metadata_json"] else {}
                })
            return chunks

    def count_indexed_chunks(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM vector_chunks").fetchone()
            return row["count"] if row else 0
