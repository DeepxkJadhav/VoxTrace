import io
import sys
import os
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app, storage

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "VoxTrace" in response.text

def test_status_endpoint():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert response.json()["service"] == "VoxTrace"

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "whisper_model" in data

def test_reject_unsupported_audio_format():
    fake_txt = io.BytesIO(b"Hello audio text")
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("malicious.exe", fake_txt, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Invalid audio format" in response.json()["detail"]

def test_reject_empty_audio_file():
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("sample.wav", empty_file, "audio/wav")}
    )
    assert response.status_code == 400
    assert "Empty audio file" in response.json()["detail"]

def test_async_job_lifecycle():
    fake_audio = io.BytesIO(b"RIFF....WAVEfmt ....data....")
    # 1. Submit async job
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("meeting_notes.wav", fake_audio, "audio/wav")},
        data={"session_name": "Executive Sync"}
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "QUEUED"
    assert "job_id" in data
    job_id = data["job_id"]

    # 2. Wait briefly for background worker execution
    time.sleep(0.5)

    # 3. Poll job status
    poll_res = client.get(f"/api/v1/jobs/{job_id}")
    assert poll_res.status_code == 200
    job_data = poll_res.json()
    assert job_data["status"] in ["PROCESSING", "COMPLETED"]
    assert job_data["filename"] == "meeting_notes.wav"

def test_sync_transcription_and_rag_query():
    fake_audio = io.BytesIO(b"RIFF....WAVEfmt ....data....")
    response = client.post(
        "/api/v1/transcribe-sync",
        files={"file": ("keynote.wav", fake_audio, "audio/wav")},
        data={"session_name": "AI Summit"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert "transcript" in data
    assert len(data["transcript"]["full_text"]) > 0

    # Query RAG pipeline against the indexed keynote
    query_res = client.post(
        "/api/v1/query",
        json={"query": "What is VoxTrace?"}
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "answer" in q_data
    assert len(q_data["context"]) > 0

def test_query_validation():
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 400

def test_job_not_found():
    response = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
