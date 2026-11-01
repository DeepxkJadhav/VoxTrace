import os
import uuid
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from app.transcriber import AudioTranscriber
from app.rag_engine import VoxTraceRAGEngine
from app.storage import VoxTraceStorage
from app.worker import VoxTraceWorkerPool
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Audio Intelligence Pipeline with Asynchronous Whisper Workers & Persistent Vector RAG"
)

# Explicit allowed origins to eliminate wildcard CORS credential vulnerabilities
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB upload ceiling
ALLOWED_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')

storage = VoxTraceStorage()
transcriber = AudioTranscriber()
rag_engine = VoxTraceRAGEngine(storage=storage)
worker_pool = VoxTraceWorkerPool(storage=storage, transcriber=transcriber, rag_engine=rag_engine)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer from indexed audio transcripts")
    top_k: Optional[int] = Field(default=settings.TOP_K_RESULTS, ge=1, le=20)

class QueryResponse(BaseModel):
    query: str
    answer: str
    context: List[Dict[str, Any]]

@app.get("/")
def root():
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "total_indexed_chunks": len(rag_engine.vectors)
    }

@app.get("/api/v1/status")
def status_endpoint():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "total_indexed_chunks": len(rag_engine.vectors),
        "db_records": storage.count_indexed_chunks()
    }

@app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
async def submit_transcription_job(
    file: UploadFile = File(...),
    session_name: Optional[str] = Form(None)
):
    """
    Asynchronous non-blocking audio transcription endpoint.
    Accepts audio, generates a unique Job ID, dispatches to worker pool,
    and returns HTTP 202 Accepted immediately.
    """
    filename = file.filename or "audio.wav"
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")
    
    if len(audio_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 50MB size limit.")

    job_id = str(uuid.uuid4())
    job = storage.create_job(job_id=job_id, filename=filename, session_name=session_name)
    
    # Offload transcription to background worker pool
    worker_pool.submit_transcription_job(
        job_id=job_id,
        audio_bytes=audio_bytes,
        filename=filename,
        session_name=session_name
    )

    return {
        "status": "QUEUED",
        "job_id": job_id,
        "filename": filename,
        "session_name": session_name or filename,
        "message": "Audio transcription job accepted and queued for worker processing.",
        "poll_url": f"/api/v1/jobs/{job_id}"
    }

@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    """Poll status of an asynchronous transcription job."""
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@app.post("/api/v1/transcribe-sync")
async def transcribe_audio_synchronous(
    file: UploadFile = File(...),
    session_name: Optional[str] = Form(None)
):
    """Synchronous fallback endpoint for immediate script execution."""
    filename = file.filename or "audio.wav"
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid audio format.")
    
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")
    
    if len(audio_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds 50MB.")

    job_id = str(uuid.uuid4())
    storage.create_job(job_id=job_id, filename=filename, session_name=session_name)
    
    # Process inline
    worker_pool._process_job(job_id, audio_bytes, filename, session_name)
    return storage.get_job(job_id)

@app.post("/api/v1/query", response_model=QueryResponse)
def query_rag_pipeline(payload: QueryRequest):
    """Query indexed speech transcripts using vector similarity search."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = rag_engine.generate_rag_answer(query=payload.query, top_k=payload.top_k or 3)
    return result

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "whisper_model": settings.WHISPER_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "active_vectors": len(rag_engine.vectors),
        "db_records": storage.count_indexed_chunks()
    }
