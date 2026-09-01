import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid

from app.transcriber import AudioTranscriber
from app.rag_engine import VoxTraceRAGEngine
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Audio Intelligence Pipeline with Whisper Speech-to-Text & FAISS RAG"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transcriber = AudioTranscriber()
rag_engine = VoxTraceRAGEngine()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = settings.TOP_K_RESULTS

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
def status():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "total_indexed_chunks": len(rag_engine.vectors)
    }

@app.post("/api/v1/transcribe-and-index")
async def transcribe_and_index_audio(
    file: UploadFile = File(...),
    session_name: Optional[str] = Form(None)
):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Invalid audio format. Allowed: .wav, .mp3, .m4a, .ogg, .flac")
    
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")
    
    # 1. Transcribe audio via Whisper
    transcription_result = transcriber.transcribe_audio(audio_bytes, filename=file.filename)
    
    # 2. Index transcript into FAISS vector database
    transcript_id = str(uuid.uuid4())
    chunks_indexed = rag_engine.index_transcript(
        transcript_id=transcript_id,
        text=transcription_result["text"],
        source_metadata={
            "filename": file.filename,
            "session_name": session_name or file.filename,
            "language": transcription_result.get("language")
        }
    )

    return {
        "status": "success",
        "transcript_id": transcript_id,
        "filename": file.filename,
        "language": transcription_result.get("language"),
        "full_text": transcription_result.get("text"),
        "segments": transcription_result.get("segments"),
        "chunks_indexed": chunks_indexed,
        "total_index_size": len(rag_engine.vectors)
    }

@app.post("/api/v1/query", response_model=QueryResponse)
def query_rag_pipeline(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = rag_engine.generate_rag_answer(query=payload.query, top_k=payload.top_k or 3)
    return result

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "whisper_model": settings.WHISPER_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "indexed_vectors": len(rag_engine.vectors)
    }
