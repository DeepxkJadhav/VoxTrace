import concurrent.futures
import time
import logging
import traceback
from typing import Dict, Any, Optional

from app.storage import VoxTraceStorage
from app.transcriber import AudioTranscriber
from app.rag_engine import VoxTraceRAGEngine

logger = logging.getLogger("voxtrace.worker")

class VoxTraceWorkerPool:
    """
    Dedicated worker pool for offloading CPU-intensive Whisper audio transcription
    and embedding computations away from the async FastAPI event loop.
    """
    def __init__(
        self,
        storage: VoxTraceStorage,
        transcriber: AudioTranscriber,
        rag_engine: VoxTraceRAGEngine,
        max_workers: int = 2
    ):
        self.storage = storage
        self.transcriber = transcriber
        self.rag_engine = rag_engine
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="voxtrace_worker"
        )

    def submit_transcription_job(
        self,
        job_id: str,
        audio_bytes: bytes,
        filename: str,
        session_name: Optional[str] = None
    ):
        """Submit audio bytes for background processing and return immediately."""
        self.executor.submit(
            self._process_job,
            job_id,
            audio_bytes,
            filename,
            session_name
        )

    def _process_job(
        self,
        job_id: str,
        audio_bytes: bytes,
        filename: str,
        session_name: Optional[str]
    ):
        start_time = time.time()
        try:
            logger.info(f"Starting background transcription for job {job_id} ({filename})")
            self.storage.update_job_status(job_id, "PROCESSING")

            # 1. Run Whisper transcription (synchronous CPU/GPU bound)
            result = self.transcriber.transcribe_audio(audio_bytes, filename=filename)
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            language = result.get("language", "en")

            # 2. Chunk text and generate vector embeddings
            chunks = self.rag_engine.chunk_text(text)
            embeddings = self.rag_engine._embed(chunks)

            chunk_records = []
            for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
                chunk_records.append({
                    "chunk_id": idx,
                    "text": chunk,
                    "embedding": vec,
                    "source": {
                        "filename": filename,
                        "session_name": session_name or filename,
                        "language": language
                    }
                })

            # 3. Atomically persist transcript and vector chunks to SQLite
            self.storage.save_transcript_and_chunks(
                job_id=job_id,
                language=language,
                full_text=text,
                segments=segments,
                chunks=chunk_records
            )

            # 4. Update in-memory index for instant search
            for record in chunk_records:
                self.rag_engine.metadata.append({
                    "transcript_id": job_id,
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "source": record["source"]
                })
                self.rag_engine.vectors.append(record["embedding"])

            duration = round(time.time() - start_time, 2)
            self.storage.update_job_status(job_id, "COMPLETED", duration=duration)
            logger.info(f"Job {job_id} successfully completed in {duration}s")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
            duration = round(time.time() - start_time, 2)
            self.storage.update_job_status(job_id, "FAILED", error_message=str(e), duration=duration)
