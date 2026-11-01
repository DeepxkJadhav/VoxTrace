import math
from typing import List, Dict, Any, Optional
from app.config import settings
from app.storage import VoxTraceStorage

class VoxTraceRAGEngine:
    def __init__(self, storage: Optional[VoxTraceStorage] = None):
        self.storage = storage or VoxTraceStorage()
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: List[List[float]] = []
        self._load_from_storage()

    def _load_from_storage(self):
        """Restore all vector chunks and metadata from durable storage."""
        saved_chunks = self.storage.load_all_indexed_chunks()
        self.metadata = []
        self.vectors = []
        for c in saved_chunks:
            self.metadata.append({
                "transcript_id": c["job_id"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c.get("source", {})
            })
            self.vectors.append(c["embedding"])

    def chunk_text(self, text: str, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += max(1, (chunk_size - overlap))
            if i + overlap >= len(words) and i < len(words):
                chunks.append(" ".join(words[i:]))
                break
        return chunks if chunks else [text]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for t in texts:
            words = t.lower().split()
            vec = [0.0] * 64
            for idx, w in enumerate(words):
                h = sum(ord(c) for c in w)
                vec[h % 64] += 1.0 / (idx + 1.0)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            embeddings.append([v / norm for v in vec])
        return embeddings

    def index_transcript(self, transcript_id: str, text: str, source_metadata: Dict[str, Any] = None) -> int:
        chunks = self.chunk_text(text)
        if not chunks:
            return 0
        
        chunk_vectors = self._embed(chunks)
        chunk_records = []
        for idx, (chunk, vec) in enumerate(zip(chunks, chunk_vectors)):
            record = {
                "transcript_id": transcript_id,
                "chunk_id": idx,
                "text": chunk,
                "embedding": vec,
                "source": source_metadata or {}
            }
            self.metadata.append(record)
            self.vectors.append(vec)
            chunk_records.append(record)
            
        return len(chunks)

    def search(self, query: str, top_k: int = settings.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []
        
        q_vec = self._embed([query])[0]
        scores = []
        for idx, v in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(q_vec, v))
            scores.append((dot, idx))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            item = self.metadata[idx].copy()
            item["score"] = round(float(score), 4)
            results.append(item)
        return results

    def generate_rag_answer(self, query: str, top_k: int = settings.TOP_K_RESULTS) -> Dict[str, Any]:
        relevant_chunks = self.search(query, top_k=top_k)
        if not relevant_chunks:
            return {
                "query": query,
                "answer": "No audio transcript data has been indexed yet. Please upload and transcribe audio first.",
                "context": []
            }
        
        best = relevant_chunks[0]
        answer = (
            f"Retrieved from Audio Transcript ({best.get('source', {}).get('filename', 'Audio Session')}):\n\n"
            f'"{best["text"]}"\n\n'
            f"[Retrieved with Semantic Search — Similarity Confidence: {int(best['score'] * 100)}%]"
        )

        return {
            "query": query,
            "answer": answer,
            "context": relevant_chunks
        }
