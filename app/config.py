import os

class Settings:
    PROJECT_NAME: str = "VoxTrace"
    VERSION: str = "1.0.0"
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))

settings = Settings()
