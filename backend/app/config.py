from pydantic_settings import BaseSettings
from typing import Optional
import secrets


class Settings(BaseSettings):
    SECRET_KEY: str = secrets.token_hex(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    DATABASE_URL: str = "sqlite:////app/data/rag_system.db"

    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000

    OLLAMA_HOST: str = "ollama"
    OLLAMA_PORT: int = 11434
    DEFAULT_LLM_MODEL: str = "llama3.2:3b"

    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_FILES: int = 5
    MAX_UPLOAD_FILE_MB: int = 15
    MAX_UPLOAD_BATCH_MB: int = 50
    HYBRID_LEXICAL_MAX_DOCS: int = 2000
    ENABLE_CROSS_ENCODER_RERANK: bool = False
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FAST_MODE: bool = True
    QUERY_CACHE_TTL_SEC: int = 120
    QUERY_CACHE_MAX: int = 256
    INGEST_WORKERS: int = 2
    INGEST_QUEUE_MAX: int = 200
    VLM_MAX_PAGES: int = 8
    VLM_DPI: int = 120
    VLM_CONCURRENCY: int = 2
    VLM_TIMEOUT_SEC: int = 75
    AUDIT_LOG_PATH: str = "/app/logs/audit.log"
    AUDIT_LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
