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
    VECTOR_DB_PROVIDER: str = "chroma"
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333

    OLLAMA_HOST: str = "ollama"
    OLLAMA_PORT: int = 11434
    DEFAULT_LLM_MODEL: str = "llama3.2:3b"
    GRADER_LLM_MODEL: Optional[str] = None

    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_FILES: int = 5
    MAX_UPLOAD_FILE_MB: int = 15
    MAX_UPLOAD_BATCH_MB: int = 50
    HYBRID_LEXICAL_MAX_DOCS: int = 2000
    HYBRID_BM25_DIR: str = "/app/data/hybrid_bm25"
    ENABLE_CROSS_ENCODER_RERANK: bool = False
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FAST_MODE: bool = True
    FAST_MODE_MAX_TOKENS: int = 320
    FAST_MODE_CONTEXT_CHARS_PER_DOC: int = 900
    QUALITY_MODE_CONTEXT_CHARS_PER_DOC: int = 2200
    QUERY_CACHE_TTL_SEC: int = 120
    QUERY_CACHE_MAX: int = 256
    ENABLE_SEMANTIC_CACHE: bool = True
    SEMANTIC_CACHE_SIMILARITY: float = 0.84
    SEMANTIC_CACHE_TTL_SEC: int = 300
    SEMANTIC_CACHE_MAX: int = 512
    INGEST_WORKERS: int = 2
    INGEST_QUEUE_MAX: int = 200
    ENABLE_CONTEXTUAL_RETRIEVAL: bool = True
    GRAPH_MEMORY_DIR: str = "/app/data/graph_memory"
    ENABLE_GRAPH_RAG: bool = True
    MEMORY_STORE_DIR: str = "/app/data/long_term_memory"
    ENABLE_LONG_TERM_MEMORY: bool = True
    ENABLE_WEB_FALLBACK: bool = False
    WEB_SEARCH_PROVIDER: str = "duckduckgo"
    WEB_SEARCH_MAX_RESULTS: int = 5
    WEB_SEARCH_TIMEOUT_SEC: int = 7
    VLM_MAX_PAGES: int = 8
    VLM_DPI: int = 120
    VLM_CONCURRENCY: int = 2
    VLM_TIMEOUT_SEC: int = 75
    MAX_DOCS_FOR_GRADING: int = 6
    MAX_DOCS_FOR_GENERATION: int = 6
    AUDIT_LOG_PATH: str = "/app/logs/audit.log"
    AUDIT_LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
