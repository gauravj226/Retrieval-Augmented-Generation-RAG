import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from .config import settings
from .database import SessionLocal, init_db
from .models.models import KnowledgeBase, Personality, User
from .routers import admin, auth, chat, documents, groups
from .routers.mcp_router import setup_mcp
from .services.rag_service import get_chroma_client
from .services.audit import configure_audit_logger, audit_event
from .services.ingest_queue import start_ingest_workers, stop_ingest_workers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
configure_audit_logger()

app = FastAPI(
    title="RAG Platform API",
    description="Agentic RAG with LangGraph, Ollama (llama3.2:3b), ChromaDB",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────

# Trust IIS forwarded headers
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(admin.router)
app.include_router(groups.router)

# ── MCP (exposes chat endpoints as MCP tools) ─────────────────────────────────
setup_mcp(app)

# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Initialising database...")
    audit_event("system.startup", details={"service": "backend"})
    init_db()
    
    db = SessionLocal()
    try:
        _seed_personalities(db)
    finally:
        db.close()
    
    # Warm up Chroma connection
    try:
        get_chroma_client().heartbeat()
        logger.info("ChromaDB connected ✓")
    except Exception as e:
        logger.warning(f"ChromaDB not ready yet: {e}")
        
    logger.info("RAG Platform (Agentic v2) ready ✓")
    audit_event("system.ready", details={"service": "backend"})
    await start_ingest_workers()

@app.on_event("shutdown")
async def shutdown():
    await stop_ingest_workers()
    audit_event("system.shutdown", details={"service": "backend"})

def _seed_personalities(db):
    """Seed built-in personality presets if they don't exist."""
    presets = [
        {
            "name": "Professional Assistant",
            "system_prompt": "You are a professional, precise assistant. Answer clearly and concisely using only the provided context. Avoid speculation.",
            "tone": "professional",
        },
        {
            "name": "Friendly Tutor",
            "system_prompt": "You are a friendly and patient tutor. Explain concepts clearly with examples. Encourage the user and make complex topics accessible.",
            "tone": "friendly",
        },
        {
            "name": "Technical Expert",
            "system_prompt": "You are a senior technical expert. Provide detailed, accurate technical answers. Use precise terminology and include relevant technical details from the context.",
            "tone": "technical",
        },
        {
            "name": "Concise Summariser",
            "system_prompt": "You are a concise summariser. Give brief, bullet-pointed answers. Maximum 5 bullet points. No preamble.",
            "tone": "concise",
        },
    ]
    for p in presets:
        if not db.query(Personality).filter(Personality.name == p["name"]).first():
            db.add(Personality(**p, is_preset=True))
            db.commit()

# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "mode": "agentic"}
