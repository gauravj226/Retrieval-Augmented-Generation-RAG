from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# check_same_thread=False required for SQLite in async/multi-thread context
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from .models import models  # noqa
    Base.metadata.create_all(bind=engine)
    _ensure_kb_columns()
    _ensure_entity_relationships_table()


def _ensure_kb_columns():
    """Best-effort additive migration for KnowledgeBase retrieval config fields."""
    required = {
        "score_threshold": "ALTER TABLE knowledge_bases ADD COLUMN score_threshold VARCHAR DEFAULT '0.35'",
        "memory_window": "ALTER TABLE knowledge_bases ADD COLUMN memory_window INTEGER DEFAULT 5",
        "chunk_size": "ALTER TABLE knowledge_bases ADD COLUMN chunk_size INTEGER DEFAULT 800",
        "chunk_overlap": "ALTER TABLE knowledge_bases ADD COLUMN chunk_overlap INTEGER DEFAULT 120",
    }
    inspector = inspect(engine)
    existing = {c["name"] for c in inspector.get_columns("knowledge_bases")}
    missing = [name for name in required if name not in existing]
    if not missing:
        return

    with engine.begin() as conn:
        for col in missing:
            conn.execute(text(required[col]))


def _ensure_entity_relationships_table():
    if engine.dialect.name != "sqlite":
        return
    inspector = inspect(engine)
    if inspector.has_table("entity_relationships"):
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE entity_relationships (
                    id INTEGER PRIMARY KEY,
                    kb_id INTEGER NOT NULL,
                    source_entity VARCHAR NOT NULL,
                    relation_type VARCHAR NOT NULL DEFAULT 'references',
                    target_entity VARCHAR NOT NULL,
                    source_doc VARCHAR NOT NULL,
                    target_doc VARCHAR,
                    confidence FLOAT NOT NULL DEFAULT 1.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT uq_entity_relationship UNIQUE
                    (kb_id, source_entity, relation_type, target_entity, source_doc, target_doc)
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX ix_entity_relationships_kb_id ON entity_relationships (kb_id)"))
        conn.execute(text("CREATE INDEX ix_entity_relationships_source_entity ON entity_relationships (source_entity)"))
        conn.execute(text("CREATE INDEX ix_entity_relationships_target_entity ON entity_relationships (target_entity)"))
