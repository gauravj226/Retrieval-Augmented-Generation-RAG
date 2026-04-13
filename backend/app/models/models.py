from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, UniqueConstraint, Numeric, JSON, Date
)
from sqlalchemy.orm import relationship
from ..database import Base

# ── Personalities ─────────────────────────────────────────────────────────────

class Personality(Base):
    __tablename__ = "personalities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=False)
    tone = Column(String, default="professional")
    is_preset = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    knowledge_bases = relationship("KnowledgeBase", back_populates="personality")

# ── Groups ────────────────────────────────────────────────────────────────────

class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    members = relationship("UserGroupMapping", back_populates="group", cascade="all, delete-orphan")

# ── Knowledge Bases ───────────────────────────────────────────────────────────

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    personality_id = Column(Integer, ForeignKey("personalities.id"))
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    personality = relationship("Personality", back_populates="knowledge_bases")
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")

# ── Documents ─────────────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")

# ── Invoice Metadata ──────────────────────────────────────────────────────────

class InvoiceMetadata(Base):
    __tablename__ = "invoice_metadata"

    id = Column(Integer, primary_key=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"))
    document_id = Column(Integer, ForeignKey("documents.id"))
    filename = Column(String)
    vendor_name = Column(String, index=True)
    invoice_number = Column(String, index=True)
    invoice_date = Column(Date)
    total_amount = Column(Numeric(12, 2))
    currency = Column(String(3), default="GBP")
    line_items = Column(JSON)   # list of {desc, qty, unit_price, total}
    raw_extracted = Column(JSON)   # full VLM extraction for audit
