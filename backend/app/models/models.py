from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, UniqueConstraint,
    Numeric, JSON, Date, Float,
)
from sqlalchemy.orm import relationship
from ..database import Base


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


class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    members = relationship("UserGroupMapping", back_populates="group", cascade="all, delete-orphan")
    kb_permissions = relationship("GroupKBPermission", back_populates="group", cascade="all, delete-orphan")


class UserGroupMapping(Base):
    __tablename__ = "user_group_mappings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "group_id", name="uq_user_group"),)

    user = relationship("User", back_populates="group_memberships")
    group = relationship("Group", back_populates="members")


class GroupKBPermission(Base):
    __tablename__ = "group_kb_permissions"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=False)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    permission = Column(String, default="read")

    __table_args__ = (UniqueConstraint("group_id", "kb_id", name="uq_group_kb"),)

    group = relationship("Group", back_populates="kb_permissions")
    knowledge_base = relationship("KnowledgeBase", back_populates="group_permissions")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    department = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    group_memberships = relationship("UserGroupMapping", back_populates="user", cascade="all, delete-orphan")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    department = Column(String, index=True, nullable=False)

    llm_model = Column(String, default="llama3.2:3b")
    embedding_model = Column(String, default="all-MiniLM-L6-v2")

    personality_id = Column(Integer, ForeignKey("personalities.id"), nullable=True)
    system_prompt = Column(Text, nullable=True)

    temperature = Column(String, default="0.7")
    max_tokens = Column(Integer, default=1024)

    top_k_docs = Column(Integer, default=4)
    mmr_fetch_k = Column(Integer, default=16)
    mmr_lambda = Column(String, default="0.7")
    score_threshold = Column(String, default="0.35")
    memory_window = Column(Integer, default=5)
    chunk_size = Column(Integer, default=800)
    chunk_overlap = Column(Integer, default=120)

    chroma_collection = Column(String, unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    personality = relationship("Personality", back_populates="knowledge_bases")
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")
    group_permissions = relationship("GroupKBPermission", back_populates="knowledge_base", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="processing")
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


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
    line_items = Column(JSON)
    raw_extracted = Column(JSON)


class EntityRelationship(Base):
    __tablename__ = "entity_relationships"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), index=True, nullable=False)
    source_entity = Column(String, index=True, nullable=False)
    relation_type = Column(String, default="references", nullable=False)
    target_entity = Column(String, index=True, nullable=False)
    source_doc = Column(String, nullable=False)
    target_doc = Column(String, nullable=True)
    confidence = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "kb_id",
            "source_entity",
            "relation_type",
            "target_entity",
            "source_doc",
            "target_doc",
            name="uq_entity_relationship",
        ),
    )
