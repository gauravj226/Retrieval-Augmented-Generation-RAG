from __future__ import annotations
from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, EmailStr


# ── Auth ─────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    department: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    department: Optional[str]
    created_at: datetime
    group_ids: List[int] = []

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    group_ids: Optional[List[int]] = None


# ── Personalities ─────────────────────────────────────────────────────────────

class PersonalityCreate(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str
    tone: str = "professional"


class PersonalityUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tone: Optional[str] = None


class PersonalityResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    system_prompt: str
    tone: str
    is_preset: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Groups ────────────────────────────────────────────────────────────────────

class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class GroupKBPermissionSet(BaseModel):
    kb_id: int
    permission: str   # read | manage


class GroupResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    member_count: int = 0
    kb_permissions: List[dict] = []

    model_config = {"from_attributes": True}


# ── Knowledge Bases ───────────────────────────────────────────────────────────

class KBCreate(BaseModel):
    name: str
    description: Optional[str] = None
    department: str
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"
    personality_id: Optional[int] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_k_docs: int = 4
    mmr_fetch_k: int = 16
    mmr_lambda: float = 0.7
    score_threshold: float = 0.35
    memory_window: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120


class KBUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    llm_model: Optional[str] = None
    personality_id: Optional[int] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k_docs: Optional[int] = None
    mmr_fetch_k: Optional[int] = None
    mmr_lambda: Optional[float] = None
    score_threshold: Optional[float] = None
    memory_window: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    is_active: Optional[bool] = None


class KBResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    department: str
    llm_model: str
    embedding_model: str
    personality_id: Optional[int]
    system_prompt: Optional[str]
    temperature: str
    max_tokens: int
    top_k_docs: int
    mmr_fetch_k: int
    mmr_lambda: str
    score_threshold: str
    memory_window: int
    chunk_size: int
    chunk_overlap: int
    is_active: bool
    created_at: datetime
    document_count: int = 0

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: int
    original_filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    uploaded_at: datetime
    uploaded_by: Optional[int] = None

    model_config = {"from_attributes": True}


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    kb_id: int
    session_id: Optional[int] = None
    fast_mode: Optional[bool] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Any]
    reasoning_trace: List[str] = []
    ui_payload: Optional[dict] = None
    session_id: int



class SessionResponse(BaseModel):
    id: int
    kb_id: int
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}

