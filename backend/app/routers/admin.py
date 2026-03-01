import json
import uuid
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models.models import (
    Document, Group, KnowledgeBase, Personality, User, UserGroupMapping,
)
from ..routers.auth import get_admin_user
from ..schemas.schemas import (
    KBCreate, KBResponse, KBUpdate,
    PersonalityCreate, PersonalityResponse, PersonalityUpdate,
    UserResponse, UserUpdate,
)
from ..services.rag_service import delete_kb_collection
from ..services.audit import audit_event

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Ollama models ─────────────────────────────────────────────────────────────

@router.get("/ollama/models")
async def get_ollama_models(admin: User = Depends(get_admin_user)):
    """List models available in the local Ollama instance."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(
                f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}/api/tags"
            )
            data = res.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return [settings.DEFAULT_LLM_MODEL]


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats(
    db: Session = Depends(get_db), admin: User = Depends(get_admin_user)
):
    return {
        "total_users": db.query(User).count(),
        "total_kbs": db.query(KnowledgeBase).count(),
        "active_kbs": db.query(KnowledgeBase).filter(KnowledgeBase.is_active == True).count(),
        "total_documents": db.query(Document).count(),
        "total_groups": db.query(Group).count(),
        "total_personalities": db.query(Personality).count(),
    }


# ── Personalities ─────────────────────────────────────────────────────────────

@router.get("/personalities", response_model=List[PersonalityResponse])
async def list_personalities(
    db: Session = Depends(get_db), admin: User = Depends(get_admin_user)
):
    return db.query(Personality).all()


@router.post("/personalities", response_model=PersonalityResponse)
async def create_personality(
    data: PersonalityCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    if db.query(Personality).filter(Personality.name == data.name).first():
        raise HTTPException(status_code=400, detail="Name already exists")
    p = Personality(**data.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    audit_event(
        "personality.created",
        actor=admin,
        target_type="personality",
        target_id=p.id,
        details={"name": p.name},
    )
    return p


@router.put("/personalities/{pid}", response_model=PersonalityResponse)
async def update_personality(
    pid: int,
    data: PersonalityUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    p = db.query(Personality).filter(Personality.id == pid).first()
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    if p.is_preset:
        raise HTTPException(status_code=400, detail="Cannot edit built-in presets")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(p, field, value)
    db.commit()
    db.refresh(p)
    audit_event(
        "personality.updated",
        actor=admin,
        target_type="personality",
        target_id=p.id,
        details={"name": p.name},
    )
    return p


@router.delete("/personalities/{pid}")
async def delete_personality(
    pid: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    p = db.query(Personality).filter(Personality.id == pid).first()
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    if p.is_preset:
        raise HTTPException(status_code=400, detail="Cannot delete built-in presets")
    p_name = p.name
    db.delete(p)
    db.commit()
    audit_event(
        "personality.deleted",
        actor=admin,
        target_type="personality",
        target_id=pid,
        details={"name": p_name},
    )
    return {"message": "Deleted"}


# ── Knowledge Bases ───────────────────────────────────────────────────────────

@router.get("/knowledge-bases", response_model=List[KBResponse])
async def list_kbs(
    db: Session = Depends(get_db), admin: User = Depends(get_admin_user)
):
    kbs = db.query(KnowledgeBase).all()
    return [{**kb.__dict__, "document_count": len(kb.documents)} for kb in kbs]


@router.post("/knowledge-bases", response_model=KBResponse)
async def create_kb(
    data: KBCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    if db.query(KnowledgeBase).filter(KnowledgeBase.name == data.name).first():
        raise HTTPException(status_code=400, detail="Name already exists")

    slug = data.department.lower().replace(" ", "_")
    collection = f"kb_{slug}_{uuid.uuid4().hex[:8]}"

    dump = data.model_dump()
    kb = KnowledgeBase(
        **{k: v for k, v in dump.items() if k not in ("temperature", "mmr_lambda")},
        temperature=str(data.temperature),
        mmr_lambda=str(data.mmr_lambda),
        chroma_collection=collection,
        created_by=admin.id,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    audit_event(
        "kb.created",
        actor=admin,
        target_type="knowledge_base",
        target_id=kb.id,
        details={"name": kb.name, "department": kb.department},
    )
    return {**kb.__dict__, "document_count": 0}


@router.put("/knowledge-bases/{kb_id}", response_model=KBResponse)
async def update_kb(
    kb_id: int,
    data: KBUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Not found")

    for field, value in data.model_dump(exclude_unset=True).items():
        if field in ("temperature", "mmr_lambda"):
            setattr(kb, field, str(value))
        else:
            setattr(kb, field, value)

    db.commit()
    db.refresh(kb)
    audit_event(
        "kb.updated",
        actor=admin,
        target_type="knowledge_base",
        target_id=kb.id,
        details={"name": kb.name},
    )
    return {**kb.__dict__, "document_count": len(kb.documents)}


@router.delete("/knowledge-bases/{kb_id}")
async def delete_kb(
    kb_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Not found")
    kb_name = kb.name
    delete_kb_collection(kb.chroma_collection)
    db.delete(kb)
    db.commit()
    audit_event(
        "kb.deleted",
        actor=admin,
        target_type="knowledge_base",
        target_id=kb_id,
        details={"name": kb_name},
    )
    return {"message": "Deleted"}


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    db: Session = Depends(get_db), admin: User = Depends(get_admin_user)
):
    users = db.query(User).all()
    result = []
    for u in users:
        group_ids = [m.group_id for m in u.group_memberships]
        d = {**u.__dict__, "group_ids": group_ids}
        result.append(d)
    return result


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Not found")

    updates = data.model_dump(exclude_unset=True)
    group_ids: Optional[List[int]] = updates.pop("group_ids", None)

    for field, value in updates.items():
        setattr(user, field, value)

    if group_ids is not None:
        db.query(UserGroupMapping).filter(UserGroupMapping.user_id == user_id).delete()
        for gid in group_ids:
            db.add(UserGroupMapping(user_id=user_id, group_id=gid))

    db.commit()
    db.refresh(user)
    audit_event(
        "user.updated",
        actor=admin,
        target_type="user",
        target_id=user.id,
        details={"username": user.username},
    )
    return {**user.__dict__, "group_ids": group_ids or []}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Not found")
    if user.is_admin:
        raise HTTPException(status_code=400, detail="Cannot delete admin")
    username = user.username
    db.delete(user)
    db.commit()
    audit_event(
        "user.deleted",
        actor=admin,
        target_type="user",
        target_id=user_id,
        details={"username": username},
    )
    return {"message": "Deleted"}
