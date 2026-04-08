import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.models import ChatMessage, ChatSession, KnowledgeBase, User
from ..routers.auth import get_current_active_user
from ..routers.permissions import get_accessible_kb_ids, require_kb_access
from ..schemas.schemas import (
    ChatRequest, ChatResponse, MessageResponse, SessionResponse,
)
from ..services.agentic_rag import run_agentic_rag
from ..services.audit import audit_event

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get("/knowledge-bases")
async def get_available_kbs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    accessible = get_accessible_kb_ids(current_user, db)
    q = db.query(KnowledgeBase).filter(KnowledgeBase.is_active == True)
    if accessible is not None:
        q = q.filter(KnowledgeBase.id.in_(accessible))
    return [
        {
            "id": kb.id, "name": kb.name, "department": kb.department,
            "description": kb.description, "llm_model": kb.llm_model,
        }
        for kb in q.all()
    ]


@router.get("/kb-permission/{kb_id}")
async def get_my_kb_permission(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from ..routers.permissions import get_kb_permission
    perm = get_kb_permission(current_user, kb_id, db)
    return {
        "kb_id":      kb_id,
        "permission": perm,
        "can_manage": perm == "manage",
        "can_read":   perm in ("read", "manage"),
        "is_admin":   current_user.is_admin,
    }


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    require_kb_access(current_user, request.kb_id, "read", db)

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == request.kb_id,
        KnowledgeBase.is_active == True,
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found or inactive")

    # ── Session management ────────────────────────────────────────────────────
    if request.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == request.session_id,
            ChatSession.user_id == current_user.id,
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        title   = request.message[:50] + ("..." if len(request.message) > 50 else "")
        session = ChatSession(user_id=current_user.id, kb_id=kb.id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
        audit_event(
            "chat.session_created",
            actor=current_user,
            target_type="chat_session",
            target_id=session.id,
            details={"kb_id": kb.id, "title": title},
        )

    # ── Build chat history ────────────────────────────────────────────────────
    history_msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    chat_history = [
        (history_msgs[i].content, history_msgs[i + 1].content)
        for i in range(0, len(history_msgs) - 1, 2)
        if i + 1 < len(history_msgs)
    ]

    # ── Run agentic RAG graph ─────────────────────────────────────────────────
    try:
        answer, sources, trace, ui_payload = await run_agentic_rag(
            kb=kb,
            question=request.message,
            chat_history=chat_history,
            db=db,
            user_id=current_user.id,
            session_id=session.id,
            fast_mode_override=request.fast_mode,
        )
    except Exception as e:
        audit_event(
            "chat.message_failed",
            actor=current_user,
            target_type="knowledge_base",
            target_id=kb.id,
            status="failed",
            details={"session_id": session.id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))

        # ── Persist messages ──────────────────────────────────────────────────────
    db.add_all([
        ChatMessage(
            session_id=session.id,
            role="user",
            content=request.message,
        ),
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            # Store both sources and trace together for session replay
            sources=json.dumps({
                "sources":         sources,
                "reasoning_trace": trace,
                "ui_payload": ui_payload,
            }),
        ),
    ])
    db.commit()
    audit_event(
        "chat.message_sent",
        actor=current_user,
        target_type="chat_session",
        target_id=session.id,
        details={"kb_id": kb.id, "sources_count": len(sources)},
    )

    # ── Return trace explicitly in response ───────────────────────────────────
    return {
        "answer":          answer,
        "sources":         sources,
        "reasoning_trace": trace,
        "ui_payload":      ui_payload,
        "session_id":      session.id,
    }



@router.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .limit(50)
        .all()
    )


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Not found")
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Not found")
    title = session.title
    db.delete(session)
    db.commit()
    audit_event(
        "chat.session_deleted",
        actor=current_user,
        target_type="chat_session",
        target_id=session_id,
        details={"title": title},
    )
    return {"message": "Deleted"}

