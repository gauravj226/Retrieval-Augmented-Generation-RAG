import os
import uuid
from typing import List, Tuple

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models.models import Document, KnowledgeBase, User
from ..routers.auth import get_current_active_user
from ..routers.permissions import require_kb_access
from ..schemas.schemas import DocumentResponse
from ..services.document_processor import SUPPORTED_EXTENSIONS
from ..services.rag_service import delete_document_vectors
from ..services.ingest_queue import IngestTask, enqueue_ingest_task
from ..services.audit import audit_event

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_UPLOAD_FILES = settings.MAX_UPLOAD_FILES
MAX_UPLOAD_FILE_BYTES = settings.MAX_UPLOAD_FILE_MB * 1024 * 1024
MAX_UPLOAD_BATCH_BYTES = settings.MAX_UPLOAD_BATCH_MB * 1024 * 1024


def _validate_extension(raw_name: str) -> str:
    ext = raw_name.rsplit(".", 1)[-1].lower() if "." in raw_name else ""
    if not ext or ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )
    return ext


async def _read_validated_upload(file: UploadFile) -> Tuple[str, str, bytes]:
    raw_name = file.filename or ""
    if not raw_name:
        raise HTTPException(status_code=400, detail="File has no filename")

    ext = _validate_extension(raw_name)
    content = await file.read()
    size = len(content)
    if size == 0:
        raise HTTPException(status_code=400, detail=f"'{raw_name}' is empty")
    if size > MAX_UPLOAD_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"'{raw_name}' exceeds max size of {settings.MAX_UPLOAD_FILE_MB} MB "
                f"per file"
            ),
        )
    return raw_name, ext, content


async def _save_and_queue_document(
    kb_id: int,
    db: Session,
    current_user: User,
    original_filename: str,
    ext: str,
    content: bytes,
) -> Document:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    doc = Document(
        filename=unique_name,
        original_filename=original_filename,
        file_type=ext,
        file_size=len(content),
        kb_id=kb_id,
        status="processing",
        uploaded_by=current_user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    audit_event(
        "document.upload_started",
        actor=current_user,
        target_type="document",
        target_id=doc.id,
        details={"kb_id": kb_id, "filename": original_filename, "size": len(content)},
    )
    try:
        await enqueue_ingest_task(
            IngestTask(
                doc_id=doc.id,
                kb_id=kb_id,
                file_path=file_path,
                original_filename=original_filename,
                uploaded_by=current_user.id,
                stored_filename=doc.filename,
            )
        )
        audit_event(
            "document.upload_queued",
            actor=current_user,
            target_type="document",
            target_id=doc.id,
            details={"kb_id": kb_id, "filename": original_filename},
        )
    except Exception as e:
        doc.status = "error"
        db.commit()
        audit_event(
            "document.upload_failed",
            actor=current_user,
            target_type="document",
            target_id=doc.id,
            status="failed",
            details={"kb_id": kb_id, "filename": original_filename, "error": str(e)},
        )
        raise HTTPException(status_code=503, detail=str(e))

    db.commit()
    db.refresh(doc)
    return doc


@router.post("/upload/{kb_id}", response_model=DocumentResponse)
async def upload_document(
    kb_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    require_kb_access(current_user, kb_id, "manage", db)

    if not db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first():
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    raw_name, ext, content = await _read_validated_upload(file)
    return await _save_and_queue_document(
        kb_id=kb_id,
        db=db,
        current_user=current_user,
        original_filename=raw_name,
        ext=ext,
        content=content,
    )


@router.post("/upload-multiple/{kb_id}", response_model=List[DocumentResponse])
async def upload_documents(
    kb_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    require_kb_access(current_user, kb_id, "manage", db)

    if not db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first():
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_UPLOAD_FILES} files allowed per upload",
        )

    validated: List[Tuple[str, str, bytes]] = []
    total_bytes = 0
    for file in files:
        raw_name, ext, content = await _read_validated_upload(file)
        total_bytes += len(content)
        validated.append((raw_name, ext, content))

    if total_bytes > MAX_UPLOAD_BATCH_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Combined upload exceeds {settings.MAX_UPLOAD_BATCH_MB} MB "
                f"per batch"
            ),
        )

    created: List[Document] = []
    audit_event(
        "document.batch_upload_started",
        actor=current_user,
        target_type="knowledge_base",
        target_id=kb_id,
        details={"file_count": len(validated)},
    )
    for raw_name, ext, content in validated:
        doc = await _save_and_queue_document(
            kb_id=kb_id,
            db=db,
            current_user=current_user,
            original_filename=raw_name,
            ext=ext,
            content=content,
        )
        created.append(doc)
    audit_event(
        "document.batch_upload_queued",
        actor=current_user,
        target_type="knowledge_base",
        target_id=kb_id,
        details={"file_count": len(created)},
    )

    return created


@router.get("/{kb_id}", response_model=List[DocumentResponse])
async def list_documents(
    kb_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    require_kb_access(current_user, kb_id, "read", db)
    if not db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first():
        raise HTTPException(status_code=404, detail="Not found")
    return db.query(Document).filter(Document.kb_id == kb_id).all()


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # ── Ownership + permission rules ──────────────────────────────────────────
    # Admin            → can delete ANY document in ANY KB
    # manage-level     → can only delete documents THEY uploaded
    # read-level/none  → cannot delete at all
    if current_user.is_admin:
        pass  # full access
    else:
        # Must have at least manage permission on this KB
        require_kb_access(current_user, doc.kb_id, "manage", db)

        # Must be the uploader
        if doc.uploaded_by != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="You can only delete documents you uploaded yourself"
            )

    fp = os.path.join(settings.UPLOAD_DIR, doc.filename)
    if os.path.exists(fp):
        os.remove(fp)

    # Remove vectors tied to this specific document from the KB collection.
    delete_document_vectors(
        kb=doc.knowledge_base,
        doc_id=doc.id,
        stored_filename=doc.filename,
        original_filename=doc.original_filename,
    )

    deleted_name = doc.original_filename
    kb_id = doc.kb_id
    db.delete(doc)
    db.commit()
    audit_event(
        "document.deleted",
        actor=current_user,
        target_type="document",
        target_id=doc_id,
        details={"kb_id": kb_id, "filename": deleted_name},
    )
    return {"message": "Deleted"}
