from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from ..config import settings
from ..database import SessionLocal
from ..models.models import Document, KnowledgeBase, User
from .audit import audit_event
from .rag_service import delete_document_vectors, ingest_document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestTask:
    doc_id: int
    kb_id: int
    file_path: str
    original_filename: str
    uploaded_by: Optional[int]
    stored_filename: str


_STOP = object()
_queue: Optional[asyncio.Queue] = None
_workers: list[asyncio.Task] = []


def _get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        maxsize = max(1, int(settings.INGEST_QUEUE_MAX))
        _queue = asyncio.Queue(maxsize=maxsize)
    return _queue


def ingest_queue_size() -> int:
    q = _get_queue()
    return q.qsize()


async def enqueue_ingest_task(task: IngestTask) -> None:
    q = _get_queue()
    if not _workers:
        raise RuntimeError("Ingestion workers are not running")
    try:
        q.put_nowait(task)
    except asyncio.QueueFull as e:
        raise RuntimeError("Ingestion queue is full; try again shortly") from e


async def _process_task(task: IngestTask, worker_name: str) -> None:
    db = SessionLocal()
    actor = None
    try:
        doc = db.query(Document).filter(Document.id == task.doc_id).first()
        if not doc:
            logger.info("[%s] Skipping missing document id=%s", worker_name, task.doc_id)
            return

        if task.uploaded_by:
            actor = db.query(User).filter(User.id == task.uploaded_by).first()

        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == task.kb_id).first()
        if not kb:
            doc.status = "error"
            db.commit()
            audit_event(
                "document.upload_failed",
                actor=actor,
                target_type="document",
                target_id=doc.id,
                status="failed",
                details={"kb_id": task.kb_id, "filename": task.original_filename, "error": "KB not found"},
            )
            return

        # Run ingestion in a background thread. Parsing/OCR/VLM paths contain
        # heavy sync work that can block the main event loop otherwise.
        kb_ref = SimpleNamespace(
            chroma_collection=kb.chroma_collection,
            embedding_model=kb.embedding_model,
        )
        chunk_count = await asyncio.to_thread(
            _run_ingest_sync,
            task,
            int(doc.id),
            kb_ref,
        )

        # Re-read in case it changed while job was running.
        doc = db.query(Document).filter(Document.id == task.doc_id).first()
        if not doc:
            logger.info("[%s] Document deleted before completion id=%s", worker_name, task.doc_id)
            delete_document_vectors(
                kb=kb,
                doc_id=task.doc_id,
                stored_filename=task.stored_filename,
                original_filename=task.original_filename,
            )
            return

        doc.chunk_count = int(chunk_count)
        doc.status = "ready"
        db.commit()

        audit_event(
            "document.upload_completed",
            actor=actor,
            target_type="document",
            target_id=doc.id,
            details={
                "kb_id": task.kb_id,
                "filename": task.original_filename,
                "chunk_count": int(chunk_count),
                "worker": worker_name,
            },
        )
    except Exception as e:
        db.rollback()
        doc = db.query(Document).filter(Document.id == task.doc_id).first()
        if doc:
            doc.status = "error"
            db.commit()

        audit_event(
            "document.upload_failed",
            actor=actor,
            target_type="document",
            target_id=task.doc_id,
            status="failed",
            details={
                "kb_id": task.kb_id,
                "filename": task.original_filename,
                "error": str(e),
                "worker": worker_name,
            },
        )
        logger.exception("[%s] Ingestion failed for doc_id=%s: %s", worker_name, task.doc_id, e)
    finally:
        db.close()


def _run_ingest_sync(task: IngestTask, doc_id: int, kb_ref: SimpleNamespace) -> int:
    return asyncio.run(
        ingest_document(
            task.file_path,
            task.original_filename,
            kb_ref,
            metadata={
                "doc_id": doc_id,
                "stored_filename": task.stored_filename,
                "uploaded_by": task.uploaded_by or "",
            },
        )
    )


async def _worker_loop(worker_idx: int) -> None:
    q = _get_queue()
    worker_name = f"ingest-worker-{worker_idx}"
    logger.info("[%s] started", worker_name)
    while True:
        item = await q.get()
        try:
            if item is _STOP:
                logger.info("[%s] stopping", worker_name)
                return
            await _process_task(item, worker_name)
        finally:
            q.task_done()


async def start_ingest_workers() -> None:
    if _workers:
        return
    count = max(1, int(settings.INGEST_WORKERS))
    for i in range(count):
        _workers.append(asyncio.create_task(_worker_loop(i + 1), name=f"ingest-worker-{i+1}"))
    logger.info("Started %s ingestion worker(s)", count)


async def stop_ingest_workers() -> None:
    if not _workers:
        return
    q = _get_queue()
    for _ in _workers:
        await q.put(_STOP)
    await asyncio.gather(*_workers, return_exceptions=True)
    _workers.clear()
    logger.info("Ingestion workers stopped")
