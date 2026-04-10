from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain.schema import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower())


def _to_text(doc: Any) -> str:
    content = str(getattr(doc, "page_content", "") or "")
    meta = dict(getattr(doc, "metadata", {}) or {})
    raw = str(meta.get("raw", "") or "")
    source = str(meta.get("source", "") or "")
    return f"{source}\n{raw}\n{content}".strip()


def _chunk_hash(doc: Any) -> str:
    payload = _to_text(doc).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


@dataclass
class _CompiledIndex:
    mtime: float
    rows: List[Dict[str, Any]]
    bm25: BM25Okapi


class HybridBM25Index:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._compiled: Dict[int, _CompiledIndex] = {}

    def _path(self, kb_id: int) -> Path:
        return self.base / f"kb_{int(kb_id)}.json"

    def _load_rows(self, kb_id: int) -> List[Dict[str, Any]]:
        path = self._path(kb_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("rows", [])
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        except Exception:
            return []
        return []

    def _save_rows(self, kb_id: int, rows: List[Dict[str, Any]]) -> None:
        path = self._path(kb_id)
        payload = {"kb_id": int(kb_id), "rows": rows}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _compile_if_needed(self, kb_id: int) -> Optional[_CompiledIndex]:
        path = self._path(kb_id)
        if not path.exists():
            self._compiled.pop(int(kb_id), None)
            return None
        mtime = float(path.stat().st_mtime)
        cached = self._compiled.get(int(kb_id))
        if cached and cached.mtime == mtime:
            return cached
        rows = self._load_rows(kb_id)
        if not rows:
            self._compiled.pop(int(kb_id), None)
            return None
        corpus = [_tokenize(str(r.get("text", ""))) for r in rows]
        bm25 = BM25Okapi(corpus)
        compiled = _CompiledIndex(mtime=mtime, rows=rows, bm25=bm25)
        self._compiled[int(kb_id)] = compiled
        return compiled

    def upsert_chunks(self, kb_id: int, docs: Sequence[Any]) -> None:
        rows = self._load_rows(kb_id)
        seen_hashes = {str(r.get("chunk_hash", "")) for r in rows}
        for doc in docs:
            meta = dict(getattr(doc, "metadata", {}) or {})
            chunk_hash = _chunk_hash(doc)
            if chunk_hash in seen_hashes:
                continue
            text = _to_text(doc)
            rows.append(
                {
                    "chunk_hash": chunk_hash,
                    "doc_id": int(meta.get("doc_id")) if str(meta.get("doc_id", "")).isdigit() else None,
                    "source": str(meta.get("source", "") or ""),
                    "page": str(meta.get("page", "") or ""),
                    "pipeline": str(meta.get("pipeline", "") or ""),
                    "type": str(meta.get("type", "") or "text"),
                    "text": text,
                    "metadata": meta,
                }
            )
            seen_hashes.add(chunk_hash)
        with self._lock:
            self._save_rows(kb_id, rows)
            self._compiled.pop(int(kb_id), None)

    def remove_document(
        self,
        kb_id: int,
        *,
        doc_id: Optional[int] = None,
        stored_filename: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> int:
        rows = self._load_rows(kb_id)
        before = len(rows)
        out = []
        for row in rows:
            meta = dict(row.get("metadata", {}) or {})
            row_doc_id = row.get("doc_id")
            row_stored = str(meta.get("stored_filename", "") or "")
            row_source = str(row.get("source", "") or "")
            drop = False
            if doc_id is not None and row_doc_id == int(doc_id):
                drop = True
            if stored_filename and row_stored == str(stored_filename):
                drop = True
            if original_filename and row_source == str(original_filename):
                drop = True
            if not drop:
                out.append(row)
        removed = before - len(out)
        if removed > 0:
            with self._lock:
                self._save_rows(kb_id, out)
                self._compiled.pop(int(kb_id), None)
        return removed

    def search(self, kb_id: int, query: str, top_k: int = 8) -> List[Document]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        with self._lock:
            compiled = self._compile_if_needed(kb_id)
        if not compiled:
            return []
        scores = compiled.bm25.get_scores(q_terms)
        scored_rows = []
        for row, score in zip(compiled.rows, scores):
            scored_rows.append((float(score), row))
        scored_rows.sort(key=lambda x: x[0], reverse=True)

        out: List[Document] = []
        for score, row in scored_rows[: max(1, int(top_k))]:
            meta = dict(row.get("metadata", {}) or {})
            meta["bm25_score"] = float(score)
            out.append(
                Document(
                    page_content=str(meta.get("raw", "") or row.get("text", "")),
                    metadata=meta,
                )
            )
        return out
