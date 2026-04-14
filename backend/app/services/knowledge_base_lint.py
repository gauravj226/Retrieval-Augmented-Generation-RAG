import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..models.models import KnowledgeBase, Document
from .rag_service import get_chroma_client, _chunk_quality_score
from ..config import settings

logger = logging.getLogger(__name__)

class KBLinter:
    def __init__(self, db: Session):
        self.db = db
        self.chroma_client = get_chroma_client()

    def lint_kb(self, kb_id: int) -> Dict[str, Any]:
        """Run all linting checks for a specific Knowledge Base."""
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            return {"error": "Knowledge Base not found"}

        results = {
            "low_quality_chunks": self._check_ocr_quality(kb),
            "superseded_documents": self._check_superseded(kb),
            "irrelevant_chunks": self._check_relevance_history(kb),
        }
        
        return results

    def _check_ocr_quality(self, kb: KnowledgeBase) -> List[Dict[str, Any]]:
        """Find chunks with OCR quality score below threshold."""
        threshold = 0.55
        flagged = []
        try:
            collection = self.chroma_client.get_collection(name=kb.chroma_collection)
            # Fetch chunks (limited for performance, in real-world use pagination)
            data = collection.get(limit=1000, include=["documents", "metadatas"])
            
            for doc_text, meta in zip(data["documents"], data["metadatas"]):
                score = _chunk_quality_score(doc_text)
                if score < threshold:
                    flagged.append({
                        "source": meta.get("source"),
                        "doc_id": meta.get("doc_id"),
                        "quality_score": round(score, 3),
                        "snippet": doc_text[:100] + "..."
                    })
        except Exception as e:
            logger.error(f"OCR quality lint failed for KB {kb.id}: {e}")
        
        return flagged

    def _check_superseded(self, kb: KnowledgeBase) -> List[Dict[str, Any]]:
        """Find documents that might be older versions of existing ones."""
        flagged = []
        docs = self.db.query(Document).filter(Document.kb_id == kb.id).all()
        
        # Simple heuristic: same original filename prefix but different upload time
        from collections import defaultdict
        groups = defaultdict(list)
        for d in docs:
            # Strip extension and common version suffixes
            base_name = d.original_filename.rsplit('.', 1)[0].lower()
            groups[base_name].append(d)
            
        for base, group in groups.items():
            if len(group) > 1:
                group.sort(key=lambda x: x.created_at, reverse=True)
                latest = group[0]
                for old in group[1:]:
                    flagged.append({
                        "filename": old.original_filename,
                        "id": old.id,
                        "superseded_by": latest.original_filename,
                        "uploaded_at": str(old.created_at)
                    })
        return flagged

    def _check_relevance_history(self, kb: KnowledgeBase) -> List[Dict[str, Any]]:
        """Identify chunks that consistently score 0 in grading logs."""
        # This requires an audit/grading log table which might not be fully implemented yet.
        # For now, we return an empty list or a placeholder.
        return []

def run_global_lint(db: Session):
    """Run linting across all Knowledge Bases."""
    kbs = db.query(KnowledgeBase).all()
    linter = KBLinter(db)
    for kb in kbs:
        logger.info(f"Starting lint for KB {kb.id} ({kb.name})")
        results = linter.lint_kb(kb.id)
        # In a real app, these would be saved to a 'LintResults' table or emailed to admin
        if any(results.values()):
            logger.warning(f"Lint issues found for KB {kb.id}: {results}")
