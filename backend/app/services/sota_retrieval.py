import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "about",
    "from",
    "have",
    "your",
    "into",
}


@dataclass
class SparseHit:
    content: str
    metadata: dict
    score: float


def _tokenise(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_\-]{2,}", (text or "").lower()) if t not in _STOPWORDS]


def build_document_summary(texts: Sequence[str], max_sentences: int = 2) -> str:
    joined = " ".join(t.strip() for t in texts if t and t.strip())
    if not joined:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", joined)
    shortlist = [s.strip() for s in sentences if len(s.strip()) > 20]
    selected = shortlist[:max_sentences] if shortlist else [joined[:240]]
    return " ".join(selected)[:360]


def _clone_document(doc: Any, page_content: str, metadata: dict):
    try:
        return doc.__class__(page_content=page_content, metadata=metadata)
    except Exception:
        doc.page_content = page_content
        doc.metadata = metadata
        return doc


def contextualize_documents(docs: Sequence[Any]) -> List[Any]:
    texts = [getattr(d, "page_content", "") for d in docs]
    summary = build_document_summary(texts, max_sentences=2)
    output: List[Any] = []
    for doc in docs:
        raw = getattr(doc, "page_content", "")
        meta = dict(getattr(doc, "metadata", {}) or {})
        meta["raw"] = raw
        meta["context_summary"] = summary
        contextualized = f"Document context: {summary}\n\nChunk:\n{raw}".strip()
        output.append(_clone_document(doc, contextualized, meta))
    return output


def route_mode_for_query(query: str) -> str:
    q = (query or "").strip().lower()
    if not q:
        return "clarify"
    if q in {"hi", "hello", "hey", "thanks", "thank you"}:
        return "general"
    if len(q.split()) < 2:
        return "clarify"
    if re.search(r"\b[a-z]{2,}\-[a-z0-9]{2,}\b", q) or re.search(r"\b[a-f0-9]{6,}\b", q):
        return "sparse"
    if any(t in q for t in ("relationship", "depends on", "connected to", "impact of", "upstream", "downstream")):
        return "graph"
    return "retrieve"


def score_sparse_hits(query: str, docs: Iterable[Any], top_k: int = 8) -> List[SparseHit]:
    q_terms = _tokenise(query)
    if not q_terms:
        return []
    scored: List[SparseHit] = []
    for doc in docs:
        content = str(getattr(doc, "page_content", "") or "")
        meta = dict(getattr(doc, "metadata", {}) or {})
        source = str(meta.get("source", "")).lower()
        corpus = (content + "\n" + str(meta.get("raw", ""))).lower()
        filename_hits = sum(1 for t in q_terms if t in source)
        token_hits = sum(corpus.count(t) for t in q_terms)
        exact_id_boost = 2 if re.search(r"\b[a-z]{2,}\-\d{2,}\b", query.lower()) else 0
        score = (filename_hits * 6) + min(token_hits, 20) + exact_id_boost
        if score > 0:
            scored.append(SparseHit(content=content, metadata=meta, score=float(score)))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
