import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence
from rank_bm25 import BM25Okapi



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
    src = (text or "").lower()
    tokens = [t for t in re.findall(r"[a-zA-Z0-9_\-]{2,}", src) if t not in _STOPWORDS]
    # Preserve discriminative entities so sparse retrieval can separate close intents.
    tokens.extend([f"drive_{letter}" for letter in re.findall(r"\b([a-z])\s*:?\s*drive\b", src)])
    tokens.extend([f"win_{ver}" for ver in re.findall(r"\bwindows\s*(10|11)\b", src)])
    tokens.extend([f"path_{p}" for p in re.findall(r"(\\\\[a-z0-9._$-]+\\[a-z0-9._$\\-]+)", src)])
    tokens.extend(
        [
            f"id_{i}"
            for i in re.findall(
                r"\b(?:[a-z]{2,}[._-]?\d+[a-z0-9._-]*|\d+[a-z][a-z0-9._-]*)\b",
                src,
            )
        ]
    )
    return tokens


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
    # Group by source document
    from collections import defaultdict
    groups = defaultdict(list)
    for doc in docs:
        src = (getattr(doc, "metadata", {}) or {}).get("source", "__default__")
        groups[src].append(doc)
    
    output = []
    for src, group_docs in groups.items():
        texts = [getattr(d, "page_content", "") for d in group_docs]
        summary = build_document_summary(texts, max_sentences=2)  # per-doc summary
        for doc in group_docs:
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


def score_sparse_hits(query: str, docs, top_k: int = 8) -> List[SparseHit]:
    q_terms = _tokenise(query)
    if not q_terms:
        return []
    
    corpus = []
    for doc in docs:
        content = str(getattr(doc, "page_content", "") or "")
        raw = str((getattr(doc, "metadata", {}) or {}).get("raw", ""))
        corpus.append(_tokenise(content + " " + raw))
    
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q_terms)
    
    scored = []
    for i, (doc, score) in enumerate(zip(docs, scores)):
        if score > 0:
            meta = dict(getattr(doc, "metadata", {}) or {})
            scored.append(SparseHit(
                content=str(getattr(doc, "page_content", "")),
                metadata=meta,
                score=float(score)
            ))
    
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
