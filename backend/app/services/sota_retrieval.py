import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from rank_bm25 import BM25Okapi

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "when", "where", "which", "about", "from", "have", "your", "into",
}

_SQL_PATTERNS = [
    r"\bhow many (invoices|bills|payments)",
    r"\btotal (amount|spend|value|cost).*(vendor|supplier|from)",
    r"\blist (all|every) (invoices|bills).*(from|vendor|supplier)",
    r"\bcount (invoices|documents|bills)",
    r"\bsum.*(invoices|payments)",
    r"\bwhich vendor",
    r"\bhighest|lowest.*(invoice|amount|spend)",
]

@dataclass
class SparseHit:
    content: str
    metadata: dict
    score: float

def _tokenise(text: str) -> List[str]:
    src = (text or "").lower()
    tokens = [t for t in re.findall(r"[a-zA-Z0-9_\-]{2,}", src) if t not in _STOPWORDS]
    return tokens

def route_mode_for_query(query: str) -> str:
    """
    Decides the retrieval mode: 'sql' for aggregations, 'retrieve' for semantic search.
    """
    q = (query or "").strip().lower()
    
    # Check for SQL patterns
    for pat in _SQL_PATTERNS:
        if re.search(pat, q):
            return "sql"
            
    # Default to standard retrieval
    return "retrieve"

async def contextualize_documents(docs: List[Any], kb: Any = None) -> List[Any]:
    """
    Stubs for contextual retrieval if enabled.
    """
    return docs
