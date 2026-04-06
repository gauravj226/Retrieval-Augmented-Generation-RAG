import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    denom = max(1, min(len(a), len(b)))
    return inter / denom


@dataclass
class CacheItem:
    query: str
    answer: str
    sources: List[dict]
    trace: List[str]
    kb_id: int
    mode: str
    ts: float


class SemanticAnswerCache:
    def __init__(self, similarity_threshold: float = 0.82, ttl_seconds: int = 300, max_entries: int = 512):
        self.similarity_threshold = float(similarity_threshold)
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self._items: Dict[str, CacheItem] = {}
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        stale = [k for k, v in self._items.items() if (now - v.ts) > self.ttl_seconds]
        for key in stale:
            self._items.pop(key, None)
        while len(self._items) > self.max_entries:
            oldest_key = min(self._items, key=lambda k: self._items[k].ts)
            self._items.pop(oldest_key, None)

    def put(
        self,
        query: str,
        answer: str,
        kb_id: int,
        mode: str,
        sources: Optional[List[dict]] = None,
        trace: Optional[List[str]] = None,
    ) -> None:
        now = time.time()
        key = f"{kb_id}:{mode}:{now:.6f}"
        with self._lock:
            self._items[key] = CacheItem(
                query=query,
                answer=answer,
                sources=sources or [],
                trace=trace or [],
                kb_id=int(kb_id),
                mode=str(mode),
                ts=now,
            )
            self._purge(now)

    def get(self, query: str, kb_id: int, mode: str) -> Optional[Tuple[str, List[dict], List[str], float]]:
        now = time.time()
        q_tokens = _tokens(query)
        best: Optional[Tuple[str, List[dict], List[str], float]] = None
        with self._lock:
            self._purge(now)
            for item in self._items.values():
                if item.kb_id != int(kb_id) or item.mode != str(mode):
                    continue
                candidate_tokens = _tokens(item.query)
                score = max(_jaccard(q_tokens, candidate_tokens), _overlap_ratio(q_tokens, candidate_tokens))
                if score >= self.similarity_threshold and (best is None or score > best[3]):
                    best = (item.answer, item.sources, item.trace, score)
        return best
