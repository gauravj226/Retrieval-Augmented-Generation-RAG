import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _tokens(text: str) -> set[str]:
    src = (text or "").lower()
    base = set(re.findall(r"[a-zA-Z0-9]{2,}", src))
    # Preserve single-letter drive intents (q drive vs u drive) for cache separation.
    for letter in re.findall(r"\b([a-z])\s*:?\s*drive\b", src):
        base.add(f"drive_{letter}")
    return base


def _focus_entities(text: str) -> Dict[str, set[str]]:
    raw = text or ""
    lower = raw.lower()
    entities: Dict[str, set[str]] = {
        "drive": set(re.findall(r"\b([a-z])\s*:?\s*drive\b", lower)),
        "windows": set(re.findall(r"\bwindows\s*(10|11)\b", lower)),
        "path": set(re.findall(r"(\\\\[a-z0-9._$-]+\\[a-z0-9._$\\-]+)", lower)),
        "id": set(
            re.findall(
                r"\b(?:[a-z]{2,}[._-]?\d+[a-z0-9._-]*|\d+[a-z][a-z0-9._-]*)\b",
                lower,
            )
        ),
    }
    acronyms = [a.lower() for a in re.findall(r"\b[A-Z]{2,8}\b", raw)]
    if acronyms:
        entities["acronym"] = set(acronyms)
    return {k: v for k, v in entities.items() if v}


def _entities_compatible(query_entities: Dict[str, set[str]], candidate_entities: Dict[str, set[str]]) -> bool:
    if not query_entities:
        return True
    saw_candidate_family = False
    saw_overlap = False
    for family, q_vals in query_entities.items():
        c_vals = candidate_entities.get(family, set())
        if not c_vals:
            continue
        saw_candidate_family = True
        if q_vals.intersection(c_vals):
            saw_overlap = True
        else:
            # Candidate contains same entity family with conflicting value(s).
            return False
    # If candidate had no tracked family, let similarity decide.
    # If candidate had tracked family, require at least one overlap.
    return (not saw_candidate_family) or saw_overlap


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
    scope: str
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
        scope: Optional[str] = None,
    ) -> None:
        now = time.time()
        norm_scope = (scope or "global").strip() or "global"
        key = f"{kb_id}:{mode}:{norm_scope}:{now:.6f}"
        with self._lock:
            self._items[key] = CacheItem(
                query=query,
                answer=answer,
                sources=sources or [],
                trace=trace or [],
                kb_id=int(kb_id),
                mode=str(mode),
                scope=norm_scope,
                ts=now,
            )
            self._purge(now)

    def get(
        self,
        query: str,
        kb_id: int,
        mode: str,
        scope: Optional[str] = None,
    ) -> Optional[Tuple[str, List[dict], List[str], float]]:
        now = time.time()
        q_tokens = _tokens(query)
        q_entities = _focus_entities(query)
        norm_scope = (scope or "global").strip() or "global"
        best: Optional[Tuple[str, List[dict], List[str], float]] = None
        with self._lock:
            self._purge(now)
            for item in self._items.values():
                if item.kb_id != int(kb_id) or item.mode != str(mode):
                    continue
                if item.scope != norm_scope:
                    continue
                if q_entities:
                    c_entities = _focus_entities(item.query)
                    if not _entities_compatible(q_entities, c_entities):
                        continue
                candidate_tokens = _tokens(item.query)
                score = max(_jaccard(q_tokens, candidate_tokens), _overlap_ratio(q_tokens, candidate_tokens))
                if score >= self.similarity_threshold and (best is None or score > best[3]):
                    best = (item.answer, item.sources, item.trace, score)
        return best
