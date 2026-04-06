import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List


class GraphMemoryStore:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kb_id: int) -> Path:
        return self.base_dir / f"kb_{int(kb_id)}_graph.json"

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        # Lightweight entity extraction for proper nouns, service names and acronyms.
        entities = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text or "")
        entities += re.findall(r"\b[A-Z]{2,}[0-9A-Z_-]*\b", text or "")
        deduped = []
        seen = set()
        for entity in entities:
            key = entity.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped[:40]

    def _load(self, kb_id: int) -> Dict[str, Dict[str, int]]:
        path = self._path(kb_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, kb_id: int, graph: Dict[str, Dict[str, int]]) -> None:
        self._path(kb_id).write_text(json.dumps(graph, ensure_ascii=True, indent=2), encoding="utf-8")

    def index_documents(self, kb_id: int, docs: Iterable[Any]) -> None:
        graph = self._load(kb_id)
        adj = defaultdict(dict, graph)

        for doc in docs:
            text = str(getattr(doc, "page_content", "") or "")
            raw = str((getattr(doc, "metadata", {}) or {}).get("raw", "") or "")
            entities = self._extract_entities(raw if raw else text)
            for a, b in combinations([e.lower() for e in entities], 2):
                adj[a][b] = int(adj[a].get(b, 0)) + 1
                adj[b][a] = int(adj[b].get(a, 0)) + 1

        self._save(kb_id, {k: dict(v) for k, v in adj.items()})

    def expand_query(self, kb_id: int, query: str, max_neighbors: int = 5) -> List[str]:
        graph = self._load(kb_id)
        entities = [e.lower() for e in self._extract_entities(query)]
        expansions: List[str] = []
        for ent in entities:
            neighbors = graph.get(ent, {})
            top = sorted(neighbors.items(), key=lambda kv: int(kv[1]), reverse=True)[:max_neighbors]
            for n, _ in top:
                if n not in expansions:
                    expansions.append(n)
        return expansions[:max_neighbors]
