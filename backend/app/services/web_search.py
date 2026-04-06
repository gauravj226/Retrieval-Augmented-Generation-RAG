import logging
from typing import List

from ..config import settings

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> List[dict]:
    if not settings.ENABLE_WEB_FALLBACK:
        return []
    provider = (settings.WEB_SEARCH_PROVIDER or "duckduckgo").lower()
    if provider != "duckduckgo":
        return []
    try:
        from duckduckgo_search import DDGS

        with DDGS(timeout=settings.WEB_SEARCH_TIMEOUT_SEC) as ddgs:
            results = ddgs.text(
                keywords=query,
                max_results=min(max_results, settings.WEB_SEARCH_MAX_RESULTS),
            )
            output = []
            for row in results or []:
                output.append(
                    {
                        "source": row.get("href", "web"),
                        "title": row.get("title", "Web Result"),
                        "content": row.get("body", ""),
                        "pipeline": "web",
                        "type": "web",
                    }
                )
            return output
    except Exception as exc:
        logger.warning("Web fallback search failed: %s", exc)
        return []
