import logging

from langchain_chroma import Chroma

from ..config import settings
from ..models.models import KnowledgeBase

logger = logging.getLogger(__name__)


def get_vector_store(kb: KnowledgeBase, embedding_function, chroma_client):
    provider = (settings.VECTOR_DB_PROVIDER or "chroma").strip().lower()
    if provider == "qdrant":
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

            client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            return QdrantVectorStore(
                client=client,
                collection_name=kb.chroma_collection,
                embedding=embedding_function,
            )
        except Exception as exc:
            logger.warning("Qdrant provider unavailable (%s). Falling back to Chroma.", exc)
    return Chroma(
        client=chroma_client,
        collection_name=kb.chroma_collection,
        embedding_function=embedding_function,
    )
