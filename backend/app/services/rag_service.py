import logging
import uuid
from typing import Any, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sqlalchemy.orm import Session
import torch

from ..config import settings
from ..models.models import KnowledgeBase, Personality
from .document_processor import process_file
from .graph_memory import GraphMemoryStore
from .sota_retrieval import contextualize_documents
from .vector_store_factory import get_vector_store

logger = logging.getLogger(__name__)
_chroma_client = None
_embeddings_cache: dict = {}


#  Chroma client 

def get_chroma_client() -> chromadb.HttpClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


#  Metadata sanitiser 
# ChromaDB 0.5.x ONLY allows str / int / float / bool metadata values.
# Keys starting with '_' (LangChain internals like _type, _id) are also dropped.

def _sanitize_metadata(meta: dict) -> dict:
    """Return a copy of meta containing only ChromaDB-safe primitive values."""
    safe = {}
    for k, v in meta.items():
        k = str(k)
        if k.startswith("_"):          # drop LangChain internal keys
            continue
        if isinstance(v, bool):
            safe[k] = v
        elif isinstance(v, int):
            safe[k] = v
        elif isinstance(v, float):
            safe[k] = v
        elif isinstance(v, str):
            safe[k] = v
        elif v is None:
            safe[k] = ""              # None not allowed  use empty string
        else:
            safe[k] = str(v)          # lists/dicts/objects â†’ stringify
    return safe


#  Embeddings (local sentence-transformers) 

def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    if model_name not in _embeddings_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _embeddings_cache[model_name] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            if device == "cuda":
                logger.warning(
                    "Embedding model '%s' failed on CUDA, falling back to CPU: %s",
                    model_name,
                    e,
                )
                _embeddings_cache[model_name] = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            else:
                raise
    return _embeddings_cache[model_name]


#  Local LLM via Ollama 

def get_llm(kb: KnowledgeBase) -> ChatOllama:
    return ChatOllama(
        model=kb.llm_model or settings.DEFAULT_LLM_MODEL,
        base_url=f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}",
        temperature=float(kb.temperature),
        num_predict=kb.max_tokens,
    )


#  Vector store 

def get_vectorstore(kb: KnowledgeBase):
    return get_vector_store(
        kb=kb,
        embedding_function=get_embeddings(kb.embedding_model),
        chroma_client=get_chroma_client(),
    )


#  MMR retriever 

def get_mmr_retriever(kb: KnowledgeBase, vectorstore: Any):
    """Maximal Marginal Relevance  balances relevance vs diversity."""
    top_k    = kb.top_k_docs or 4
    fetch_k  = max(kb.mmr_fetch_k or top_k * 4, top_k + 1)
    lmb      = float(kb.mmr_lambda or 0.7)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lmb},
    )


#  Personality resolver 

def resolve_system_prompt(kb: KnowledgeBase, db: Session) -> str:
    if kb.system_prompt and kb.system_prompt.strip():
        return kb.system_prompt
    if kb.personality_id:
        p = db.query(Personality).filter(Personality.id == kb.personality_id).first()
        if p:
            return p.system_prompt
    return (
        "You are a helpful AI assistant. Use the provided context to answer questions "
        "accurately and concisely. If the answer is not in the context, say so clearly."
    )


#  Document ingestion 

async def ingest_document(
    file_path: str,
    original_filename: str,
    kb: KnowledgeBase,
    metadata: Optional[dict] = None,
) -> int:
    """
    Route document to correct ingestion pipeline based on complexity,
    then embed summaries into ChromaDB while preserving raw content
    in metadata for retrieval-time LLM context (multi-vector strategy).
    """
    from .document_classifier import classify_document
    from .docling_parser       import parse_with_docling
    from .vlm_parser           import parse_with_vlm

    doc_type = classify_document(file_path)
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
    logger.info(f"[ingest] '{original_filename}' classified as: {doc_type}")

    #  Route to appropriate parser 
    if doc_type == "structured":
        if ext == "pdf":
            try:
                chunks = parse_with_docling(file_path, original_filename)
            except Exception as e:
                logger.warning(f"[ingest] Docling failed ({e}), falling back to standard")
                chunks = await _standard_chunks(file_path, original_filename, kb)
        else:
            chunks = await _standard_chunks(file_path, original_filename, kb)

    elif doc_type == "visual":
        try:
            chunks = parse_with_vlm(file_path, original_filename)
        except Exception as e:
            logger.warning(f"[ingest] VLM failed ({e}), falling back to Docling")
            try:
                chunks = parse_with_docling(file_path, original_filename)
            except Exception:
                chunks = await _standard_chunks(file_path, original_filename, kb)

    else:
        # Standard text pipeline  existing behaviour unchanged
        chunks = await _standard_chunks(file_path, original_filename, kb)

    if not chunks:
        raise ValueError(f"No content extracted from {original_filename}")

    if metadata:
        for chunk in chunks:
            base = chunk.metadata or {}
            chunk.metadata = _sanitize_metadata({**base, **metadata})
    else:
        for chunk in chunks:
            chunk.metadata = _sanitize_metadata(chunk.metadata or {})

    if settings.ENABLE_CONTEXTUAL_RETRIEVAL:
        chunks = contextualize_documents(chunks)

    vectorstore = get_vector_store(
        kb=kb,
        embedding_function=get_embeddings(kb.embedding_model),
        chroma_client=get_chroma_client(),
    )

    # page_content = summary (what gets embedded and searched)
    # metadata.raw = full content (what gets sent to the LLM)
    # ChromaDB stores both  retrieval returns summary+metadata together
    vectorstore.add_documents(chunks)
    if settings.ENABLE_GRAPH_RAG:
        GraphMemoryStore(settings.GRAPH_MEMORY_DIR).index_documents(kb_id=kb.id, docs=chunks)

    logger.info(f"[ingest] '{original_filename}': {len(chunks)} chunks â†’ ChromaDB")
    return len(chunks)


async def _standard_chunks(
    file_path: str,
    original_filename: str,
    kb: KnowledgeBase,
) -> List[Document]:
    """
    Standard fallback chunking pipeline.
    """
    return await process_file(
        file_path=file_path,
        original_filename=original_filename,
        metadata={
            "source": original_filename,
            "type": "text",
            "pipeline": "standard",
            "raw": "",
        },
    )

async def query_kb(
    kb: KnowledgeBase,
    question: str,
    chat_history: Optional[List[Tuple[str, str]]],
    db: Session,
) -> Tuple[str, List[dict]]:
    vectorstore   = get_vectorstore(kb)
    retriever     = get_mmr_retriever(kb, vectorstore)
    llm           = get_llm(kb)
    system_prompt = resolve_system_prompt(kb, db)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, k=5, output_key="answer"
    )
    if chat_history:
        for human, ai in chat_history[-5:]:
            memory.chat_memory.add_user_message(human)
            memory.chat_memory.add_ai_message(ai)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    result = chain({"question": f"[System: {system_prompt}]\n\n{question}"})
    answer = result["answer"]

    sources, seen = [], set()
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append({
                "source": src,
                "content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            })

    return answer, sources


#  Cleanup 

def delete_kb_collection(collection_name: str):
    try:
        get_chroma_client().delete_collection(collection_name)
    except Exception as e:
        logger.warning(f"Could not delete collection '{collection_name}': {e}")


def _delete_by_where(collection, where: dict) -> int:
    """Delete vectors matching a where clause and return removed count."""
    result = collection.get(where=where)
    ids = result.get("ids") or []
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)


def delete_document_vectors(
    kb: KnowledgeBase,
    *,
    doc_id: int,
    stored_filename: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> int:
    """
    Delete vectors for a specific document in a KB collection.
    Tries unique keys first (doc_id, stored_filename), then legacy source fallback.
    """
    try:
        collection = get_chroma_client().get_collection(name=kb.chroma_collection)
    except Exception as e:
        logger.warning(
            f"Could not open collection '{kb.chroma_collection}' for document delete: {e}"
        )
        return 0

    removed = 0

    # New ingests carry doc_id metadata (most reliable).
    removed += _delete_by_where(collection, {"doc_id": int(doc_id)})

    # Extra guard for new ingests.
    if stored_filename:
        removed += _delete_by_where(collection, {"stored_filename": str(stored_filename)})

    # Legacy fallback: best-effort for older chunks that only had source/kb_id metadata.
    if removed == 0 and original_filename:
        try:
            removed += _delete_by_where(
                collection,
                {"$and": [{"source": str(original_filename)}, {"kb_id": int(kb.id)}]},
            )
        except Exception:
            # Some older Chroma builds have limited $and handling in where filters.
            removed += _delete_by_where(collection, {"source": str(original_filename)})

    logger.info(
        f"Deleted {removed} vector chunks for doc_id={doc_id} from '{kb.chroma_collection}'"
    )
    return removed


