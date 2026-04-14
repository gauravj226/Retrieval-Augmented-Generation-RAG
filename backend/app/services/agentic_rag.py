"""
System 2 Agentic RAG using LangGraph + llama3.2:3b via Ollama.

Graph nodes (all pure functions: AgentState -> dict):
  route_query        -> decides retrieve / general / clarify / introspect
  retrieve           -> MMR + BM25 hybrid search with cross-encoder rerank
  grade_documents    -> LLM grades relevance (Stage 1) + sufficiency (Stage 2)
  rewrite_query      -> rewrites question if docs were poor or insufficient
  generate           -> generates answer from filtered docs
  check_hallucination-> grades generation vs source docs
  check_answer_quality -> grades answer vs original question + sources
  reflect            -> LLM reflects on why answer was poor
  direct_answer      -> answers general questions without retrieval
  clarify            -> asks user for clarification
  introspect         -> answers meta-queries about KB contents from manifest
  finalise           -> packages final answer with hallucination warning if needed

Conditional edges:
  route_query        -> retrieve | direct_answer | clarify | introspect
  grade_documents    -> generate | rewrite_query
  check_hallucination-> check_answer_quality | reflect
  check_answer_quality -> finalise | reflect
  reflect            -> rewrite_query (loop)
  rewrite_query      -> retrieve (loop, capped at max_rewrites)
"""
from __future__ import annotations

import logging
import re
import time
import hashlib
import threading
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session
import torch

from ..config import settings
from ..models.models import KnowledgeBase, Personality, InvoiceMetadata
from .agent_state import AgentState
from .rag_service import get_chroma_client, get_embeddings, query_invoices
from .graph_memory import GraphMemoryStore
from .long_term_memory import LongTermMemoryStore
from .semantic_cache import SemanticAnswerCache
from .bm25_index import HybridBM25Index
from .sota_retrieval import route_mode_for_query, score_sparse_hits
from .kb_manifest import load_manifest
from .vector_store_factory import get_vector_store
from .web_search import web_search
from .ui_payload import build_ui_payload

logger = logging.getLogger(__name__)

_cross_encoder_cache: dict = {}
_query_cache: Dict[str, Tuple[float, Tuple[str, List[dict], List[str], Optional[dict]]]] = {}
_query_cache_lock = threading.Lock()
_semantic_cache = SemanticAnswerCache(
    similarity_threshold=settings.SEMANTIC_CACHE_SIMILARITY,
    ttl_seconds=settings.SEMANTIC_CACHE_TTL_SEC,
    max_entries=settings.SEMANTIC_CACHE_MAX,
)

DEFAULT_MAX_REWRITES = 2
DEFAULT_MAX_RETRIES = 2


# ── Utility helpers ───────────────────────────────────────────────────────────

def _normalise_query(text: str) -> str:
    return " ".join((text or "").lower().split())


def _history_tail_hash(chat_history: Optional[List[Tuple[str, str]]]) -> str:
    if not chat_history:
        return "none"
    tail = chat_history[-2:]
    payload = "||".join(f"{h}::{a}" for h, a in tail)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _cache_key(
    kb: KnowledgeBase,
    question: str,
    chat_history: Optional[List[Tuple[str, str]]],
    fast_mode: bool,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
) -> str:
    mode = "fast" if fast_mode else "quality"
    # KB-global scope unless personal memory is active (cross-user cache hits)
    scope = f"u:{int(user_id)}" if (user_id is not None and settings.ENABLE_LONG_TERM_MEMORY) else "global"
    return f"{kb.id}:{mode}:{scope}:{_normalise_query(question)}:{_history_tail_hash(chat_history)}"


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm(
    kb: KnowledgeBase,
    temperature: float = 0.0,
    model_override: Optional[str] = None,
    num_predict_override: Optional[int] = None,
) -> ChatOllama:
    return ChatOllama(
        model=model_override or kb.llm_model or settings.DEFAULT_LLM_MODEL,
        base_url=f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}",
        temperature=temperature,
        num_predict=(num_predict_override if num_predict_override is not None else kb.max_tokens),
    )


def _grade_llm(kb: KnowledgeBase) -> ChatOllama:
    return _llm(kb, temperature=0.0, model_override=(getattr(settings, "GRADER_LLM_MODEL", None) or None))


def _gen_llm(kb: KnowledgeBase, *, max_tokens_override: Optional[int] = None) -> ChatOllama:
    return _llm(kb, temperature=float(kb.temperature or 0.4), num_predict_override=max_tokens_override)


def _parse_binary(text: str, positive: str, negative: str) -> str:
    """
    Safely parses a binary grader response. Checks negative patterns FIRST
    so 'not grounded' -> hallucinating, 'not_useful' -> not_useful.
    """
    cleaned = text.strip().lower()
    cleaned = "".join(c if c.isalnum() or c in ("_", " ") else " " for c in cleaned)
    tokens = cleaned.split()[-3:]
    joined = " ".join(tokens)
    neg_patterns = [f"not {positive}", f"not_{positive}", negative, f"not_{negative}"]
    for pat in neg_patterns:
        if pat in joined:
            return negative
    if positive in joined:
        return positive
    logger.warning("[_parse_binary] Unexpected: '%s' — defaulting to %s", text[:60], negative)
    return negative


# ── Query helpers ─────────────────────────────────────────────────────────────

def _expand_query(query: str) -> List[str]:
    q, ql, expanded = query.strip(), query.strip().lower(), [query.strip()]
    expansions = {
        "toil": "time off in lieu policy",
        "sar": "subject access request policy",
        "ppe": "personal protective equipment policy",
    }
    for short, full in expansions.items():
        if re.search(rf"\b{re.escape(short)}\b", ql):
            expanded.append(f"{q} ({full})")
            expanded.append(full)
    return expanded


def _query_terms(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())
        if t not in {"what", "which", "with", "from", "about", "how", "when", "where"}
    ]


def _anchor_terms(query: str) -> List[str]:
    return [t for t in _query_terms(query) if len(t) >= 5]


def _doc_has_anchor(doc: Document, anchors: List[str]) -> bool:
    if not anchors:
        return True
    corpus = " ".join([
        str((doc.metadata or {}).get("source", "")),
        str((doc.metadata or {}).get("raw", "")),
        str(getattr(doc, "page_content", "")),
    ]).lower()
    return any(a in corpus for a in anchors)


def _doc_overlap_score(query: str, doc: Document) -> int:
    terms = _query_terms(query)
    if not terms:
        return 0
    src = str((doc.metadata or {}).get("source", "")).lower()
    raw = str((doc.metadata or {}).get("raw", "")).lower()
    txt = str(getattr(doc, "page_content", "")).lower()
    corpus = f"{src}\n{raw}\n{txt}"
    return (sum(1 for t in terms if t in src) * 4) + sum(1 for t in terms if t in corpus)


def _rewrite_is_compatible(original: str, rewritten: str) -> bool:
    o_terms, r_terms = set(_query_terms(original)), set(_query_terms(rewritten))
    if not o_terms:
        return True
    if not r_terms:
        return False
    return len(o_terms.intersection(r_terms)) / max(1, len(o_terms)) >= 0.5


# ── Entity helpers ────────────────────────────────────────────────────────────

def _extract_focus_entities(text: str) -> dict:
    raw, lower = text or "", (text or "").lower()
    entities: dict = {"drive": set(), "windows": set(), "path": set(), "id": set(), "acronym": set()}
    entities["drive"].update(re.findall(r"\b([a-z])\s*:?\s*drive\b", lower))
    entities["windows"].update(re.findall(r"\bwindows\s*(10|11)\b", lower))
    entities["path"].update(re.findall(r"(\\\\[a-z0-9._$-]+\\[a-z0-9._$\\-]+)", lower))
    entities["id"].update(re.findall(r"\b(?:[a-z]{2,}[._-]?\d+[a-z0-9._-]*|\d+[a-z][a-z0-9._-]*)\b", lower))
    entities["acronym"].update([a.lower() for a in re.findall(r"\b[A-Z]{2,8}\b", raw)])
    return {k: v for k, v in entities.items() if v}


def _entity_alignment_score(query_entities: dict, doc_entities: dict, source: str) -> int:
    if not query_entities:
        return 0
    weights = {"drive": 24, "windows": 18, "path": 14, "id": 12, "acronym": 10}
    score, source_l = 0, (source or "").lower()
    for family, q_vals in query_entities.items():
        d_vals = doc_entities.get(family, set())
        if not d_vals:
            continue
        overlap = q_vals.intersection(d_vals)
        if overlap:
            score += weights.get(family, 8) * len(overlap)
            for val in overlap:
                if val and re.search(rf"\b{re.escape(val)}\b", source_l):
                    score += max(4, weights.get(family, 8) // 2)
        else:
            score -= max(8, int(weights.get(family, 8) * 0.7))
    return score


def _keyword_rerank(docs: List[Document], query: str, keep: int) -> List[Document]:
    terms = [t for t in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
             if t not in {"what", "which", "with", "from", "about", "policy"}]
    query_entities = _extract_focus_entities(query)
    scored = []
    for doc in docs:
        src = str(doc.metadata.get("source", "")).lower()
        raw = str(doc.metadata.get("raw", "")).lower()
        corpus = f"{doc.page_content.lower()}\n{raw}" if raw else doc.page_content.lower()
        doc_entities = _extract_focus_entities(f"{src}\n{corpus}")
        filename_hits = sum(1 for t in terms if t in src)
        content_hits = min(sum(corpus.count(t) for t in terms), 8)
        path_hint = 3 if any(t in {"path", "server", "drive"} for t in terms) and ("\\\\" in corpus or "net use " in corpus) else 0
        entity_hint = _entity_alignment_score(query_entities, doc_entities, src)
        score = (filename_hits * 5) + content_hits + path_hint + entity_hint
        try:
            doc.metadata["retrieval_score"] = float(score)
        except Exception:
            pass
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:keep]]


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _lexical_candidates(kb: KnowledgeBase, query: str, limit: int) -> List[Document]:
    terms = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
    if not terms:
        return []
    try:
        indexed = HybridBM25Index(settings.HYBRID_BM25_DIR).search(
            kb_id=int(kb.id), query=query, top_k=max(1, int(limit))
        )
        if not indexed:
            return []
        docs = []
        for hit in indexed:
            docs.append(Document(
                page_content=hit.content,
                metadata={**hit.metadata, "lexical_score": hit.score}
            ))
        return docs
    except Exception as e:
        logger.warning("bm25 index retrieval failed; falling back to chroma scan: %s", e)

    try:
        collection = get_chroma_client().get_collection(name=kb.chroma_collection)
        data = collection.get(
            limit=settings.HYBRID_LEXICAL_MAX_DOCS,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.warning("lexical retrieval skipped: %s", e)
        return []

    docs = data.get("documents") or []
    metas = data.get("metadatas") or [{} for _ in docs]
    warmup_docs: List[Document] = []
    scored = []
    for text, meta in zip(docs, metas):
        if not text:
            continue
        meta = meta or {}
        content = str(text).lower()
        raw = str(meta.get("raw", "")).lower()
        source = str(meta.get("source", "")).lower()
        corpus = f"{content}\n{raw}" if raw else content
        filename_hits = sum(1 for t in terms if t in source)
        content_hits = sum(corpus.count(t) for t in terms)
        score = (filename_hits * 6) + min(content_hits, 12)
        doc_obj = Document(page_content=str(text), metadata=meta)
        warmup_docs.append(doc_obj)
        if score > 0:
            scored.append((score, doc_obj))
    scored.sort(key=lambda x: x[0], reverse=True)
    try:
        if warmup_docs:
            HybridBM25Index(settings.HYBRID_BM25_DIR).upsert_chunks(kb_id=int(kb.id), docs=warmup_docs)
    except Exception:
        pass
    return [d for _, d in scored[:limit]]


def _cross_encoder_rerank(query: str, docs: List[Document], keep: int) -> List[Document]:
    if not docs or not settings.ENABLE_CROSS_ENCODER_RERANK:
        return docs[:keep]
    model_name = settings.CROSS_ENCODER_MODEL
    if model_name not in _cross_encoder_cache:
        try:
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _cross_encoder_cache[model_name] = CrossEncoder(model_name, device=device)
        except Exception as e:
            logger.warning("cross-encoder unavailable, skipping rerank: %s", e)
            return docs[:keep]
    ce = _cross_encoder_cache.get(model_name)
    if ce is None:
        return docs[:keep]
    try:
        pairs = [[query, d.page_content[:1400]] for d in docs]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
        return [d for _, d in ranked[:keep]]
    except Exception as e:
        logger.warning("cross-encoder rerank failed: %s", e)
        return docs[:keep]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_introspect(kb: KnowledgeBase):
    llm = _gen_llm(kb)

    async def introspect(state):
        question = state['question']
        manifest = load_manifest(kb.id)
        if not manifest:
            ans = 'I do not yet have a document catalogue. Try a specific question.'
            return {'generation': ans, 'final_answer': ans, 'sources': [], 'reasoning_trace': ['Introspect: manifest empty']}
        lines_out = []
        for e in manifest.values():
            name = e.get('display_name', e['filename'])
            hs = e.get('headings', [])
            lines_out.append(name + ((' covers: ' + ', '.join(hs[:4])) if hs else ''))
        prompt = (
            f'You are a helpful assistant for the {kb.department} department.\n'
            f'Based on the documents below, answer what topics you can help with.\n\n'
            f'Documents:\n'
            + '\n'.join(f'- {l}' for l in lines_out)
            + f'\n\nUser question: {question}\n\nAnswer:'
        )
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        ans = res.content.strip()
        return {'generation': ans, 'final_answer': ans, 'sources': [], 'reasoning_trace': [f'Introspect: {len(manifest)} documents']}

    return introspect


# Node 1 — route_query
def make_route_query(kb: KnowledgeBase):
    async def route_query(state: AgentState) -> dict:
        question = state["question"]
        mode_hint = route_mode_for_query(question)
        if mode_hint == "introspect":
            return {"route_decision": "introspect", "route_mode": "introspect", "reasoning_trace": ["Route: introspect"]}
        if bool(state.get("fast_mode")):
            q = question.strip().lower()
            if q in {"hi", "hello", "hey", "thanks", "thank you"}:
                decision = "general"
            elif len(q.split()) <= 1:
                decision = "clarify"
            else:
                decision = "retrieve"
            return {
                "route_decision": decision,
                "route_mode": mode_hint,
                "reasoning_trace": [f"Route decision: **{decision}** ({mode_hint})"],
            }
        llm = _grade_llm(kb)
        prompt = f"""You are a query router for a knowledge base in the '{kb.department}' department.
KB description: {kb.description or 'General purpose knowledge base'}
Classify the user question into exactly ONE of:
- "retrieve" → question is about content that would be in the knowledge base
- "general" → simple greeting, small talk, or completely off-topic
- "clarify" → too vague to answer without more details
Reply with ONLY the single word: retrieve, general, or clarify.
Question: {question}"""
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        decision = result.content.strip().lower()
        if decision not in ("retrieve", "general", "clarify"):
            decision = "retrieve"
        logger.info(f"[route_query] → {decision}")
        return {
            "route_decision": decision,
            "route_mode": mode_hint if decision == "retrieve" else decision,
            "reasoning_trace": [f"🔀 Route decision: **{decision}** ({mode_hint})"],
        }

    return route_query


# Node 2 — retrieve
def make_retrieve(kb: KnowledgeBase):
    vectorstore = get_vector_store(
        kb=kb,
        embedding_function=get_embeddings(kb.embedding_model),
        chroma_client=get_chroma_client(),
    )
    top_k = kb.top_k_docs or 4
    fetch_k = max(kb.mmr_fetch_k or top_k * 6, top_k + 2)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
            "lambda_mult": float(kb.mmr_lambda or 0.7),
            "score_threshold": float(kb.score_threshold or 0.35),
        },
    )
    fast_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k, "score_threshold": float(getattr(kb, "score_threshold", None) or 0.35)},
    )

    async def retrieve(state: AgentState) -> dict:
        fast_mode = bool(state.get("fast_mode"))
        query = state.get("rewritten_query") or state["question"]
        anchors = _anchor_terms(state.get("question") or query)
        queries = [query] if fast_mode else _expand_query(query)
        route_mode = state.get("route_mode", "retrieve")

        if settings.ENABLE_GRAPH_RAG and route_mode == "graph":
            graph_terms = GraphMemoryStore(settings.GRAPH_MEMORY_DIR).expand_query(kb_id=kb.id, query=query)
            queries.extend(graph_terms[:4])

        active_retriever = fast_retriever if fast_mode else retriever
        merged: List[Document] = []
        seen = set()
        for q in queries:
            docs = await active_retriever.ainvoke(q)
            for d in docs:
                key = (str(d.metadata.get("source", "")), d.page_content[:220])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)

        lexical = [] if fast_mode else _lexical_candidates(kb, query, limit=max(top_k * 3, 6))
        all_docs = []
        seen2 = set()
        for d in merged + lexical:
            key = (str(d.metadata.get("source", "")), d.page_content[:220])
            if key in seen2:
                continue
            seen2.add(key)
            all_docs.append(d)

        quality_keep = top_k if fast_mode else max(top_k * 2, 6)
        reranked = _keyword_rerank(all_docs, query, keep=quality_keep)
        anchored = [d for d in reranked if _doc_has_anchor(d, anchors)]
        if anchored:
            reranked = anchored[:quality_keep]

        query_entities = _extract_focus_entities(query)
        top_keyword_score = 0.0
        entity_match_count = 0
        for d in reranked[:5]:
            try:
                top_keyword_score = max(top_keyword_score, float(d.metadata.get("retrieval_score") or 0.0))
            except Exception:
                pass
            src = str(d.metadata.get("source", "")).lower()
            raw = str(d.metadata.get("raw", "")).lower()
            txt = str(getattr(d, "page_content", "")).lower()
            d_entities = _extract_focus_entities(f"{src}\n{raw}\n{txt}")
            if _entity_alignment_score(query_entities, d_entities, src) > 0:
                entity_match_count += 1

        if (not fast_mode) and route_mode == "sparse":
            sparse_hits = score_sparse_hits(query, reranked, top_k=quality_keep)
            reranked = [Document(page_content=h.content, metadata=h.metadata) for h in sparse_hits]

        if not fast_mode:
            reranked = _cross_encoder_rerank(query, reranked, keep=quality_keep)

        if not reranked:
            confidence, conf_reason = "low", "no_relevant_docs"
        elif query_entities and entity_match_count == 0:
            confidence, conf_reason = "low", "no_exact_entity_match"
        elif top_keyword_score < 10:
            confidence, conf_reason = "low", "low_rerank_score"
        elif top_keyword_score < 18:
            confidence, conf_reason = "medium", "moderate_rerank_score"
        else:
            confidence, conf_reason = "high", "strong_match"

        return {
            "documents": reranked,
            "retrieval_confidence": confidence,
            "retrieval_score": float(top_keyword_score),
            "reasoning_trace": [
                f"📚 Retrieved {len(reranked)} docs for: *{query[:60]}* [{route_mode}{', fast-lite' if fast_mode else ''}]",
                (f"📌 Anchor filter active: {', '.join(anchors[:4])}" if anchors else "📌 Anchor filter inactive"),
                f"📉 Retrieval confidence: **{confidence}** ({conf_reason}, score={top_keyword_score:.1f})",
            ],
        }

    return retrieve


# Node 3 — grade_documents  (Self-RAG Stage 1: Relevance + Stage 2: Sufficiency)
def make_grade_documents(kb: KnowledgeBase):
    """
    Self-RAG grading with two stages:
      Stage 1 — Relevance:   Is this chunk about the same topic as the query?   (ISREL)
      Stage 2 — Sufficiency: Can I extract the direct answer from this chunk?   (ISSUP)
    Docs are classified into: sufficient → partial → insufficient (dropped from preferred set).
    A rewrite_hint is generated when relevant docs exist but none are sufficient, so
    route_after_grading can trigger a targeted rewrite even when docs pass relevance.
    """
    llm = _grade_llm(kb)

    async def grade_documents(state: AgentState) -> dict:
        question = state.get("rewritten_query") or state["question"]
        documents = state["documents"][: max(1, int(settings.MAX_DOCS_FOR_GRADING))]

        grades, filtered = [], []
        sufficient_docs: List[Document] = []
        partial_docs: List[Document] = []
        heuristic_kept = 0

        for doc in documents:
            # ── Stage 1: Relevance ────────────────────────────────────────────
            relevance_prompt = f"""Grade whether this document is relevant to the question.
Department context: {kb.department} — {kb.description or 'General knowledge-base operations'}
Question: {question}
Source filename: {doc.metadata.get("source", "unknown")}
Document excerpt: {doc.page_content[:1200]}

Rules:
- Treat synonyms/paraphrases/domain-specific wording as relevant when intent matches.
- Filename hints are meaningful evidence (do not ignore filename-topic overlap).
- Terms like link/unlink/pair/connect/disconnect may be department-specific operations.
- Reply "yes" if this chunk could directly help answer the user's request, even if wording differs.
- Reply "no" only when clearly unrelated.
Reply ONLY: yes or no"""
            result = await llm.ainvoke([HumanMessage(content=relevance_prompt)])
            _rg = result.content.strip().lower()
            grade = (
                "yes" if _rg.startswith("yes")
                else "no" if _rg.startswith("no")
                else _parse_binary(result.content, "yes", "no")
            )
            grades.append(grade)

            heuristic_score = _doc_overlap_score(question, doc)
            heuristic_threshold = max(3, len(_query_terms(question)))

            if grade == "yes" or heuristic_score >= heuristic_threshold:
                filtered.append(doc)
                if grade != "yes" and heuristic_score >= heuristic_threshold:
                    heuristic_kept += 1

            # ── Stage 2: Sufficiency (ISSUP) — only for LLM-relevant docs ────
            if grade == "yes":
                suf_prompt = f"""Given this document chunk, can you extract a direct answer to the question?
Question: {question}
Chunk: {(doc.metadata.get("raw") or doc.page_content)[:1200]}

Reply ONLY one word:
- "sufficient"   — chunk contains a direct, complete answer
- "partial"      — chunk is relevant but only partially answers the question
- "insufficient" — chunk is relevant in topic but does not contain the actual answer

Reply:"""
                suf_result = await llm.ainvoke([HumanMessage(content=suf_prompt)])
                suf_raw = suf_result.content.strip().lower()

                # Normalise: "sufficient" must not be preceded by "in" (catches "insufficient")
                if re.search(r"\bsufficient\b", suf_raw) and not re.search(r"\binsufficient\b", suf_raw):
                    suf_label = "sufficient"
                elif re.search(r"\bpartial\b", suf_raw):
                    suf_label = "partial"
                else:
                    suf_label = "insufficient"

                doc.metadata["sufficiency"] = suf_label

                if suf_label == "sufficient":
                    sufficient_docs.append(doc)
                elif suf_label == "partial":
                    partial_docs.append(doc)
                # "insufficient": stays in filtered (for grounding) but not preferred

        # ── Fallback: LLM over-strict but retrieval found plausible docs ─────
        if not filtered and documents:
            retrieval_conf = str(state.get("retrieval_confidence") or "").lower()
            try:
                best_score = max(float((d.metadata or {}).get("retrieval_score") or 0.0) for d in documents)
            except Exception:
                best_score = 0.0
            if retrieval_conf == "high" or best_score >= 18.0:
                filtered = documents[: min(4, len(documents))]
                logger.info(
                    "[grade_documents] fallback keep=%s (conf=%s, score=%.1f)",
                    len(filtered), retrieval_conf, best_score,
                )

        has_sufficient = bool(sufficient_docs)

        # Build a targeted rewrite hint when relevant docs exist but none answer directly
        rewrite_hint = ""
        if filtered and not has_sufficient and partial_docs:
            rewrite_hint = (
                "previous results were relevant but did not contain the direct answer — "
                "search more specifically for the exact detail"
            )

        logger.info(
            "[grade_documents] %d/%d relevant | sufficient=%d partial=%d (heuristic_kept=%d)",
            len(filtered), len(documents), len(sufficient_docs), len(partial_docs), heuristic_kept,
        )

        return {
            "doc_grades": grades,
            "filtered_docs": filtered,
            "has_sufficient_docs": has_sufficient,
            "rewrite_hint": rewrite_hint,
            "reasoning_trace": [
                f"🔍 Grading: {len(filtered)}/{len(documents)} relevant"
                + (
                    f" | ✅ {len(sufficient_docs)} sufficient, ⚠️ {len(partial_docs)} partial"
                    if filtered else ""
                )
                + (f" (heuristic rescue: {heuristic_kept})" if heuristic_kept else "")
            ],
        }
    return grade_documents


# Node 4 — rewrite_query
def make_rewrite_query(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def rewrite_query(state: AgentState) -> dict:
        question = state["question"]
        rewrite_n = state.get("rewrite_count", 0)
        reflection = state.get("reflection", "")
        rewrite_hint = state.get("rewrite_hint", "")

        hint_lines = []
        if reflection:
            hint_lines.append(f"Previous reflection: {reflection}")
        if rewrite_hint:
            hint_lines.append(f"Hint: {rewrite_hint}")
        context = ("\n" + "\n".join(hint_lines)) if hint_lines else ""

        prompt = f"""Rewrite this question to improve retrieval quality.
Preserve core entities, abbreviations, and intent. Do NOT answer. Do NOT broaden topic. Keep to one short query (3-12 words).{context}
Do NOT reinterpret the task into a different domain.
Keep the user's concrete nouns from the original question whenever possible.
Original: {question}
Rewritten (reply with ONLY the rewritten question):"""

        result = await llm.ainvoke([HumanMessage(content=prompt)])
        rewritten = result.content.strip()
        if not _rewrite_is_compatible(question, rewritten):
            logger.info("[rewrite_query] rejected drift rewrite='%s' original='%s'", rewritten[:80], question[:80])
            rewritten = question

        logger.info(f"[rewrite_query] → '{rewritten[:60]}'")
        return {
            "rewritten_query": rewritten,
            "rewrite_count": rewrite_n + 1,
            "rewrite_hint": "",  # clear hint after use so it doesn't persist across iterations
            "reasoning_trace": [
                f"✏️ Rewrite {rewrite_n+1}: *{rewritten[:80]}*"
                + (f" (hint: {rewrite_hint[:60]})" if rewrite_hint else "")
            ],
        }

    return rewrite_query


# Node 5 — generate
def make_generate(kb: KnowledgeBase):
    async def generate(state: AgentState) -> dict:
        question = state["question"]
        docs = state.get("filtered_docs") or state.get("documents", [])
        docs = docs[: max(1, int(settings.MAX_DOCS_FOR_GENERATION))]
        fast_mode = bool(state.get("fast_mode"))
        memory_window = int(getattr(kb, "memory_window", 5) or 5)
        retrieval_confidence = (state.get("retrieval_confidence") or "medium").lower()
        system_prompt = state.get("system_prompt", "")
        chat_history = state.get("chat_history", [])
        gen_count = state.get("generation_count", 0)
        reflection = state.get("reflection")
        memory_context = state.get("memory_context", "")
        has_sufficient = state.get("has_sufficient_docs", True)

        if fast_mode:
            memory_context = ""

        fast_token_cap = int(getattr(settings, "FAST_MODE_MAX_TOKENS", 320) or 320)
        llm = _gen_llm(
            kb,
            max_tokens_override=min(int(kb.max_tokens or fast_token_cap), fast_token_cap) if fast_mode else None,
        )
        per_doc_chars = int(
            getattr(settings, "FAST_MODE_CONTEXT_CHARS_PER_DOC", 900)
            if fast_mode
            else getattr(settings, "QUALITY_MODE_CONTEXT_CHARS_PER_DOC", 2200)
        )

        context_parts = []
        for i, doc in enumerate(docs):
            raw = doc.metadata.get("raw", "").strip()
            content = (raw if raw else doc.page_content)[: max(400, per_doc_chars)]
            doc_type = doc.metadata.get("type", "text")
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            label = (
                f"[Source {i+1}: {source}"
                + (f", page {page}" if page else "")
                + (f", {doc_type}" if doc_type != "text" else "")
                + "]"
            )
            context_parts.append(f"{label}\n{content}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        history_str = ""
        if (not fast_mode) and chat_history:
            history_str = "\n".join(
                f"Human: {h}\nAssistant: {a}" for h, a in chat_history[-memory_window:]
            )
            history_str = f"\n\nConversation history:\n{history_str}"

        reflection_hint = (
            f"\n\nPrevious answer rejected — reason:\n{reflection}\nFix this."
            if reflection else ""
        )

        # ── Sufficiency hint injected into generation prompt ─────────────────
        sufficiency_note = ""
        if not has_sufficient and docs:
            sufficiency_note = (
                "\n\nNOTE: Retrieved documents are partially relevant. "
                "Only answer what is directly supported by the context provided. "
                "If the complete answer is not present, state what you found and "
                "explicitly indicate what information is missing."
            )

        grounding_guard = (
            "You are a precise AI assistant. "
            "Non-negotiable rule: answer using ONLY the provided context snippets. "
            "If only part of the answer is supported, provide the best available answer from context and clearly label any uncertainty. "
            "Ask a clarification question only when context has no actionable information at all. "
            "Do not use unstated prior knowledge. Preserve Markdown tables when present. "
            "Treat this as an internal enterprise documentation assistant; procedural IT/AV operations in provided docs are benign context."
        )
        merged_system = (
            f"{grounding_guard}\n\nAdditional behavior instructions:\n{system_prompt}"
            if system_prompt else grounding_guard
        )

        messages = [
            SystemMessage(content=merged_system),
            HumanMessage(
                content=(
                    f'Context:\n"""\n{context}\n"""\n\n'
                    f'{memory_context}{history_str}{reflection_hint}{sufficiency_note}'
                    f'\n\nQuestion: {question}\n\nAnswer:'
                )
            ),
        ]
        result = await llm.ainvoke(messages)
        generation = result.content.strip()

        if retrieval_confidence == "low":
            caution = "I found loosely related documents — please verify with the cited sources."
            if caution.lower() not in generation.lower():
                generation = f"{caution}\n\n{generation}"

        sources, seen = [], set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                pipeline = doc.metadata.get("pipeline", "standard")
                sources.append({
                    "source": src,
                    "pipeline": pipeline,
                    "type": doc.metadata.get("type", "text"),
                    "page": doc.metadata.get("page", ""),
                    "content": (doc.metadata.get("raw") or doc.page_content)[:200] + "…",
                    "sufficiency": doc.metadata.get("sufficiency", "n/a"),
                })

        return {
            "generation": generation,
            "generation_count": gen_count + 1,
            "sources": sources,
            "reasoning_trace": [
                f"💬 Generated answer via **{docs[0].metadata.get('pipeline','standard') if docs else 'standard'}** pipeline "
                f"(attempt {gen_count+1}, {len(generation)} chars)"
                + ("" if has_sufficient else " ⚠️ partial context")
            ],
        }

    return generate


# Node 6 — check_hallucination
def make_check_hallucination(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def check_hallucination(state: AgentState) -> dict:
        generation = state["generation"]
        docs = state.get("filtered_docs") or state.get("documents", [])
        context_parts = []
        for d in docs[:4]:
            raw = str(d.metadata.get("raw", "")).strip()
            context_parts.append((raw if raw else d.page_content)[:1200])
        context = "\n\n".join(context_parts)
        prompt = f"""Is this AI answer grounded in the source documents?
Sources: {context}
Answer: {generation}
Reply ONLY: grounded or hallucinating"""
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        rh = result.content.strip().lower()
        verdict = (
            "grounded" if rh.startswith("grounded")
            else "hallucinating" if rh.startswith("hallucinating")
            else _parse_binary(result.content, "grounded", "hallucinating")
        )
        logger.info(f"[check_hallucination] → {verdict}")
        return {
            "hallucination_check": verdict,
            "reasoning_trace": [f"{'✅' if verdict == 'grounded' else '⚠️'} Hallucination: **{verdict}**"],
        }
    return check_hallucination


# Node 7 — check_answer_quality
def make_check_answer_quality(kb: KnowledgeBase):
    def check_answer_quality(state: AgentState) -> dict:
        q = state["question"]
        gen = state["generation"]
        llm = _grade_llm(kb)
        
        prompt = (
            f"Does this answer fully address the question using the provided sources?\n"
            f"Department: {kb.department}\nQuestion: {question}\n"
            f"Sources:\n{_sp}\nAnswer: {generation}\n\n"
            f"Reply ONLY: useful or not_useful"
        )
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        quality = _parse_binary(result.content, "useful", "not_useful")
        logger.info(f"[check_answer_quality] → {quality}")
        return {
            "answer_quality": quality,
            "reasoning_trace": [f"{'✅' if quality == 'useful' else '🔄'} Quality: **{quality}**"],
        }
    return check_answer_quality


# Node 8 — reflect
def make_reflect(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def reflect(state: AgentState) -> dict:
        question = state["question"]
        generation = state["generation"]
        hcheck = state.get("hallucination_check")
        issue = (
            "Answer contained unsupported information (hallucination)."
            if hcheck == "hallucinating"
            else "Answer did not fully address the question."
        )
        prompt = f"""A RAG answer was rejected.
Issue: {issue}
Question: {question}
Rejected answer: {generation}
In 1-2 sentences, what was wrong and how should the approach change?"""
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        reflection = result.content.strip()
        logger.info(f"[reflect] {reflection[:80]}")
        return {
            "reflection": reflection,
            "generation": None,
            "hallucination_check": None,
            "answer_quality": None,
            "reasoning_trace": [f"🪞 Reflection: *{reflection[:120]}*"],
        }

    return reflect


# Node 9 — direct_answer
def make_direct_answer(kb: KnowledgeBase):
    llm = _gen_llm(kb)

    async def direct_answer(state: AgentState) -> dict:
        question = state["question"]
        system_prompt = state.get("system_prompt", "")
        chat_history = state.get("chat_history", [])
        memory_window = int(getattr(kb, "memory_window", 5) or 5)
        history_str = "\n".join(
            f"Human: {h}\nAssistant: {a}" for h, a in chat_history[-memory_window:]
        ) if chat_history else ""
        messages = [
            SystemMessage(content=system_prompt or "You are a helpful AI assistant."),
            HumanMessage(content=(f"{history_str}\n\n" if history_str else "") + question),
        ]
        result = await llm.ainvoke(messages)
        answer = result.content.strip()
        return {
            "generation": answer,
            "final_answer": answer,
            "sources": [],
            "reasoning_trace": ["💬 Answered directly (no retrieval needed)"],
        }

    return direct_answer


# Node 10 — clarify
def clarify(state: AgentState) -> dict:
    question = state["question"]
    clarification = (
        f"Your question *\"{question}\"* is a bit broad for me to search effectively. "
        "Could you provide more details? For example:\n"
        "- What specific aspect are you interested in?\n"
        "- What context or timeframe are you asking about?\n"
        "- What format would be most helpful (summary, steps, comparison)?"
    )
    return {
        "final_answer": clarification,
        "sources": [],
        "reasoning_trace": ["❓ Asked user for clarification (question too vague)"],
    }


# Node 11 — finalise
def finalise(state: AgentState) -> dict:
    answer = state.get("final_answer") or state.get("generation") or "I was unable to generate an answer."
    if state.get("hallucination_check") == "hallucinating":
        answer = (
            "⚠️ I could not verify this answer against the source documents. "
            "Please treat it with caution and check the original files.\n\n"
            + answer
        )
    return {"final_answer": answer}


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_routing(state: AgentState) -> str:
    return state.get("route_decision", "retrieve")


def route_after_grading(state: AgentState) -> str:
    """
    Extended Self-RAG routing:
      - No filtered docs                     → rewrite (existing behaviour)
      - Filtered docs but 0 sufficient docs  → targeted rewrite (NEW)
      - Filtered docs with sufficient docs   → generate
      - Max rewrites exhausted               → generate regardless
    """
    filtered = state.get("filtered_docs", [])
    max_rewrites = state.get("max_rewrites", DEFAULT_MAX_REWRITES)
    rewrite_count = state.get("rewrite_count", 0)

    if not filtered:
        if rewrite_count >= max_rewrites:
            logger.info("[route_after_grading] Max rewrites reached — generating with what we have")
            return "generate"
        return "rewrite"

    # NEW: relevant docs found but none contain the direct answer
    has_sufficient = state.get("has_sufficient_docs", True)
    rewrite_hint = state.get("rewrite_hint", "")
    if not has_sufficient and rewrite_hint and rewrite_count < max_rewrites:
        logger.info("[route_after_grading] Relevant but insufficient — targeted rewrite")
        return "rewrite"

    return "generate"


def route_after_hallucination(state: AgentState) -> str:
    verdict = state.get("hallucination_check", "grounded")
    if verdict == "hallucinating":
        if state.get("generation_count", 0) >= state.get("max_retries", DEFAULT_MAX_RETRIES):
            return "quality"
        return "reflect"
    return "quality"


def route_after_quality(state: AgentState) -> str:
    quality = state.get("answer_quality", "useful")
    if quality == "not_useful":
        conf = str(state.get("retrieval_confidence") or "").lower()
        if conf in {"high", "medium"} and state.get("rewrite_count", 0) >= 1:
            return "end"
        if state.get("rewrite_count", 0) >= state.get("max_rewrites", DEFAULT_MAX_REWRITES):
            return "end"
        return "reflect"
    return "end"


def route_after_reflect(state: AgentState) -> str:
    return "rewrite"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agentic_rag_graph(kb: KnowledgeBase, db: Session, fast_mode: bool = False) -> StateGraph:
    from .rag_service import resolve_system_prompt
    resolve_system_prompt(kb, db)  # pre-warm — result stored in initial_state by caller

    graph = StateGraph(AgentState)

    if fast_mode:
        graph.add_node("route_query",         make_route_query(kb))
        graph.add_node("retrieve",            make_retrieve(kb))
        graph.add_node("generate",            make_generate(kb))
        graph.add_node("check_hallucination", make_check_hallucination(kb))

        graph.add_edge(START, "route_query")
        graph.add_conditional_edges(
            "route_query", route_after_routing,
            {"retrieve": "retrieve", "general": "direct_answer", "clarify": "clarify", "introspect": "introspect"},
        )
        graph.add_edge("introspect", "finalise")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "check_hallucination")
        graph.add_edge("check_hallucination", "finalise")
        graph.add_edge("direct_answer", "finalise")
        graph.add_edge("clarify",       "finalise")
        graph.add_edge("finalise",      END)

    else:
        graph.add_node("route_query",          make_route_query(kb))
        graph.add_node("retrieve",             make_retrieve(kb))
        graph.add_node("grade_documents",      make_grade_documents(kb))
        graph.add_node("rewrite_query",        make_rewrite_query(kb))
        graph.add_node("generate",             make_generate(kb))
        graph.add_node("check_hallucination",  make_check_hallucination(kb))
        graph.add_node("check_answer_quality", make_check_answer_quality(kb))
        graph.add_node("reflect", make_reflect(kb))
        graph.add_node("direct_answer", make_direct_answer(kb))
        graph.add_node("clarify", clarify)
        graph.add_node("finalise", finalise)
        graph.add_node("introspect", make_introspect(kb))

        graph.add_edge(START, "route_query")
        graph.add_conditional_edges(
            "route_query", route_after_routing,
            {"retrieve": "retrieve", "general": "direct_answer", "clarify": "clarify", "introspect": "introspect"},
        )
        graph.add_edge("introspect", "finalise")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents", route_after_grading,
            {"generate": "generate", "rewrite": "rewrite_query"},
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate", "check_hallucination")
        graph.add_conditional_edges(
            "check_hallucination", route_after_hallucination,
            {"quality": "check_answer_quality", "reflect": "reflect"},
        )
        graph.add_conditional_edges(
            "check_answer_quality", route_after_quality,
            {"end": "finalise", "reflect": "reflect"},
        )
        graph.add_conditional_edges(
            "reflect", route_after_reflect,
            {"rewrite": "rewrite_query"},
        )
        graph.add_edge("direct_answer", "finalise")
        graph.add_edge("clarify", "finalise")
        graph.add_edge("finalise", END)
        return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────

async def run_agentic_rag(
    kb: KnowledgeBase,
    question: str,
    chat_history: Optional[List[Tuple[str, str]]],
    db: Session,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    fast_mode_override: Optional[bool] = None,
) -> Tuple[str, List[dict], List[str], Optional[dict]]:
    from .rag_service import resolve_system_prompt
    system_prompt = resolve_system_prompt(kb, db)

    effective_fast_mode = bool(settings.FAST_MODE) if fast_mode_override is None else bool(fast_mode_override)
    # KB-global scope unless personal memory is active
    cache_scope = f"u:{int(user_id)}" if (user_id is not None and settings.ENABLE_LONG_TERM_MEMORY) else "global"
    cache_key = _cache_key(kb, question, chat_history, effective_fast_mode, user_id=user_id, session_id=session_id)

    now = time.time()
    ttl = max(1, int(settings.QUERY_CACHE_TTL_SEC))
    with _query_cache_lock:
        hit = _query_cache.get(cache_key)
        if hit and (now - hit[0]) <= ttl:
            return hit[1]

    if settings.ENABLE_SEMANTIC_CACHE:
        sem_hit = _semantic_cache.get(
            query=question,
            kb_id=kb.id,
            mode="fast" if effective_fast_mode else "quality",
            scope=cache_scope,
        )
        if sem_hit:
            answer, sources, trace, score = sem_hit
            trace = list(trace) + [f"Semantic cache hit (score={score:.2f})"]
            return answer, sources, trace, build_ui_payload(question=question, answer=answer)

    memory_context = ""
    memory_store = None
    if (
        (not effective_fast_mode)
        and settings.ENABLE_LONG_TERM_MEMORY
        and user_id is not None
        and session_id is not None
    ):
        memory_store = LongTermMemoryStore(settings.MEMORY_STORE_DIR)
        profile = memory_store.load(user_id=user_id, session_id=session_id)
        memory_context = LongTermMemoryStore.to_prompt_context(profile)

    initial_state: AgentState = {
        "question":          question,
        "kb_id":             kb.id,
        "chat_history":      chat_history or [],
        "fast_mode":         effective_fast_mode,
        "route_decision":    "",
        "route_mode":        "retrieve",
        "documents":         [],
        "rewrite_count":     0,
        "rewritten_query":   None,
        "retrieval_confidence": None,
        "retrieval_score":   None,
        "doc_grades":        [],
        "filtered_docs":     [],
        "has_sufficient_docs": True,
        "rewrite_hint":      "",
        "generation":        None,
        "generation_count":  0,
        "hallucination_check": None,
        "answer_quality":    None,
        "reflection":        None,
        "final_answer":      None,
        "sources":           [],
        "reasoning_trace":   ["Fast mode enabled"] if effective_fast_mode else [],
        "system_prompt":     system_prompt,
        "memory_context":    memory_context,
        "user_id":           user_id,
        "session_id":        session_id,
        "max_rewrites":      DEFAULT_MAX_REWRITES,
        "max_retries":       DEFAULT_MAX_RETRIES,
    }

    try:
        app = build_agentic_rag_graph(kb, db, fast_mode=effective_fast_mode)
        final = await app.ainvoke(initial_state)
        answer = final.get("final_answer") or final.get("generation") or "Unable to answer."
        sources = final.get("sources", [])
        trace = final.get("reasoning_trace", [])

        # Web fallback
        if settings.ENABLE_WEB_FALLBACK and (not sources or "not in the context" in answer.lower()):
            web_sources = web_search(question, max_results=settings.WEB_SEARCH_MAX_RESULTS)
            if web_sources:
                snippets = "\n\n".join(
                    f"[{i + 1}] {row.get('title', 'Web Result')}\n{row.get('content', '')}"
                    for i, row in enumerate(web_sources[: settings.WEB_SEARCH_MAX_RESULTS])
                )
                llm = _gen_llm(kb)
                web_prompt = (
                    "Local knowledge base could not fully answer. "
                    "Use the web snippets below and cite uncertainty.\n\n"
                    f"{snippets}\n\nQuestion: {question}\n\nAnswer:"
                )
                web_answer = await llm.ainvoke([HumanMessage(content=web_prompt)])
                answer = (web_answer.content or "").strip() or answer
                sources = sources + web_sources
                trace = list(trace) + ["Web fallback used after weak local retrieval."]

        ui_payload = build_ui_payload(question=question, answer=answer)
        result = (answer, sources, trace, ui_payload)

        # Store in caches
        with _query_cache_lock:
            _query_cache[cache_key] = (now, result)
            if len(_query_cache) > int(settings.QUERY_CACHE_MAX):
                oldest_key = min(_query_cache, key=lambda k: _query_cache[k][0])
                _query_cache.pop(oldest_key, None)

        if settings.ENABLE_SEMANTIC_CACHE:
            _semantic_cache.set(
                query=question,
                kb_id=kb.id,
                mode="fast" if effective_fast_mode else "quality",
                scope=cache_scope,
                answer=answer,
                sources=sources,
                trace=trace,
            )

        # Update long-term memory
        if memory_store is not None:
            try:
                memory_store.update(
                    user_id=int(user_id),
                    session_id=int(session_id),
                    user_message=question,
                    assistant_message=answer,
                )
            except Exception as e:
                logger.warning("[run_agentic_rag] memory update failed: %s", e)

        return result

    except Exception as e:
        logger.exception("[run_agentic_rag] pipeline error: %s", e)
        return (
            "I encountered an error processing your request. Please try again.",
            [],
            [f"Pipeline error: {type(e).__name__}"],
            None,
        )
