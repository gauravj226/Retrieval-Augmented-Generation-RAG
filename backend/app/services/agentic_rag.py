""" System 2 Agentic RAG using LangGraph + llama3.2:3b via Ollama. Graph nodes (all pure functions: AgentState → dict): route_query → decides retrieve / general / clarify retrieve → MMR search against ChromaDB grade_documents → LLM grades each doc for relevance + sufficiency (Self-RAG) rewrite_query → rewrites question if docs were poor or insufficient generate → generates answer from filtered docs check_hallucination → grades generation vs source docs check_answer_quality → grades answer vs original question reflect → LLM reflects on why answer was poor direct_answer → answers general questions without retrieval clarify → asks user for clarification Conditional edges: after route_query → retrieve | direct_answer | clarify after grade_documents → generate | rewrite_query after generate → check_hallucination after check_hallucination→ check_answer_quality | reflect | generate after check_answer_quality → END | reflect after reflect → retrieve (loop) rewrite_query → retrieve (loop, capped at max_rewrites) """
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
    scope = f"u:{int(user_id)}" if (user_id is not None and settings.ENABLE_LONG_TERM_MEMORY) else "global"
    return f"{kb.id}:{mode}:{scope}:{_normalise_query(question)}:{_history_tail_hash(chat_history)}"


# ── LLM helpers ──────────────────────────────────────────────────────────────

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
    return _llm(
        kb,
        temperature=0.0,
        model_override=(getattr(settings, "GRADER_LLM_MODEL", None) or None),
    )


def _gen_llm(kb: KnowledgeBase, *, max_tokens_override: Optional[int] = None) -> ChatOllama:
    return _llm(
        kb,
        temperature=float(kb.temperature or 0.4),
        num_predict_override=max_tokens_override,
    )


def _parse_binary(text: str, positive: str, negative: str) -> str:
    cleaned = text.strip().lower()
    cleaned = ''.join(c if c.isalnum() or c in ('_', ' ') else ' ' for c in cleaned)
    tokens = cleaned.split()[-3:]
    joined = ' '.join(tokens)
    neg_patterns = [f"not {positive}", f"not_{positive}", negative, f"not_{negative}"]
    for pat in neg_patterns:
        if pat in joined:
            return negative
    if positive in joined:
        return positive
    logger.warning(f"[_parse_binary] Unexpected: '{text[:60]}' — defaulting to {negative}")
    return negative


def _expand_query(query: str) -> List[str]:
    q = query.strip()
    ql = q.lower()
    expanded = [q]
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
    anchors = []
    for t in _query_terms(query):
        if len(t) >= 5:
            anchors.append(t)
    return anchors


def _doc_has_anchor(doc: Document, anchors: List[str]) -> bool:
    if not anchors:
        return True
    src = str((doc.metadata or {}).get("source", "")).lower()
    raw = str((doc.metadata or {}).get("raw", "")).lower()
    txt = str(getattr(doc, "page_content", "")).lower()
    corpus = f"{src}\n{raw}\n{txt}"
    return any(a in corpus for a in anchors)


def _doc_overlap_score(query: str, doc: Document) -> int:
    terms = _query_terms(query)
    if not terms:
        return 0
    src = str((doc.metadata or {}).get("source", "")).lower()
    raw = str((doc.metadata or {}).get("raw", "")).lower()
    txt = str(getattr(doc, "page_content", "")).lower()
    corpus = f"{src}\n{raw}\n{txt}"
    filename_hits = sum(1 for t in terms if t in src)
    content_hits = sum(1 for t in terms if t in corpus)
    return (filename_hits * 4) + content_hits


def _rewrite_is_compatible(original: str, rewritten: str) -> bool:
    o_terms = set(_query_terms(original))
    r_terms = set(_query_terms(rewritten))
    if not o_terms:
        return True
    if not r_terms:
        return False
    overlap_ratio = len(o_terms.intersection(r_terms)) / max(1, len(o_terms))
    return overlap_ratio >= 0.5


def _extract_focus_entities(text: str) -> dict[str, set[str]]:
    raw = text or ""
    lower = raw.lower()
    entities: dict[str, set[str]] = {
        "drive": set(),
        "windows": set(),
        "path": set(),
        "id": set(),
        "acronym": set(),
    }
    entities["drive"].update(re.findall(r"\b([a-z])\s*:?\s*drive\b", lower))
    entities["windows"].update(re.findall(r"\bwindows\s*(10|11)\b", lower))
    entities["path"].update(re.findall(r"(\\\\[a-z0-9._$-]+\\[a-z0-9._$\\-]+)", lower))
    id_matches = re.findall(
        r"\b(?:[a-z]{2,}[._-]?\d+[a-z0-9._-]*|\d+[a-z][a-z0-9._-]*)\b",
        lower,
    )
    entities["id"].update(id_matches)
    entities["acronym"].update([a.lower() for a in re.findall(r"\b[A-Z]{2,8}\b", raw)])
    return {k: v for k, v in entities.items() if v}


def _entity_alignment_score(
    query_entities: dict[str, set[str]],
    doc_entities: dict[str, set[str]],
    source: str,
) -> int:
    if not query_entities:
        return 0
    weights = {
        "drive": 24,
        "windows": 18,
        "path": 14,
        "id": 12,
        "acronym": 10,
    }
    score = 0
    source_l = (source or "").lower()
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
    terms = [
        t for t in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
        if t not in {"what", "which", "with", "from", "about", "policy"}
    ]
    query_entities = _extract_focus_entities(query)
    scored = []
    for doc in docs:
        src = str(doc.metadata.get("source", "")).lower()
        text = doc.page_content.lower()
        raw = str(doc.metadata.get("raw", "")).lower()
        corpus = f"{text}\n{raw}" if raw else text
        doc_entities = _extract_focus_entities(f"{src}\n{corpus}")
        filename_hits = sum(1 for t in terms if t in src)
        content_hits = sum(corpus.count(t) for t in terms)
        path_hint = 0
        if any(t in {"path", "server", "drive"} for t in terms):
            if "\\\\" in corpus or "net use " in corpus:
                path_hint = 3
        entity_hint = _entity_alignment_score(query_entities, doc_entities, src)
        score = (filename_hits * 5) + min(content_hits, 8) + path_hint + entity_hint
        try:
            doc.metadata["retrieval_score"] = float(score)
        except Exception:
            pass
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:keep]]


def _lexical_candidates(kb: KnowledgeBase, query: str, limit: int) -> List[Document]:
    terms = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
    if not terms:
        return []
    try:
        indexed = HybridBM25Index(settings.HYBRID_BM25_DIR).search(
            kb_id=int(kb.id),
            query=query,
            top_k=max(1, int(limit)),
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
        logger.warning(f"Lexical search failed: {e}")
        return []


# ── Graph nodes ──────────────────────────────────────────────────────────────

def make_route_query(kb: KnowledgeBase):
    def route_query(state: AgentState) -> dict:
        q = state["question"]
        mode = route_mode_for_query(q)
        return {"route_mode": mode, "reasoning_trace": state["reasoning_trace"] + [f"Routing decision: {mode}"]}
    return route_query


def make_retrieve(kb: KnowledgeBase):
    def retrieve(state: AgentState) -> dict:
        q = state.get("rewritten_query") or state["question"]
        vectorstore = get_vector_store(
            kb=kb,
            embedding_function=get_embeddings(kb.embedding_model),
            chroma_client=get_chroma_client(),
        )
        
        top_k = kb.top_k_docs or 4
        fetch_k = max(kb.mmr_fetch_k or top_k * 4, top_k + 1)
        lmb = float(kb.mmr_lambda or 0.7)
        score_threshold = float(kb.score_threshold or 0.35)
        
        docs = vectorstore.max_marginal_relevance_search(
            q,
            k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lmb,
            score_threshold=score_threshold,
        )
        
        if state.get("route_mode") == "sparse" or not docs:
            lexical = _lexical_candidates(kb, q, top_k)
            if lexical:
                existing_sources = {d.metadata.get("source") for d in docs}
                for l_doc in lexical:
                    if l_doc.metadata.get("source") not in existing_sources:
                        docs.append(l_doc)
        
        docs = _keyword_rerank(docs, q, top_k)
        return {"documents": docs, "reasoning_trace": state["reasoning_trace"] + [f"Retrieved {len(docs)} docs"]}
    return retrieve


def make_grade_documents(kb: KnowledgeBase):
    def grade_documents(state: AgentState) -> dict:
        q = state["question"]
        docs = state["documents"]
        llm = _grade_llm(kb)
        
        grades = []
        filtered = []
        for doc in docs:
            prompt = (
                f"Question: {q}\nDocument: {doc.page_content}\n"
                "Is this document relevant? Answer 'yes' or 'no'."
            )
            res = llm.invoke(prompt)
            verdict = _parse_binary(res.content, "yes", "no")
            grades.append(verdict)
            if verdict == "yes":
                filtered.append(doc)
        
        return {
            "doc_grades": grades,
            "filtered_docs": filtered,
            "has_sufficient_docs": len(filtered) > 0,
            "reasoning_trace": state["reasoning_trace"] + [f"Grader: {len(filtered)}/{len(docs)} relevant"]
        }
    return grade_documents


def make_rewrite_query(kb: KnowledgeBase):
    def rewrite_query(state: AgentState) -> dict:
        q = state["question"]
        llm = _gen_llm(kb)
        prompt = f"Rewrite this search query to be more effective for a vector database: {q}"
        res = llm.invoke(prompt)
        new_q = res.content.strip()
        return {
            "rewritten_query": new_q,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "reasoning_trace": state["reasoning_trace"] + [f"Rewrote query to: {new_q}"]
        }
    return rewrite_query


def make_generate(kb: KnowledgeBase):
    def generate(state: AgentState) -> dict:
        q = state["question"]
        docs = state.get("filtered_docs") or state.get("documents") or []
        llm = _gen_llm(kb)
        
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\n"
            "Answer the question based on the context."
        )
        res = llm.invoke(prompt)
        return {
            "generation": res.content,
            "generation_count": state.get("generation_count", 0) + 1,
            "reasoning_trace": state["reasoning_trace"] + ["Generated answer"]
        }
    return generate


def make_check_hallucination(kb: KnowledgeBase):
    def check_hallucination(state: AgentState) -> dict:
        gen = state["generation"]
        docs = state.get("filtered_docs") or state.get("documents") or []
        llm = _grade_llm(kb)
        
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            f"Answer: {gen}\nContext: {context}\n"
            "Is the answer grounded in the context? Answer 'grounded' or 'hallucinating'."
        )
        res = llm.invoke(prompt)
        verdict = _parse_binary(res.content, "grounded", "hallucinating")
        return {
            "hallucination_check": verdict,
            "reasoning_trace": state["reasoning_trace"] + [f"Hallucination check: {verdict}"]
        }
    return check_hallucination


def make_check_answer_quality(kb: KnowledgeBase):
    def check_answer_quality(state: AgentState) -> dict:
        q = state["question"]
        gen = state["generation"]
        llm = _grade_llm(kb)
        
        prompt = (
            f"Question: {q}\nAnswer: {gen}\n"
            "Is the answer useful? Answer 'useful' or 'not_useful'."
        )
        res = llm.invoke(prompt)
        verdict = _parse_binary(res.content, "useful", "not_useful")
        return {
            "answer_quality": verdict,
            "reasoning_trace": state["reasoning_trace"] + [f"Quality check: {verdict}"]
        }
    return check_answer_quality


def make_reflect(kb: KnowledgeBase):
    def reflect(state: AgentState) -> dict:
        q = state["question"]
        gen = state.get("generation", "")
        llm = _gen_llm(kb)
        prompt = f"The previous answer was poor. Why? Question: {q}\nAnswer: {gen}\nReflect and suggest a better approach."
        res = llm.invoke(prompt)
        return {"reflection": res.content, "reasoning_trace": state["reasoning_trace"] + ["Reflected on failure"]}
    return reflect


def make_direct_answer(kb: KnowledgeBase):
    def direct_answer(state: AgentState) -> dict:
        q = state["question"]
        llm = _gen_llm(kb)
        res = llm.invoke(q)
        return {"generation": res.content, "reasoning_trace": state["reasoning_trace"] + ["Direct answer generated"]}
    return direct_answer


def clarify(state: AgentState) -> dict:
    return {"generation": "Could you please provide more details?", "reasoning_trace": state["reasoning_trace"] + ["Requested clarification"]}


def make_introspect(kb: KnowledgeBase):
    def introspect(state: AgentState) -> dict:
        manifest = load_manifest(kb.id)
        if not manifest:
            return {"generation": "I don't have any documents indexed yet.", "reasoning_trace": state["reasoning_trace"] + ["Introspect: no docs"]}
        
        topics = []
        for doc in manifest.values():
            topics.extend(doc.get("headings", []))
        
        topic_str = ", ".join(set(topics[:15]))
        return {"generation": f"I can help with: {topic_str}", "reasoning_trace": state["reasoning_trace"] + ["Introspect successful"]}
    return introspect


def finalise(state: AgentState) -> dict:
    gen = state["generation"]
    if state.get("hallucination_check") == "hallucinating":
        gen = "⚠️ " + gen
    
    docs = state.get("filtered_docs") or state.get("documents") or []
    sources = []
    for d in docs:
        sources.append({"source": d.metadata.get("source"), "content": d.page_content})
    
    return {"final_answer": gen, "sources": sources}


# Gap 6: No SQL route for aggregation queries
def make_sql_query(kb: KnowledgeBase):
    async def sql_query(state: AgentState) -> dict:
        from ..database import SessionLocal
        db = SessionLocal()
        try:
            invoices = query_invoices(db, kb_id=int(kb.id))
            if not invoices:
                return {"generation": "No invoice data found to aggregate.", "reasoning_trace": state["reasoning_trace"] + ["SQL: No invoices"]}
            
            # Simple aggregation logic or LLM-based summarization of the list
            q = state["question"]
            llm = _gen_llm(kb)
            data_str = json.dumps([
                {"vendor": i.vendor_name, "amount": float(i.total_amount or 0), "date": str(i.invoice_date)}
                for i in invoices
            ])
            prompt = (
                f"Question: {q}\n"
                f"Data: {data_str}\n\n"
                "Answer the question using the provided invoice data."
            )
            res = await llm.ainvoke(prompt)
            return {"generation": res.content, "reasoning_trace": state["reasoning_trace"] + [f"SQL query over {len(invoices)} invoices"]}
        finally:
            db.close()
    return sql_query


# ── Routing logic ────────────────────────────────────────────────────────────

def route_after_routing(state: AgentState) -> str:
    return state.get("route_mode", "retrieve")


def route_after_grading(state: AgentState) -> str:
    if state.get("has_sufficient_docs"):
        return "generate"
    return "rewrite"


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
    system_prompt = resolve_system_prompt(kb, db)

    graph = StateGraph(AgentState)

    if fast_mode:
        graph.add_node("route_query", make_route_query(kb))
        graph.add_node("retrieve", make_retrieve(kb))
        graph.add_node("generate", make_generate(kb))
        graph.add_node("direct_answer", make_direct_answer(kb))
        graph.add_node("clarify", clarify)
        graph.add_node("finalise", finalise)
        graph.add_node("introspect", make_introspect(kb))
        graph.add_node("check_hallucination", make_check_hallucination(kb))
        graph.add_node("sql_query", make_sql_query(kb))

        graph.add_edge(START, "route_query")
        graph.add_conditional_edges(
            "route_query", route_after_routing,
            {
                "retrieve": "retrieve", 
                "general": "direct_answer", 
                "clarify": "clarify", 
                "introspect": "introspect",
                "sql": "sql_query",
                "sparse": "retrieve",
                "graph": "retrieve"
            },
        )
        graph.add_edge("introspect", "finalise")
        graph.add_edge("sql_query", "finalise")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "check_hallucination")
        
        # Gap 4: Replace direct edge with conditional routing
        graph.add_conditional_edges(
            "check_hallucination",
            route_after_hallucination,
            {"quality": "finalise", "reflect": "finalise"},  # fast: no loop, just flag
        )
        
        graph.add_edge("direct_answer", "finalise")
        graph.add_edge("clarify", "finalise")
        graph.add_edge("finalise", END)
        return graph.compile()

    else:
        graph.add_node("route_query", make_route_query(kb))
        graph.add_node("retrieve", make_retrieve(kb))
        graph.add_node("grade_documents", make_grade_documents(kb))
        graph.add_node("rewrite_query", make_rewrite_query(kb))
        graph.add_node("generate", make_generate(kb))
        graph.add_node("check_hallucination", make_check_hallucination(kb))
        graph.add_node("check_answer_quality", make_check_answer_quality(kb))
        graph.add_node("reflect", make_reflect(kb))
        graph.add_node("direct_answer", make_direct_answer(kb))
        graph.add_node("clarify", clarify)
        graph.add_node("finalise", finalise)
        graph.add_node("introspect", make_introspect(kb))
        graph.add_node("sql_query", make_sql_query(kb))

        graph.add_edge(START, "route_query")
        graph.add_conditional_edges(
            "route_query", route_after_routing,
            {
                "retrieve": "retrieve", 
                "general": "direct_answer", 
                "clarify": "clarify", 
                "introspect": "introspect",
                "sql": "sql_query",
                "sparse": "retrieve",
                "graph": "retrieve"
            },
        )
        graph.add_edge("introspect", "finalise")
        graph.add_edge("sql_query", "finalise")
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
        "question": question,
        "kb_id": kb.id,
        "chat_history": chat_history or [],
        "fast_mode": effective_fast_mode,
        "route_decision": "",
        "route_mode": "retrieve",
        "documents": [],
        "rewrite_count": 0,
        "rewritten_query": None,
        "retrieval_confidence": None,
        "retrieval_score": None,
        "doc_grades": [],
        "filtered_docs": [],
        "has_sufficient_docs": True,
        "rewrite_hint": "",
        "generation": None,
        "generation_count": 0,
        "hallucination_check": None,
        "answer_quality": None,
        "reflection": None,
        "final_answer": None,
        "sources": [],
        "reasoning_trace": ["Fast mode enabled"] if effective_fast_mode else [],
        "system_prompt": system_prompt,
        "memory_context": memory_context,
        "user_id": user_id,
        "session_id": session_id,
        "max_rewrites": DEFAULT_MAX_REWRITES,
        "max_retries": DEFAULT_MAX_RETRIES,
    }

    try:
        app = build_agentic_rag_graph(kb, db, fast_mode=effective_fast_mode)
        final = await app.ainvoke(initial_state)
        answer = final.get("final_answer") or final.get("generation") or "Unable to answer."
        sources = final.get("sources", [])
        trace = final.get("reasoning_trace", [])

        if settings.ENABLE_WEB_FALLBACK and (not sources or "not in the" in answer.lower()):
            web_results = await web_search(question)
            if web_results:
                answer += f"\n\n---\n**Web results:**\n{web_results}"
                trace.append("🌐 Web fallback triggered")

        ui_payload = build_ui_payload(question=question, answer=answer)

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

        result = (answer, sources, trace, ui_payload)
        with _query_cache_lock:
            _query_cache[cache_key] = (now, result)

        if memory_store is not None:
            try:
                memory_store.update(
                    user_id=user_id,
                    session_id=session_id,
                    question=question,
                    answer=answer,
                )
            except Exception as e:
                logger.warning("[run_agentic_rag] memory update failed: %s", e)

        return result

    except Exception as e:
        logger.exception("[run_agentic_rag] pipeline error: %s", e)
        return (
            "I encountered an error processing your request. Please try again.",
            [],
            [f"❌ Pipeline error: {type(e).__name__}"],
            None,
        )
