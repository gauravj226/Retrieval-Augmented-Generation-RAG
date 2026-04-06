"""
System 2 Agentic RAG using LangGraph + llama3.2:3b via Ollama.

Graph nodes (all pure functions: AgentState → dict):
  route_query      → decides retrieve / general / clarify
  retrieve         → MMR search against ChromaDB
  grade_documents  → LLM grades each doc for relevance
  rewrite_query    → rewrites question if docs were poor
  generate         → generates answer from filtered docs
  check_hallucination → grades generation vs source docs
  check_answer_quality → grades answer vs original question
  reflect          → LLM reflects on why answer was poor
  direct_answer    → answers general questions without retrieval
  clarify          → asks user for clarification

Conditional edges:
  after route_query        → retrieve | direct_answer | clarify
  after grade_documents    → generate | rewrite_query
  after generate           → check_hallucination
  after check_hallucination→ check_answer_quality | reflect | generate
  after check_answer_quality → END | reflect
  after reflect            → retrieve (loop)
  rewrite_query            → retrieve (loop, capped at max_rewrites)
"""
from __future__ import annotations

import logging
import re
import time
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session
import torch

from ..config import settings
from ..models.models import KnowledgeBase, Personality
from .agent_state import AgentState
from .rag_service import get_chroma_client, get_embeddings
from .graph_memory import GraphMemoryStore
from .long_term_memory import LongTermMemoryStore
from .semantic_cache import SemanticAnswerCache
from .sota_retrieval import route_mode_for_query, score_sparse_hits
from .vector_store_factory import get_vector_store
from .web_search import web_search

logger = logging.getLogger(__name__)
_cross_encoder_cache: dict = {}
_query_cache: Dict[str, Tuple[float, Tuple[str, List[dict], List[str]]]] = {}
_query_cache_lock = threading.Lock()
_semantic_cache = SemanticAnswerCache(
    similarity_threshold=settings.SEMANTIC_CACHE_SIMILARITY,
    ttl_seconds=settings.SEMANTIC_CACHE_TTL_SEC,
    max_entries=settings.SEMANTIC_CACHE_MAX,
)
DEFAULT_MAX_REWRITES = 2
DEFAULT_MAX_RETRIES  = 2


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
) -> str:
    mode = "fast" if fast_mode else "quality"
    return f"{kb.id}:{mode}:{_normalise_query(question)}:{_history_tail_hash(chat_history)}"
# Helper: build LLM
def _llm(kb: KnowledgeBase, temperature: float = 0.0) -> ChatOllama:
    """Return a ChatOllama instance. temperature=0 for graders/routers."""
    return ChatOllama(
        model=kb.llm_model or settings.DEFAULT_LLM_MODEL,
        base_url=f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}",
        temperature=temperature,
        num_predict=kb.max_tokens,
    )


def _grade_llm(kb: KnowledgeBase) -> ChatOllama:
    """Deterministic LLM for binary graders."""
    return _llm(kb, temperature=0.0)


def _gen_llm(kb: KnowledgeBase) -> ChatOllama:
    """Creative LLM for generation."""
    return _llm(kb, temperature=float(kb.temperature or 0.4))


def _parse_binary(text: str, positive: str, negative: str) -> str:
    """
    Safely parses a binary grader response.
    Checks for negative patterns FIRST so 'not grounded' → hallucinating
    and 'not_useful' → not_useful — not the wrong positive.
    """
    cleaned = text.strip().lower()
    cleaned = ''.join(c if c.isalnum() or c in ('_', ' ') else ' ' for c in cleaned)
    tokens  = cleaned.split()[-3:]   # only last 3 words — model adds filler
    joined  = ' '.join(tokens)

    neg_patterns = [f"not {positive}", f"not_{positive}", negative, f"not_{negative}"]
    for pat in neg_patterns:
        if pat in joined:
            return negative

    if positive in joined:
        return positive

    logger.warning(f"[_parse_binary] Unexpected: '{text[:60]}' — defaulting to {negative}")
    return negative   # fail safe


def _expand_query(query: str) -> List[str]:
    """
    Add targeted expansions for common enterprise acronyms/phrases so
    dense retrieval has stronger semantic anchors.
    """
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


def _keyword_rerank(docs: List[Document], query: str, keep: int) -> List[Document]:
    """
    Lightweight lexical reranker on top of vector retrieval:
    - boosts filename term overlap heavily
    - boosts content term overlap
    """
    terms = [
        t for t in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
        if t not in {"what", "which", "with", "from", "about", "policy"}
    ]
    if not terms:
        return docs[:keep]

    scored = []
    for doc in docs:
        src = str(doc.metadata.get("source", "")).lower()
        text = doc.page_content.lower()
        raw = str(doc.metadata.get("raw", "")).lower()
        corpus = f"{text}\n{raw}" if raw else text

        filename_hits = sum(1 for t in terms if t in src)
        content_hits = sum(corpus.count(t) for t in terms)

        # Boost explicit network-path evidence for "path"/"server" questions.
        path_hint = 0
        if any(t in {"path", "server", "drive"} for t in terms):
            if "\\\\" in corpus or "net use " in corpus:
                path_hint = 3

        score = (filename_hits * 5) + min(content_hits, 8) + path_hint
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:keep]]


def _lexical_candidates(kb: KnowledgeBase, query: str, limit: int) -> List[Document]:
    """
    Sparse lexical retrieval over existing chunks (hybrid with dense retriever).
    Pulls a bounded corpus snapshot from Chroma and scores by keyword overlap.
    """
    terms = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
    if not terms:
        return []

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
        if score > 0:
            scored.append((score, Document(page_content=str(text), metadata=meta)))

    scored.sort(key=lambda x: x[0], reverse=True)
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
# Node 1 — route_query
def make_route_query(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def route_query(state: AgentState) -> dict:
        question = state["question"]
        mode_hint = route_mode_for_query(question)
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

        prompt = f"""You are a query router for a knowledge base in the '{kb.department}' department.
KB description: {kb.description or 'General purpose knowledge base'}

Classify the user question into exactly ONE of:
- "retrieve"  → question is about content that would be in the knowledge base
- "general"   → simple greeting, small talk, or completely off-topic
- "clarify"   → too vague to answer without more details

Reply with ONLY the single word: retrieve, general, or clarify.

Question: {question}"""

        result   = await llm.ainvoke([HumanMessage(content=prompt)])
        decision = result.content.strip().lower()
        if decision not in ("retrieve", "general", "clarify"):
            decision = "retrieve"
        logger.info(f"[route_query] → {decision}")
        return {
            "route_decision":   decision,
            "route_mode":       mode_hint if decision == "retrieve" else decision,
            "reasoning_trace":  [f"🔀 Route decision: **{decision}** ({mode_hint})"],
        }
    return route_query


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
            "k":           top_k,
            "fetch_k":     fetch_k,
            "lambda_mult": float(kb.mmr_lambda or 0.7),
        },
    )

    async def retrieve(state: AgentState) -> dict:
        query = state.get("rewritten_query") or state["question"]
        queries = [query] if bool(state.get("fast_mode")) else _expand_query(query)
        route_mode = state.get("route_mode", "retrieve")
        if settings.ENABLE_GRAPH_RAG and route_mode == "graph":
            graph_terms = GraphMemoryStore(settings.GRAPH_MEMORY_DIR).expand_query(kb_id=kb.id, query=query)
            queries.extend(graph_terms[:4])

        merged: List[Document] = []
        seen = set()
        for q in queries:
            docs = await retriever.ainvoke(q)
            for d in docs:
                key = (
                    str(d.metadata.get("source", "")),
                    d.page_content[:220],
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)

        lexical = [] if bool(state.get("fast_mode")) else _lexical_candidates(kb, query, limit=max(top_k * 3, 6))

        all_docs = []
        seen2 = set()
        for d in merged + lexical:
            key = (
                str(d.metadata.get("source", "")),
                d.page_content[:220],
            )
            if key in seen2:
                continue
            seen2.add(key)
            all_docs.append(d)

        reranked = _keyword_rerank(all_docs, query, keep=max(top_k * 4, 10))
        if route_mode == "sparse":
            sparse_hits = score_sparse_hits(query, reranked, top_k=max(top_k * 3, 8))
            reranked = [Document(page_content=h.content, metadata=h.metadata) for h in sparse_hits]
        reranked = _cross_encoder_rerank(query, reranked, keep=max(top_k * 3, 8))
        return {
            "documents":        reranked,
            "reasoning_trace":  [f"📚 Retrieved {len(reranked)} docs for: *{query[:60]}* [{route_mode}]"],
        }
    return retrieve


def make_grade_documents(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def grade_documents(state: AgentState) -> dict:
        question  = state.get("rewritten_query") or state["question"]
        documents = state["documents"]
        grades, filtered = [], []

        for doc in documents:
            prompt = f"""Grade whether this document is relevant to the question.

Question: {question}
Source filename: {doc.metadata.get("source", "unknown")}
Document excerpt: {doc.page_content[:1200]}

Rules:
- Reply "yes" only if this document likely contains direct policy details for the question topic.
- Reply "no" for generic policy text that is not specifically about the asked topic.

Reply ONLY: yes or no"""
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            grade = _parse_binary(result.content, "yes", "no")
            grades.append(grade)
            if grade == "yes":
                filtered.append(doc)

        logger.info(f"[grade_documents] {len(filtered)}/{len(documents)} relevant")
        return {
            "doc_grades":       grades,
            "filtered_docs":    filtered,
            "reasoning_trace":  [f"🔍 Grading: {len(filtered)}/{len(documents)} relevant"],
        }
    return grade_documents


def make_rewrite_query(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def rewrite_query(state: AgentState) -> dict:
        question   = state["question"]
        rewrite_n  = state.get("rewrite_count", 0)
        reflection = state.get("reflection", "")
        context    = f"\nPrevious reflection: {reflection}" if reflection else ""

        prompt = f"""Rewrite this question to improve retrieval quality.
Preserve core entities, abbreviations, and intent.
Do NOT answer. Do NOT broaden topic. Keep to one short query (3-12 words).{context}

Original: {question}
Rewritten (reply with ONLY the rewritten question):"""

        result    = await llm.ainvoke([HumanMessage(content=prompt)])
        rewritten = result.content.strip()
        logger.info(f"[rewrite_query] → '{rewritten[:60]}'")
        return {
            "rewritten_query":  rewritten,
            "rewrite_count":    rewrite_n + 1,
            "reasoning_trace":  [f"✏️ Rewrite {rewrite_n+1}: *{rewritten[:80]}*"],
        }
    return rewrite_query

def make_generate(kb: KnowledgeBase):
    llm = _gen_llm(kb)

    async def generate(state: AgentState) -> dict:
        question      = state["question"]
        docs          = state.get("filtered_docs") or state.get("documents", [])
        system_prompt = state.get("system_prompt", "")
        chat_history  = state.get("chat_history", [])
        gen_count     = state.get("generation_count", 0)
        reflection    = state.get("reflection")
        memory_context = state.get("memory_context", "")
        # Docling/VLM chunks embed a summary but store full content in raw.
        # Standard chunks have no raw (page_content is already the full text).
        context_parts = []
        for i, doc in enumerate(docs):
            raw      = doc.metadata.get("raw", "").strip()
            content  = raw if raw else doc.page_content
            doc_type = doc.metadata.get("type", "text")
            source   = doc.metadata.get("source", "unknown")
            page     = doc.metadata.get("page", "")
            label    = f"[Source {i+1}: {source}" + (f", page {page}" if page else "") + \
                       (f", {doc_type}" if doc_type != "text" else "") + "]"
            context_parts.append(f"{label}\n{content}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        
        history_str = ""
        if chat_history:
            history_str = "\n".join(
                f"Human: {h}\nAssistant: {a}" for h, a in chat_history[-3:]
            )
            history_str = f"\n\nConversation history:\n{history_str}"

        reflection_hint = (
            f"\n\nPrevious answer rejected — reason:\n{reflection}\nFix this."
            if reflection else ""
        )

        messages = [
            SystemMessage(content=system_prompt or
                "You are a precise AI assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say so clearly. "
                "When context contains Markdown tables, preserve them in your answer."),
            HumanMessage(content=f'Context:\n"""\n{context}\n"""\n\n{memory_context}{history_str}{reflection_hint}\n\nQuestion: {question}\n\nAnswer:'),
        ]

        result     = await llm.ainvoke(messages)
        generation = result.content.strip()

        sources, seen = [], set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                pipeline = doc.metadata.get("pipeline", "standard")
                sources.append({
                    "source":   src,
                    "pipeline": pipeline,
                    "type":     doc.metadata.get("type", "text"),
                    "page":     doc.metadata.get("page", ""),
                    "content":  (doc.metadata.get("raw") or doc.page_content)[:200] + "…",
                })

        return {
            "generation":       generation,
            "generation_count": gen_count + 1,
            "sources":          sources,
            "reasoning_trace":  [
                f"💬 Generated answer via **{docs[0].metadata.get('pipeline','standard') if docs else 'standard'}** pipeline "
                f"(attempt {gen_count+1}, {len(generation)} chars)"
            ],
        }
    return generate


def make_check_hallucination(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def check_hallucination(state: AgentState) -> dict:
        generation = state["generation"]
        docs       = state.get("filtered_docs") or state.get("documents", [])
        context_parts = []
        for d in docs[:4]:
            raw = str(d.metadata.get("raw", "")).strip()
            context_parts.append((raw if raw else d.page_content)[:1200])
        context = "\n\n".join(context_parts)

        prompt = f"""Is this AI answer grounded in the source documents?

Sources:
{context}

Answer:
{generation}

Reply ONLY: grounded or hallucinating"""

        result  = await llm.ainvoke([HumanMessage(content=prompt)])
        verdict = _parse_binary(result.content, "grounded", "hallucinating")
        logger.info(f"[check_hallucination] → {verdict}")
        return {
            "hallucination_check": verdict,
            "reasoning_trace":     [f"{'✅' if verdict == 'grounded' else '⚠️'} Hallucination: **{verdict}**"],
        }
    return check_hallucination


def make_check_answer_quality(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def check_answer_quality(state: AgentState) -> dict:
        question   = state["question"]
        generation = state["generation"]

        prompt = f"""Does this answer fully address the question?

Question: {question}
Answer: {generation}

Reply ONLY: useful or not_useful"""

        result  = await llm.ainvoke([HumanMessage(content=prompt)])
        quality = _parse_binary(result.content, "useful", "not_useful")
        logger.info(f"[check_answer_quality] → {quality}")
        return {
            "answer_quality":  quality,
            "reasoning_trace": [f"{'✅' if quality == 'useful' else '🔄'} Quality: **{quality}**"],
        }
    return check_answer_quality


def make_reflect(kb: KnowledgeBase):
    llm = _grade_llm(kb)

    async def reflect(state: AgentState) -> dict:
        question   = state["question"]
        generation = state["generation"]
        hcheck     = state.get("hallucination_check")

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

        result     = await llm.ainvoke([HumanMessage(content=prompt)])
        reflection = result.content.strip()
        logger.info(f"[reflect] {reflection[:80]}")
        return {
            "reflection":          reflection,
            "generation":          None,
            "hallucination_check": None,
            "answer_quality":      None,
            "reasoning_trace":     [f"🪞 Reflection: *{reflection[:120]}*"],
        }
    return reflect


def make_direct_answer(kb: KnowledgeBase):
    llm = _gen_llm(kb)

    async def direct_answer(state: AgentState) -> dict:
        question      = state["question"]
        system_prompt = state.get("system_prompt", "")
        chat_history  = state.get("chat_history", [])

        history_str = "\n".join(
            f"Human: {h}\nAssistant: {a}" for h, a in chat_history[-3:]
        ) if chat_history else ""

        messages = [
            SystemMessage(content=system_prompt or "You are a helpful AI assistant."),
            HumanMessage(content=(f"{history_str}\n\n" if history_str else "") + question),
        ]

        result = await llm.ainvoke(messages)
        answer = result.content.strip()
        return {
            "generation":      answer,
            "final_answer":    answer,
            "sources":         [],
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
        "final_answer":   clarification,
        "sources":        [],
        "reasoning_trace": ["❓ Asked user for clarification (question too vague)"],
    }
# Finalise node — collects final_answer from generation
def finalise(state: AgentState) -> dict:
    return {
        "final_answer": state.get("final_answer") or state.get("generation", "I was unable to generate an answer."),
    }
# Conditional edge functions

def route_after_routing(state: AgentState) -> str:
    return state.get("route_decision", "retrieve")


def route_after_grading(state: AgentState) -> str:
    if state.get("filtered_docs"):
        return "generate"
    if state.get("rewrite_count", 0) >= state.get("max_rewrites", DEFAULT_MAX_REWRITES):
        logger.info("[route_after_grading] Max rewrites reached — generating with what we have")
        return "generate"
    return "rewrite"


def route_after_hallucination(state: AgentState) -> str:
    verdict = state.get("hallucination_check", "grounded")
    if verdict == "hallucinating":
        if state.get("generation_count", 0) >= state.get("max_retries", DEFAULT_MAX_RETRIES):
            return "quality"   # give up on hallucination check, check quality
        return "reflect"
    return "quality"


def route_after_quality(state: AgentState) -> str:
    quality = state.get("answer_quality", "useful")
    if quality == "not_useful":
        if state.get("rewrite_count", 0) >= state.get("max_rewrites", DEFAULT_MAX_REWRITES):
            return "end"   # accept imperfect answer rather than loop forever
        return "reflect"
    return "end"


def route_after_reflect(state: AgentState) -> str:
    """After reflection — rewrite the query and re-retrieve."""
    return "rewrite"
# Graph builder

def build_agentic_rag_graph(kb: KnowledgeBase, db: Session, fast_mode: bool = False) -> StateGraph:
    """Compile and return the full agentic RAG graph for a given KB."""
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

        graph.add_edge(START, "route_query")
        graph.add_conditional_edges("route_query", route_after_routing, {
            "retrieve": "retrieve",
            "general": "direct_answer",
            "clarify": "clarify",
        })
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "finalise")
        graph.add_edge("direct_answer", "finalise")
        graph.add_edge("clarify", "finalise")
        graph.add_edge("finalise", END)
        return graph.compile()
    graph.add_node("route_query",          make_route_query(kb))
    graph.add_node("retrieve",             make_retrieve(kb))
    graph.add_node("grade_documents",      make_grade_documents(kb))
    graph.add_node("rewrite_query",        make_rewrite_query(kb))
    graph.add_node("generate",             make_generate(kb))
    graph.add_node("check_hallucination",  make_check_hallucination(kb))
    graph.add_node("check_answer_quality", make_check_answer_quality(kb))
    graph.add_node("reflect",              make_reflect(kb))
    graph.add_node("direct_answer",        make_direct_answer(kb))
    graph.add_node("clarify",              clarify)
    graph.add_node("finalise",             finalise)
    graph.add_edge(START, "route_query")
    graph.add_conditional_edges("route_query", route_after_routing, {
        "retrieve": "retrieve",
        "general":  "direct_answer",
        "clarify":  "clarify",
    })
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading, {
        "generate": "generate",
        "rewrite":  "rewrite_query",
    })
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges("check_hallucination", route_after_hallucination, {
        "quality": "check_answer_quality",
        "reflect": "reflect",
    })
    graph.add_conditional_edges("check_answer_quality", route_after_quality, {
        "end":     "finalise",
        "reflect": "reflect",
    })
    graph.add_conditional_edges("reflect", route_after_reflect, {
        "rewrite": "rewrite_query",
    })
    graph.add_edge("direct_answer", "finalise")
    graph.add_edge("clarify",       "finalise")
    graph.add_edge("finalise",      END)

    return graph.compile()
# Public interface — called from routers/chat.py

async def run_agentic_rag(
    kb: KnowledgeBase,
    question: str,
    chat_history: Optional[List[Tuple[str, str]]],
    db: Session,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    fast_mode_override: Optional[bool] = None,
) -> Tuple[str, List[dict], List[str]]:
    """
    Run the agentic RAG graph.
    Returns (answer, sources, reasoning_trace).
    """
    from .rag_service import resolve_system_prompt

    system_prompt = resolve_system_prompt(kb, db)
    effective_fast_mode = bool(settings.FAST_MODE) if fast_mode_override is None else bool(fast_mode_override)
    cache_key = _cache_key(kb, question, chat_history, effective_fast_mode)

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
        )
        if sem_hit:
            answer, sources, trace, score = sem_hit
            trace = list(trace) + [f"Semantic cache hit (score={score:.2f})"]
            return answer, sources, trace

    memory_context = ""
    memory_store = None
    if settings.ENABLE_LONG_TERM_MEMORY and user_id is not None and session_id is not None:
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
        "doc_grades":        [],
        "filtered_docs":     [],
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
        final     = await app.ainvoke(initial_state)
        answer    = final.get("final_answer") or final.get("generation") or "Unable to answer."
        sources   = final.get("sources", [])
        trace     = final.get("reasoning_trace", [])

        if settings.ENABLE_WEB_FALLBACK and (not sources or "not in the context" in answer.lower()):
            web_sources = web_search(question, max_results=settings.WEB_SEARCH_MAX_RESULTS)
            if web_sources:
                snippets = "\n\n".join(
                    f"[{i+1}] {row.get('title', 'Web Result')}\n{row.get('content', '')}"
                    for i, row in enumerate(web_sources[: settings.WEB_SEARCH_MAX_RESULTS])
                )
                llm = _gen_llm(kb)
                web_prompt = (
                    "Local knowledge base could not fully answer. Use the web snippets below and cite uncertainty.\n\n"
                    f"{snippets}\n\nQuestion: {question}\n\nAnswer:"
                )
                web_answer = await llm.ainvoke([HumanMessage(content=web_prompt)])
                answer = (web_answer.content or "").strip() or answer
                sources = sources + web_sources
                trace = list(trace) + ["Web fallback used after weak local retrieval."]

        if memory_store is not None:
            memory_store.update(
                user_id=int(user_id),
                session_id=int(session_id),
                user_message=question,
                assistant_message=answer,
            )
        result = (answer, sources, trace)

        with _query_cache_lock:
            _query_cache[cache_key] = (time.time(), result)
            if len(_query_cache) > int(settings.QUERY_CACHE_MAX):
                oldest_key = min(_query_cache, key=lambda k: _query_cache[k][0])
                _query_cache.pop(oldest_key, None)
        if settings.ENABLE_SEMANTIC_CACHE:
            _semantic_cache.put(
                query=question,
                answer=answer,
                kb_id=kb.id,
                mode="fast" if effective_fast_mode else "quality",
                sources=sources,
                trace=trace,
            )

        return result
    except Exception as e:
        logger.exception(f"Agentic RAG graph failed: {e}")
        raise


