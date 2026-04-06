"""
Shared state schema for the Agentic RAG LangGraph graph.
All nodes read from and write to AgentState.
"""
from __future__ import annotations
from typing import Annotated, Any, List, Optional, TypedDict
import operator


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    question:       str                        # original user question
    kb_id:          int                        # which KB to search
    chat_history:   List[tuple[str, str]]      # (human, ai) pairs
    fast_mode:      bool                       # per-request fast mode flag

    # ── Routing ───────────────────────────────────────────────────────────────
    route_decision: str                        # "retrieve" | "general" | "clarify"
    route_mode: str                            # "retrieve" | "sparse" | "graph"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    documents:      List[Any]                  # retrieved LangChain Documents
    rewrite_count:  int                        # how many times query was rewritten
    rewritten_query: Optional[str]             # last rewritten query

    # ── Grading ───────────────────────────────────────────────────────────────
    doc_grades:     List[str]                  # "yes"|"no" per document
    filtered_docs:  List[Any]                  # only relevant documents

    # ── Generation ────────────────────────────────────────────────────────────
    generation:     Optional[str]              # current draft answer
    generation_count: int                      # retry counter

    # ── Quality checks ────────────────────────────────────────────────────────
    hallucination_check: Optional[str]         # "grounded" | "hallucinating"
    answer_quality:      Optional[str]         # "useful" | "not_useful"
    reflection:          Optional[str]         # LLM self-reflection text

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer:   Optional[str]
    sources:        List[dict]
    reasoning_trace: Annotated[List[str], operator.add]  # append-only trace
    system_prompt:  Optional[str]              # resolved personality prompt
    memory_context: Optional[str]              # persistent user preferences and facts
    user_id: Optional[int]
    session_id: Optional[int]

    # ── Limits (prevent infinite loops) ──────────────────────────────────────
    max_rewrites:   int
    max_retries:    int
