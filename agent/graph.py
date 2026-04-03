"""
agent/graph.py
- LangGraph StateGraph를 조립하는 모듈
- 노드(처리단계)와 엣지(연결)로 에이전트 흐름을 정의
- 복합 질문("both")은 RAG → DB → Synthesize 순서로 loop
"""

import config
from langgraph.graph import StateGraph, END
from agent.state import GraphState
from agent.nodes import (
    router_node,
    rag_node,
    db_node,
    general_node,
    synthesize_node,
)

_compiled_graph = None  # 싱글톤 캐시


def _route_after_router(state: GraphState) -> str:
    """
    Router 노드 이후 분기 결정
    - "both" 이면 RAG를 먼저 실행
    - 나머지는 바로 해당 노드로
    """
    route = state.get("route", "general")
    if route == "both":
        return "rag"   # both: rag → db → synthesize 순서
    return route       # rag | db | general


def _route_after_rag(state: GraphState) -> str:
    """
    RAG 노드 이후 분기 결정
    - "both" 이면 DB 노드로 계속 진행
    - 그 외에는 바로 synthesize
    """
    if state.get("route") == "both":
        return "db"
    return "synthesize"


def _check_iteration(state: GraphState) -> str:
    """
    복합 질문 loop 종료 조건 검사
    - MAX_ITERATIONS 초과 시 강제 종료
    """
    iteration = state.get("iteration", 0)
    if iteration >= config.MAX_ITERATIONS:
        return "synthesize"
    return "continue"


def build_graph() -> StateGraph:
    """
    LangGraph StateGraph를 빌드하고 컴파일하여 반환
    """
    graph = StateGraph(GraphState)

    # ── 노드 등록 ──────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("db", db_node)
    graph.add_node("general", general_node)
    graph.add_node("synthesize", synthesize_node)

    # ── 시작점 설정 ────────────────────────────
    graph.set_entry_point("router")

    # ── Router → 조건부 분기 ───────────────────
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "rag": "rag",
            "db": "db",
            "general": "general",
            "both": "rag",   # both → rag 먼저 (안전망: 함수가 rag 반환하지만 명시)
        },
    )

    # ── RAG → 조건부 분기 (both이면 DB로 계속) ─
    graph.add_conditional_edges(
        "rag",
        _route_after_rag,
        {
            "db": "db",
            "synthesize": "synthesize",
        },
    )

    # ── DB → Synthesize ────────────────────────
    graph.add_edge("db", "synthesize")

    # ── General → Synthesize ───────────────────
    graph.add_edge("general", "synthesize")

    # ── Synthesize → END ──────────────────────
    graph.add_edge("synthesize", END)

    return graph.compile()


def get_graph():
    """컴파일된 그래프 싱글톤 반환 (최초 1회만 빌드)"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
