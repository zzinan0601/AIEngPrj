"""
agent/graph.py
- LangGraph StateGraph 조립
- 차트 요청 시 synthesize 이후 chart_node 로 분기
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
    chart_node,
)

_compiled_graph = None


def _route_after_router(state: GraphState) -> str:
    route = state.get("route", "general")
    if route == "both":
        return "rag"
    return route


def _route_after_rag(state: GraphState) -> str:
    if state.get("route") == "both":
        return "db"
    return "synthesize"


def _route_after_synthesize(state: GraphState) -> str:
    """
    차트 요청이 있으면 chart_node 로, 없으면 END
    """
    if state.get("chart_request"):
        return "chart"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # ── 노드 등록 ──────────────────────────────
    graph.add_node("router",     router_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("db",         db_node)
    graph.add_node("general",    general_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("chart",      chart_node)      # 신규

    # ── 시작점 ────────────────────────────────
    graph.set_entry_point("router")

    # ── Router → 분기 ─────────────────────────
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"rag": "rag", "db": "db", "general": "general", "both": "rag"},
    )

    # ── RAG → 분기 ────────────────────────────
    graph.add_conditional_edges(
        "rag",
        _route_after_rag,
        {"db": "db", "synthesize": "synthesize"},
    )

    # ── DB / General → Synthesize ─────────────
    graph.add_edge("db",      "synthesize")
    graph.add_edge("general", "synthesize")

    # ── Synthesize → Chart or END ─────────────
    graph.add_conditional_edges(
        "synthesize",
        _route_after_synthesize,
        {"chart": "chart", END: END},
    )

    # ── Chart → END ───────────────────────────
    graph.add_edge("chart", END)

    return graph.compile()


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
