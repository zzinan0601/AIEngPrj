"""
agent/graph.py
- planner_node 가 질문을 분석해 실행 계획(plan) 수립
- executor_node 가 plan 에서 단계를 꺼내 route 세팅
- 각 단계(rag/db/general) 실행 후 step_done_node 로 plan_idx 증가
- 남은 단계 있으면 executor_node 로 돌아가 반복
- 모든 단계 완료 시 synthesize → chart(선택) → END

흐름:
  planner → executor → [rag|db|general] → step_done
                ↑                              |
                └──────(남은 단계 있음)─────────┘
                        (완료 시) → synthesize → [chart] → END
"""

from langgraph.graph import StateGraph, END
from agent.state import GraphState
from agent.nodes import (
    planner_node,
    executor_node,
    api_node,
    step_done_node,
    rag_node,
    db_node,
    general_node,
    synthesize_node,
    chart_node,
)

_compiled_graph = None


def _route_from_executor(state: GraphState) -> str:
    """executor_node 이후 분기: route 값으로 실행할 노드 결정"""
    route = state.get("route", "general")
    if route == "__done__":
        return "synthesize"
    return route   # "rag" | "db" | "general"


def _route_after_step(state: GraphState) -> str:
    """
    step_done_node 이후: 남은 단계 있으면 executor, 없으면 synthesize
    plan_idx 는 step_done_node 에서 이미 증가된 상태
    """
    plan     = state.get("plan", [])
    plan_idx = state.get("plan_idx", 0)
    if plan_idx < len(plan):
        return "executor"   # 다음 단계 실행
    return "synthesize"


def _route_after_synthesize(state: GraphState) -> str:
    """차트 요청 있으면 chart, 없으면 END"""
    if state.get("chart_request"):
        return "chart"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # ── 노드 등록 ──────────────────────────────────────────────────
    graph.add_node("planner",    planner_node)
    graph.add_node("executor",   executor_node)
    graph.add_node("api",        api_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("db",         db_node)
    graph.add_node("general",    general_node)
    graph.add_node("step_done",  step_done_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("chart",      chart_node)

    # ── 시작점 ────────────────────────────────────────────────────
    graph.set_entry_point("planner")

    # ── planner → executor ────────────────────────────────────────
    graph.add_edge("planner", "executor")

    # ── executor → rag | db | general | synthesize ────────────────
    graph.add_conditional_edges(
        "executor",
        _route_from_executor,
        {
            "rag":        "rag",
            "db":         "db",
            "api":        "api",
            "general":    "general",
            "synthesize": "synthesize",
        },
    )

    # ── 각 실행 노드 → step_done ──────────────────────────────────
    graph.add_edge("rag",     "step_done")
    graph.add_edge("db",      "step_done")
    graph.add_edge("general", "step_done")
    graph.add_edge("api",     "step_done")

    # ── step_done → executor(반복) | synthesize(완료) ─────────────
    graph.add_conditional_edges(
        "step_done",
        _route_after_step,
        {
            "executor":   "executor",
            "synthesize": "synthesize",
        },
    )

    # ── synthesize → chart | END ──────────────────────────────────
    graph.add_conditional_edges(
        "synthesize",
        _route_after_synthesize,
        {"chart": "chart", END: END},
    )

    # ── chart → END ───────────────────────────────────────────────
    graph.add_edge("chart", END)

    return graph.compile()


def get_graph():
    """컴파일된 그래프 싱글톤 반환"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
