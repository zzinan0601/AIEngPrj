"""
agent/state.py
- LangGraph 에이전트가 공유하는 상태(State) 정의
- plan: planner_node 가 생성한 실행 단계 목록 ["rag","db"] 등
- step_results: 각 단계별 중간 결과 누적
"""

import operator
from typing import TypedDict, Annotated, List, Optional


class A2AMessage(TypedDict):
    """에이전트 간 통신(A2A) 메시지 구조"""
    sender:   str
    receiver: str
    content:  str
    msg_type: str


class GraphState(TypedDict):
    """LangGraph 전체 상태 정의"""

    # 사용자 원본 질문
    question: str

    # planner 가 생성한 실행 계획 (순서대로)
    # 예: ["rag", "db"]  /  ["general"]  /  ["db", "rag"]
    plan: List[str]

    # 현재 실행 중인 단계 인덱스
    plan_idx: int

    # 라우팅 결정값 (executor 가 plan 에서 꺼낸 현재 단계)
    route: str

    # RAG 검색 결과
    context: str

    # RAG 출처 목록
    sources: Annotated[List[dict], operator.add]

    # DB 조회 결과
    db_results: str

    # 실행된 SQL
    generated_sql: str

    # 최종 답변
    answer: str

    # 처리 로그 (누적)
    logs: Annotated[List[str], operator.add]

    # A2A 메시지 (누적)
    a2a_messages: Annotated[List[A2AMessage], operator.add]

    # 복합 실행 반복 횟수
    iteration: int

    # 프롬프트/퓨샷 설정
    prompt_config: dict

    # MCP 사용 여부
    use_mcp: bool

    # 선택된 LLM 모델명
    selected_model: Optional[str]

    # 선택된 DB 타입
    db_type: Optional[str]

    # 이전 대화 기록
    chat_history: List[dict]

    # 차트 요청 여부
    chart_request: bool

    # 차트 설정
    chart_config: Optional[dict]

