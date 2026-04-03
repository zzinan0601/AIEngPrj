"""
agent/state.py
- LangGraph 에이전트가 공유하는 상태(State) 정의
- 모든 노드(Node)는 이 상태를 읽고 업데이트
- Annotated + operator.add : 리스트 필드는 덮어쓰지 않고 누적
"""

import operator
from typing import TypedDict, Annotated, List, Optional


class A2AMessage(TypedDict):
    """에이전트 간 통신(A2A) 메시지 구조"""
    sender: str      # 메시지 보낸 에이전트
    receiver: str    # 메시지 받을 에이전트
    content: str     # 메시지 내용
    msg_type: str    # 메시지 유형: request / response / info


class GraphState(TypedDict):
    """LangGraph 전체 상태 정의"""

    # 사용자 원본 질문
    question: str

    # 라우팅 결정값: "rag" | "db" | "both" | "general"
    route: str

    # RAG 검색 결과 (문서 컨텍스트)
    context: str

    # DB 조회 결과
    db_results: str

    # 실행된 SQL 쿼리 (투명성 확보)
    generated_sql: str

    # 최종 답변
    answer: str

    # 처리 로그 목록 (누적 - operator.add)
    logs: Annotated[List[str], operator.add]

    # 에이전트 간 A2A 메시지 (누적)
    a2a_messages: Annotated[List[A2AMessage], operator.add]

    # 복합 질문 loop 반복 횟수
    iteration: int

    # 프롬프트/퓨샷 설정 (UI에서 주입)
    prompt_config: dict

    # MCP 사용 여부 (True면 MCP 경유)
    use_mcp: bool
