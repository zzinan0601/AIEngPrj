"""
agent/nodes.py
- LangGraph 각 노드(처리 단계) 함수 정의
- Router  : 질문을 분석해 처리 경로 결정
- RAG     : 문서 벡터 검색 + 재랭킹
- DB      : SQL 생성 → DB 조회
- General : LLM 직접 응답
- Synthesize : 모든 결과를 합쳐 최종 답변 생성
"""

import config
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import GraphState, A2AMessage

# LLM 싱글톤 (ollama 로컬)
_llm = None


def get_llm() -> ChatOllama:
    """LLM 인스턴스를 싱글톤으로 반환"""
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
        )
    return _llm


def _log(msg: str) -> str:
    """타임스탬프가 붙은 로그 메시지 생성"""
    ts = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] {msg}"


# ── 1. Router 노드 ─────────────────────────────────────────────────────────
def router_node(state: GraphState) -> dict:
    """
    질문을 분석해 처리 경로를 결정한다.
    - rag    : 업로드된 문서에서 답을 찾아야 하는 경우
    - db     : 데이터베이스 조회가 필요한 경우
    - both   : 문서 + DB 모두 필요한 복합 질문
    - general: 일반 대화/상식 질문
    """
    question = state["question"]
    llm = get_llm()

    system_prompt = """당신은 질문 분류 전문가입니다. 아래 4가지 중 하나만 출력하세요.
- rag    : 문서(PDF, TXT 등)에서 정보를 찾아야 하는 질문
- db     : 데이터베이스 조회(통계, 목록, 집계 등)가 필요한 질문
- both   : 문서와 DB 두 곳 모두 확인해야 하는 복합 질문
- general: 일반 대화, 인사, 상식, 계산 등
반드시 네 단어 중 하나만 출력하세요."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"질문: {question}"),
        ])
        route = response.content.strip().lower()
        # 예외 처리: 알 수 없는 응답이면 general로 폴백
        if route not in ("rag", "db", "both", "general"):
            route = "general"
    except Exception as e:
        route = "general"

    # A2A 메시지: Router → 다음 에이전트에게 라우팅 결과 전달
    a2a_msg: A2AMessage = {
        "sender": "router",
        "receiver": route,
        "content": f"질문 '{question[:40]}' 을 {route} 경로로 라우팅",
        "msg_type": "request",
    }

    return {
        "route": route,
        "logs": [_log(f"🔀 라우팅 결정: [{route.upper()}]")],
        "a2a_messages": [a2a_msg],
        "iteration": state.get("iteration", 0),
    }


# ── 2. RAG 노드 ───────────────────────────────────────────────────────────
def rag_node(state: GraphState) -> dict:
    """
    문서 벡터 검색 + bge-reranker 재정렬 후 컨텍스트 생성.
    use_mcp=True 이면 MCP 서버를 경유하여 검색한다.
    """
    question = state["question"]
    use_mcp = state.get("use_mcp", False)
    logs = [_log("📚 RAG 문서 검색 시작")]

    try:
        if use_mcp:
            # MCP 경유 검색
            from mcp.mcp_client import search_via_mcp
            raw = search_via_mcp(question)
            results = [raw] if raw else []
            logs.append(_log("🔌 MCP 경유 검색 완료"))
        else:
            # 직접 RAG 파이프라인 호출
            from rag.pipeline import search_and_rerank
            results = search_and_rerank(question)

        if results:
            context = "\n\n---\n\n".join(
                [f"[문서 {i+1}]\n{r}" for i, r in enumerate(results)]
            )
            logs.append(_log(f"✅ {len(results)}개 청크 검색·재정렬 완료"))
        else:
            context = ""
            logs.append(_log("⚠️ 관련 문서를 찾지 못했습니다"))

    except Exception as e:
        context = ""
        logs.append(_log(f"❌ RAG 오류: {str(e)}"))

    # A2A: RAG → Synthesize 에 컨텍스트 전달
    a2a_msg: A2AMessage = {
        "sender": "rag",
        "receiver": "synthesize",
        "content": f"RAG 검색 결과 {len(context)}자 전달",
        "msg_type": "response",
    }

    return {"context": context, "logs": logs, "a2a_messages": [a2a_msg]}


# ── 3. DB 노드 ────────────────────────────────────────────────────────────
def db_node(state: GraphState) -> dict:
    """
    LLM으로 SQL을 생성한 뒤 SQLite DB를 실제로 조회한다.
    use_mcp=True 이면 MCP 서버를 경유한다.
    """
    question = state["question"]
    use_mcp = state.get("use_mcp", False)
    logs = [_log("🗄️ DB 쿼리 생성·조회 시작")]

    try:
        if use_mcp:
            from mcp.mcp_client import query_db_via_mcp
            db_results = query_db_via_mcp(question)
            sql = "(MCP 경유)"
            logs.append(_log("🔌 MCP 경유 DB 조회 완료"))
        else:
            from agent.db_agent import generate_and_execute_query
            sql, rows = generate_and_execute_query(question)
            db_results = str(rows) if rows else "조회 결과 없음"
            logs.append(_log(f"✅ SQL 실행 완료 (결과 {len(rows) if isinstance(rows, list) else 0}행)"))

    except Exception as e:
        sql = ""
        db_results = ""
        logs.append(_log(f"❌ DB 오류: {str(e)}"))

    a2a_msg: A2AMessage = {
        "sender": "db",
        "receiver": "synthesize",
        "content": f"DB 조회 결과 전달 (sql: {str(sql)[:60]})",
        "msg_type": "response",
    }

    return {
        "db_results": db_results,
        "generated_sql": sql,
        "logs": logs,
        "a2a_messages": [a2a_msg],
    }


# ── 4. General 노드 ───────────────────────────────────────────────────────
def general_node(state: GraphState) -> dict:
    """일반 대화: 문서/DB 조회 없이 LLM이 직접 답변"""
    logs = [_log("💬 일반 LLM 응답 모드")]
    a2a_msg: A2AMessage = {
        "sender": "general",
        "receiver": "synthesize",
        "content": "일반 응답 경로로 처리",
        "msg_type": "info",
    }
    # context / db_results 비워서 synthesize에서 순수 LLM 응답하도록
    return {
        "context": "",
        "db_results": "",
        "logs": logs,
        "a2a_messages": [a2a_msg],
    }


# ── 5. Synthesize 노드 ────────────────────────────────────────────────────
def synthesize_node(state: GraphState) -> dict:
    """
    RAG 컨텍스트 + DB 결과 + 프롬프트/퓨샷 설정을 합쳐
    최종 답변을 생성한다.
    """
    question = state["question"]
    context = state.get("context", "")
    db_results = state.get("db_results", "")
    prompt_config = state.get("prompt_config", {})
    llm = get_llm()
    logs = [_log("✨ 최종 답변 생성 중...")]

    # 시스템 프롬프트 (UI 설정값 우선)
    system_prompt = prompt_config.get(
        "system_prompt",
        "당신은 친절하고 정확한 AI 어시스턴트입니다. 한국어로 답변하세요.",
    )

    messages = [SystemMessage(content=system_prompt)]

    # 퓨샷 예제 추가
    for fs in prompt_config.get("fewshots", []):
        messages.append(HumanMessage(content=fs["question"]))
        # AIMessage 대신 HumanMessage prefix 방식으로 처리
        messages.append(SystemMessage(content=f"[예시 답변] {fs['answer']}"))

    # 사용자 질문 + 수집된 정보 구성
    user_content = f"질문: {question}\n"
    if context:
        user_content += f"\n[참고 문서]\n{context}\n"
    if db_results:
        user_content += f"\n[DB 조회 결과]\n{db_results}\n"
    if context or db_results:
        user_content += "\n위 정보를 참고하여 정확하게 답변해 주세요."

    messages.append(HumanMessage(content=user_content))

    try:
        response = llm.invoke(messages)
        answer = response.content
        logs.append(_log("✅ 답변 생성 완료"))
    except Exception as e:
        answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        logs.append(_log(f"❌ 생성 오류: {str(e)}"))

    return {"answer": answer, "logs": logs}
