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

# 모델명별 LLM 인스턴스 캐시 (여러 모델 동시 보관)
_llm_cache: dict = {}


def get_llm(model_name: str = None) -> ChatOllama:
    """
    모델명별로 ChatOllama 인스턴스를 캐시하여 반환.
    model_name 미지정 시 config.LLM_MODEL(기본값) 사용.
    """
    global _llm_cache
    name = model_name or config.LLM_MODEL
    if name not in _llm_cache:
        _llm_cache[name] = ChatOllama(
            model=name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
        )
    return _llm_cache[name]


def _log(msg: str) -> str:
    """타임스탬프가 붙은 로그 메시지 생성"""
    ts = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] {msg}"


# ── 1. Router 노드 ─────────────────────────────────────────────────────────
def _has_documents() -> bool:
    """Qdrant에 저장된 문서가 1개 이상 있으면 True"""
    try:
        from rag.vector_store import get_file_list
        return len(get_file_list()) > 0
    except Exception:
        return False


def _parse_route(raw: str) -> str:
    """
    LLM 응답에서 라우팅 키워드를 추출한다.
    llama3.1:8b 는 "- rag", "rag.", "RAG (문서 관련)" 등 다양하게 반환하므로
    완전 일치 대신 포함 여부로 판단한다.
    """
    text = raw.strip().lower()
    # 우선순위: both > db > rag > general
    if "both" in text:
        return "both"
    if "db" in text or "database" in text or "sql" in text or "데이터베이스" in text:
        return "db"
    if "rag" in text or "document" in text or "문서" in text:
        return "rag"
    if "general" in text or "일반" in text:
        return "general"
    return None   # 판단 불가 → 호출부에서 처리


def _has_explicit_doc_reference(question: str) -> bool:
    """
    사용자가 명시적으로 문서 참조를 요청했는지 확인.
    이 경우 LLM 판단 없이 즉시 rag로 강제한다.
    """
    keywords = [
        # 한국어
        "문서", "파일", "첨부", "업로드", "pdf", "txt", "docx",
        "참조", "참고", "기반으로", "바탕으로", "내용에서",
        "문서에서", "파일에서", "자료에서", "보고서에서",
        # 영어
        "document", "file", "uploaded", "attached", "based on",
        "according to", "from the", "in the document", "in the file",
        "refer to", "reference",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


def router_node(state: GraphState) -> dict:
    """
    질문을 분석해 처리 경로를 결정한다.
    - rag    : 업로드된 문서에서 답을 찾아야 하는 경우
    - db     : 데이터베이스 조회가 필요한 경우
    - both   : 문서 + DB 모두 필요한 복합 질문
    - general: 일반 대화/상식 질문

    우선순위:
    1. 명시적 문서 참조 키워드 → 즉시 rag (LLM 호출 생략)
    2. LLM 분류 (영어 프롬프트 + 퓨샷)
    3. LLM 판단 불가 + 문서 존재 → rag 폴백
    4. 문서 없음 → general
    """
    question = state["question"]
    docs_exist = _has_documents()
    selected_model = state.get("selected_model") or config.LLM_MODEL

    # ── 우선순위 1: 명시적 문서 참조 키워드 감지 ──────────────────
    # "document를 참조해서", "문서에서", "파일 기반으로" 등이 있으면
    # LLM 호출 없이 즉시 rag로 강제
    if docs_exist and _has_explicit_doc_reference(question):
        route = "rag"
        log_msg = _log(f"🔀 라우팅 결정: [RAG] (명시적 문서 참조 감지)")
        a2a_msg: A2AMessage = {
            "sender": "router",
            "receiver": "rag",
            "content": f"명시적 문서 참조 키워드 감지 → RAG 강제",
            "msg_type": "request",
        }
        return {
            "route": route,
            "logs": [log_msg],
            "a2a_messages": [a2a_msg],
            "iteration": state.get("iteration", 0),
        }

    # ── 우선순위 2: LLM 분류 ──────────────────────────────────────
    llm = get_llm(selected_model)
    system_prompt = """You are a routing classifier. Output ONLY one word from: rag, db, both, general

Rules:
- rag    : question about uploaded documents, files, manuals, reports, papers
- db     : question requiring database query (statistics, counts, lists from DB)
- both   : question requiring BOTH documents AND database
- general: greeting, math, common knowledge, anything else

Examples:
Q: "What does the document say about refund policy?" -> rag
Q: "How many users registered last month?" -> db
Q: "Summarize the report and show total sales" -> both
Q: "Hello, how are you?" -> general
Q: "What is the main topic of the uploaded file?" -> rag
Q: "안녕하세요" -> general
Q: "업로드한 문서에서 환불 정책을 알려줘" -> rag
Q: "지난달 매출 합계는?" -> db

Output ONLY the single word. No explanation."""

    route = None
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Q: {question}"),
        ])
        route = _parse_route(response.content)
    except Exception:
        pass

    # ── 우선순위 3 & 4: 폴백 ─────────────────────────────────────
    if route is None:
        route = "rag" if docs_exist else "general"

    if route in ("rag", "both") and not docs_exist:
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
            from mcp.mcp_client import search_via_mcp
            raw = search_via_mcp(question)
            results = [{"text": raw, "filename": "MCP", "chunk_index": 0}] if raw else []
            logs.append(_log("🔌 MCP 경유 검색 완료"))
        else:
            from rag.pipeline import search_and_rerank
            results = search_and_rerank(question)

        if results:
            context = "\n\n---\n\n".join(
                [f"[문서 {i+1}]\n{r['text']}" for i, r in enumerate(results)]
            )
            # 출처 목록: 중복 제거 후 순서 유지
            seen = set()
            sources = []
            for r in results:
                key = (r["filename"], r["chunk_index"])
                if key not in seen:
                    seen.add(key)
                    sources.append(r)
            logs.append(_log(f"✅ {len(results)}개 청크 검색·재정렬 완료"))
        else:
            context = ""
            sources = []
            logs.append(_log("⚠️ 관련 문서를 찾지 못했습니다"))

    except Exception as e:
        context = ""
        sources = []
        logs.append(_log(f"❌ RAG 오류: {str(e)}"))

    # A2A: RAG → Synthesize 에 컨텍스트 전달
    a2a_msg: A2AMessage = {
        "sender": "rag",
        "receiver": "synthesize",
        "content": f"RAG 검색 결과 {len(context)}자 전달",
        "msg_type": "response",
    }

    return {"context": context, "sources": sources, "logs": logs, "a2a_messages": [a2a_msg]}


# ── 3. DB 노드 ────────────────────────────────────────────────────────────
def db_node(state: GraphState) -> dict:
    """
    LLM으로 SQL을 생성한 뒤 SQLite DB를 실제로 조회한다.
    use_mcp=True 이면 MCP 서버를 경유한다.
    """
    question = state["question"]
    use_mcp  = state.get("use_mcp", False)
    db_type  = state.get("db_type") or config.DB_TYPE
    rag_context = state.get("context", "")   # both 경로에서 RAG 결과 참조
    logs = [_log(f"🗄️ DB 쿼리 생성·조회 시작 ({db_type})")]

    try:
        if use_mcp:
            from mcp.mcp_client import query_db_via_mcp
            db_results = query_db_via_mcp(question)
            sql = "(MCP 경유)"
            logs.append(_log("🔌 MCP 경유 DB 조회 완료"))
        else:
            from agent.db_agent import generate_and_execute_query
            sql, rows = generate_and_execute_query(question, db_type, rag_context)
            db_results = str(rows) if rows else "조회 결과 없음"
            logs.append(_log(f"✅ SQL 실행 완료 (결과 {len(rows) if isinstance(rows, list) else 0}행)"))
            if rag_context:
                logs.append(_log("📚 RAG 컨텍스트를 SQL 생성에 참조했습니다"))

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
    selected_model = state.get("selected_model") or config.LLM_MODEL
    llm = get_llm(selected_model)
    logs = [_log(f"✨ 최종 답변 생성 중... (모델: {selected_model})")]

    # 시스템 프롬프트 (UI 설정값 우선, 없으면 기본값)
    system_prompt = prompt_config.get("system_prompt", "")
    if not system_prompt.strip():
        system_prompt = "You are a helpful AI assistant. Answer in Korean."

    messages = [SystemMessage(content=system_prompt)]

    # 퓨샷 예제 추가
    for fs in prompt_config.get("fewshots", []):
        messages.append(HumanMessage(content=fs["question"]))
        messages.append(SystemMessage(content=f"[예시 답변] {fs['answer']}"))

    # 사용자 질문 + 수집된 정보 구성
    user_content = f"Question: {question}\n"
    if context:
        user_content += f"\n[Reference Documents]\n{context}\n"
    if db_results:
        user_content += f"\n[DB Query Results]\n{db_results}\n"
    if context or db_results:
        user_content += "\nPlease answer accurately based on the above information."

    # llama3.1:8b 는 한국어 질문이 오면 시스템 프롬프트를 무시하고
    # 한국어로 응답하는 경향이 있으므로 HumanMessage 끝에도 지시를 강제 삽입한다.
    user_content += f"\n\n[Important] Follow this instruction strictly: {system_prompt}"

    messages.append(HumanMessage(content=user_content))

    try:
        response = llm.invoke(messages)
        answer = response.content

        # 출처가 있으면 답변 뒤에 추가
        sources = state.get("sources", [])
        if sources:
            seen = set()
            source_lines = []
            for s in sources:
                fname = s.get("filename", "")
                cidx  = s.get("chunk_index", 0)
                key   = (fname, cidx)
                if key not in seen and fname:
                    seen.add(key)
                    source_lines.append(f"- {fname}  (청크 {cidx + 1})")
            if source_lines:
                answer += "\n\n---\n**📚 출처**\n" + "\n".join(source_lines)

        logs.append(_log("✅ 답변 생성 완료"))
    except Exception as e:
        answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        logs.append(_log(f"❌ 생성 오류: {str(e)}"))

    return {"answer": answer, "logs": logs}
