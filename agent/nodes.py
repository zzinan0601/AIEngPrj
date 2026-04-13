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


# ── 1. Planner 노드 ────────────────────────────────────────────────────────
def _has_documents() -> bool:
    """Qdrant에 저장된 문서가 1개 이상 있으면 True"""
    try:
        from rag.vector_store import get_file_list
        return len(get_file_list()) > 0
    except Exception:
        return False


def _has_explicit_doc_reference(question: str) -> bool:
    """명시적 문서 참조 키워드 감지"""
    keywords = [
        "문서", "파일", "첨부", "업로드", "pdf", "txt", "docx",
        "참조", "참고", "기반으로", "바탕으로", "내용에서",
        "문서에서", "파일에서", "자료에서", "보고서에서",
        "document", "file", "uploaded", "attached", "based on",
        "according to", "from the", "in the document", "in the file",
        "refer to", "reference",
    ]
    return any(kw in question.lower() for kw in keywords)


def _parse_plan(raw: str, docs_exist: bool) -> list:
    """
    LLM 응답에서 실행 계획(단계 목록)을 추출한다.
    JSON 배열 형태로 반환받고, 파싱 실패 시 폴백.

    반환 예시: ["rag", "db"] / ["db"] / ["general"] / ["api"]
    """
    import json, re

    raw = raw.strip()
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            steps = json.loads(match.group())
            # rag / db / general / api 허용
            valid = [s.strip().lower() for s in steps
                     if s.strip().lower() in ("rag", "db", "general", "api")]
            if valid:
                if not docs_exist:
                    valid = [s for s in valid if s != "rag"]
                return valid if valid else ["general"]
        except Exception:
            pass

    # 폴백
    text = raw.lower()
    steps = []
    if ("rag" in text or "document" in text or "문서" in text) and docs_exist:
        steps.append("rag")
    if "db" in text or "database" in text or "sql" in text or "데이터베이스" in text:
        steps.append("db")
    if "api" in text or "rest" in text:
        steps.append("api")
    if not steps:
        steps = ["general"]
    return steps


def planner_node(state: GraphState) -> dict:
    """
    LLM이 질문을 분석해 실행 계획(plan)을 수립한다.
    - 단순 질문: ["general"] / ["rag"] / ["db"]
    - 복합 질문: ["rag", "db"] / ["db", "rag"] 등 순서까지 결정
    - both 개념 없음: LLM이 직접 단계와 순서를 결정
    """
    import json

    question       = state["question"]
    docs_exist     = _has_documents()
    selected_model = state.get("selected_model") or config.LLM_MODEL
    llm            = get_llm(selected_model)

    # ── 명시적 문서 참조 키워드 → rag 우선 포함 ───────────────────
    if docs_exist and _has_explicit_doc_reference(question):
        plan = ["rag"]
        # DB 관련 키워드도 있으면 db 추가
        q = question.lower()
        if any(kw in q for kw in ["db", "조회", "데이터", "통계", "집계", "매출", "수량", "목록"]):
            plan.append("db")
        log_msg = _log(f"📋 계획 수립: {plan} (문서 키워드 감지)")
        a2a_msg: A2AMessage = {
            "sender": "planner", "receiver": str(plan),
            "content": f"키워드 감지로 계획 수립: {plan}", "msg_type": "request",
        }
        return {
            "plan": plan, "plan_idx": 0, "route": plan[0],
            "chart_request": _is_chart_request(question),
            "logs": [log_msg], "a2a_messages": [a2a_msg],
            "iteration": 0,
        }

    # ── LLM 으로 계획 수립 ────────────────────────────────────────
    avail_steps = []
    if docs_exist:
        avail_steps.append("rag (uploaded documents)")
    avail_steps.append("db (database query)")
    avail_steps.append("api (REST API call via MCP)")
    avail_steps.append("general (LLM direct answer)")

    system_prompt = f"""You are a task planner for an AI assistant.
Analyze the question and create an execution plan as a JSON array.

Available steps: {', '.join(avail_steps)}

Rules:
- Use ONLY step names: "rag", "db", "api", "general"
- Order matters: put the step that should run FIRST first
- Use only the steps actually needed
- Output ONLY a JSON array, nothing else

Examples:
Q: "What is the refund policy in the document?" → ["rag"]
Q: "How many orders last month?" → ["db"]
Q: "Hello!" → ["general"]
Q: "API로 사용자 1번 정보 조회해줘" → ["api"]
Q: "REST API 호출해서 결과 보여줘" → ["api"]
Q: "사용자 1번 API 조회하고 DB에서 관련 주문도 보여줘" → ["api", "db"]
Q: "Summarize the document and show related sales data" → ["rag", "db"]
Q: "2+2?" → ["general"]"""

    plan = None
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Q: {question}"),
        ])
        plan = _parse_plan(response.content, docs_exist)
    except Exception:
        pass

    if not plan:
        plan = ["rag"] if docs_exist else ["general"]

    log_msg = _log(f"📋 실행 계획: {plan}")
    a2a_msg: A2AMessage = {
        "sender": "planner", "receiver": str(plan),
        "content": f"계획 수립 완료: {plan}", "msg_type": "request",
    }
    return {
        "plan": plan, "plan_idx": 0, "route": plan[0],
        "chart_request": _is_chart_request(question),
        "logs": [log_msg], "a2a_messages": [a2a_msg],
        "iteration": 0,
    }


# ── 1-1. Executor 노드 ─────────────────────────────────────────────────────
def executor_node(state: GraphState) -> dict:
    """
    plan 에서 현재 단계를 꺼내 route 를 세팅한다.
    실제 실행은 graph 의 조건부 엣지가 route 값으로 rag/db/general 로 분기한다.
    """
    plan     = state.get("plan", ["general"])
    plan_idx = state.get("plan_idx", 0)

    if plan_idx >= len(plan):
        # 모든 단계 완료 → synthesize 로
        return {"route": "__done__", "logs": [_log("✅ 모든 단계 완료 → 답변 생성")]}

    current = plan[plan_idx]
    return {
        "route":    current,
        "plan_idx": plan_idx,   # 아직 증가 안 함 (실행 후 증가)
        "logs":     [_log(f"▶️  단계 {plan_idx+1}/{len(plan)}: [{current.upper()}] 실행")],
    }


# ── 1-2. API 노드 ─────────────────────────────────────────────────────────
def api_node(state: GraphState) -> dict:
    """
    MCP SSE 서버에 등록된 REST API 도구를 호출한다.
    질문에서 파라미터를 LLM으로 추출한 뒤 call_rest_api_sample 도구를 실행.
    실제 운영 시 mcp_server.py에 원하는 REST API 도구를 추가하면 됨.
    """
    question       = state["question"]
    selected_model = state.get("selected_model") or config.LLM_MODEL
    llm            = get_llm(selected_model)
    logs           = [_log("🌐 REST API 호출 시작")]

    # ── LLM으로 질문에서 파라미터 추출 ──────────────────────────────
    param_prompt = """Extract the user_id parameter from the question.
Output ONLY a number (1-10). If not found, output 1."""

    user_id = "1"
    try:
        resp = llm.invoke([
            SystemMessage(content=param_prompt),
            HumanMessage(content=question),
        ])
        extracted = resp.content.strip()
        # 숫자만 추출
        import re
        nums = re.findall(r'\d+', extracted)
        if nums:
            user_id = nums[0]
        logs.append(_log(f"🔍 파라미터 추출: user_id={user_id}"))
    except Exception as e:
        logs.append(_log(f"⚠️ 파라미터 추출 실패, 기본값 사용: {e}"))

    # ── MCP SSE 서버 도구 호출 ───────────────────────────────────────
    api_result = ""
    try:
        from call_mcp.mcp_client import call_tool
        api_result = call_tool("call_rest_api_sample", {"user_id": user_id})
        logs.append(_log(f"✅ API 호출 완료 (user_id={user_id})"))
    except Exception as e:
        api_result = f"API 호출 오류: {str(e)}"
        logs.append(_log(f"❌ API 오류: {str(e)}"))

    # db_results 필드에 저장 (synthesize_node가 참조)
    a2a_msg: A2AMessage = {
        "sender": "api", "receiver": "synthesize",
        "content": f"REST API 결과 전달 (user_id={user_id})",
        "msg_type": "response",
    }
    return {
        "db_results": api_result,   # synthesize가 db_results로 포함
        "logs": logs,
        "a2a_messages": [a2a_msg],
    }


# ── 1-3. Step Done 노드 ────────────────────────────────────────────────────
def step_done_node(state: GraphState) -> dict:
    """
    한 단계 실행 완료 후 plan_idx 를 증가시킨다.
    다음 executor_node 호출 시 다음 단계로 넘어간다.
    """
    plan     = state.get("plan", [])
    plan_idx = state.get("plan_idx", 0) + 1
    return {
        "plan_idx": plan_idx,
        "logs": [_log(f"✔️  단계 완료 ({plan_idx}/{len(plan)})")],
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
    question    = state["question"]
    use_mcp     = state.get("use_mcp", False)
    db_type     = state.get("db_type") or config.DB_TYPE
    rag_context = state.get("context", "")
    # 이전 대화에서 값·조건 힌트 추출을 위해 chat_history 전달
    chat_history = state.get("chat_history", [])
    logs = [_log(f"🗄️ DB 쿼리 생성·조회 시작 ({db_type})")]

    try:
        if use_mcp:
            from mcp.mcp_client import query_db_via_mcp
            db_results = query_db_via_mcp(question)
            sql = "(MCP 경유)"
            logs.append(_log("🔌 MCP 경유 DB 조회 완료"))
        else:
            from agent.db_agent import generate_and_execute_query
            sql, rows = generate_and_execute_query(
                question, db_type, rag_context, chat_history
            )
            db_results = str(rows) if rows else "조회 결과 없음"
            logs.append(_log(f"✅ SQL 실행 완료 (결과 {len(rows) if isinstance(rows, list) else 0}행)"))
            if rag_context:
                logs.append(_log("📚 RAG 컨텍스트를 SQL 생성에 참조했습니다"))
            if chat_history:
                logs.append(_log("🧠 이전 대화를 SQL 생성에 참조했습니다"))

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

    # 시스템 프롬프트
    system_prompt = prompt_config.get("system_prompt", "")
    if not system_prompt.strip():
        system_prompt = "You are a helpful AI assistant. Answer in Korean."

    messages = [SystemMessage(content=system_prompt)]

    # 퓨샷 예제
    for fs in prompt_config.get("fewshots", []):
        messages.append(HumanMessage(content=fs["question"]))
        messages.append(SystemMessage(content=f"[예시 답변] {fs['answer']}"))

    # 이전 대화 기록 주입 (슬라이딩 윈도우)
    history = state.get("chat_history", [])
    if history and config.MEMORY_TURNS > 0:
        # 최근 N턴 = 최근 N*2개 메시지 (질문+답변 쌍)
        window = history[-(config.MEMORY_TURNS * 2):]
        for h in window:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                # AIMessage 대신 SystemMessage prefix로 처리
                messages.append(SystemMessage(content=f"[이전 답변] {h['content']}"))
        logs.append(_log(f"🧠 대화 기억 {len(window)//2}턴 주입 (최대 {config.MEMORY_TURNS}턴)"))

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


# ── 차트 감지 헬퍼 ────────────────────────────────────────────────────────
_CHART_KEYWORDS = [
    # 한국어
    "차트", "그래프", "그려", "시각화", "플롯", "막대", "선그래프",
    "파이차트", "분포", "추이", "트렌드", "히스토그램",
    # 영어
    "chart", "graph", "plot", "visualize", "bar chart", "line chart",
    "pie chart", "histogram", "trend", "distribution",
]

def _is_chart_request(question: str) -> bool:
    """질문에 차트/시각화 키워드가 있으면 True"""
    q = question.lower()
    return any(kw in q for kw in _CHART_KEYWORDS)


# ── 6. Chart 노드 ─────────────────────────────────────────────────────────
def chart_node(state: GraphState) -> dict:
    """
    RAG 컨텍스트 또는 DB 결과를 분석해 차트 설정(JSON)을 생성.
    LLM 에게 JSON 형식만 출력하도록 강제한 뒤 파싱한다.

    chart_config 구조:
    {
      "type"    : "bar" | "line" | "pie" | "scatter",
      "title"   : "차트 제목",
      "x_labels": ["1월","2월",...],
      "series"  : [{"name":"매출", "values":[100,200,...]}],
      "summary" : "텍스트 요약 (차트 아래 표시)"
    }
    """
    import json
    import re

    question     = state["question"]
    context      = state.get("context", "")
    db_results   = state.get("db_results", "")
    selected_model = state.get("selected_model") or config.LLM_MODEL
    llm = get_llm(selected_model)
    logs = [_log("📊 차트 설정 생성 중...")]

    # 데이터 소스 결정
    data_source = ""
    if db_results and db_results != "조회 결과 없음":
        data_source = f"[DB 조회 결과]\n{db_results}"
    elif context:
        data_source = f"[문서 내용]\n{context[:2000]}"

    if not data_source:
        logs.append(_log("⚠️ 차트 생성에 사용할 데이터가 없습니다"))
        return {"chart_config": None, "logs": logs}

    system_prompt = """You are a data visualization expert.
Analyze the given data and generate chart configuration as JSON.
Output ONLY valid JSON, no explanation, no markdown code block.

JSON structure:
{
  "type": "bar" or "line" or "pie" or "scatter",
  "title": "chart title in Korean",
  "x_labels": ["label1", "label2", ...],
  "series": [{"name": "series name", "values": [number, number, ...]}],
  "summary": "one sentence summary in Korean"
}

Rules:
- values must be numbers only
- x_labels and values must have same length
- choose chart type that best fits the data
- if data cannot be visualized, return {"type": "none", "summary": "이유"}"""

    user_content = f"질문: {question}\n\n{data_source}"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        raw = response.content.strip()

        # 마크다운 코드블록 제거
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        chart_config = json.loads(raw)

        if chart_config.get("type") == "none":
            logs.append(_log(f"⚠️ 차트 생성 불가: {chart_config.get('summary','')}"))
            return {"chart_config": None, "logs": logs}

        logs.append(_log(f"✅ 차트 생성 완료: {chart_config.get('type')} - {chart_config.get('title')}"))
        return {"chart_config": chart_config, "logs": logs}

    except Exception as e:
        logs.append(_log(f"❌ 차트 생성 오류: {str(e)}"))
        return {"chart_config": None, "logs": logs}
