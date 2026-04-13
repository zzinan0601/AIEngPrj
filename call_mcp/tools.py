"""
call_mcp/tools.py
- LangChain @tool 데코레이터를 사용한 도구 정의
- 에이전트가 직접 호출하거나 MCP 서버에 등록하는 함수들
- 각 도구는 독립적으로 동작하며 에러를 안전하게 처리
"""

from langchain_core.tools import tool


@tool
def search_document_tool(query: str) -> str:
    """
    업로드된 문서에서 관련 내용을 검색합니다.
    PDF, TXT, DOCX 파일에서 의미적으로 유사한 내용을 찾아 반환합니다.
    """
    try:
        from rag.pipeline import search_and_rerank
        results = search_and_rerank(query)
        if not results:
            return "관련 문서를 찾을 수 없습니다."
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"문서 검색 오류: {str(e)}"


@tool
def query_database_tool(question: str) -> str:
    """
    데이터베이스에서 SQL로 정보를 조회합니다.
    자연어 질문을 SQL로 변환한 후 DB를 조회하여 결과를 반환합니다.
    """
    try:
        from agent.db_agent import generate_and_execute_query
        sql, rows = generate_and_execute_query(question)
        if not rows:
            return f"조회 결과 없음\n실행 SQL: {sql}"
        result_text = f"실행 SQL: {sql}\n\n결과 ({len(rows)}행):\n"
        for i, row in enumerate(rows[:20]):  # 최대 20행만 표시
            result_text += f"{i+1}. {row}\n"
        return result_text
    except Exception as e:
        return f"DB 조회 오류: {str(e)}"


@tool
def get_db_schema_tool(dummy: str = "") -> str:
    """
    현재 데이터베이스의 테이블 구조(스키마)를 반환합니다.
    어떤 테이블과 컬럼이 있는지 확인할 때 사용합니다.
    """
    try:
        from agent.db_agent import get_db_schema
        return get_db_schema()
    except Exception as e:
        return f"스키마 조회 오류: {str(e)}"


# 에이전트에 바인딩할 도구 전체 목록
TOOLS = [
    search_document_tool,
    query_database_tool,
    get_db_schema_tool,
]
