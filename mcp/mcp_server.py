"""
call_mcp/mcp_server.py
- FastMCP SSE 모드로 실행 (HTTP 서버)
- 실행: python mcp_server.py
- 기본 포트: config.MCP_SERVER_PORT (기본 8765)
- 클라이언트는 HTTP SSE로 연결 → subprocess 방식의 stdout 오염 문제 없음
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

mcp_server = FastMCP(
    name="RAG-Agent-Tools",
    instructions="RAG 문서 검색과 DB 조회를 제공하는 MCP 서버입니다.",
)


@mcp_server.tool()
def search_documents(query: str) -> str:
    """업로드된 문서에서 관련 내용을 벡터 검색 + 재정렬하여 반환합니다."""
    try:
        from rag.pipeline import search_and_rerank
        results = search_and_rerank(query)
        if not results:
            return "관련 문서를 찾을 수 없습니다."
        parts = []
        for r in results:
            text     = r.get("text", "")      if isinstance(r, dict) else str(r)
            filename = r.get("filename", "")  if isinstance(r, dict) else ""
            cidx     = r.get("chunk_index", 0) if isinstance(r, dict) else 0
            parts.append(f"[출처: {filename} 청크{cidx+1}]\n{text}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"문서 검색 오류: {str(e)}"


@mcp_server.tool()
def query_database(question: str) -> str:
    """자연어 질문을 SQL로 변환하여 데이터베이스를 조회합니다."""
    try:
        from agent.db_agent import generate_and_execute_query
        sql, rows = generate_and_execute_query(question)
        if not rows:
            return f"조회 결과 없음\nSQL: {sql}"
        return f"SQL: {sql}\n결과: {rows[:10]}"
    except Exception as e:
        return f"DB 조회 오류: {str(e)}"


@mcp_server.tool()
def get_schema() -> str:
    """현재 데이터베이스의 테이블 스키마를 반환합니다."""
    try:
        from agent.db_agent import get_db_schema
        return get_db_schema()
    except Exception as e:
        return f"스키마 조회 오류: {str(e)}"



@mcp_server.tool()
def call_rest_api_sample(user_id: str, name: str) -> str:
    """POST 방식 REST API 호출 샘플"""
    import httpx

    base_url = "https://your-api-server.com/api"

    # 헤더 설정
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_TOKEN",   # 필요 시
    }

    # POST body
    payload = {
        "userId": user_id,
        "name": name,
    }

    try:
        resp = httpx.post(
            f"{base_url}/endpoint",
            headers=headers,
            json=payload,      # ← dict를 JSON body로 전송
            timeout=10,
        )
        if resp.status_code != 200:
            return f"API 호출 실패: HTTP {resp.status_code}\n{resp.text}"

        data = resp.json()
        return str(data)

    except httpx.TimeoutException:
        return "REST API 호출 timeout"
    except Exception as e:
        return f"REST API 호출 오류: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    port = config.MCP_SERVER_PORT
    print(f"[MCP Server] SSE 모드로 시작: http://localhost:{port}/sse")
    # mcp_server.run(transport="sse") 가 port 파라미터를 지원하지 않는 경우
    # uvicorn 으로 직접 sse_app 을 실행
    uvicorn.run(mcp_server.sse_app(), host="127.0.0.1", port=port)
