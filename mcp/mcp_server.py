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
def call_rest_api_sample(user_id: str) -> str:
    """
    REST API 호출 샘플 툴.
    JSONPlaceholder(https://jsonplaceholder.typicode.com) 의 무료 테스트 API를
    호출해서 사용자 정보와 해당 사용자의 게시물 목록을 반환합니다.

    Args:
        user_id: 조회할 사용자 ID (1~10)

    실제 사용 시 이 함수를 복사해서 URL/헤더/인증 등을 수정하세요.
    """
    import httpx

    base_url = "https://jsonplaceholder.typicode.com"

    try:
        # 1. 사용자 정보 조회
        user_resp = httpx.get(f"{base_url}/users/{user_id}", timeout=10)
        if user_resp.status_code != 200:
            return f"사용자 조회 실패: HTTP {user_resp.status_code}"
        user = user_resp.json()

        # 2. 해당 사용자의 게시물 목록 조회
        posts_resp = httpx.get(
            f"{base_url}/posts",
            params={"userId": user_id},
            timeout=10,
        )
        posts = posts_resp.json() if posts_resp.status_code == 200 else []

        # 3. 결과 포매팅
        result_lines = [
            f"[사용자 정보]",
            f"이름: {user.get('name')}",
            f"이메일: {user.get('email')}",
            f"회사: {user.get('company', {}).get('name')}",
            f"",
            f"[게시물 목록] 총 {len(posts)}건",
        ]
        for p in posts[:5]:   # 최대 5건만 표시
            result_lines.append(f"- [{p['id']}] {p['title']}")
        if len(posts) > 5:
            result_lines.append(f"  ... 외 {len(posts)-5}건")

        return "\n".join(result_lines)

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
