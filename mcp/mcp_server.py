"""
mcp/mcp_server.py
- FastMCP를 이용한 MCP(Model Context Protocol) 서버
- 문서 검색, DB 조회 도구를 MCP 프로토콜로 노출
- 실행 방법: python -m mcp.mcp_server  (별도 터미널)
- 클라이언트(mcp_client.py)는 stdio를 통해 이 서버와 통신
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp.server.fastmcp import FastMCP          # mcp >= 1.0
except ImportError:
    from fastmcp import FastMCP                     # fastmcp 독립 패키지 폴백

# MCP 서버 인스턴스 생성
mcp_server = FastMCP(
    name="RAG-Agent-Tools",
    instructions="RAG 문서 검색과 DB 조회를 제공하는 MCP 서버입니다.",
)


# ── 도구 1: 문서 검색 ───────────────────────────────────────────────
@mcp_server.tool()
def search_documents(query: str) -> str:
    """
    업로드된 문서에서 관련 내용을 벡터 검색 + 재정렬하여 반환합니다.

    Args:
        query: 검색할 질문 또는 키워드
    """
    try:
        from rag.pipeline import search_and_rerank
        results = search_and_rerank(query)
        if not results:
            return "관련 문서를 찾을 수 없습니다."
        # search_and_rerank 는 [{text, filename, chunk_index}] 반환
        parts = []
        for r in results:
            text     = r.get("text", "")     if isinstance(r, dict) else str(r)
            filename = r.get("filename", "") if isinstance(r, dict) else ""
            cidx     = r.get("chunk_index", 0) if isinstance(r, dict) else 0
            parts.append(f"[출처: {filename} 청크{cidx+1}]\n{text}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"문서 검색 오류: {str(e)}"


# ── 도구 2: DB 조회 ─────────────────────────────────────────────────
@mcp_server.tool()
def query_database(question: str) -> str:
    """
    자연어 질문을 SQL로 변환하여 데이터베이스를 조회합니다.

    Args:
        question: 데이터를 조회하는 자연어 질문
    """
    try:
        from agent.db_agent import generate_and_execute_query
        sql, rows = generate_and_execute_query(question)
        if not rows:
            return f"조회 결과 없음\nSQL: {sql}"
        return f"SQL: {sql}\n결과: {rows[:10]}"  # 최대 10행
    except Exception as e:
        return f"DB 조회 오류: {str(e)}"


# ── 도구 3: DB 스키마 조회 ──────────────────────────────────────────
@mcp_server.tool()
def get_schema() -> str:
    """현재 데이터베이스의 테이블 스키마를 반환합니다."""
    try:
        from agent.db_agent import get_db_schema
        return get_db_schema()
    except Exception as e:
        return f"스키마 조회 오류: {str(e)}"


if __name__ == "__main__":
    # stdio 모드로 MCP 서버 실행 (클라이언트가 subprocess로 실행)
    mcp_server.run(transport="stdio")
