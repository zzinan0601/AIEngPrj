"""
call_mcp/mcp_server_stdio.py
- UI의 'MCP 경유 호출' 토글용 stdio 방식 MCP 서버
- mcp_server.py (SSE) 는 그대로 유지하고 이 파일을 추가로 사용
- 핵심: 시작 즉시 stdout → stderr 리다이렉트
  → 모듈 임포트 시 출력되는 모든 텍스트가 MCP 프로토콜을 오염시키지 않음
- 클라이언트(mcp_client.py)가 subprocess로 이 파일을 직접 실행
"""

import sys
import os

# ── stdout 즉시 차단 (MCP 프로토콜 보호) ──────────────────────────────
# 이 줄 이후의 모든 print/logging stdout 출력이 stderr로 리다이렉트됨
# MCP stdio 프로토콜은 stdout만 사용하므로 stdout을 깨끗하게 유지해야 함
_original_stdout = sys.stdout
sys.stdout = sys.stderr

# 이제부터 임포트해도 stdout 오염 없음
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

mcp_stdio = FastMCP(
    name="RAG-Agent-Stdio",
    instructions="stdio 방식 RAG 문서 검색과 DB 조회 MCP 서버",
)


@mcp_stdio.tool()
def search_documents(query: str) -> str:
    """업로드된 문서에서 관련 내용을 벡터 검색 + 재정렬하여 반환합니다."""
    try:
        from rag.pipeline import search_and_rerank
        results = search_and_rerank(query)
        if not results:
            return "관련 문서를 찾을 수 없습니다."
        parts = []
        for r in results:
            text     = r.get("text", "")       if isinstance(r, dict) else str(r)
            filename = r.get("filename", "")   if isinstance(r, dict) else ""
            cidx     = r.get("chunk_index", 0) if isinstance(r, dict) else 0
            parts.append(f"[출처: {filename} 청크{cidx+1}]\n{text}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"문서 검색 오류: {str(e)}"


@mcp_stdio.tool()
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


@mcp_stdio.tool()
def get_schema() -> str:
    """현재 데이터베이스의 테이블 스키마를 반환합니다."""
    try:
        from agent.db_agent import get_db_schema
        return get_db_schema()
    except Exception as e:
        return f"스키마 조회 오류: {str(e)}"


if __name__ == "__main__":
    # stdout 을 원래대로 복원 후 MCP stdio 프로토콜 시작
    # (FastMCP 내부에서 sys.stdout 을 직접 사용)
    sys.stdout = _original_stdout
    mcp_stdio.run(transport="stdio")
