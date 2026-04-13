"""
call_mcp/mcp_server_stdio.py
- stdio 방식 MCP 서버 (UI MCP 경유 토글용)
- FastMCP 가 transport="stdio" 실행 시 sys.stdout 을 직접 사용하므로
  임포트 구간에서만 stdout 을 차단하고 run() 직전에 복원
"""

import sys
import os

# ── 임포트 중 stdout 차단 ─────────────────────────────────────────────
# 모듈 임포트 시 발생하는 print/logging 이 MCP 프로토콜 stdout 을 오염시키는 것을 방지
_real_stdout = sys.stdout
sys.stdout   = open(os.devnull, "w")   # 임포트 구간만 /dev/null 로

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

# stdout 복원 (FastMCP 내부에서 sys.stdout 을 직접 사용)
sys.stdout = _real_stdout

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
    # 서버 시작 시 임베딩 모델 미리 로드 → 첫 도구 호출 시 timeout 방지
    # (bge-m3 로드에 수십 초 소요, 미리 해두면 call_tool 응답이 빠름)
    try:
        from rag.embeddings import get_embeddings
        get_embeddings()
        print("[MCP Stdio] 임베딩 모델 로드 완료", file=sys.stderr)
    except Exception as e:
        print(f"[MCP Stdio] 임베딩 모델 로드 실패 (무시): {e}", file=sys.stderr)

    mcp_stdio.run(transport="stdio")
