"""
call_mcp/mcp_client.py
- MCP 서버에 SSE(HTTP) 방식으로 연결
- 서버는 별도 터미널에서 python mcp_server.py 로 먼저 실행해야 함
- stdio 방식 대비 장점:
    · 서버가 한 번만 뜨므로 모듈 로드 1회
    · stdout 오염으로 인한 프로토콜 파괴 없음
    · 도구 호출마다 subprocess 생성 오버헤드 없음
"""

import sys
import os
import asyncio
import threading
import logging

import config

logger = logging.getLogger(__name__)

# MCP 서버 SSE 엔드포인트
_SERVER_URL = f"http://localhost:{config.MCP_SERVER_PORT}/sse"


def _run_in_new_loop(coro):
    """
    새 이벤트 루프를 가진 스레드에서 async 코루틴을 실행.
    Python 3.11+ ExceptionGroup 안전 처리 포함.
    """
    result_holder = [None]
    error_holder  = [None]

    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder[0] = loop.run_until_complete(coro)
        except BaseException as e:
            eg_type = type(e).__name__
            if eg_type in ("ExceptionGroup", "BaseExceptionGroup"):
                real = [
                    ex for ex in e.exceptions
                    if not isinstance(ex, (asyncio.CancelledError, GeneratorExit))
                ]
                if real:
                    error_holder[0] = real[0]
            else:
                error_holder[0] = e
        finally:
            loop.close()

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()
    t.join(timeout=30)   # SSE는 빠르므로 30초면 충분

    if not t.is_alive() and result_holder[0] is None and error_holder[0] is None:
        error_holder[0] = TimeoutError("MCP 서버 응답 없음 (30초 초과) - 서버가 실행 중인지 확인하세요")

    if error_holder[0]:
        logger.error(f"[MCP] 오류: {error_holder[0]}")
        raise error_holder[0]

    return result_holder[0]


def _extract_text(result) -> str:
    """call_tool 반환값에서 텍스트를 안전하게 추출"""
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if content:
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
        if parts:
            return "\n".join(parts)
    if isinstance(result, str):
        return result
    return str(result)


async def _call_tool_async(tool_name: str, arguments: dict) -> str:
    """SSE 방식으로 MCP 서버에 도구 호출"""
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _extract_text(result)


async def _list_tools_async() -> list:
    """SSE 방식으로 도구 목록 조회"""
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [
                {"name": t.name, "description": t.description}
                for t in tools.tools
            ]


def call_tool(tool_name: str, arguments: dict) -> str:
    """MCP 도구를 동기 방식으로 호출"""
    return _run_in_new_loop(_call_tool_async(tool_name, arguments))


def list_mcp_tools() -> list:
    """MCP 서버에 등록된 도구 목록 반환"""
    return _run_in_new_loop(_list_tools_async())


def search_via_mcp(query: str) -> str:
    """MCP를 통해 문서 검색 실행"""
    return call_tool("search_documents", {"query": query})


def query_db_via_mcp(question: str) -> str:
    """MCP를 통해 DB 조회 실행"""
    return call_tool("query_database", {"question": question})


def test_mcp_connection() -> dict:
    """
    MCP 연결 테스트.
    실행: python -c "from call_mcp.mcp_client import test_mcp_connection; print(test_mcp_connection())"
    """
    result = {
        "server_url": _SERVER_URL,
        "tools":      [],
        "get_schema": "",
        "error":      "",
    }
    try:
        print(f"[MCP TEST] 서버 URL: {_SERVER_URL}")
        result["tools"] = list_mcp_tools()
        print(f"[MCP TEST] 도구 목록: {[t['name'] for t in result['tools']]}")

        print(f"[MCP TEST] get_schema 호출 중...")
        result["get_schema"] = call_tool("get_schema", {})
        print(f"[MCP TEST] get_schema 결과: {str(result['get_schema'])[:200]}")

    except Exception as e:
        result["error"] = str(e)
        print(f"[MCP TEST] 오류: {e}")

    return result


# ── stdio 방식 클라이언트 (UI MCP 경유 토글용) ─────────────────────────────
_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STDIO_SERVER   = os.path.join(_PROJECT_ROOT, "call_mcp", "mcp_server_stdio.py")


async def _call_tool_stdio_async(tool_name: str, arguments: dict) -> str:
    """stdio 방식으로 MCP 서버 subprocess 를 띄워 도구 호출"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import subprocess

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[_STDIO_SERVER],
        env=None,
        stderr=subprocess.PIPE,   # DEVNULL → PIPE 로 변경해 오류 캡처
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

            # 디버그: result 전체 구조 stderr 로 출력
            print(f"[STDIO DEBUG] result type : {type(result)}", file=sys.stderr)
            print(f"[STDIO DEBUG] result raw  : {str(result)[:300]}", file=sys.stderr)
            print(f"[STDIO DEBUG] isError     : {getattr(result, 'isError', 'N/A')}", file=sys.stderr)

            content = getattr(result, "content", None)
            print(f"[STDIO DEBUG] content     : {content}", file=sys.stderr)

            if content:
                for i, item in enumerate(content):
                    print(f"[STDIO DEBUG] content[{i}]: type={type(item)}, "
                          f"text={str(getattr(item,'text',''))[:200]}", file=sys.stderr)

            extracted = _extract_text(result)
            print(f"[STDIO DEBUG] extracted   : {extracted[:200]}", file=sys.stderr)
            return extracted


def call_tool_stdio(tool_name: str, arguments: dict) -> str:
    """stdio 방식 도구 호출 (동기)"""
    return _run_in_new_loop(_call_tool_stdio_async(tool_name, arguments))


def search_via_mcp(query: str) -> str:
    """
    MCP를 통해 문서 검색.
    SSE 서버가 떠있으면 SSE 방식, 아니면 stdio 방식으로 자동 폴백.
    """
    try:
        return call_tool("search_documents", {"query": query})
    except Exception:
        return call_tool_stdio("search_documents", {"query": query})


def query_db_via_mcp(question: str) -> str:
    """
    MCP를 통해 DB 조회.
    SSE 서버가 떠있으면 SSE 방식, 아니면 stdio 방식으로 자동 폴백.
    """
    try:
        return call_tool("query_database", {"question": question})
    except Exception:
        return call_tool_stdio("query_database", {"question": question})
