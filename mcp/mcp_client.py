"""
mcp/mcp_client.py
- MCP 서버(mcp_server.py)에 연결하는 클라이언트
- stdio 전송 방식: 서버를 subprocess로 실행하고 stdin/stdout으로 통신
- Streamlit 호환: 새 스레드에서 async 이벤트 루프를 실행해 동기 함수처럼 사용

[디버깅 주의]
- async 함수 안의 print 는 daemon 스레드에서 실행되므로 Streamlit 화면에 안 찍힘
- 로그는 result_holder / error_holder 로 메인 스레드에서 확인
- result 객체는 str 이 아니므로 "str" + result 는 TypeError → str(result) 사용
"""

import sys
import os
import asyncio
import threading
import logging

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER_SCRIPT = os.path.join(_PROJECT_ROOT, "mcp", "mcp_server.py")


def _run_in_new_loop(coro):
    """
    새 이벤트 루프를 가진 스레드에서 async 코루틴을 실행.
    Streamlit 등 이미 이벤트 루프가 있는 환경에서도 안전하게 동작.
    """
    result_holder = [None]
    error_holder  = [None]

    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder[0] = loop.run_until_complete(coro)
        except Exception as e:
            error_holder[0] = e
        finally:
            loop.close()

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()
    t.join(timeout=60)

    if error_holder[0]:
        # 메인 스레드에서 로그 출력 (Streamlit 콘솔에 표시됨)
        logger.error(f"[MCP] 오류: {error_holder[0]}")
        raise error_holder[0]

    logger.debug(f"[MCP] 결과: {result_holder[0]}")
    return result_holder[0]


def _extract_text(result) -> str:
    """
    call_tool 반환값에서 텍스트를 안전하게 추출.
    MCP 버전마다 구조가 다를 수 있으므로 여러 방식으로 시도.
    """
    if result is None:
        return ""

    # 방식 1: result.content 리스트
    content = getattr(result, "content", None)
    if content:
        parts = []
        for item in content:
            # TextContent 타입
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
            # dict 형태
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
        if parts:
            return "\n".join(parts)

    # 방식 2: result 자체가 문자열
    if isinstance(result, str):
        return result

    # 방식 3: str 변환
    return str(result)


async def _call_tool_async(tool_name: str, arguments: dict) -> str:
    """MCP 서버에 도구 호출 요청을 보내고 결과 반환 (async)"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[_SERVER_SCRIPT],
        env=None,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

            # async 안에서는 logger 사용 (daemon 스레드지만 메인 루프와 공유)
            logger.debug(f"[MCP] raw result type: {type(result)}")
            logger.debug(f"[MCP] raw result: {str(result)[:200]}")

            return _extract_text(result)


def call_tool(tool_name: str, arguments: dict) -> str:
    """MCP 도구를 동기 방식으로 호출"""
    return _run_in_new_loop(_call_tool_async(tool_name, arguments))


def search_via_mcp(query: str) -> str:
    """MCP를 통해 문서 검색 실행"""
    return call_tool("search_documents", {"query": query})


def query_db_via_mcp(question: str) -> str:
    """MCP를 통해 DB 조회 실행"""
    return call_tool("query_database", {"question": question})


def list_mcp_tools() -> list:
    """MCP 서버에 등록된 도구 목록 반환"""
    async def _list():
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[_SERVER_SCRIPT],
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [
                    {"name": t.name, "description": t.description}
                    for t in tools.tools
                ]

    return _run_in_new_loop(_list())


def test_mcp_connection() -> dict:
    """
    MCP 연결 및 도구 호출 테스트.
    UI 또는 콘솔에서 직접 호출해 결과 확인 가능.
    예: from mcp.mcp_client import test_mcp_connection; print(test_mcp_connection())
    """
    result = {"tools": [], "test_call": "", "error": ""}
    try:
        result["tools"] = list_mcp_tools()
        if result["tools"]:
            # 첫 번째 도구로 테스트 호출
            first_tool = result["tools"][0]["name"]
            result["test_call"] = call_tool(first_tool, {})
    except Exception as e:
        result["error"] = str(e)
    return result
