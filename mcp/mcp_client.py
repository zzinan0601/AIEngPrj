"""
mcp/mcp_client.py
- MCP 서버(mcp_server.py)에 연결하는 클라이언트
- stdio 전송 방식: 서버를 subprocess로 실행하고 stdin/stdout으로 통신
- Streamlit 호환: 새 스레드에서 async 이벤트 루프를 실행해 동기 함수처럼 사용
"""

import sys
import os
import asyncio
import threading

# 프로젝트 루트 경로
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER_SCRIPT = os.path.join(_PROJECT_ROOT, "mcp", "mcp_server.py")


def _run_in_new_loop(coro):
    """
    새 이벤트 루프를 가진 스레드에서 async 코루틴을 실행.
    Streamlit 등 이미 이벤트 루프가 있는 환경에서도 안전하게 동작.
    """
    result_holder = [None]
    error_holder = [None]

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
    t.join(timeout=60)  # 최대 60초 대기

    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0]


async def _call_tool_async(tool_name: str, arguments: dict) -> str:
    """MCP 서버에 도구 호출 요청을 보내고 결과 반환 (async)"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,   # 현재 Python 인터프리터
        args=[_SERVER_SCRIPT],    # mcp_server.py를 subprocess로 실행
        env=None,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            # content[0].text 에 결과가 있음
            if result.content:
                return result.content[0].text
            return ""


def call_tool(tool_name: str, arguments: dict) -> str:
    """MCP 도구를 동기 방식으로 호출 (외부에서 사용하는 공개 함수)"""
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
