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
_SERVER_SCRIPT = os.path.join(_PROJECT_ROOT, "call_mcp", "mcp_server.py")


def _run_in_new_loop(coro):
    """
    새 이벤트 루프를 가진 스레드에서 async 코루틴을 실행.
    Python 3.11+ ExceptionGroup (TaskGroup cleanup 에러) 안전 처리.
    """
    result_holder = [None]
    error_holder  = [None]

    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder[0] = loop.run_until_complete(coro)
        except BaseException as e:
            # Python 3.11+ TaskGroup 에러는 ExceptionGroup 으로 래핑됨
            eg_type = type(e)
            if eg_type.__name__ in ("ExceptionGroup", "BaseExceptionGroup"):
                # cleanup 과정의 CancelledError 는 무해 → 무시
                real = [
                    ex for ex in e.exceptions
                    if not isinstance(ex, (asyncio.CancelledError, GeneratorExit))
                ]
                if real:
                    error_holder[0] = real[0]
                # real 이 없으면 cleanup 노이즈 → 무시 (result_holder 는 이미 채워짐)
            else:
                error_holder[0] = e
        finally:
            loop.close()

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()
    t.join(timeout=60)

    if error_holder[0]:
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
    예: from mcp.mcp_client import test_mcp_connection; print(test_mcp_connection())
    """
    import json

    result = {
        "tools":        [],
        "test_tool":    "",
        "test_call":    "",
        "raw_result":   "",   # content 원시 구조
        "error":        "",
    }
    try:
        # 1. 도구 목록 확인
        result["tools"] = list_mcp_tools()
        print(f"[MCP TEST] 도구 목록: {result['tools']}")

        if not result["tools"]:
            result["error"] = "도구 목록이 비어있음 - 서버 연결 실패 가능성"
            return result

        # 2. search_documents 로 테스트 (query 인자 포함)
        result["test_tool"] = "search_documents"
        raw = _run_in_new_loop(
            _call_tool_async_debug("search_documents", {"query": "테스트"})
        )
        # raw 가 None 이면 스레드 timeout 또는 내부 예외
        if raw is None:
            result["error"] = "도구 호출 결과가 None - 서버 응답 없음 또는 timeout"
            print(f"[MCP TEST] raw=None: 서버 응답 없음")
            return result

        result["test_call"] = raw.get("text", "")
        result["raw_result"] = raw.get("raw", "")
        print(f"[MCP TEST] raw 결과: {result['raw_result'][:300]}")

    except Exception as e:
        result["error"] = str(e)
        print(f"[MCP TEST] 오류: {e}")

    return result


async def _call_tool_async_debug(tool_name: str, arguments: dict) -> dict:
    """디버그용 - raw result 구조까지 반환"""
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

            raw_str = str(result)
            print(f"[MCP DEBUG] result type : {type(result)}")
            print(f"[MCP DEBUG] result raw  : {raw_str[:500]}")
            print(f"[MCP DEBUG] isError     : {getattr(result, 'isError', 'N/A')}")

            content = getattr(result, "content", None)
            if content:
                print(f"[MCP DEBUG] content len : {len(content)}")
                for i, item in enumerate(content):
                    print(f"[MCP DEBUG] content[{i}] type: {type(item)}, text: {str(getattr(item,'text',''))[:200]}")

            return {
                "text": _extract_text(result),
                "raw":  raw_str,
            }
