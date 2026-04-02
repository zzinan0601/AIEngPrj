# tool_executor.py
"""
Tool 실행기

Agent가 선택한 Tool을 실제 실행한다.
- Internal Tool 실행
- MCP Tool 호출 (다음 단계에서 연결)
"""

from typing import Dict, Any
from registry.tool.service import ToolService
from registry.tool.models import ToolDefinition


class ToolExecutor:
    def __init__(self, tool_service: ToolService):
        self.tool_service = tool_service

        # 내부 툴 실행 함수 매핑
        # tool_id -> function
        self.internal_tool_map = {}

    # -------------------------------
    # Internal Tool 등록 (코드 함수 연결)
    # -------------------------------
    def register_internal_function(self, tool_id: str, func):
        """
        내부 Tool을 실제 파이썬 함수와 연결

        예:
        executor.register_internal_function(
            "document_search",
            document_search_function
        )
        """
        self.internal_tool_map[tool_id] = func

    # -------------------------------
    # Tool 실행 진입점
    # -------------------------------
    def execute(self, tool_id: str, params: Dict[str, Any]) -> Any:
        """
        Agent가 Tool 실행 요청하는 메인 함수
        """

        tool = self.tool_service.repo.get_tool(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")

        if not tool.enabled:
            raise ValueError(f"Tool disabled: {tool_id}")

        # 파라미터 검증
        self._validate_params(tool, params)

        # Tool 타입 분기
        if tool.tool_type == "internal":
            return self._execute_internal(tool, params)

        elif tool.tool_type == "mcp":
            return self._execute_mcp(tool, params)

        else:
            raise ValueError(f"Unknown tool type: {tool.tool_type}")

    # -------------------------------
    # 파라미터 검증
    # -------------------------------
    def _validate_params(self, tool: ToolDefinition, params: Dict[str, Any]):
        for name, param_def in tool.parameters.items():
            if param_def.required and name not in params:
                raise ValueError(f"Missing required parameter: {name}")

    # -------------------------------
    # Internal Tool 실행
    # -------------------------------
    def _execute_internal(self, tool: ToolDefinition, params: Dict[str, Any]):
        func = self.internal_tool_map.get(tool.tool_id)

        if not func:
            raise ValueError(f"Internal function not registered: {tool.tool_id}")

        return func(**params)

    # -------------------------------
    # MCP Tool 실행 (다음 파일에서 구현)
    # -------------------------------
    def _execute_mcp(self, tool: ToolDefinition, params: Dict[str, Any]):
        """
        MCP Client 호출
        다음 단계에서 구현됨
        """
        raise NotImplementedError("MCP execution not implemented yet")