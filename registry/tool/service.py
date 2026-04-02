# registry/tool/service.py
"""
Tool Registry 서비스 레이어

Repository 위에서 동작하는 비즈니스 로직 담당
- Tool 등록 검증
- Tool 타입별 처리
- Agent용 Tool 목록 제공
"""

from typing import List
from .models import ToolDefinition
from .repository import ToolRepository


class ToolService:
    def __init__(self, repo: ToolRepository):
        self.repo = repo

    # -------------------------------
    # Tool 등록
    # -------------------------------
    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Tool 등록 시 검증 수행
        """

        # Tool 타입 검증
        if tool.tool_type not in ["internal", "mcp"]:
            raise ValueError("tool_type must be 'internal' or 'mcp'")

        # MCP Tool 필수값 검증
        if tool.tool_type == "mcp":
            if not tool.mcp_server or not tool.mcp_endpoint:
                raise ValueError("MCP tool must have mcp_server and mcp_endpoint")

        # 파라미터 이름 중복 검사
        param_names = list(tool.parameters.keys())
        if len(param_names) != len(set(param_names)):
            raise ValueError("Duplicate parameter names detected")

        self.repo.add_tool(tool)

    # -------------------------------
    # Tool 수정
    # -------------------------------
    def update_tool(self, tool_id: str, updated: ToolDefinition) -> None:
        """
        Tool 수정 시 동일한 검증 수행
        """
        self.register_tool_validation(updated)
        self.repo.update_tool(tool_id, updated)

    def register_tool_validation(self, tool: ToolDefinition):
        if tool.tool_type not in ["internal", "mcp"]:
            raise ValueError("tool_type must be 'internal' or 'mcp'")

        if tool.tool_type == "mcp":
            if not tool.mcp_server or not tool.mcp_endpoint:
                raise ValueError("MCP tool must have mcp_server and mcp_endpoint")

    # -------------------------------
    # Tool 삭제
    # -------------------------------
    def delete_tool(self, tool_id: str) -> None:
        self.repo.delete_tool(tool_id)

    # -------------------------------
    # Agent용 Tool 목록
    # -------------------------------
    def get_available_tools(self) -> List[ToolDefinition]:
        """
        Agent가 사용할 Tool 목록 반환
        (활성화된 Tool만)
        """
        return self.repo.list_tools(enabled_only=True)