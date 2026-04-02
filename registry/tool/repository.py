# registry/tool/repository.py
"""
Tool Registry 저장소

툴 등록 / 조회 / 수정 / 삭제 담당
나중에 DB로 교체 가능하도록 Repository 패턴 사용
"""

from typing import Dict, List, Optional
from .models import ToolDefinition


class ToolRepository:
    def __init__(self):
        # tool_id -> ToolDefinition
        self.tools: Dict[str, ToolDefinition] = {}

    # -------------------------------
    # Tool 등록
    # -------------------------------
    def add_tool(self, tool: ToolDefinition) -> None:
        """
        새로운 Tool 등록
        """
        if tool.tool_id in self.tools:
            raise ValueError(f"Tool already exists: {tool.tool_id}")

        self.tools[tool.tool_id] = tool

    # -------------------------------
    # Tool 조회 (단건)
    # -------------------------------
    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """
        tool_id로 Tool 조회
        """
        return self.tools.get(tool_id)

    # -------------------------------
    # Tool 전체 조회
    # -------------------------------
    def list_tools(self, enabled_only: bool = True) -> List[ToolDefinition]:
        """
        등록된 Tool 목록 조회

        enabled_only=True:
            활성화된 Tool만 반환 (Agent 사용용)
        """
        if enabled_only:
            return [t for t in self.tools.values() if t.enabled]
        return list(self.tools.values())

    # -------------------------------
    # Tool 수정
    # -------------------------------
    def update_tool(self, tool_id: str, updated: ToolDefinition) -> None:
        """
        Tool 정보 수정
        """
        if tool_id not in self.tools:
            raise ValueError(f"Tool not found: {tool_id}")

        self.tools[tool_id] = updated

    # -------------------------------
    # Tool 삭제
    # -------------------------------
    def delete_tool(self, tool_id: str) -> None:
        """
        Tool 제거
        """
        if tool_id not in self.tools:
            raise ValueError(f"Tool not found: {tool_id}")

        del self.tools[tool_id]