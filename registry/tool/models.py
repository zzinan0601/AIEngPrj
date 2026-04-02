# registry/tool/models.py
"""
Tool Registry 데이터 모델

Agent가 사용할 '툴'의 정의를 저장한다.

툴 종류 예:
- document_search
- db_query
- log_search
- external_api (MCP)

나중에 UI에서 사용자가
툴을 등록/수정/삭제 가능하도록 설계됨.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# -------------------------------
# Tool 파라미터 정의
# -------------------------------
@dataclass
class ToolParameter:
    """
    툴 입력 파라미터 정의

    예:
    query: str
    top_k: int
    """

    name: str              # 파라미터 이름
    type: str              # string / int / float / bool
    description: str       # LLM에게 설명 제공
    required: bool = True  # 필수 여부


# -------------------------------
# Tool 정의
# -------------------------------
@dataclass
class ToolDefinition:
    """
    시스템에 등록되는 Tool 전체 정의
    """

    tool_id: str                # 내부 고유 ID
    name: str                   # LLM에게 보여줄 이름
    description: str            # LLM에게 보여줄 설명

    # tool 종류
    # internal : 시스템 내부 함수
    # mcp      : MCP 서버 호출
    tool_type: str              # "internal" | "mcp"

    # MCP용 필드 (internal이면 None)
    mcp_server: Optional[str] = None
    mcp_endpoint: Optional[str] = None

    # 입력 파라미터 정의
    parameters: Dict[str, ToolParameter] = field(default_factory=dict)

    # 활성화 여부 (툴 ON/OFF 기능 대비)
    enabled: bool = True