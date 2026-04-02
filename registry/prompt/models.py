"""
registry/prompt/models.py

Prompt Registry에서 사용하는 데이터 모델 정의 파일.

이 파일은 '프롬프트를 DB에 저장하기 위한 스키마 역할'을 한다.
앞으로 Agent들은 실행 시 이 모델을 통해 프롬프트를 로드하게 된다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Prompt:
    """
    하나의 프롬프트 레코드를 표현하는 모델.

    예:
    - docs_agent_system
    - db_agent_system
    - supervisor_prompt
    """

    id: Optional[int]          # DB PK (자동 증가 예정)
    name: str                  # 프롬프트 고유 이름 (유니크 키)
    content: str               # 실제 프롬프트 텍스트
    version: int               # 버전 관리 (수정 시 증가)
    is_active: bool            # 현재 사용 중인 버전인지 여부
    created_at: datetime       # 생성 시간
    updated_at: datetime       # 마지막 수정 시간


@dataclass
class FewShotExample:
    """
    Few-shot 예시 데이터 모델.

    하나의 Prompt에는 여러 개의 Few-shot 예시가 연결될 수 있다.
    Agent 실행 시 Prompt + Few-shot을 함께 로드한다.
    """

    id: Optional[int]          # DB PK
    prompt_name: str           # 어떤 Prompt에 속하는지 (Prompt.name 참조)
    input_text: str            # 예시 입력 (사용자 질문)
    output_text: str           # 예시 출력 (모범 답변)
    created_at: datetime       # 생성 시간