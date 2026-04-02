"""
registry/prompt/service.py

Prompt Registry의 비즈니스 로직 레이어.
프롬프트 버전 관리 / 롤백 / Few-shot 관리 기능을 담당한다.
"""

from datetime import datetime
from typing import List, Optional

from .models import Prompt, FewShotExample
from .repository import PromptRepository


class PromptService:
    """
    Prompt Registry 핵심 서비스.

    Agent들은 이 Service를 통해서만 프롬프트를 사용한다.
    """

    def __init__(self, repo: PromptRepository):
        self.repo = repo

    # -------------------------------------------------
    # Prompt 생성 / 수정 (버전 관리 핵심)
    # -------------------------------------------------

    def create_or_update_prompt(self, name: str, content: str):
        """
        프롬프트 생성 또는 수정.

        규칙:
        - 기존 프롬프트가 없으면 version=1 생성
        - 기존 프롬프트가 있으면 version+1 새 레코드 생성
        """

        existing = self.repo.get_active_prompt(name)

        now = datetime.utcnow()

        if existing is None:
            # 최초 생성
            new_prompt = Prompt(
                id=None,
                name=name,
                content=content,
                version=1,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            self.repo.create_prompt(new_prompt)
            return new_prompt

        # 기존 버전 비활성화
        self._deactivate_prompt(name)

        # 새 버전 생성
        new_prompt = Prompt(
            id=None,
            name=name,
            content=content,
            version=existing.version + 1,
            is_active=True,
            created_at=now,
            updated_at=now,
        )

        self.repo.create_prompt(new_prompt)
        return new_prompt

    def get_prompt(self, name: str) -> Optional[Prompt]:
        """현재 활성 프롬프트 조회"""
        return self.repo.get_active_prompt(name)

    # -------------------------------------------------
    # 롤백 기능
    # -------------------------------------------------

    def rollback_prompt(self, name: str, target_version: int):
        """
        특정 버전으로 롤백.

        1) 현재 active 비활성화
        2) target_version 활성화
        """
        conn = self.repo._connect()
        cursor = conn.cursor()

        # 기존 active 비활성화
        cursor.execute("UPDATE prompts SET is_active=0 WHERE name=?", (name,))

        # 특정 버전 활성화
        cursor.execute(
            "UPDATE prompts SET is_active=1 WHERE name=? AND version=?",
            (name, target_version),
        )

        conn.commit()
        conn.close()

    def _deactivate_prompt(self, name: str):
        """현재 활성 프롬프트 비활성화 (내부용)"""
        conn = self.repo._connect()
        cursor = conn.cursor()
        cursor.execute("UPDATE prompts SET is_active=0 WHERE name=?", (name,))
        conn.commit()
        conn.close()

    # -------------------------------------------------
    # Few-shot 관리
    # -------------------------------------------------

    def add_few_shot(self, prompt_name: str, input_text: str, output_text: str):
        """Few-shot 예시 추가"""
        example = FewShotExample(
            id=None,
            prompt_name=prompt_name,
            input_text=input_text,
            output_text=output_text,
            created_at=datetime.utcnow(),
        )
        self.repo.add_few_shot(example)

    def get_few_shots(self, prompt_name: str) -> List[FewShotExample]:
        """Few-shot 목록 조회"""
        return self.repo.get_few_shots(prompt_name)