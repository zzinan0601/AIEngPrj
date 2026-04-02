"""
registry/prompt/repository.py

Prompt / Few-shot 데이터를 SQLite DB에 저장하고 조회하는 레이어.
DB와 직접 통신하는 코드는 이 파일에만 존재해야 한다.
"""

import sqlite3
from datetime import datetime
from typing import List, Optional

from .models import Prompt, FewShotExample


class PromptRepository:
    """
    Prompt 테이블 접근 전용 Repository.
    모든 DB 쿼리는 여기서만 수행한다.
    """

    def __init__(self, db_path: str = "platform.db"):
        self.db_path = db_path
        self._create_tables()

    def _connect(self):
        """DB 연결 생성"""
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        """최초 실행 시 테이블 생성"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            content TEXT,
            version INTEGER,
            is_active INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS few_shots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_name TEXT,
            input_text TEXT,
            output_text TEXT,
            created_at TEXT
        )
        """)

        conn.commit()
        conn.close()

    # ---------------------------
    # Prompt CRUD
    # ---------------------------

    def create_prompt(self, prompt: Prompt):
        """프롬프트 신규 저장"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO prompts (name, content, version, is_active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            prompt.name,
            prompt.content,
            prompt.version,
            int(prompt.is_active),
            prompt.created_at.isoformat(),
            prompt.updated_at.isoformat()
        ))

        conn.commit()
        conn.close()

    def get_active_prompt(self, name: str) -> Optional[Prompt]:
        """현재 활성화된 프롬프트 조회"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT * FROM prompts WHERE name=? AND is_active=1
        """, (name,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Prompt(
            id=row[0],
            name=row[1],
            content=row[2],
            version=row[3],
            is_active=bool(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
        )

    # ---------------------------
    # Few-shot CRUD
    # ---------------------------

    def add_few_shot(self, example: FewShotExample):
        """Few-shot 예시 추가"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO few_shots (prompt_name, input_text, output_text, created_at)
        VALUES (?, ?, ?, ?)
        """, (
            example.prompt_name,
            example.input_text,
            example.output_text,
            example.created_at.isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_few_shots(self, prompt_name: str) -> List[FewShotExample]:
        """특정 프롬프트의 Few-shot 목록 조회"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT * FROM few_shots WHERE prompt_name=?
        """, (prompt_name,))

        rows = cursor.fetchall()
        conn.close()

        examples = []
        for row in rows:
            examples.append(FewShotExample(
                id=row[0],
                prompt_name=row[1],
                input_text=row[2],
                output_text=row[3],
                created_at=datetime.fromisoformat(row[4]),
            ))

        return examples