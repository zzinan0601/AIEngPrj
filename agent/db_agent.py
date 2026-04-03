"""
agent/db_agent.py
- 자연어 질문을 SQL로 변환하고 SQLite DB를 조회
- DB 스키마는 Qdrant 벡터 DB에서 검색해 LLM 프롬프트에 주입
- 결과: (생성된 SQL, 조회 결과 rows) 튜플 반환
"""

import sqlite3
import re
import config
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


def get_llm() -> ChatOllama:
    """LLM 인스턴스 반환"""
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.0,  # SQL 생성은 결정론적으로
    )


def get_db_schema() -> str:
    """
    SQLite DB의 실제 테이블/컬럼 스키마를 문자열로 반환.
    DB가 없거나 테이블이 없으면 빈 문자열 반환.
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()

        # sqlite_master에서 테이블 목록 조회
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        schema_lines = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            col_defs = ", ".join(
                [f"{col[1]} {col[2]}" for col in columns]
            )
            schema_lines.append(f"테이블: {table} ({col_defs})")

        conn.close()
        return "\n".join(schema_lines) if schema_lines else "DB 테이블 없음"
    except Exception as e:
        return f"스키마 조회 실패: {str(e)}"


def get_schema_from_vector(question: str) -> str:
    """
    Qdrant 스키마 컬렉션에서 질문과 관련된 스키마 검색.
    벡터 DB에 스키마가 없으면 실제 DB 스키마를 사용.
    """
    try:
        from rag.vector_store import search_documents
        results = search_documents(
            question,
            top_k=3,
            collection_name=config.SCHEMA_COLLECTION,
        )
        if results:
            return "\n".join(results)
    except Exception:
        pass
    # 폴백: 실제 DB에서 직접 스키마 읽기
    return get_db_schema()


def clean_sql(raw_sql: str) -> str:
    """LLM 응답에서 순수 SQL만 추출"""
    # 마크다운 코드블록 제거
    raw_sql = re.sub(r"```(?:sql)?", "", raw_sql, flags=re.IGNORECASE)
    raw_sql = raw_sql.replace("```", "")
    # 앞뒤 공백 제거 후 첫 번째 SQL 문만 추출
    lines = [line.strip() for line in raw_sql.strip().splitlines() if line.strip()]
    sql = " ".join(lines)
    # SELECT로 시작하는 부분만 추출 (안전)
    match = re.search(r"(SELECT\s.+)", sql, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else sql.strip()


def generate_sql(question: str, schema_context: str) -> str:
    """LLM을 이용해 자연어 → SQLite SQL 변환"""
    llm = get_llm()

    system_prompt = f"""당신은 SQLite SQL 전문가입니다.
아래 DB 스키마를 참고하여 질문에 맞는 SELECT SQL 쿼리를 생성하세요.

[DB 스키마]
{schema_context}

규칙:
- SELECT 문만 생성 (INSERT/UPDATE/DELETE 금지)
- SQLite 문법 사용
- SQL만 출력 (설명 없이)"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"질문: {question}"),
    ])
    return clean_sql(response.content)


def execute_sql(sql: str) -> list:
    """SQL을 실행하고 결과를 dict 리스트로 반환"""
    conn = sqlite3.connect(config.DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        # 컬럼명과 함께 dict로 변환
        return [dict(zip(columns, row)) for row in rows]
    except sqlite3.Error as e:
        raise RuntimeError(f"SQL 실행 오류: {str(e)}\nSQL: {sql}")
    finally:
        conn.close()


def generate_and_execute_query(question: str) -> tuple:
    """
    질문 → SQL 생성 → 실행 → 결과 반환
    Returns: (sql 문자열, 결과 rows 리스트)
    """
    schema = get_schema_from_vector(question)
    sql = generate_sql(question, schema)
    rows = execute_sql(sql)
    return sql, rows


def embed_db_schema():
    """
    현재 SQLite DB의 스키마를 Qdrant 스키마 컬렉션에 임베딩하여 저장.
    UI에서 '스키마 임베딩' 버튼 클릭 시 호출된다.
    """
    from rag.vector_store import save_chunks

    schema_text = get_db_schema()
    if "DB 테이블 없음" in schema_text or "실패" in schema_text:
        return 0, schema_text

    # 테이블별로 분리해서 저장
    chunks = schema_text.splitlines()
    chunks = [c for c in chunks if c.strip()]

    count = save_chunks(
        chunks=chunks,
        metadata={"source": "db_schema", "filename": "db_schema"},
        collection_name=config.SCHEMA_COLLECTION,
    )
    return count, schema_text
