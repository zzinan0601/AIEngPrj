"""
agent/db_agent.py
- 자연어 질문을 SQL로 변환하고 DB를 조회
- DB 타입: sqlite | postgresql | oracle (session_state 또는 config.DB_TYPE)
- SQLAlchemy로 DB 연결을 통일해 DB 타입별 분기를 최소화
"""

import re
import config
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ── LLM ──────────────────────────────────────────────────────────────
def get_llm(model_name: str = None) -> ChatOllama:
    """LLM 인스턴스 반환 (model_name 미지정 시 config 기본값)"""
    return ChatOllama(
        model=model_name or config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.0,   # SQL 생성은 결정론적으로
    )


# ── DB 연결 ───────────────────────────────────────────────────────────
def get_engine(db_type: str = None):
    """
    SQLAlchemy 엔진을 생성하여 반환.
    db_type 미지정 시 config.DB_TYPE 사용.
    """
    from sqlalchemy import create_engine

    t = (db_type or config.DB_TYPE).lower()

    if t == "postgresql":
        url = (
            f"postgresql+psycopg2://{config.PG_USER}:{config.PG_PASSWORD}"
            f"@{config.PG_HOST}:{config.PG_PORT}/{config.PG_DBNAME}"
        )
    elif t == "oracle":
        url = (
            f"oracle+oracledb://{config.ORA_USER}:{config.ORA_PASSWORD}"
            f"@{config.ORA_HOST}:{config.ORA_PORT}/?service_name={config.ORA_SERVICE}"
        )
    else:
        # sqlite (기본)
        url = f"sqlite:///{config.DB_PATH}"

    return create_engine(url)


# ── 스키마 조회 ───────────────────────────────────────────────────────
def get_db_schema(db_type: str = None) -> str:
    """
    DB 타입에 맞는 스키마 조회 SQL로 테이블/컬럼 정보를 반환.
    """
    t = (db_type or config.DB_TYPE).lower()
    try:
        engine = get_engine(t)
        with engine.connect() as conn:
            if t == "postgresql":
                # information_schema 에서 테이블/컬럼 조회
                result = conn.execute(__import__('sqlalchemy').text("""
                    SELECT table_name, column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """))
                rows = result.fetchall()
                table_map = {}
                for table, col, dtype in rows:
                    table_map.setdefault(table, []).append(f"{col} {dtype}")
                return "\n".join(
                    f"테이블: {t} ({', '.join(cols)})"
                    for t, cols in sorted(table_map.items())
                ) or "DB 테이블 없음"

            elif t == "oracle":
                result = conn.execute(__import__('sqlalchemy').text("""
                    SELECT table_name, column_name, data_type
                    FROM all_tab_columns
                    WHERE owner = SYS_CONTEXT('USERENV','SESSION_USER')
                    ORDER BY table_name, column_id
                """))
                rows = result.fetchall()
                table_map = {}
                for table, col, dtype in rows:
                    table_map.setdefault(table, []).append(f"{col} {dtype}")
                return "\n".join(
                    f"테이블: {t} ({', '.join(cols)})"
                    for t, cols in sorted(table_map.items())
                ) or "DB 테이블 없음"

            else:
                # SQLite: PRAGMA 방식
                from sqlalchemy import inspect as sa_inspect
                inspector = sa_inspect(engine)
                tables = inspector.get_table_names()
                lines = []
                for table in tables:
                    cols = inspector.get_columns(table)
                    col_defs = ", ".join(
                        f"{c['name']} {c['type']}" for c in cols
                    )
                    lines.append(f"테이블: {table} ({col_defs})")
                return "\n".join(lines) if lines else "DB 테이블 없음"

    except Exception as e:
        return f"스키마 조회 실패: {str(e)}"


def get_schema_from_vector(question: str) -> str:
    """
    Qdrant 스키마 컬렉션에서 질문과 관련된 스키마 검색.
    벡터 DB에 스키마가 없으면 실제 DB 스키마를 폴백으로 사용.
    """
    try:
        from rag.vector_store import search_documents
        results = search_documents(
            question, top_k=3, collection_name=config.SCHEMA_COLLECTION
        )
        if results:
            return "\n".join(results)
    except Exception:
        pass
    return get_db_schema()


# ── SQL 생성 ──────────────────────────────────────────────────────────
def clean_sql(raw_sql: str) -> str:
    """LLM 응답에서 순수 SQL만 추출"""
    raw_sql = re.sub(r"```(?:sql)?", "", raw_sql, flags=re.IGNORECASE)
    raw_sql = raw_sql.replace("```", "")
    lines = [l.strip() for l in raw_sql.strip().splitlines() if l.strip()]
    sql = " ".join(lines)
    match = re.search(r"(SELECT\s.+)", sql, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else sql.strip()


def generate_sql(question: str, schema_context: str, db_type: str = None) -> str:
    """LLM을 이용해 자연어 → SQL 변환 (DB 타입별 문법 안내 포함)"""
    t = (db_type or config.DB_TYPE).lower()
    dialect = {"postgresql": "PostgreSQL", "oracle": "Oracle", "sqlite": "SQLite"}.get(t, "SQLite")

    llm = get_llm()
    system_prompt = f"""당신은 {dialect} SQL 전문가입니다.
아래 DB 스키마를 참고하여 질문에 맞는 SELECT SQL 쿼리를 생성하세요.

[DB 스키마]
{schema_context}

규칙:
- SELECT 문만 생성 (INSERT/UPDATE/DELETE 금지)
- {dialect} 문법 사용
- SQL만 출력 (설명 없이)"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"질문: {question}"),
    ])
    return clean_sql(response.content)


# ── SQL 실행 ──────────────────────────────────────────────────────────
def execute_sql(sql: str, db_type: str = None) -> list:
    """SQL을 실행하고 결과를 dict 리스트로 반환"""
    from sqlalchemy import text
    engine = get_engine(db_type)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        raise RuntimeError(f"SQL 실행 오류: {str(e)}\nSQL: {sql}")


# ── 통합 함수 ─────────────────────────────────────────────────────────
def generate_and_execute_query(question: str, db_type: str = None) -> tuple:
    """
    질문 → SQL 생성 → 실행 → 결과 반환.
    db_type 미지정 시 config.DB_TYPE 사용.
    Returns: (sql 문자열, 결과 rows 리스트)
    """
    t = db_type or config.DB_TYPE
    schema = get_schema_from_vector(question)
    sql = generate_sql(question, schema, t)
    rows = execute_sql(sql, t)
    return sql, rows


def embed_db_schema(db_type: str = None):
    """
    현재 DB 스키마를 Qdrant 스키마 컬렉션에 임베딩하여 저장.
    db_type 미지정 시 config.DB_TYPE 사용.
    """
    from rag.vector_store import save_chunks

    schema_text = get_db_schema(db_type)
    if "DB 테이블 없음" in schema_text or "실패" in schema_text:
        return 0, schema_text

    chunks = [c for c in schema_text.splitlines() if c.strip()]
    count = save_chunks(
        chunks=chunks,
        metadata={"source": "db_schema", "filename": "db_schema"},
        collection_name=config.SCHEMA_COLLECTION,
    )
    return count, schema_text
