"""
config.py
- .env 파일을 읽어 전역 설정값을 제공
- 다른 모든 모듈이 이 파일에서 설정을 가져옴
- 데이터 디렉토리를 자동 생성
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).parent

# ── Ollama LLM ──────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1:8b")

# ── 임베딩 / 리랭커 모델 ───────────────────
EMBEDDING_MODEL_PATH: str = os.getenv(
    "EMBEDDING_MODEL_PATH", str(BASE_DIR / "models" / "bge-m3")
)
RERANKER_MODEL_PATH: str = os.getenv(
    "RERANKER_MODEL_PATH", str(BASE_DIR / "models" / "bge-reranker-v2-m3")
)

# ── Qdrant 벡터 DB ──────────────────────────
QDRANT_PATH: str = os.getenv("QDRANT_PATH", str(BASE_DIR / "data" / "qdrant_data"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
SCHEMA_COLLECTION: str = os.getenv("SCHEMA_COLLECTION", "schema")
VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "1024"))  # bge-m3 출력 차원

# ── DB 타입 및 접속 정보 ─────────────────────
# DB_TYPE: sqlite | postgresql | oracle
DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")

# SQLite
DB_PATH: str = os.getenv("DB_PATH", str(BASE_DIR / "data" / "app.db"))

# PostgreSQL
PG_HOST: str     = os.getenv("PG_HOST", "localhost")
PG_PORT: str     = os.getenv("PG_PORT", "5432")
PG_USER: str     = os.getenv("PG_USER", "")
PG_PASSWORD: str = os.getenv("PG_PASSWORD", "")
PG_DBNAME: str   = os.getenv("PG_DBNAME", "")

# Oracle
ORA_HOST: str     = os.getenv("ORA_HOST", "localhost")
ORA_PORT: str     = os.getenv("ORA_PORT", "1521")
ORA_USER: str     = os.getenv("ORA_USER", "")
ORA_PASSWORD: str = os.getenv("ORA_PASSWORD", "")
ORA_SERVICE: str  = os.getenv("ORA_SERVICE", "")

# ── 파일 경로 ────────────────────────────────
PROMPTS_PATH: str = os.getenv("PROMPTS_PATH", str(BASE_DIR / "data" / "prompts"))
UPLOADS_PATH: str = os.getenv("UPLOADS_PATH", str(BASE_DIR / "data" / "uploads"))

# ── 청킹 ────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── 검색 ────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))

# ── MCP 서버 ─────────────────────────────────
MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8765"))

# ── 에이전트 루프 ─────────────────────────────
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

# ── 대화 메모리 ───────────────────────────────
# 0 이면 메모리 비활성화, 최근 N턴만 LLM에 전달 (슬라이딩 윈도우)
MEMORY_TURNS: int = int(os.getenv("MEMORY_TURNS", "10"))


def init_data_dirs():
    """데이터 디렉토리를 자동으로 생성"""
    for path in [QDRANT_PATH, PROMPTS_PATH, UPLOADS_PATH]:
        os.makedirs(path, exist_ok=True)


# 모듈 임포트 시 자동 실행
init_data_dirs()
