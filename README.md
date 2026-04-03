# 🤖 RAG AI 어시스턴트

LangChain + LangGraph + Ollama + Qdrant 기반의 로컬 RAG 에이전트 시스템.  
**폐쇄망 Windows 환경**에서 완전 오프라인으로 동작합니다.

---

## 📁 프로젝트 구조

```
rag_agent/
├── app.py                  # Streamlit 진입점
├── config.py               # 환경설정 로딩 (.env)
├── .env                    # 환경변수 (모델경로, DB경로 등)
├── requirements.txt        # 패키지 목록
│
├── agent/                  # LangGraph 에이전트
│   ├── state.py            # GraphState 정의
│   ├── nodes.py            # 노드 함수 (Router/RAG/DB/General/Synthesize)
│   ├── graph.py            # StateGraph 조립 (노드+엣지)
│   └── db_agent.py         # SQL 생성 및 DB 조회
│
├── rag/                    # RAG 파이프라인
│   ├── embeddings.py       # bge-m3 임베딩 (로컬)
│   ├── reranker.py         # bge-reranker 재정렬
│   ├── vector_store.py     # Qdrant CRUD
│   └── pipeline.py         # 문서처리 · 검색 통합
│
├── mcp/                    # MCP · Tool · A2A
│   ├── tools.py            # LangChain Tool 정의
│   ├── mcp_server.py       # MCP Server (FastMCP)
│   └── mcp_client.py       # MCP Client
│
├── ui/                     # Streamlit UI 컴포넌트
│   ├── layout.py           # 3패널 레이아웃 조립
│   ├── sidebar_left.py     # 왼쪽: 시스템정보·업로드·파일목록
│   ├── chat.py             # 중앙: 채팅 화면
│   ├── sidebar_right.py    # 오른쪽: 처리 흐름 로그
│   ├── modals.py           # 모달 팝업 (청크상세·프롬프트)
│   └── prompt_manager.py   # 프롬프트/퓨샷 CRUD
│
├── models/                 # 로컬 모델 디렉토리 (직접 다운)
│   ├── bge-m3/             # 임베딩 모델
│   └── bge-reranker-v2-m3/ # 리랭커 모델
│
└── data/                   # 런타임 자동 생성
    ├── qdrant_data/        # Qdrant 벡터 DB 로컬 저장소
    ├── prompts/            # 프롬프트·퓨샷 JSON
    ├── uploads/            # 업로드 파일 임시 저장
    └── app.db              # SQLite DB
```

---

## ⚙️ 사전 준비

### 1. Ollama 설치 및 모델 다운로드
```bash
# Ollama 설치 후 (https://ollama.com)
ollama pull llama3.1:8b
ollama serve   # 백그라운드 실행
```

### 2. 허깅페이스 모델 로컬 다운로드
인터넷이 되는 환경에서 다운로드 후 `models/` 폴더에 복사:
```python
# 다운로드 스크립트 (인터넷 환경에서 실행)
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-m3",              local_dir="models/bge-m3")
snapshot_download("BAAI/bge-reranker-v2-m3",  local_dir="models/bge-reranker-v2-m3")
```

### 3. 패키지 설치
```bash
# 온라인 환경
pip install -r requirements.txt

# 폐쇄망 환경 (wheels 폴더에 whl 파일 사전 준비 후)
pip install --no-index --find-links=./wheels -r requirements.txt
```

---

## 🚀 실행 방법

```bash
# 프로젝트 루트에서 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 🖥️ 화면 구성

| 영역 | 기능 |
|------|------|
| **왼쪽 사이드바** | 시스템정보 / 문서업로드 / 파일목록(삭제) / 청크상세 / DB스키마임베딩 / 프롬프트관리 |
| **중앙 채팅** | 질문입력 → 에이전트 자동 라우팅 → 스트리밍 답변 |
| **오른쪽 패널** | 처리 흐름 로그 누적 / A2A 메시지 흐름 표시 |

---

## 🔀 에이전트 라우팅 흐름

```
사용자 질문
    │
    ▼
[Router Node] ── 문서 관련 ──► [RAG Node] ──────────────► [Synthesize] → 답변
                │                  │ (both일 때)              ▲
                ├── DB 조회 ──────► [DB Node] ───────────────┤
                │                                            │
                └── 일반 대화 ──► [General Node] ────────────┘
```

- **복합 질문(both)**: RAG → DB 순서로 실행 후 결과 합산
- **loop**: 최대 `MAX_ITERATIONS`(기본 3)회 반복 후 강제 종료

---

## 📝 프롬프트 / 퓨샷 관리

왼쪽 사이드바 하단 **"프롬프트/퓨샷 관리"** 버튼 클릭  
→ 모달 팝업에서 시스템 프롬프트 수정 / 퓨샷 예제 추가·수정·삭제  
→ 저장 즉시 다음 질문부터 반영

---

## 🔌 MCP 사용

왼쪽 사이드바의 **"MCP 경유 호출"** 토글 활성화 시  
모든 도구 호출이 `mcp_server.py`를 subprocess로 경유합니다.

---

## 🗄️ DB 조회 기능

1. `data/app.db` (SQLite)에 테이블 생성
2. 사이드바 **"DB 스키마 임베딩"** 클릭 → 스키마를 벡터 DB에 저장
3. "총 매출은?" 같은 DB 조회 질문 입력 → 자동으로 SQL 생성·실행

---

## ❓ 주요 설정 (.env)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LLM_MODEL` | `llama3.1:8b` | Ollama 모델명 |
| `EMBEDDING_MODEL_PATH` | `./models/bge-m3` | 임베딩 모델 경로 |
| `RERANKER_MODEL_PATH` | `./models/bge-reranker-v2-m3` | 리랭커 경로 |
| `CHUNK_SIZE` | `500` | 청킹 크기 |
| `TOP_K` | `5` | 벡터 검색 결과 수 |
| `RERANK_TOP_K` | `3` | 리랭킹 후 최종 결과 수 |
| `MAX_ITERATIONS` | `3` | 복합질문 최대 반복 수 |
