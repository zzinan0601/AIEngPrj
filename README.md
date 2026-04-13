# 🤖 RAG AI 어시스턴트

LangChain + LangGraph + Ollama + Qdrant 기반의 로컬 RAG 에이전트 시스템.  
**폐쇄망 Windows / Python 3.12 환경**에서 완전 오프라인으로 동작합니다.

---

## 📁 프로젝트 구조

```
rag_agent/
├── app.py                       # Streamlit 진입점
├── config.py                    # 환경설정 로딩 (.env)
├── .env                         # 환경변수
├── requirements.txt             # 패키지 목록
│
├── agent/                       # LangGraph 에이전트
│   ├── state.py                 # GraphState 정의
│   ├── nodes.py                 # 노드 함수 (Planner/Executor/RAG/DB/General/Synthesize/Chart)
│   ├── graph.py                 # StateGraph 조립
│   └── db_agent.py              # SQL 생성 및 DB 조회
│
├── rag/                         # RAG 파이프라인
│   ├── embeddings.py            # bge-m3 임베딩 (로컬)
│   ├── reranker.py              # bge-reranker 재정렬 (XLMRoberta 직접 로드)
│   ├── vector_store.py          # Qdrant CRUD (@st.cache_resource)
│   └── pipeline.py              # 문서처리·검색 통합
│
├── call_mcp/                    # MCP · Tool · A2A
│   ├── tools.py                 # LangChain Tool 정의
│   ├── mcp_server.py            # MCP Server SSE 방식 (uvicorn)
│   ├── mcp_server_stdio.py      # MCP Server stdio 방식 (UI 토글용)
│   └── mcp_client.py            # MCP Client (SSE + stdio 자동 폴백)
│
├── ui/                          # Streamlit UI 컴포넌트
│   ├── layout.py                # 3패널 레이아웃 조립
│   ├── sidebar_left.py          # 왼쪽: LLM선택·DB선택·문서관리·프롬프트
│   ├── chat.py                  # 중앙: 채팅 화면 + 차트 렌더링
│   ├── sidebar_right.py         # 오른쪽: 처리 흐름 로그 + A2A 메시지
│   ├── modals.py                # 모달 팝업 (문서관리·DB스키마·프롬프트/퓨샷)
│   └── prompt_manager.py        # 프롬프트/퓨샷 JSON CRUD
│
├── models/                      # 로컬 모델 (직접 다운로드)
│   ├── bge-m3/                  # 임베딩 모델
│   └── bge-reranker-v2-m3/      # 리랭커 모델 (snapshots/해시값/ 경로 확인)
│
└── data/                        # 런타임 자동 생성
    ├── qdrant_data/             # Qdrant 벡터 DB 로컬 저장소
    ├── prompts/                 # 프롬프트·퓨샷 JSON
    ├── uploads/                 # 업로드 파일 임시 저장
    └── app.db                   # SQLite DB (DB_TYPE=sqlite 시)
```

---

## ⚙️ 사전 준비

### 1. Ollama 설치 및 모델 다운로드
```bash
# Ollama 설치 후 (https://ollama.com)
ollama pull llama3.1:8b
ollama serve
```

### 2. 허깅페이스 모델 로컬 다운로드
인터넷 환경에서 다운로드 후 `models/` 폴더에 복사:
```python
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-m3",              local_dir="models/bge-m3")
snapshot_download("BAAI/bge-reranker-v2-m3",  local_dir="models/bge-reranker-v2-m3")
```

> **중요**: bge-reranker 모델 파일이 `snapshots/해시값/` 하위에 있을 경우  
> `.env`의 `RERANKER_MODEL_PATH`를 실제 경로로 수정하세요.  
> 예: `RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3/snapshots/abc123`

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

### MCP SSE 서버 별도 실행 (선택)
MCP 경유 호출 토글 사용 시 SSE 서버를 먼저 실행합니다:
```bash
# 별도 터미널
python call_mcp/mcp_server.py
# → http://localhost:8765/sse 로 서비스
```

---

## 🖥️ 화면 구성

| 영역 | 기능 |
|---|---|
| **왼쪽 사이드바** | LLM 모델 선택 / DB 타입 선택 / 📂 문서 관리 / ⚙️ 프롬프트·퓨샷 / 🗄️ DB 스키마 임베딩 / MCP 토글 |
| **중앙 채팅** | 질문 입력 → Planner가 계획 수립 → 자동 실행 → 답변 + 차트 |
| **오른쪽 패널** | 실행 계획·처리 흐름 로그 누적 / A2A 메시지 흐름 |

---

## 🔀 에이전트 실행 흐름

```
사용자 질문
    ↓
[Planner Node]  ← LLM이 질문 분석 → 실행 계획(plan) 수립
    예: ["rag", "db"] / ["general"] / ["db", "rag"]
    ↓
[Executor Node] ← plan에서 현재 단계 꺼내 route 세팅
    ↓
[RAG Node]      ← 문서 벡터 검색 + bge-reranker 재정렬
[DB Node]       ← SQL 자동 생성 + DB 조회 (이전 대화·RAG 결과 참조)
[General Node]  ← LLM 직접 응답
    ↓
[Step Done]     ← plan_idx 증가
    ↓ 남은 단계 있으면 Executor로 돌아가 반복
    ↓ 모든 단계 완료
[Synthesize]    ← 모든 결과 합쳐 최종 답변 생성 (대화 메모리 반영)
    ↓ 차트 요청이면
[Chart Node]    ← Plotly 차트 설정 JSON 생성
    ↓
UI에 답변 + 출처 + 차트 표시
```

**Planner 예시:**

| 질문 | 생성된 plan |
|---|---|
| "안녕하세요" | `["general"]` |
| "문서 요약해줘" | `["rag"]` |
| "지난달 매출은?" | `["db"]` |
| "계약서 내용과 관련 매출 알려줘" | `["rag", "db"]` |
| "매출부터 보고 보고서랑 비교해줘" | `["db", "rag"]` |
| "월별 매출 차트로 그려줘" | `["db"]` + chart_request=True |

---

## 📂 문서 관리 모달

왼쪽 사이드바 **📂 문서 관리** 클릭:

| 탭 | 기능 |
|---|---|
| ⬆️ 파일 업로드 | PDF·TXT·DOCX → 청킹·임베딩 → Qdrant 저장 |
| 📋 저장된 파일 | 파일명·청크수·용량·등록일 목록 / 파일명 클릭 → 청크 내용 확인 / 🗑 삭제 |
| 🗄️ DB 스키마 현황 | DB 타입별 임베딩된 스키마 확인 / 내용 보기 / 삭제 |

---

## ⚙️ 프롬프트 / 퓨샷 관리

왼쪽 사이드바 **⚙️ 프롬프트 / 퓨샷** 클릭:
- **시스템 프롬프트**: 모든 질문에 적용되는 역할 지시문 등록·수정
- **퓨샷 관리**: 질문·답변 예시 등록·수정·삭제

> 저장 즉시 다음 질문부터 반영됩니다.

---

## 🗄️ DB 조회 기능

1. `.env`에서 `DB_TYPE` 설정 (sqlite / postgresql / oracle)
2. 해당 DB 접속 정보 입력
3. 사이드바 DB 타입 콤보박스에서 선택
4. **🗄️ DB 스키마 임베딩** 클릭 → 스키마를 벡터 DB에 저장
5. "지난달 매출 합계는?" 등 질문 → Planner가 `["db"]` 계획 수립 → SQL 자동 생성·실행

### DB별 접속 설정 (.env)

```ini
# SQLite
DB_TYPE=sqlite
DB_PATH=./data/app.db

# PostgreSQL
DB_TYPE=postgresql
PG_HOST=localhost
PG_PORT=5432
PG_USER=myuser
PG_PASSWORD=mypassword
PG_DBNAME=mydb

# Oracle (oracledb 드라이버, Python 3.12 지원)
DB_TYPE=oracle
ORA_HOST=localhost
ORA_PORT=1521
ORA_USER=myuser
ORA_PASSWORD=mypassword
ORA_SERVICE=myservice
```

---

## 📊 차트 기능

질문에 차트/그래프 키워드가 있으면 자동으로 차트를 생성합니다.

- **지원 타입**: bar / line / pie / scatter
- **데이터 소스**: DB 조회 결과 (우선) 또는 RAG 문서 내용
- **렌더링**: Plotly (없으면 Streamlit 기본 차트 폴백)

예시 질문:
```
월별 매출을 막대 그래프로 보여줘
부서별 인원수를 파이 차트로 그려줘
최근 6개월 매출 추이를 선 그래프로 시각화해줘
```

---

## 🔌 MCP (Model Context Protocol)

### SSE 방식 (권장)
```bash
# 서버 실행
python call_mcp/mcp_server.py

# 연결 테스트
python -c "from call_mcp.mcp_client import test_mcp_connection; print(test_mcp_connection())"
```

### stdio 방식 (UI 토글)
별도 서버 실행 없이 UI의 **🔌 MCP 경유 호출** 토글만 켜면 됩니다.  
SSE 서버가 떠있으면 SSE 방식, 없으면 stdio 자동 폴백.

> **주의**: Qdrant 로컬 모드는 단일 프로세스만 접근 가능합니다.  
> MCP 서버와 Streamlit 앱이 동시에 같은 `qdrant_data/` 경로를 사용하면  
> "already accessed" 오류가 발생합니다. Qdrant를 서버 모드로 운영하거나  
> MCP 도구에서 Qdrant 직접 접근을 제거하세요.

---

## 🧠 대화 메모리

이전 대화를 기억해 문맥 연속 질문이 가능합니다.

```ini
# .env
MEMORY_TURNS=10   # 최근 10턴 기억 (0=비활성화)
```

| 값 | 설명 |
|---|---|
| `0` | 메모리 비활성화 |
| `5` | 가볍게, 빠른 응답 |
| `10` | 기본값 (권장) |
| `15` | 긴 문맥, 응답 느려질 수 있음 |

---

## ❓ 주요 .env 설정

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b` | Ollama 모델명 |
| `EMBEDDING_MODEL_PATH` | `./models/bge-m3` | 임베딩 모델 경로 |
| `RERANKER_MODEL_PATH` | `./models/bge-reranker-v2-m3` | 리랭커 경로 (snapshots 하위 확인) |
| `DB_TYPE` | `sqlite` | DB 종류 (sqlite/postgresql/oracle) |
| `CHUNK_SIZE` | `500` | 문서 청킹 크기 |
| `CHUNK_OVERLAP` | `50` | 청크 오버랩 크기 |
| `TOP_K` | `5` | 벡터 검색 결과 수 |
| `RERANK_TOP_K` | `3` | 리랭킹 후 최종 결과 수 |
| `MEMORY_TURNS` | `10` | 대화 메모리 턴 수 |
| `MCP_SERVER_PORT` | `8765` | MCP SSE 서버 포트 |

---

## 🔧 알려진 이슈 및 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `already accessed` | Qdrant 로컬 동시 접근 | Qdrant 서버 모드 사용 |
| reranker 모델 로드 실패 | `config.json` model_type 없음 | `RERANKER_MODEL_PATH`를 snapshots 실제 경로로 수정 |
| MCP stdio timeout | subprocess 내 모델 로드 시간 초과 | SSE 방식 사용 권장 |
| LLM이 한국어 무시 | llama3.1:8b 특성 | 프롬프트에 언어 지시 이중 삽입 (자동 적용) |
| 버튼 두 번 클릭 | `@st.dialog` rerun 특성 | `on_click` 콜백 방식으로 처리 (적용됨) |
