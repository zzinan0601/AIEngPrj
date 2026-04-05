"""
ui/sidebar_left.py
- 왼쪽 사이드바: 버튼 + 모델 선택 콤보박스
- 모델 목록은 Ollama API에서 실시간으로 가져옴
- 실패 시 .env 기본값으로 폴백
"""

import streamlit as st
import config


@st.cache_data(ttl=60)   # 60초 캐시 (매번 API 호출 방지)
def get_ollama_models() -> list:
    """
    Ollama 서버에서 다운로드된 모델 목록을 가져온다.
    실패 시 config 기본 모델만 포함한 리스트 반환.
    """
    try:
        import httpx
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return models if models else [config.LLM_MODEL]
    except Exception:
        pass
    return [config.LLM_MODEL]


def render_sidebar_left():
    """왼쪽 사이드바 렌더링"""
    with st.sidebar:

        # ── 시스템 정보 ──────────────────────────────────────────────
        st.markdown("#### 🖥️ 시스템 정보")
        st.markdown(
            f"""<div style='font-size:11px;line-height:1.9;color:#555'>
            🔢 <b>임베딩</b> : bge-m3 (로컬)<br>
            🎯 <b>리랭커</b> : bge-reranker-v2-m3<br>
            🗄️ <b>벡터DB</b> : Qdrant (로컬)<br>
            💾 <b>RDBMS</b> : SQLite
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── LLM 모델 선택 콤보박스 ──────────────────────────────────
        st.markdown("#### 🤖 LLM 모델 선택")
        models = get_ollama_models()

        # 현재 선택값 유지 (세션에 없으면 config 기본값)
        current = st.session_state.get("selected_model", config.LLM_MODEL)
        # 현재 선택값이 목록에 없으면 0번 인덱스로
        default_idx = models.index(current) if current in models else 0

        selected = st.selectbox(
            "모델 선택",
            options=models,
            index=default_idx,
            label_visibility="collapsed",
            help="Ollama에 설치된 모델 중 선택 (질문마다 즉시 반영)",
        )
        st.session_state["selected_model"] = selected

        # 현재 선택 모델 표시
        st.markdown(
            f"<div style='font-size:11px;color:#888;margin-top:-8px'>"
            f"현재: <b>{selected}</b></div>",
            unsafe_allow_html=True,
        )

        # 모델 목록 새로고침 버튼
        if st.button("🔄 목록 새로고침", use_container_width=True, key="refresh_models"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # ── DB 타입 선택 ─────────────────────────────────────────────
        st.markdown("#### 🗄️ DB 설정")

        db_types = ["sqlite", "postgresql", "oracle"]
        current_db = st.session_state.get("db_type", config.DB_TYPE)
        default_db_idx = db_types.index(current_db) if current_db in db_types else 0

        selected_db = st.selectbox(
            "DB 타입",
            options=db_types,
            index=default_db_idx,
            label_visibility="collapsed",
            help="접속 정보는 .env 파일에 설정 후 사용하세요",
        )
        st.session_state["db_type"] = selected_db
        st.markdown(
            f"<div style='font-size:11px;color:#888;margin-top:-8px'>"
            f"현재: <b>{selected_db}</b></div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── 기능 버튼 ───────────────────────────────────────────────
        if st.button("📂 문서 관리", use_container_width=True,
                     help="문서 업로드 및 파일 목록 관리"):
            st.session_state["show_doc_modal"] = True
            st.session_state["show_prompt_modal"] = False

        st.markdown("")

        if st.button("⚙️ 프롬프트 / 퓨샷", use_container_width=True,
                     help="시스템 프롬프트 및 퓨샷 예제 관리"):
            st.session_state["show_prompt_modal"] = True
            st.session_state["show_doc_modal"] = False

        st.markdown("")

        if st.button("🗄️ DB 스키마 임베딩", use_container_width=True,
                     help="현재 DB 스키마를 벡터 DB에 저장"):
            with st.spinner("스키마 임베딩 중..."):
                try:
                    from agent.db_agent import embed_db_schema
                    db_type = st.session_state.get("db_type", config.DB_TYPE)
                    count, _ = embed_db_schema(db_type)
                    if count > 0:
                        st.success(f"✅ {count}개 항목 임베딩 완료")
                        st.session_state["logs"].append(
                            f"🗄️ DB 스키마 임베딩({db_type}): {count}항목"
                        )
                    else:
                        st.warning("임베딩할 스키마가 없습니다")
                except Exception as e:
                    st.error(f"오류: {str(e)}")

        st.divider()

        # ── MCP 토글 ────────────────────────────────────────────────
        st.session_state["use_mcp"] = st.toggle(
            "🔌 MCP 경유 호출",
            value=st.session_state.get("use_mcp", False),
            help="활성화하면 모든 도구 호출이 MCP 서버를 경유합니다",
        )
