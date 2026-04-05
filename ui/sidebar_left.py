"""
ui/sidebar_left.py
- 왼쪽 사이드바: 버튼 + 모델/DB 선택 콤보박스
- selectbox 변경 시 모달 플래그를 초기화해 팝업 방지
"""

import streamlit as st
import config


@st.cache_resource   # 앱 전체에서 1회만 호출 (세션/rerun 무관)
def get_ollama_models() -> list:
    """Ollama 서버에서 설치된 모델 목록을 가져온다. 실패 시 기본값 반환."""
    try:
        import httpx
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return models if models else [config.LLM_MODEL]
    except Exception:
        pass
    return [config.LLM_MODEL]


def _close_all_modals():
    """selectbox 변경 등으로 rerun 발생 시 모달이 뜨지 않도록 플래그 초기화"""
    st.session_state["show_doc_modal"]    = False
    st.session_state["show_prompt_modal"] = False


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

        # ── LLM 모델 선택 ────────────────────────────────────────────
        st.markdown("#### 🤖 LLM 모델")
        models = get_ollama_models()
        current_model = st.session_state.get("selected_model", config.LLM_MODEL)
        model_idx = models.index(current_model) if current_model in models else 0

        selected_model = st.selectbox(
            "모델 선택",
            options=models,
            index=model_idx,
            label_visibility="collapsed",
            key="sb_model",
            help="Ollama에 설치된 모델 중 선택",
            on_change=_close_all_modals,   # 변경 시 모달 닫기
        )
        st.session_state["selected_model"] = selected_model

        if st.button("🔄 목록 새로고침", use_container_width=True, key="refresh_models"):
            st.cache_resource.clear()
            from rag.vector_store import warmup
            warmup()
            _close_all_modals()
            st.rerun()

        st.divider()

        # ── 기능 버튼 ───────────────────────────────────────────────
        if st.button("📂 문서 관리", use_container_width=True,
                     help="문서 업로드 및 파일 목록 관리"):
            st.session_state["show_doc_modal"]    = True
            st.session_state["show_prompt_modal"] = False

        st.markdown("")

        if st.button("⚙️ 프롬프트 / 퓨샷", use_container_width=True,
                     help="시스템 프롬프트 및 퓨샷 예제 관리"):
            st.session_state["show_prompt_modal"] = True
            st.session_state["show_doc_modal"]    = False

        st.markdown("")

        # ── DB 타입 선택 (스키마 임베딩 버튼 바로 위) ────────────────
        db_types = ["sqlite", "postgresql", "oracle"]
        current_db = st.session_state.get("db_type", config.DB_TYPE)
        db_idx = db_types.index(current_db) if current_db in db_types else 0

        selected_db = st.selectbox(
            "DB 타입",
            options=db_types,
            index=db_idx,
            label_visibility="visible",
            key="sb_db_type",
            help="접속 정보는 .env 파일에 설정하세요",
            on_change=_close_all_modals,   # 변경 시 모달 닫기
        )
        st.session_state["db_type"] = selected_db

        if st.button("🗄️ DB 스키마 임베딩", use_container_width=True,
                     help="현재 DB 스키마를 벡터 DB에 저장"):
            with st.spinner("스키마 임베딩 중..."):
                try:
                    from agent.db_agent import embed_db_schema
                    count, _ = embed_db_schema(selected_db)
                    if count > 0:
                        st.success(f"✅ {count}개 항목 임베딩 완료")
                        st.session_state["logs"].append(
                            f"🗄️ DB 스키마 임베딩({selected_db}): {count}항목"
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
