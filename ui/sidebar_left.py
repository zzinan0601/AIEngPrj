"""
ui/sidebar_left.py
- 왼쪽 사이드바: 아이콘 버튼만 노출
- 실제 기능(업로드, 파일목록, 프롬프트)은 모달 팝업으로 위임
"""

import streamlit as st
import config


def render_sidebar_left():
    """왼쪽 사이드바 렌더링 - 버튼만 표시"""
    with st.sidebar:

        # ── 시스템 정보 ──────────────────────────────────────────────
        st.markdown("#### 🖥️ 시스템 정보")
        st.markdown(
            f"""<div style='font-size:11px;line-height:1.9;color:#555'>
            🤖 <b>LLM</b> : {config.LLM_MODEL}<br>
            🔢 <b>임베딩</b> : bge-m3 (로컬)<br>
            🎯 <b>리랭커</b> : bge-reranker-v2-m3<br>
            🗄️ <b>벡터DB</b> : Qdrant (로컬)<br>
            💾 <b>RDBMS</b> : SQLite
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── 기능 버튼 목록 ───────────────────────────────────────────
        if st.button("📂 문서 관리", use_container_width=True,
                     help="문서 업로드 및 파일 목록 관리"):
            st.session_state["show_doc_modal"] = True

        st.markdown("")

        if st.button("⚙️ 프롬프트 / 퓨샷", use_container_width=True,
                     help="시스템 프롬프트 및 퓨샷 예제 관리"):
            st.session_state["show_prompt_modal"] = True

        st.markdown("")

        if st.button("🗄️ DB 스키마 임베딩", use_container_width=True,
                     help="현재 DB 스키마를 벡터 DB에 저장"):
            with st.spinner("스키마 임베딩 중..."):
                try:
                    from agent.db_agent import embed_db_schema
                    count, _ = embed_db_schema()
                    if count > 0:
                        st.success(f"✅ {count}개 항목 임베딩 완료")
                        st.session_state["logs"].append(
                            f"🗄️ DB 스키마 임베딩: {count}항목"
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
