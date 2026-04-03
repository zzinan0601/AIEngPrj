"""
ui/sidebar_left.py
- 왼쪽 사이드바 전체 렌더링
  0. 현재 시스템 정보 (모델/임베딩/DB)
  1. 문서 업로드 → 벡터 DB 저장
  2. 저장된 파일 목록 (표 형태, 파일명(청크수), 삭제 버튼)
  3. 파일명 클릭 → 청크 상세 모달 트리거
  4. DB 스키마 임베딩 버튼
  5. 프롬프트/퓨샷 관리 버튼
"""

import os
import streamlit as st
import config
from rag.pipeline import process_and_store
from rag.vector_store import get_file_list, delete_by_filename


def render_sidebar_left():
    """왼쪽 사이드바 전체 렌더링"""
    with st.sidebar:

        # ── 0. 시스템 정보 ────────────────────────────────────────────
        st.markdown("#### 🖥️ 시스템 정보")
        st.markdown(
            f"""
            <div style='font-size:11px; line-height:1.9; color:#555'>
            🤖 <b>LLM</b> : {config.LLM_MODEL}<br>
            🔢 <b>임베딩</b> : bge-m3 (로컬)<br>
            🎯 <b>리랭커</b> : bge-reranker-v2-m3<br>
            🗄️ <b>벡터DB</b> : Qdrant (로컬)<br>
            💾 <b>RDBMS</b> : SQLite
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── 1. 문서 업로드 ────────────────────────────────────────────
        st.markdown("#### 📎 문서 업로드")
        uploaded = st.file_uploader(
            "파일 선택 (PDF · TXT · DOCX)",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )

        if uploaded:
            if st.button("💾 벡터 DB 저장", use_container_width=True):
                with st.spinner("처리 중... 임베딩 시간이 걸릴 수 있습니다"):
                    try:
                        # 업로드 디렉토리에 파일 저장
                        save_path = os.path.join(config.UPLOADS_PATH, uploaded.name)
                        os.makedirs(config.UPLOADS_PATH, exist_ok=True)
                        with open(save_path, "wb") as f:
                            f.write(uploaded.getbuffer())

                        # RAG 파이프라인 실행
                        chunk_count = process_and_store(save_path, uploaded.name)
                        st.success(f"✅ {chunk_count}개 청크 저장 완료")
                        st.session_state.logs.append(
                            f"📄 파일 저장: {uploaded.name} ({chunk_count}청크)"
                        )
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 오류: {str(e)}")
                        st.session_state.logs.append(f"❌ 파일 저장 오류: {str(e)}")

        st.divider()

        # ── 2 & 3. 저장된 파일 목록 ──────────────────────────────────
        st.markdown("#### 📁 저장된 파일")
        file_list = get_file_list()

        if not file_list:
            st.caption("저장된 파일이 없습니다")
        else:
            st.caption(f"총 {len(file_list)}개 파일")
            for f_info in file_list:
                fname = f_info["filename"]
                chunks = f_info["chunks"]

                col_name, col_del = st.columns([5, 1])

                with col_name:
                    # 파일명 클릭 → 청크 상세 모달 트리거
                    if st.button(
                        f"📄 {fname[:22]}({chunks})",
                        key=f"file_btn_{fname}",
                        use_container_width=True,
                        help=f"{fname} - {chunks}개 청크",
                    ):
                        st.session_state["modal_file"] = fname
                        st.session_state["show_chunk_modal"] = True

                with col_del:
                    if st.button(
                        "🗑", key=f"del_btn_{fname}", help="이 파일 삭제"
                    ):
                        try:
                            delete_by_filename(fname)
                            st.session_state.logs.append(f"🗑️ 파일 삭제: {fname}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"삭제 실패: {str(e)}")

        st.divider()

        # ── 4. DB 스키마 임베딩 ───────────────────────────────────────
        st.markdown("#### 🗄️ DB 스키마 관리")
        if st.button("🔄 DB 스키마 임베딩", use_container_width=True):
            with st.spinner("스키마 임베딩 중..."):
                try:
                    from agent.db_agent import embed_db_schema
                    count, schema_text = embed_db_schema()
                    if count > 0:
                        st.success(f"✅ 스키마 {count}개 항목 임베딩 완료")
                        st.session_state.logs.append(f"🗄️ DB 스키마 임베딩: {count}항목")
                    else:
                        st.warning("임베딩할 스키마가 없습니다 (DB 테이블 없음)")
                except Exception as e:
                    st.error(f"스키마 임베딩 오류: {str(e)}")

        st.divider()

        # ── 5. 프롬프트 관리 ─────────────────────────────────────────
        st.markdown("#### ⚙️ 프롬프트 설정")
        if st.button("📝 프롬프트/퓨샷 관리", use_container_width=True):
            st.session_state["show_prompt_modal"] = True

        # MCP 사용 여부 토글
        st.session_state["use_mcp"] = st.toggle(
            "🔌 MCP 경유 호출",
            value=st.session_state.get("use_mcp", False),
            help="활성화하면 모든 도구 호출이 MCP 서버를 경유합니다",
        )
