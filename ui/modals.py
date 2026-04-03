"""
ui/modals.py
- Streamlit @st.dialog 데코레이터를 사용한 모달 팝업
- chunk_detail_modal : 파일의 청크 내용을 상세 확인
- prompt_fewshot_modal : 시스템 프롬프트 + 퓨샷 등록/수정/삭제
"""

import streamlit as st
from rag.vector_store import get_file_chunks
from ui.prompt_manager import (
    load_prompt_config,
    save_prompt_config,
    update_system_prompt,
    add_fewshot,
    update_fewshot,
    delete_fewshot,
    reset_to_default,
)


# ── 청크 상세 모달 ────────────────────────────────────────────────────
@st.dialog("📄 청크 상세 내용", width="large")
def chunk_detail_modal(filename: str):
    """선택한 파일의 청크 내용을 모달로 표시"""
    st.markdown(f"**파일명**: `{filename}`")
    st.markdown("---")

    chunks = get_file_chunks(filename)

    if not chunks:
        st.info("청크가 없거나 파일을 찾을 수 없습니다.")
        return

    st.caption(f"총 {len(chunks)}개 청크")

    for i, chunk in enumerate(chunks):
        with st.expander(f"청크 {i + 1}  ({len(chunk)}자)", expanded=(i == 0)):
            st.text(chunk)

    if st.button("닫기", key="close_chunk_modal"):
        st.rerun()


# ── 프롬프트 / 퓨샷 관리 모달 ────────────────────────────────────────
@st.dialog("⚙️ 프롬프트 & 퓨샷 관리", width="large")
def prompt_fewshot_modal():
    """시스템 프롬프트와 퓨샷 예제를 관리하는 모달"""
    data = load_prompt_config()

    tab_sys, tab_few = st.tabs(["🤖 시스템 프롬프트", "📝 퓨샷 관리"])

    # ── 탭 1: 시스템 프롬프트 ────────────────────────────────────────
    with tab_sys:
        st.caption("모든 질문에 공통으로 적용되는 역할 지시문")
        new_prompt = st.text_area(
            "시스템 프롬프트",
            value=data.get("system_prompt", ""),
            height=200,
            label_visibility="collapsed",
            key="sys_prompt_input",
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💾 저장", use_container_width=True, key="save_sys"):
                update_system_prompt(new_prompt)
                st.success("시스템 프롬프트가 저장되었습니다.")
                st.session_state.logs.append("⚙️ 시스템 프롬프트 업데이트")
        with col2:
            if st.button("🔄 기본값 복원", use_container_width=True, key="reset_sys"):
                reset_to_default()
                st.rerun()

    # ── 탭 2: 퓨샷 관리 ─────────────────────────────────────────────
    with tab_few:
        fewshots = data.get("fewshots", [])

        # 등록된 퓨샷 목록
        if fewshots:
            st.caption(f"등록된 퓨샷: {len(fewshots)}개")
            for i, fs in enumerate(fewshots):
                with st.expander(
                    f"퓨샷 {i + 1}: {fs['question'][:40]}...",
                    expanded=False,
                ):
                    edit_q = st.text_area(
                        "질문", value=fs["question"], key=f"edit_q_{i}"
                    )
                    edit_a = st.text_area(
                        "답변", value=fs["answer"], key=f"edit_a_{i}"
                    )
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("✏️ 수정", key=f"upd_{i}", use_container_width=True):
                            update_fewshot(i, edit_q, edit_a)
                            st.success("수정 완료")
                            st.rerun()
                    with c2:
                        if st.button("🗑️ 삭제", key=f"del_{i}", use_container_width=True):
                            delete_fewshot(i)
                            st.session_state.logs.append(f"📝 퓨샷 {i+1} 삭제")
                            st.rerun()
        else:
            st.info("등록된 퓨샷이 없습니다.")

        st.markdown("---")
        st.markdown("**새 퓨샷 추가**")
        new_q = st.text_area("질문 예시", key="new_fs_q", height=80)
        new_a = st.text_area("답변 예시", key="new_fs_a", height=100)

        if st.button("➕ 퓨샷 추가", use_container_width=True, key="add_fs"):
            if new_q.strip() and new_a.strip():
                add_fewshot(new_q.strip(), new_a.strip())
                st.success("퓨샷이 추가되었습니다.")
                st.session_state.logs.append("📝 새 퓨샷 등록")
                st.rerun()
            else:
                st.warning("질문과 답변을 모두 입력하세요.")
