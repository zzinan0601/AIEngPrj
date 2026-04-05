"""
ui/modals.py
- @st.dialog 안에서 st.rerun() 을 호출하면 모달이 닫힌다 (Streamlit 설계).
- 해결: st.rerun() 제거 → 버튼 클릭 시 자동 rerun에 맡김.
- 모달 닫기는 명시적 '닫기' 버튼에서만 show_*_modal = False 세팅.
"""

import os
import streamlit as st
import config

from rag.vector_store import get_file_list, get_file_chunks, delete_by_filename
from rag.pipeline import process_and_store
from ui.prompt_manager import (
    load_prompt_config,
    update_system_prompt,
    add_fewshot,
    update_fewshot,
    delete_fewshot,
    reset_to_default,
)


def _fmt_size(size_bytes: int) -> str:
    """바이트 → 읽기 쉬운 단위 변환"""
    if not size_bytes:
        return "-"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 ** 2:.1f} MB"


# ── 문서 관리 모달 ─────────────────────────────────────────────────────
@st.dialog("📂 문서 관리", width="large")
def doc_management_modal():
    """
    뷰 전환: session_state['doc_modal_view'] 로 관리
      - "list"   : 업로드 탭 + 파일 목록 탭
      - "detail" : 청크 상세 (← 뒤로 버튼으로 list 복귀)

    핵심 규칙:
      - st.rerun() 절대 호출하지 않음 → 버튼 클릭의 자동 rerun 사용
      - 닫기 버튼에서만 show_doc_modal = False 세팅
    """

    view = st.session_state.get("doc_modal_view", "list")

    # ══════════════════════════════════════════════════
    # 청크 상세 화면
    # ══════════════════════════════════════════════════
    if view == "detail":
        fname = st.session_state.get("doc_modal_file", "")

        col_back, col_title = st.columns([1, 6])
        with col_back:
            # st.rerun() 없이 session_state만 변경 → 자동 rerun으로 list 복귀
            if st.button("← 뒤로", key="back_to_list"):
                st.session_state["doc_modal_view"] = "list"
        with col_title:
            st.markdown(f"**📄 {fname}** 청크 상세")

        st.divider()

        chunks = get_file_chunks(fname)
        if not chunks:
            st.info("청크 내용을 불러올 수 없습니다.")
            return

        st.caption(f"총 {len(chunks)}개 청크")
        for i, chunk in enumerate(chunks):
            with st.expander(f"청크 {i + 1}  ({len(chunk)}자)", expanded=(i == 0)):
                st.text(chunk)
        return

    # ══════════════════════════════════════════════════
    # 파일 목록 + 업로드 화면 (기본)
    # ══════════════════════════════════════════════════
    tab_upload, tab_list = st.tabs(["⬆️ 파일 업로드", "📋 저장된 파일"])

    # ── 업로드 탭 ────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "파일 선택 (PDF · TXT · DOCX)",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"선택: **{uploaded.name}**  ({_fmt_size(uploaded.size)})")
            if st.button("💾 벡터 DB에 저장", use_container_width=True,
                         key="modal_upload_btn"):
                with st.spinner("임베딩 중..."):
                    try:
                        os.makedirs(config.UPLOADS_PATH, exist_ok=True)
                        save_path = os.path.join(config.UPLOADS_PATH, uploaded.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded.getbuffer())
                        count = process_and_store(save_path, uploaded.name)
                        st.success(f"✅ {count}개 청크 저장 완료")
                        st.session_state["logs"].append(
                            f"📄 파일 저장: {uploaded.name} ({count}청크)"
                        )
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
                        st.session_state["logs"].append(f"❌ 파일 저장 오류: {str(e)}")

    # ── 파일 목록 탭 ─────────────────────────────────
    with tab_list:
        with st.spinner("파일 목록 불러오는 중..."):
            try:
                file_list = get_file_list()
            except Exception as e:
                st.error(f"파일 목록 조회 오류: {str(e)}")
                return

        if not file_list:
            st.info("저장된 파일이 없습니다.")
        else:
            st.caption(f"총 {len(file_list)}개 파일")

            # 헤더 행
            h0, h1, h2, h3, h4 = st.columns([4, 1, 2, 3, 1])
            for col, label in zip(
                [h0, h1, h2, h3, h4],
                ["파일명", "청크", "용량", "등록일", ""],
            ):
                col.markdown(
                    f"<span style='font-size:11px;font-weight:600;color:#888'>"
                    f"{label}</span>",
                    unsafe_allow_html=True,
                )
            st.divider()

            # 파일 행
            for f_info in file_list:
                fname  = f_info["filename"]
                chunks = f_info["chunks"]
                size   = _fmt_size(f_info.get("file_size", 0))
                date   = f_info.get("upload_date", "-")

                c0, c1, c2, c3, c4 = st.columns([4, 1, 2, 3, 1])

                with c0:
                    # st.rerun() 없이 session_state 변경만 → 자동 rerun으로 detail 전환
                    if st.button(
                        f"📄 {fname[:22]}", key=f"fn_{fname}",
                        use_container_width=True,
                        help="클릭하면 청크 상세 내용을 확인합니다",
                    ):
                        st.session_state["doc_modal_view"] = "detail"
                        st.session_state["doc_modal_file"] = fname

                with c1:
                    st.markdown(
                        f"<div style='font-size:12px;padding-top:6px'>{chunks}</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"<div style='font-size:12px;padding-top:6px'>{size}</div>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"<div style='font-size:11px;padding-top:6px;color:#888'>"
                        f"{date}</div>",
                        unsafe_allow_html=True,
                    )
                with c4:
                    if st.button("🗑", key=f"del_{fname}", help="삭제"):
                        try:
                            delete_by_filename(fname)
                            st.session_state["logs"].append(f"🗑️ 파일 삭제: {fname}")
                            # st.rerun() 없음 - 다음 자연 rerun 때 목록 갱신
                        except Exception as e:
                            st.error(f"삭제 실패: {str(e)}")


# ── 프롬프트 / 퓨샷 관리 모달 ─────────────────────────────────────────
@st.dialog("⚙️ 프롬프트 & 퓨샷 관리", width="large")
def prompt_fewshot_modal():
    """시스템 프롬프트와 퓨샷 예제를 관리하는 모달 (st.rerun() 없음)"""
    data = load_prompt_config()
    tab_sys, tab_few = st.tabs(["🤖 시스템 프롬프트", "📝 퓨샷 관리"])

    with tab_sys:
        st.caption("모든 질문에 공통으로 적용되는 역할 지시문")
        new_prompt = st.text_area(
            label="시스템 프롬프트",
            value=data.get("system_prompt", ""),
            height=200,
            label_visibility="collapsed",
            key="sys_prompt_input",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 저장", use_container_width=True, key="save_sys"):
                update_system_prompt(new_prompt)
                st.success("저장 완료")
                st.session_state["logs"].append("⚙️ 시스템 프롬프트 업데이트")
        with c2:
            if st.button("🔄 기본값 복원", use_container_width=True, key="reset_sys"):
                reset_to_default()
                st.success("기본값으로 복원됐습니다.")

    with tab_few:
        fewshots = data.get("fewshots", [])
        if fewshots:
            st.caption(f"등록된 퓨샷: {len(fewshots)}개")
            for i, fs in enumerate(fewshots):
                with st.expander(f"퓨샷 {i+1}: {fs['question'][:40]}"):
                    eq = st.text_area("질문", value=fs["question"], key=f"eq_{i}")
                    ea = st.text_area("답변", value=fs["answer"],   key=f"ea_{i}")
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✏️ 수정", key=f"upd_{i}",
                                     use_container_width=True):
                            update_fewshot(i, eq, ea)
                            st.success("수정 완료")
                    with b2:
                        if st.button("🗑️ 삭제", key=f"delf_{i}",
                                     use_container_width=True):
                            delete_fewshot(i)
                            st.session_state["logs"].append(f"📝 퓨샷 {i+1} 삭제")
        else:
            st.info("등록된 퓨샷이 없습니다.")

        st.markdown("---")
        st.markdown("**새 퓨샷 추가**")
        nq = st.text_area("질문 예시", key="new_fs_q", height=70)
        na = st.text_area("답변 예시", key="new_fs_a", height=90)
        if st.button("➕ 추가", use_container_width=True, key="add_fs"):
            if nq.strip() and na.strip():
                add_fewshot(nq.strip(), na.strip())
                st.success("추가 완료")
                st.session_state["logs"].append("📝 새 퓨샷 등록")
            else:
                st.warning("질문과 답변을 모두 입력하세요.")
