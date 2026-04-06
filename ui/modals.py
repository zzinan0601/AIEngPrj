"""
ui/modals.py

[두 번 클릭 문제]
  → 액션 지연 처리 패턴 유지 (버튼 클릭 시 session_state에 액션 저장)

[퓨샷 키입력 중 갑자기 저장 문제]
  → 원인: expander 안에 button이 있으면 expander 클릭 시 내부 button이
          의도치 않게 trigger되는 Streamlit 버그
  → 해결: 수정/삭제 button을 expander 바깥 column으로 이동
          수정은 session_state["_edit_idx"]로 인라인 편집 영역 표시
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
    if not size_bytes:
        return "-"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 ** 2:.1f} MB"


def _set_action(action: str, **kwargs):
    st.session_state["_modal_action"] = action
    st.session_state["_modal_data"]   = kwargs


def _pop_action():
    action = st.session_state.pop("_modal_action", None)
    data   = st.session_state.pop("_modal_data",   {})
    return action, data


# ── 문서 관리 모달 ─────────────────────────────────────────────────────
@st.dialog("📂 문서 관리", width="large")
def doc_management_modal():

    action, data = _pop_action()
    msg = None

    if action == "upload_file":
        raw, fname = data.get("raw"), data.get("fname")
        if raw and fname:
            with st.spinner("임베딩 중..."):
                try:
                    os.makedirs(config.UPLOADS_PATH, exist_ok=True)
                    save_path = os.path.join(config.UPLOADS_PATH, fname)
                    with open(save_path, "wb") as f:
                        f.write(raw)
                    count = process_and_store(save_path, fname)
                    msg = ("success", f"✅ {count}개 청크 저장 완료")
                    st.session_state["logs"].append(
                        f"📄 파일 저장: {fname} ({count}청크)"
                    )
                except Exception as e:
                    msg = ("error", f"❌ {str(e)}")

    elif action == "delete_file":
        fname = data.get("fname")
        if fname:
            try:
                delete_by_filename(fname)
                st.session_state["logs"].append(f"🗑️ 파일 삭제: {fname}")
                msg = ("success", f"'{fname}' 삭제 완료")
            except Exception as e:
                msg = ("error", f"삭제 실패: {str(e)}")

    elif action == "go_detail":
        st.session_state["doc_modal_view"] = "detail"
        st.session_state["doc_modal_file"] = data.get("fname")

    elif action == "go_list":
        st.session_state["doc_modal_view"] = "list"

    if msg:
        getattr(st, msg[0])(msg[1])

    view = st.session_state.get("doc_modal_view", "list")

    if view == "detail":
        fname = st.session_state.get("doc_modal_file", "")
        col_back, col_title = st.columns([1, 6])
        with col_back:
            if st.button("← 뒤로", key="back_to_list"):
                _set_action("go_list")
        with col_title:
            st.markdown(f"**📄 {fname}** 청크 상세")
        st.divider()
        chunks = get_file_chunks(fname)
        if not chunks:
            st.info("청크 내용을 불러올 수 없습니다.")
            return
        st.caption(f"총 {len(chunks)}개 청크")
        for i, chunk in enumerate(chunks):
            with st.expander(f"청크 {i+1}  ({len(chunk)}자)", expanded=(i == 0)):
                st.text(chunk)
        return

    tab_upload, tab_list = st.tabs(["⬆️ 파일 업로드", "📋 저장된 파일"])

    with tab_upload:
        uploaded = st.file_uploader(
            "파일 선택 (PDF · TXT · DOCX)",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
            key="doc_uploader",
        )
        if uploaded:
            st.caption(f"선택: **{uploaded.name}**  ({_fmt_size(uploaded.size)})")
        if st.button("💾 벡터 DB에 저장", use_container_width=True, key="upload_btn"):
            if uploaded is None:
                st.warning("파일을 선택하세요.")
            else:
                _set_action("upload_file", raw=uploaded.getvalue(), fname=uploaded.name)

    with tab_list:
        try:
            file_list = get_file_list()
        except Exception as e:
            st.error(f"파일 목록 조회 오류: {str(e)}")
            return

        if not file_list:
            st.info("저장된 파일이 없습니다.")
        else:
            st.caption(f"총 {len(file_list)}개 파일")
            h0, h1, h2, h3, h4 = st.columns([4, 1, 2, 3, 1])
            for col, label in zip([h0,h1,h2,h3,h4], ["파일명","청크","용량","등록일",""]):
                col.markdown(
                    f"<span style='font-size:11px;font-weight:600;color:#888'>"
                    f"{label}</span>", unsafe_allow_html=True)
            st.divider()
            for f_info in file_list:
                fname  = f_info["filename"]
                chunks = f_info["chunks"]
                size   = _fmt_size(f_info.get("file_size", 0))
                date   = f_info.get("upload_date", "-")
                c0,c1,c2,c3,c4 = st.columns([4,1,2,3,1])
                with c0:
                    if st.button(f"📄 {fname[:22]}", key=f"fn_{fname}",
                                 use_container_width=True):
                        _set_action("go_detail", fname=fname)
                with c1:
                    st.markdown(
                        f"<div style='font-size:12px;padding-top:6px'>{chunks}</div>",
                        unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f"<div style='font-size:12px;padding-top:6px'>{size}</div>",
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(
                        f"<div style='font-size:11px;padding-top:6px;color:#888'>"
                        f"{date}</div>", unsafe_allow_html=True)
                with c4:
                    if st.button("🗑", key=f"del_{fname}", help="삭제"):
                        _set_action("delete_file", fname=fname)


# ── 프롬프트 / 퓨샷 관리 모달 ─────────────────────────────────────────
@st.dialog("⚙️ 프롬프트 & 퓨샷 관리", width="large")
def prompt_fewshot_modal():

    # ── 액션 처리 ────────────────────────────────────────────────────
    action, data = _pop_action()
    msg = None

    if action == "save_prompt":
        update_system_prompt(data.get("prompt", ""))
        st.session_state["logs"].append("⚙️ 시스템 프롬프트 업데이트")
        msg = ("success", "저장 완료")

    elif action == "reset_prompt":
        reset_to_default()
        msg = ("success", "기본값으로 복원됐습니다.")

    elif action == "add_fewshot":
        q, a = data.get("q", ""), data.get("a", "")
        if q and a:
            add_fewshot(q, a)
            # 추가 후 입력창 초기화
            st.session_state.pop("_new_fs_q", None)
            st.session_state.pop("_new_fs_a", None)
            st.session_state["logs"].append("📝 새 퓨샷 등록")
            msg = ("success", "퓨샷이 추가됐습니다.")
        else:
            msg = ("warning", "질문과 답변을 모두 입력하세요.")

    elif action == "update_fewshot":
        update_fewshot(data["idx"], data["q"], data["a"])
        st.session_state.pop("_edit_idx", None)   # 편집 모드 종료
        msg = ("success", "수정 완료")

    elif action == "delete_fewshot":
        delete_fewshot(data["idx"])
        st.session_state.pop("_edit_idx", None)
        st.session_state["logs"].append(f"📝 퓨샷 {data['idx']+1} 삭제")
        msg = ("success", "삭제 완료")

    elif action == "start_edit":
        st.session_state["_edit_idx"] = data["idx"]

    elif action == "cancel_edit":
        st.session_state.pop("_edit_idx", None)

    if msg:
        getattr(st, msg[0])(msg[1])

    # 항상 최신 데이터 로드
    data_cfg = load_prompt_config()

    tab_sys, tab_few = st.tabs(["🤖 시스템 프롬프트", "📝 퓨샷 관리"])

    # ── 시스템 프롬프트 탭 ───────────────────────────────────────────
    with tab_sys:
        st.caption("모든 질문에 공통으로 적용되는 역할 지시문")
        new_prompt = st.text_area(
            label="시스템 프롬프트",
            value=data_cfg.get("system_prompt", ""),
            height=200,
            label_visibility="collapsed",
            key="sys_prompt_input",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 저장", use_container_width=True, key="save_sys"):
                _set_action("save_prompt", prompt=new_prompt)
        with c2:
            if st.button("🔄 기본값 복원", use_container_width=True, key="reset_sys"):
                _set_action("reset_prompt")

    # ── 퓨샷 관리 탭 ────────────────────────────────────────────────
    with tab_few:
        fewshots  = data_cfg.get("fewshots", [])
        edit_idx  = st.session_state.get("_edit_idx", None)

        if fewshots:
            st.caption(f"등록된 퓨샷: {len(fewshots)}개")

            for i, fs in enumerate(fewshots):

                # ── 일반 행: expander(내용 보기) + 수정/삭제 버튼을 바깥 column에 배치 ──
                # expander 안에 button 없음 → expander 클릭 시 button trigger 방지
                col_exp, col_upd, col_del = st.columns([6, 1, 1])

                with col_exp:
                    with st.expander(f"퓨샷 {i+1}: {fs['question'][:35]}"):
                        st.markdown(f"**Q.** {fs['question']}")
                        st.markdown(f"**A.** {fs['answer']}")

                with col_upd:
                    if st.button("✏️", key=f"upd_btn_{i}", help="수정"):
                        _set_action("start_edit", idx=i)

                with col_del:
                    if st.button("🗑", key=f"del_btn_{i}", help="삭제"):
                        _set_action("delete_fewshot", idx=i)

                # ── 인라인 편집 영역 (수정 버튼 클릭 시 해당 퓨샷 아래 표시) ──
                if edit_idx == i:
                    st.markdown(f"**퓨샷 {i+1} 수정**")
                    eq = st.text_area("질문", value=fs["question"],
                                      key=f"edit_q_{i}")
                    ea = st.text_area("답변", value=fs["answer"],
                                      key=f"edit_a_{i}")
                    s1, s2 = st.columns(2)
                    with s1:
                        if st.button("💾 저장", key=f"save_edit_{i}",
                                     use_container_width=True):
                            _set_action("update_fewshot", idx=i, q=eq, a=ea)
                    with s2:
                        if st.button("❌ 취소", key=f"cancel_edit_{i}",
                                     use_container_width=True):
                            _set_action("cancel_edit")
                    st.divider()

        else:
            st.info("등록된 퓨샷이 없습니다.")

        st.markdown("---")
        st.markdown("**새 퓨샷 추가**")

        # key를 _new_fs_q / _new_fs_a 로 관리 (추가 후 pop으로 초기화)
        nq = st.text_area(
            "질문 예시",
            value=st.session_state.get("_new_fs_q", ""),
            key="_new_fs_q",
            height=70,
        )
        na = st.text_area(
            "답변 예시",
            value=st.session_state.get("_new_fs_a", ""),
            key="_new_fs_a",
            height=90,
        )
        if st.button("➕ 추가", use_container_width=True, key="add_fs"):
            _set_action("add_fewshot", q=nq, a=na)
