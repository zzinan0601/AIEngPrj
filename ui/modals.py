"""
ui/modals.py
- 모든 버튼을 on_click + args/kwargs 방식으로 처리
- on_click 콜백은 rerun 이전에 실행되므로 두 번 클릭 문제 해결
- 수정/삭제는 expander 바깥 column에 배치 (expander 내부 button trigger 방지)
"""

import os
import streamlit as st
import config

from rag.vector_store import get_file_list, get_file_chunks, delete_by_filename
from rag.pipeline import process_and_storefrom ui.prompt_manager import (
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


# ── on_click 콜백 함수 모음 ───────────────────────────────────────────
# on_click 은 rerun 직전에 실행 → 다음 rerun에서 결과 즉시 반영

def _cb_go_detail(fname: str):
    st.session_state["doc_modal_view"] = "detail"
    st.session_state["doc_modal_file"] = fname

def _cb_go_list():
    st.session_state["doc_modal_view"] = "list"

def _cb_delete_file(fname: str):
    try:
        delete_by_filename(fname)
        st.session_state["logs"].append(f"🗑️ 파일 삭제: {fname}")
        st.session_state["_modal_msg"] = ("success", f"'{fname}' 삭제 완료")
    except Exception as e:
        st.session_state["_modal_msg"] = ("error", f"삭제 실패: {str(e)}")

def _cb_delete_schema(fname: str):
    """DB 스키마 임베딩 삭제 (schema 컬렉션)"""
    try:
        delete_by_filename(fname, collection_name=config.SCHEMA_COLLECTION)
        db_tag = fname.replace("db_schema_", "").upper()
        st.session_state["logs"].append(f"🗑️ DB 스키마 삭제: {db_tag}")
        st.session_state["_modal_msg"] = ("success", f"{db_tag} 스키마 삭제 완료")
    except Exception as e:
        st.session_state["_modal_msg"] = ("error", f"삭제 실패: {str(e)}")

def _cb_save_prompt(prompt_key: str):
    prompt = st.session_state.get(prompt_key, "")
    update_system_prompt(prompt)
    st.session_state["logs"].append("⚙️ 시스템 프롬프트 업데이트")
    st.session_state["_modal_msg"] = ("success", "저장 완료")

def _cb_reset_prompt():
    reset_to_default()
    st.session_state["_modal_msg"] = ("success", "기본값으로 복원됐습니다.")

def _cb_add_fewshot(q_key: str, a_key: str):
    q = st.session_state.get(q_key, "").strip()
    a = st.session_state.get(a_key, "").strip()
    if q and a:
        add_fewshot(q, a)
        st.session_state[q_key] = ""
        st.session_state[a_key] = ""
        st.session_state["logs"].append("📝 새 퓨샷 등록")
        st.session_state["_modal_msg"] = ("success", "퓨샷이 추가됐습니다.")
    else:
        st.session_state["_modal_msg"] = ("warning", "질문과 답변을 모두 입력하세요.")

def _cb_start_edit(idx: int):
    st.session_state["_edit_idx"] = idx

def _cb_cancel_edit():
    st.session_state.pop("_edit_idx", None)

def _cb_save_edit(idx: int, q_key: str, a_key: str):
    q = st.session_state.get(q_key, "")
    a = st.session_state.get(a_key, "")
    update_fewshot(idx, q, a)
    st.session_state.pop("_edit_idx", None)
    st.session_state["_modal_msg"] = ("success", "수정 완료")

def _cb_delete_fewshot(idx: int):
    delete_fewshot(idx)
    st.session_state.pop("_edit_idx", None)
    st.session_state["logs"].append(f"📝 퓨샷 {idx+1} 삭제")
    st.session_state["_modal_msg"] = ("success", "삭제 완료")

def _cb_upload_file(file_key: str):
    """업로드 버튼 on_click - file_uploader 값을 session_state에서 읽어 처리"""
    uploaded = st.session_state.get(file_key)
    if uploaded is None:
        st.session_state["_modal_msg"] = ("warning", "파일을 선택하세요.")
        return
    try:
        os.makedirs(config.UPLOADS_PATH, exist_ok=True)
        save_path = os.path.join(config.UPLOADS_PATH, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getvalue())
        count = process_and_store(save_path, uploaded.name)
        st.session_state["logs"].append(
            f"📄 파일 저장: {uploaded.name} ({count}청크)"
        )
        st.session_state["_modal_msg"] = ("success", f"✅ {count}개 청크 저장 완료")
    except Exception as e:
        st.session_state["_modal_msg"] = ("error", f"❌ {str(e)}")


def _show_msg():
    """저장된 메시지가 있으면 표시하고 클리어"""
    msg = st.session_state.pop("_modal_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])


# ── 문서 관리 모달 ─────────────────────────────────────────────────────
@st.dialog("📂 문서 관리", width="large")
def doc_management_modal():
    _show_msg()

    view = st.session_state.get("doc_modal_view", "list")

    # ── 청크 상세 화면 ───────────────────────────────────────────────
    if view == "detail":
        fname = st.session_state.get("doc_modal_file", "")
        col_back, col_title = st.columns([1, 6])
        with col_back:
            st.button("← 뒤로", key="back_to_list",
                      on_click=_cb_go_list)
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

    # ── 목록·업로드 화면 ─────────────────────────────────────────────
    tab_upload, tab_list, tab_schema = st.tabs(
        ["⬆️ 파일 업로드", "📋 저장된 파일", "🗄️ DB 스키마 현황"]
    )

    with tab_upload:
        st.file_uploader(
            "파일 선택 (PDF · TXT · DOCX)",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
            key="doc_uploader",
        )
        st.button(
            "💾 벡터 DB에 저장",
            use_container_width=True,
            key="upload_btn",
            on_click=_cb_upload_file,
            kwargs={"file_key": "doc_uploader"},
        )

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
            h0,h1,h2,h3,h4 = st.columns([4,1,2,3,1])
            for col, label in zip([h0,h1,h2,h3,h4],
                                   ["파일명","청크","용량","등록일",""]):
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
                    st.button(
                        f"📄 {fname[:22]}", key=f"fn_{fname}",
                        use_container_width=True,
                        on_click=_cb_go_detail,
                        args=(fname,),
                    )
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
                    st.button(
                        "🗑", key=f"del_{fname}", help="삭제",
                        on_click=_cb_delete_file,
                        args=(fname,),
                    )

    # ── DB 스키마 현황 탭 ─────────────────────────────────────────────
    with tab_schema:
        try:
            schema_list = get_file_list(collection_name=config.SCHEMA_COLLECTION)
        except Exception as e:
            st.error(f"스키마 목록 조회 오류: {str(e)}")
            return

        # db_schema_{type} 형식만 필터링
        schema_list = [
            f for f in schema_list
            if f["filename"].startswith("db_schema_")
        ]

        if not schema_list:
            st.info("임베딩된 DB 스키마가 없습니다.\n사이드바의 'DB 스키마 임베딩' 버튼으로 등록하세요.")
        else:
            st.caption(f"총 {len(schema_list)}개 DB 스키마 임베딩됨")

            # DB 타입별 섹션
            for f_info in schema_list:
                fname  = f_info["filename"]                  # db_schema_sqlite 등
                db_tag = fname.replace("db_schema_", "").upper()  # SQLITE 등
                chunks = f_info["chunks"]
                date   = f_info.get("upload_date", "-")

                # 헤더 행
                h0, h1, h2 = st.columns([4, 2, 1])
                with h0:
                    st.markdown(
                        f"<span style='font-size:13px;font-weight:600'>"
                        f"🗄️ {db_tag}</span>",
                        unsafe_allow_html=True,
                    )
                with h1:
                    st.markdown(
                        f"<div style='font-size:11px;color:#888;padding-top:4px'>"
                        f"청크 {chunks}개 · {date}</div>",
                        unsafe_allow_html=True,
                    )
                with h2:
                    st.button(
                        "🗑", key=f"del_schema_{fname}", help=f"{db_tag} 스키마 삭제",
                        on_click=_cb_delete_schema,
                        args=(fname,),
                    )

                # 청크 상세 (expander)
                with st.expander(f"{db_tag} 스키마 내용 보기"):
                    schema_chunks = get_file_chunks(
                        fname, collection_name=config.SCHEMA_COLLECTION
                    )
                    for i, chunk in enumerate(schema_chunks):
                        st.code(chunk, language="sql")

                st.divider()


# ── 프롬프트 / 퓨샷 관리 모달 ─────────────────────────────────────────
@st.dialog("⚙️ 프롬프트 & 퓨샷 관리", width="large")
def prompt_fewshot_modal():
    _show_msg()

    data_cfg = load_prompt_config()
    tab_sys, tab_few = st.tabs(["🤖 시스템 프롬프트", "📝 퓨샷 관리"])

    # ── 시스템 프롬프트 탭 ───────────────────────────────────────────
    with tab_sys:
        st.caption("모든 질문에 공통으로 적용되는 역할 지시문")
        st.text_area(
            label="시스템 프롬프트",
            value=data_cfg.get("system_prompt", ""),
            height=200,
            label_visibility="collapsed",
            key="sys_prompt_input",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.button(
                "💾 저장", use_container_width=True, key="save_sys",
                on_click=_cb_save_prompt,
                kwargs={"prompt_key": "sys_prompt_input"},
            )
        with c2:
            st.button(
                "🔄 기본값 복원", use_container_width=True, key="reset_sys",
                on_click=_cb_reset_prompt,
            )

    # ── 퓨샷 관리 탭 ────────────────────────────────────────────────
    with tab_few:
        fewshots = data_cfg.get("fewshots", [])
        edit_idx = st.session_state.get("_edit_idx", None)

        if fewshots:
            st.caption(f"등록된 퓨샷: {len(fewshots)}개")
            for i, fs in enumerate(fewshots):
                col_exp, col_upd, col_del = st.columns([6, 1, 1])
                with col_exp:
                    # expander 안에 button 없음 → trigger 방지
                    with st.expander(f"퓨샷 {i+1}: {fs['question'][:35]}"):
                        st.markdown(f"**Q.** {fs['question']}")
                        st.markdown(f"**A.** {fs['answer']}")
                with col_upd:
                    st.button(
                        "✏️", key=f"upd_btn_{i}", help="수정",
                        on_click=_cb_start_edit,
                        args=(i,),
                    )
                with col_del:
                    st.button(
                        "🗑", key=f"del_btn_{i}", help="삭제",
                        on_click=_cb_delete_fewshot,
                        args=(i,),
                    )

                # 인라인 편집 영역
                if edit_idx == i:
                    st.markdown(f"**퓨샷 {i+1} 수정**")
                    q_key = f"edit_q_{i}"
                    a_key = f"edit_a_{i}"
                    st.text_area("질문", value=fs["question"], key=q_key)
                    st.text_area("답변", value=fs["answer"],   key=a_key)
                    s1, s2 = st.columns(2)
                    with s1:
                        st.button(
                            "💾 저장", key=f"save_edit_{i}",
                            use_container_width=True,
                            on_click=_cb_save_edit,
                            args=(i, q_key, a_key),
                        )
                    with s2:
                        st.button(
                            "❌ 취소", key=f"cancel_edit_{i}",
                            use_container_width=True,
                            on_click=_cb_cancel_edit,
                        )
                    st.divider()
        else:
            st.info("등록된 퓨샷이 없습니다.")

        st.markdown("---")
        st.markdown("**새 퓨샷 추가**")
        st.text_area("질문 예시", key="new_fs_q", height=70)
        st.text_area("답변 예시", key="new_fs_a", height=90)
        st.button(
            "➕ 추가", use_container_width=True, key="add_fs",
            on_click=_cb_add_fewshot,
            kwargs={"q_key": "new_fs_q", "a_key": "new_fs_a"},
        )
