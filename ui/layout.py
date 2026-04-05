"""
ui/layout.py
- 전체 3패널 레이아웃 조립
  Left Sidebar | Center Chat | Right Log Panel
- session_state 초기화 담당
- 모달 팝업 트리거 조건 처리
"""

import streamlit as st
from ui.sidebar_left import render_sidebar_left
from ui.chat import render_chat
from ui.sidebar_right import render_sidebar_right
from ui.modals import doc_management_modal, prompt_fewshot_modal


def init_session_state():
    """
    Streamlit session_state 초기값 설정.
    앱 최초 실행 또는 새 세션 시작 시 1회 실행된다.
    """
    defaults = {
        "chat_history": [],
        "logs": [],
        "a2a_messages": [],
        "show_doc_modal": False,       # 문서 관리 모달
        "show_prompt_modal": False,    # 프롬프트 관리 모달
        "doc_modal_view": "list",      # 문서 모달 내부 뷰: list | detail
        "doc_modal_file": None,        # 상세 볼 파일명
        "use_mcp": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_layout():
    """전체 레이아웃 렌더링 (진입점)"""
    init_session_state()

    # ── 왼쪽 사이드바 (st.sidebar) ──────────────────────────────────
    render_sidebar_left()

    # ── 메인 영역: 채팅(2) + 로그(1) ──────────────────────────────
    center_col, right_col = st.columns([2, 1], gap="medium")

    with center_col:
        render_chat()

    with right_col:
        render_sidebar_right()

    # ── 모달 팝업 처리 ────────────────────────────────────────────
    # @st.dialog 내부에서 st.rerun() 호출 시 모달이 닫히므로
    # 플래그를 즉시 False로 바꾸지 않고, 모달 함수 자체가 닫힘을 관리한다.
    if st.session_state.get("show_doc_modal"):
        doc_management_modal()

    if st.session_state.get("show_prompt_modal"):
        prompt_fewshot_modal()
