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
from ui.modals import chunk_detail_modal, prompt_fewshot_modal


def init_session_state():
    """
    Streamlit session_state 초기값 설정.
    앱 최초 실행 또는 새 세션 시작 시 1회 실행된다.
    """
    defaults = {
        "chat_history": [],          # 대화 기록 [{role, content, sql?}]
        "logs": [],                  # 처리 흐름 로그 목록
        "a2a_messages": [],          # A2A 메시지 목록
        "show_chunk_modal": False,   # 청크 상세 모달 표시 여부
        "show_prompt_modal": False,  # 프롬프트 관리 모달 표시 여부
        "modal_file": None,          # 청크 모달에 표시할 파일명
        "use_mcp": False,            # MCP 경유 여부
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
    # 청크 상세 모달
    if st.session_state.get("show_chunk_modal") and st.session_state.get("modal_file"):
        chunk_detail_modal(st.session_state["modal_file"])
        st.session_state["show_chunk_modal"] = False

    # 프롬프트/퓨샷 관리 모달
    if st.session_state.get("show_prompt_modal"):
        prompt_fewshot_modal()
        st.session_state["show_prompt_modal"] = False
