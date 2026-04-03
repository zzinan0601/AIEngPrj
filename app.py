"""
app.py
- Streamlit 앱 진입점
- 페이지 설정 후 ui/layout.py의 render_layout() 호출
- 실행: streamlit run app.py
"""

import streamlit as st

# ── 페이지 기본 설정 (반드시 첫 번째 st 호출) ──────────────────────
st.set_page_config(
    page_title="RAG AI 어시스턴트",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 전역 CSS (CS 화면 스타일) ─────────────────────────────────────
st.markdown(
    """
    <style>
    /* 전체 폰트 크기 다운 (CS 화면 느낌) */
    html, body, [class*="css"] { font-size: 14px; }

    /* 사이드바 너비 조정 */
    [data-testid="stSidebar"] { min-width: 240px; max-width: 260px; }
    [data-testid="stSidebar"] .block-container { padding: 1rem 0.6rem; }

    /* 파일 목록 버튼 폰트 소형화 */
    [data-testid="stSidebar"] button { font-size: 11px !important; padding: 2px 6px !important; }

    /* 채팅 입력창 */
    [data-testid="stChatInput"] { border-radius: 8px; }

    /* 오른쪽 로그 패널 스크롤 */
    .log-container { max-height: 70vh; overflow-y: auto; }

    /* 구분선 여백 축소 */
    hr { margin: 6px 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 레이아웃 렌더링 ────────────────────────────────────────────────
from ui.layout import render_layout

render_layout()
