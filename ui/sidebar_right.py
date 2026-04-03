"""
ui/sidebar_right.py
- 오른쪽 패널: 에이전트 내부 처리 흐름 로그를 실시간으로 누적 표시
- 라우팅 결정, RAG 검색, DB 조회, 오류 등 모든 단계 로그 출력
- A2A 메시지 흐름도 함께 표시
"""

import streamlit as st


# 로그 아이콘 → 배경색 매핑 (CS 스타일)
_LOG_STYLE = {
    "🔀": "#e8f4fd",   # 라우팅
    "📚": "#e8fdf0",   # RAG
    "🗄️": "#fef9e8",  # DB
    "💬": "#f4e8fd",   # 일반
    "✅": "#e8fdf0",   # 성공
    "❌": "#fde8e8",   # 오류
    "⚠️": "#fef9e8",  # 경고
    "✨": "#e8f4fd",   # 생성
    "📄": "#f0f4fd",   # 파일
    "🔌": "#fde8f4",   # MCP
    "⚙️": "#f4f4f4",  # 설정
    "🗑️": "#fde8e8",  # 삭제
    "📝": "#f4e8fd",   # 퓨샷
}


def _get_log_style(log: str) -> str:
    """로그 내용에 따라 배경색 결정"""
    for icon, color in _LOG_STYLE.items():
        if icon in log:
            return color
    return "#f9f9f9"


def render_sidebar_right():
    """오른쪽 로그 패널 렌더링"""
    st.markdown("#### 📊 처리 흐름 로그")

    # 로그 제어 버튼
    col1, col2 = st.columns([3, 2])
    with col1:
        st.caption(f"누적 {len(st.session_state.get('logs', []))}개")
    with col2:
        if st.button("🗑️ 초기화", key="clear_logs_btn", use_container_width=True):
            st.session_state["logs"] = []
            st.rerun()

    st.markdown("---")

    logs = st.session_state.get("logs", [])

    if not logs:
        st.caption("아직 로그가 없습니다.\n채팅을 시작하면 처리 흐름이 표시됩니다.")
        return

    # 최신 로그가 위에 표시되도록 역순 출력
    for log in reversed(logs):
        bg = _get_log_style(log)
        st.markdown(
            f"""<div style='
                font-size: 11px;
                line-height: 1.5;
                padding: 4px 8px;
                margin: 2px 0;
                border-left: 3px solid #ddd;
                background: {bg};
                border-radius: 2px;
                word-break: break-all;
            '>{log}</div>""",
            unsafe_allow_html=True,
        )

    # A2A 메시지 섹션
    a2a_msgs = st.session_state.get("a2a_messages", [])
    if a2a_msgs:
        st.markdown("---")
        st.markdown("**🔗 A2A 메시지 흐름**")
        for msg in reversed(a2a_msgs[-10:]):   # 최신 10개만
            st.markdown(
                f"""<div style='font-size:10px; color:#666; padding:3px 6px;
                    border-left:2px solid #a78bfa; margin:1px 0; background:#f5f3ff'>
                    {msg.get('sender','?')} → {msg.get('receiver','?')} : {msg.get('content','')[:60]}
                </div>""",
                unsafe_allow_html=True,
            )
