"""
ui/chat.py
- 중앙 채팅 화면 렌더링
- 사용자 입력 → LangGraph 그래프 실행 → 스트리밍 답변 표시
- 대화 기록은 session_state에 유지
- 처리 중 로그/A2A 메시지를 session_state에 축적
"""

import streamlit as st
from ui.prompt_manager import load_prompt_config


def _render_chart(chart_config: dict):
    """
    chart_config 에 따라 plotly 차트를 렌더링.
    plotly 없으면 streamlit 기본 차트로 폴백.
    """
    import pandas as pd

    chart_type = chart_config.get("type", "bar")
    title      = chart_config.get("title", "")
    x_labels   = chart_config.get("x_labels", [])
    series     = chart_config.get("series", [])
    summary    = chart_config.get("summary", "")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for s in series:
            name   = s.get("name", "")
            values = s.get("values", [])
            if chart_type == "bar":
                fig.add_trace(go.Bar(x=x_labels, y=values, name=name))
            elif chart_type == "line":
                fig.add_trace(go.Scatter(x=x_labels, y=values, name=name, mode="lines+markers"))
            elif chart_type == "pie":
                fig.add_trace(go.Pie(labels=x_labels, values=values, name=name))
            elif chart_type == "scatter":
                fig.add_trace(go.Scatter(x=x_labels, y=values, name=name, mode="markers"))

        fig.update_layout(title=title, height=400)
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # plotly 없으면 st 기본 차트로 폴백
        if series and x_labels:
            df_data = {s["name"]: s["values"] for s in series}
            df = pd.DataFrame(df_data, index=x_labels)
            if chart_type == "line":
                st.line_chart(df)
            else:
                st.bar_chart(df)

    if summary:
        st.caption(f"📊 {summary}")


def _run_graph(question: str, prompt_config: dict, use_mcp: bool) -> dict:
    """
    LangGraph 그래프를 실행하고 결과 state를 반환.
    그래프는 싱글톤으로 관리 (agent/graph.py).
    """
    from agent.graph import get_graph

    initial_state = {
        "question": question,
        "route": "",
        "context": "",
        "sources": [],
        "db_results": "",
        "generated_sql": "",
        "answer": "",
        "logs": [],
        "a2a_messages": [],
        "iteration": 0,
        "prompt_config": prompt_config,
        "use_mcp": use_mcp,
        "selected_model": st.session_state.get("selected_model", None),
        "db_type": st.session_state.get("db_type", None),
        "chart_request": False,
        "chart_config": None,
        "chat_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.get("chat_history", [])
            if m["role"] in ("user", "assistant")
        ],
    }

    graph = get_graph()
    result = graph.invoke(initial_state)
    return result


def render_chat():
    """중앙 채팅 영역 렌더링"""
    st.markdown("### 💬 AI 어시스턴트")

    # 대화 기록 표시
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 이전 차트 재렌더링
            if msg.get("chart_config"):
                _render_chart(msg["chart_config"])

            # 이전 SQL 표시
            if msg.get("sql"):
                with st.expander("🗄️ 실행된 SQL", expanded=False):
                    st.code(msg["sql"], language="sql")

    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요... (문서 검색 / DB 조회 / 일반 대화 모두 가능)")

    if not user_input:
        return

    # 사용자 메시지 표시
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 생성
    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            status_placeholder.markdown("⏳ 처리 중...")

            # 현재 설정 로드
            prompt_config = load_prompt_config()
            use_mcp = st.session_state.get("use_mcp", False)

            # LangGraph 실행
            result = _run_graph(user_input, prompt_config, use_mcp)

            answer = result.get("answer", "답변을 생성할 수 없습니다.")
            logs        = result.get("logs", [])
            a2a_msgs    = result.get("a2a_messages", [])
            sql         = result.get("generated_sql", "")
            route       = result.get("route", "")
            context     = result.get("context", "")
            chart_config = result.get("chart_config")

            status_placeholder.empty()

            # 라우팅 경로 뱃지
            if route:
                route_emoji = {"rag":"📚","db":"🗄️","both":"📚🗄️","general":"💬"}.get(route,"")
                if route == "rag" and not context:
                    st.warning("⚠️ 문서를 검색했지만 관련 내용을 찾지 못했습니다.")
                else:
                    st.caption(f"{route_emoji} 처리 경로: **{route.upper()}**")

            st.markdown(answer)

            # ── 차트 렌더링 ─────────────────────────────────────────
            if chart_config and chart_config.get("type") != "none":
                _render_chart(chart_config)

            # SQL 표시
            if sql and sql not in ("", "(MCP 경유)"):
                with st.expander("🗄️ 실행된 SQL", expanded=False):
                    st.code(sql, language="sql")

            # 세션 상태 업데이트
            st.session_state["chat_history"].append({
                "role":         "assistant",
                "content":      answer,
                "sql":          sql,
                "chart_config": chart_config,   # 재표시를 위해 저장
            })
            st.session_state["logs"] = st.session_state.get("logs", []) + logs
            st.session_state["a2a_messages"] = (
                st.session_state.get("a2a_messages", []) + a2a_msgs
            )
            # 답변 생성 후 rerun 시 모달이 다시 열리지 않도록 플래그 초기화
            st.session_state["show_doc_modal"] = False
            st.session_state["show_prompt_modal"] = False

        except Exception as e:
            status_placeholder.empty()
            err_msg = f"처리 중 오류가 발생했습니다: {str(e)}"
            st.error(err_msg)
            st.session_state["logs"] = st.session_state.get("logs", []) + [f"❌ {err_msg}"]

    st.rerun()
