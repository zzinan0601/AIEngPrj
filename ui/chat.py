"""
ui/chat.py
- 중앙 채팅 화면 렌더링
- 사용자 입력 → LangGraph 그래프 실행 → 스트리밍 답변 표시
- 대화 기록은 session_state에 유지
- 처리 중 로그/A2A 메시지를 session_state에 축적
"""

import streamlit as st
from ui.prompt_manager import load_prompt_config


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
        "db_results": "",
        "generated_sql": "",
        "answer": "",
        "logs": [],
        "a2a_messages": [],
        "iteration": 0,
        "prompt_config": prompt_config,
        "use_mcp": use_mcp,
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

            # DB 노드가 실행됐으면 SQL도 표시
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
            logs = result.get("logs", [])
            a2a_msgs = result.get("a2a_messages", [])
            sql = result.get("generated_sql", "")
            route = result.get("route", "")
            context = result.get("context", "")

            # 답변 표시
            status_placeholder.empty()

            # 라우팅 경로 + 컨텍스트 확인 뱃지 (맨 위에 표시)
            if route:
                route_emoji = {"rag": "📚", "db": "🗄️", "both": "📚🗄️", "general": "💬"}.get(route, "")
                if route == "rag" and not context:
                    st.warning("⚠️ 문서를 검색했지만 관련 내용을 찾지 못했습니다. 문서가 정상적으로 업로드·저장됐는지 확인하세요.")
                else:
                    st.caption(f"{route_emoji} 처리 경로: **{route.upper()}**")

            st.markdown(answer)

            # SQL 표시 (DB 경로인 경우)
            if sql and sql not in ("", "(MCP 경유)"):
                with st.expander("🗄️ 실행된 SQL", expanded=False):
                    st.code(sql, language="sql")

            # 세션 상태 업데이트
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer,
                "sql": sql,
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
