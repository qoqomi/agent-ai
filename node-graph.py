from langgraph.graph import StateGraph, START, END


# ========================================
# 스타트업 분석용 LangGraph 워크플로우를 생성하는 함수.
# 기술 분석과 시장 분석을 병렬로 수행한 뒤, 투자 판단 및 보고서를 생성함.

# Args:
#   AgentState (TypedDict): 그래프의 상태 정의 클래스
#   startup_search_agent (callable): 스타트업 탐색 Agent
#   tech_summary_agent (callable): 기술 요약 Agent
#   market_eval_agent (callable): 시장 분석 Agent
#   investment_decision_agent (callable): 투자 판단 Agent
#   report_writer_agent (callable): 보고서 생성 Agent
#   should_continue (callable): 다음 단계 전환 여부 판단 함수

# Returns:
#    graph (CompiledGraph): 컴파일된 LangGraph 그래프 객체
# ========================================


def build_startup_graph(
    AgentState,
    startup_search_agent,
    tech_summary_agent,
    market_eval_agent,
    investment_decision_agent,
    report_writer_agent,
    should_continue,
):

    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("startup_search", startup_search_agent)
    graph.add_node("tech_summary", tech_summary_agent)
    graph.add_node("market_analysis", market_eval_agent)
    graph.add_node("investment_decision", investment_decision_agent)
    graph.add_node("report_writer", report_writer_agent)

    # 시작점 설정
    graph.set_entry_point("startup_search")

    # startup_search 이후 두 노드가 병렬로 실행되도록 지정
    graph.add_conditional_edges(
        "startup_search",
        lambda state: ["tech_summary", "market_analysis"],  # 병렬 실행 대상 리스트 반환
        {"tech_summary": "tech_summary", "market_analysis": "market_analysis"},
    )

    # 두 병렬 노드가 끝난 후 investment_decision으로 converge
    graph.add_edge("tech_summary", "investment_decision")
    graph.add_edge("market_analysis", "investment_decision")

    # 판단 이후 보고서 작성 or 재탐색 루프
    graph.add_conditional_edges(
        "investment_decision",
        should_continue,
        {"report_writer": "report_writer", "startup_search": "startup_search"},
    )

    # 종료점 설정
    graph.set_finish_point("report_writer")

    # 컴파일된 그래프 반환
    return graph.compile()
