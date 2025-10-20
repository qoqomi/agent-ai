"""
담당자별 에이전트:
- 현준: startup_search_agent (스타트업 탐색)
- 승연: tech_summary_agent (기술 요약)
- 민주: market_eval_agent (시장성 평가)
- 경남: investor_insight_agent (투자자 인사이트)

+ investment_decision_agent (투자 판단)
"""

# API KEY Loading
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END

# 현준 모듈
from start_agent import startup_search_node

# 민주 모듈
from market_eval_agent import market_eval_agent

# 승연 모듈
from tech_team_agent import tech_team_summary

# 경남 모듈
from decision_module import investment_decision_agent

# 승연 모듈2
from report_agent import report_generator_agent

# ====== AgentState (업데이트된 스키마) ======
class AgentState(TypedDict):
    startup_name: str
    startup_info: dict
    tech_summary: str
    market_analysis: str
    investment_decision: int     # 투자=1, 비투자=0
    decision_reason: str         # 판단 근거 요약 문자열
    report: str                  # 상세 리포트
    iteration_count: int
    

def should_continue(state: AgentState) -> Literal["report_writer", "startup_search"]:
    """Loop back for non-invest decisions until we hit five attempts."""
    decision = state.get("investment_decision", 0)
    iteration_count = state.get("iteration_count", 0) or 0

    if decision == 1:
        return "report_writer"

    if iteration_count >= 5:
        return "report_writer"

    return "startup_search"

def create_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("startup_search", startup_search_node) # 현준, 완
    graph.add_node("tech_summary", tech_team_summary) # 승연, 완
    graph.add_node("market_eval", market_eval_agent) # 민주, 완
    graph.add_node("investment_decision", investment_decision_agent) # 경남, 완
    graph.add_node("report_writer", report_generator_agent) # 승연, 완

    graph.set_entry_point("startup_search")

    graph.add_edge("startup_search", "tech_summary")
    graph.add_edge("tech_summary", "market_eval")
    graph.add_edge("market_eval", "investment_decision")

    graph.add_conditional_edges(
        "investment_decision",
        should_continue,
        {"report_writer": "report_writer", "startup_search": "startup_search"},
    )

    graph.set_finish_point("report_writer")

    return graph.compile()


def _initial_state() -> AgentState:
    """Helper to create a LangGraph-compatible initial state."""
    return {
        "startup_name": "",
        "startup_info": {},
        "tech_summary": "",
        "market_analysis": "",
        "investment_decision": 0,
        "decision_reason": "",
        "report": "",
        "iteration_count": 0,
    }


if __name__ == "__main__":
    workflow = create_workflow()
    final_state = workflow.invoke(_initial_state())

    print("\n=== Workflow finished ===")
    print(f"Decision : {'Invest' if final_state['investment_decision'] else 'Decline'}")
    print(f"Reason   : {final_state['decision_reason']}")
    print(f"Report   : {final_state.get('report', '(none)')}")
    print(f"Iterations: {final_state['iteration_count']}")