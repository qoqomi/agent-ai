"""
에이전트별 워크플로 개요
- 승현: startup_search_node (타깃 스타트업 탐색)
- 승연: tech_team_summary (기술/팀 평가)
- 민주: market_eval_agent (시장 분석)
- 경남: investment_decision_agent (투자 의사결정)
- 보고: report_generator_agent (투자 메모 작성)
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph

from start_agent import startup_search_node
from market_eval_agent import market_eval_agent
from tech_team_agent import tech_team_summary
from decision_module import investment_decision_agent
from report_agent import report_generator_agent

load_dotenv()


class AgentState(TypedDict):
    startup_name: str
    startup_info: dict
    tech_summary: str
    market_analysis: str
    investment_decision: int
    decision_reason: str
    report: str
    iteration_count: int


def should_continue(state: AgentState) -> Literal["report_generator", "startup_search_agent"]:
    """Loop back for non-invest outcomes until we exhaust five attempts."""
    decision = state.get("investment_decision", 0)
    iteration_count = state.get("iteration_count", 0) or 0

    if decision == 1:
        return "report_generator"

    if iteration_count >= 5:
        return "report_generator"

    return "startup_search_agent"


def create_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("startup_search_agent", startup_search_node)
    graph.add_node("tech_team_agent", tech_team_summary)
    graph.add_node("market_eval_agent", market_eval_agent)
    graph.add_node("investment_decision_agent", investment_decision_agent)
    graph.add_node("report_generator", report_generator_agent)

    graph.set_entry_point("startup_search_agent")

    graph.add_edge("startup_search_agent", "tech_team_agent")
    graph.add_edge("tech_team_agent", "market_eval_agent")
    graph.add_edge("market_eval_agent", "investment_decision_agent")

    graph.add_conditional_edges(
        "investment_decision_agent",
        should_continue,
        {
            "report_generator": "report_generator",
            "startup_search_agent": "startup_search_agent",
        },
    )

    graph.set_finish_point("report_generator")

    return graph.compile()


def _initial_state() -> AgentState:
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
    print(f"Decision   : {'Invest' if final_state['investment_decision'] else 'Decline'}")
    print(f"Reason     : {final_state['decision_reason']}")
    print(f"Report     : {final_state.get('report', '(none)')}")
    print(f"Iterations : {final_state['iteration_count']}")
