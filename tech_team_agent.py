from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import re

class AgentState(TypedDict):
    startup_name: str
    startup_info: dict
    tech_team_summary: dict
    market_opportunity: str
    investment_verdict: str
    report: str
    iteration_count: int


def tech_team_summary(state: AgentState) -> AgentState:
    """승연 담당: 기술(15-20%) + 팀(25-30%) 평가"""

    company_name = state["startup_name"]
    company_info = state["startup_info"]
    website = company_info.get("website", "")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    search_tool = TavilySearchResults(max_results=3)

    print(f"\n[승연] {company_name} 기술+팀 평가 시작\n")

    # 검색 쿼리
    search_queries = {
        "core_tech": f"{company_name} {website} 핵심 기술",
        "competitive": f"{company_name} 경쟁사 비교",
        "founders": f"{company_name} 창업자 CEO CTO",
        "team": f"{company_name} 팀 인력",
    }

    all_results = {}
    for key, query in search_queries.items():
        results = search_tool.invoke(query)
        all_results[key] = results
        print(f"{key}: {len(results)}개")

    formatted_results = format_search_results(all_results)

    evaluation_prompt = f"""
    벤처캐피탈 평가 전문가로서 기술과 팀을 분석하세요.
    
    회사: {company_name}
    분야: {company_info.get('tech_focus', 'N/A')}
    검색 결과: {formatted_results}
    
    다음 형식으로 간결하게 작성:
    
    ## 기술 평가
    핵심 기술: [1-2문장]
    개발 단계: [프로토타입/베타/상용화]
    차별화: [3가지, 각 1줄]
    강점: [3가지]
    약점: [2가지]
    기술 점수: [50-150%]
    
    ## 경쟁사 비교
    | 항목 | {company_name} | 경쟁사A | 경쟁사B |
    | 크기 | ? | ? | ? |
    | 성능 | ? | ? | ? |
    
    ## 팀 평가
    창업자: CEO [이름/경력], CTO [이름/경력]
    팀 규모: [N명]
    핵심 역량: [3가지]
    산업 경험: [있음/없음, 간단히]
    팀 점수: [50-150%]
    
    ## 종합 리스크
    - [리스크 1]
    - [리스크 2]
    """

    result = llm.invoke(evaluation_prompt)
    parsed = parse_simple(result.content)

    summary_payload = {
        "full_text": clean_markdown(result.content),
        "tech_score": parsed["tech_score"],
        "team_score": parsed["team_score"],
        "tech_summary": parsed["tech_summary"],
        "team_summary": parsed["team_summary"],
        "key_risks": parsed["risks"],
        "competitive_table": parsed["competitive_table"],
        "sources": extract_sources(all_results)[:3],
    }
    state["tech_team_summary"] = summary_payload
    state["tech_summary"] = summary_payload["full_text"]

    print(f"✓ 기술: {state['tech_team_summary']['tech_score']}%")
    print(f"✓ 팀: {state['tech_team_summary']['team_score']}%\n")

    return state


def parse_simple(content: str) -> dict:
    """파싱"""
    tech_score = 100
    team_score = 100

    if match := re.search(r"기술 점수:\s*(\d+)%", content):
        tech_score = int(match.group(1))
    if match := re.search(r"팀 점수:\s*(\d+)%", content):
        team_score = int(match.group(1))

    tech_summary = ""
    team_summary = ""

    if match := re.search(r"## 기술 평가(.+?)(?=## |$)", content, re.DOTALL):
        tech_section = match.group(1)
        if core_match := re.search(
            r"핵심 기술:(.+?)(?=\n|개발)", tech_section, re.DOTALL
        ):
            tech_summary = core_match.group(1).strip()

    if match := re.search(r"## 팀 평가(.+?)(?=## |$)", content, re.DOTALL):
        team_section = match.group(1)
        lines = [l.strip() for l in team_section.split("\n") if l.strip()]
        team_summary = " ".join(lines[:3])

    competitive_table = ""
    table_lines = [l for l in content.split("\n") if "|" in l]
    if table_lines:
        competitive_table = "\n".join(table_lines[:5])

    risks = re.findall(r"- \[?리스크.*?\]?:?\s*(.+?)(?=\n|$)", content)
    risks = [r.strip() for r in risks[:2]]

    return {
        "tech_score": tech_score,
        "team_score": team_score,
        "tech_summary": tech_summary or "기술 정보 없음",
        "team_summary": team_summary or "팀 정보 없음",
        "competitive_table": competitive_table,
        "risks": risks or ["리스크 정보 없음"],
    }


def format_search_results(results_dict: dict) -> str:
    """?? ?? ?? ??"""
    formatted = []
    for category, results in results_dict.items():
        formatted.append(f"[{category}]")
        if not results:
            continue

        first = results[0]
        if isinstance(first, dict):
            snippet = first.get("content") or first.get("snippet") or first.get("title") or ""
        elif isinstance(first, str):
            snippet = first
        else:
            snippet = str(first)

        formatted.append(snippet[:150])
    return "\n".join(formatted)


def extract_sources(results_dict: dict) -> list:
    """?? ???? URL? ??"""
    sources = []
    for results in results_dict.values():
        for r in results:
            if isinstance(r, dict):
                url = r.get("url")
                if url:
                    sources.append(url)
    return list(set(sources))


def clean_markdown(text: str) -> str:
    """Markdown 제거"""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.strip()
