"""
위성 산업 스타트업 탐색 에이전트 (국내 스타트업 전용)
- 웹 검색 (간단 구현)
- RAG 검색 (PDF 문서)
- 기본값 사용 (테스트 모드)
- 최종 1개 스타트업 선정
"""
import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from start_rag_system import RAGSystem

# 환경 변수 로드
load_dotenv()


# ============================================
# 1. State 정의 (팀원 공유 문서 형식)
# ============================================
class AgentState(TypedDict):
    """에이전트 상태 - 팀원 공유 포맷"""
    startup_name: str  # 스타트업 이름
    startup_info: dict  # 스타트업 정보


# ============================================
# 2. 스타트업 탐색 노드
# ============================================
def startup_search_node(state: AgentState) -> AgentState:
    """
    스타트업 탐색 노드
    
    역할:
    1. 웹 검색 (간단 구현)
    2. RAG 검색 (PDF 문서 - 국내 스타트업 포커스)
    3. 기본값 사용 (테스트 모드)
    4. 최종 1개 스타트업 선정
    """
    print("\n" + "="*60)
    print(" 스타트업 탐색 에이전트 시작 (국내 스타트업)")
    print("="*60)
    
    # RAG 시스템 초기화
    print("\n[INFO] RAG 시스템 로드 중...")
    rag = RAGSystem()
    rag.build()
    
    
    # ===== Step 1: 웹 검색 키워드 (국내 스타트업) =====
    print("\n[INFO] 웹 검색 키워드 준비 (국내 스타트업)...")
    
    web_keywords = [
        "위성 소형화 스타트업",
        "저궤도 위성 스타트업",
        "광학위성 스타트업",
        "한국 위성 스타트업",
        "국내 소형위성 스타트업",
        "뉴스페이스 한국 스타트업"
    ]
    print(f"   [SUCCESS] 검색 키워드: {len(web_keywords)}개 준비")
    
    
    # ===== Step 2: RAG 검색 (한국 스타트업 포커스) =====
    print("\n[INFO] RAG 검색 (PDF 문서 - 한국 스타트업)...")
    
    rag_queries = [
        "한국 위성 스타트업",
        "국내 소형위성 스타트업",
        "Korea satellite startup",
        "저궤도 위성 한국 스타트업",
        "광학위성 국내 스타트업",
        "뉴스페이스 스타트업 한국"
    ]
    
    rag_results = []
    for query in rag_queries:
        result = rag.search(query, k=2)
        rag_results.append(result)
    
    print(f"   [SUCCESS] RAG 검색 완료: {len(rag_results)}개 쿼리")
    
    # RAG 결과 결합 (참고용)
    combined_rag = "\n\n".join(rag_results)
    
    
    # ===== Step 3: 기본 스타트업 사용 (테스트 모드) =====
    print("\n[INFO] 기본 스타트업 사용 (테스트 모드)...")
    
    startup = {
        "name": "컨텍",
        "website": "https://www.contec.kr",
        "tech_focus": "지상국 운영",
        "description": "2014년 설립된 민간 위성 지상국 스타트업. 전 세계 10개국에서 12개 지상국을 운영하며 위성 데이터 수신 및 처리 서비스를 제공.",
        "founded_year": 2014,
        "location": "대한민국",
        "funding": "IPO 2023, 시가총액 3000억원",
        "key_technology": "위성 데이터 수신 및 처리, 지상국 네트워크",
        "team_size": "약 50명",
        "startup_type": "IPO"
    }
    
    
    # ===== Step 4: 결과 출력 =====
    print("\n" + "="*60)
    print(" 스타트업 탐색 완료! (최종 선정)")
    print("="*60)
    
    print(f"\n[RESULT] 선정된 스타트업: {startup['name']}")
    print(f"   [기술 분야] {startup['tech_focus']}")
    print(f"   [설명] {startup['description']}")
    print(f"   [웹사이트] {startup.get('website', 'N/A')}")
    print(f"   [설립연도] {startup.get('founded_year', 'N/A')}")
    
    
    # ===== State 업데이트 (팀원 공유 포맷) =====
    return {
        "startup_name": startup["name"],
        "startup_info": startup
    }


# ============================================
# 3. LangGraph 구성
# ============================================
def create_graph():
    """그래프 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("startup_search_agent", startup_search_node)
    
    # 시작점 설정
    workflow.set_entry_point("startup_search_agent")
    
    # 종료점 연결
    workflow.add_edge("startup_search_agent", END)
    
    return workflow.compile()


# ============================================
# 4. 실행
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("  위성 산업 스타트업 탐색 에이전트 (국내 스타트업)")
    print("="*60)
    
    # 초기 State (팀원 공유 포맷)
    initial_state = {
        "startup_name": "",
        "startup_info": {}
    }
    
    # 그래프 실행
    graph = create_graph()
    result = graph.invoke(initial_state)
    
    # 최종 결과 출력
    print("\n" + "="*60)
    print(" 최종 결과 (팀원 공유 데이터)")
    print("="*60)
    
    import json
    print(f"\n[OUTPUT] startup_name: {result['startup_name']}")
    print(f"\n[OUTPUT] startup_info:")
    print(json.dumps(result["startup_info"], 
                    ensure_ascii=False, 
                    indent=2))
    
    print("\n" + "="*60)
    print(" 다음 에이전트로 전달되는 데이터:")
    print("="*60)
    print(f"- startup_name (str): '{result['startup_name']}'")
    print(f"- startup_info (dict): {len(result['startup_info'])}개 필드")
    
    print("\n" + "="*60)
    print(" 에이전트 실행 완료!")
    print("="*60)