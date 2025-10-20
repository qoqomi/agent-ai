# -*- coding: utf-8 -*-
# 필요한 패키지:
# pip install -U chromadb sentence-transformers langchain-openai
# (리랭킹) pip install -U "sentence-transformers>=3.0,<3.1"
# (호환이슈시) pip install "tokenizers==0.19.1"

from __future__ import annotations
from typing import TypedDict, Dict, Any, List, Tuple
import json

from langchain_openai import ChatOpenAI

# ====== 경로/모델 설정 ======
PERSIST_DIR = "./chroma_store"          # 너의 Chroma persist 디렉터리
COLLECTION  = "investor_kb"             # 코어 KB 컬렉션명
EMB_MODEL   = "all-MiniLM-L6-v2"        # SBERT 임베딩 모델
LLM_MODEL   = "gpt-4o-mini"             # 멀티쿼리/판정 LLM
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 리랭킹용 CrossEncoder

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

# ====== Chroma & Embedding ======
def _get_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

def _embed(texts: List[str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    return model.encode(texts, batch_size=32, normalize_embeddings=True)

# ====== 쿼리 템플릿 ======
def _build_queries(name: str, tech: str, market: str) -> List[Tuple[str, str]]:
    b = name
    return [
        ("PMF",   f"{b}의 제품-시장 적합성 신호/체크리스트는? {tech}"),
        ("Market",f"{b}의 시장 사이징(TAM=고객수×ARPA)과 Why Now 핵심은? {market}"),
        ("Moat",  f"{b} 카테고리의 네트워크 효과/전환비용/규모의 경제 근거는?"),
        ("Traction", f"{b}의 AARRR·코호트 리텐션에서 최소 봐야 할 지표와 기준은?"),
        ("UnitEconomics", f"{b}의 LTV/CAC·Payback·Quick Ratio 해석 기준 핵심은?"),
        ("Growth", f"초기 단계에서 주간 성장률 벤치마크는?"),
        ("DefaultAlive", f"번레이트/현금/런웨이로 Alive/Dead 판단하는 규칙은?"),
        ("Team",  f"핵심 인력/역할/도메인 역량 체크포인트는?")
    ]

# ====== Pre-Retriever: Multi-Query ======
def _gen_multi_queries(pillar: str, base_q: str, n: int = 4) -> List[str]:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, model_kwargs={"response_format":{"type":"json_object"}})
    sys = "너는 검색 쿼리 생성기다. 같은 정보 목적을 서로 다른 관점/용어로 분해하라. 중복/사족 금지."
    user = {
        "pillar": pillar,
        "base_query": base_q,
        "instructions": "서로 의미가 겹치지 않게 3~5개 쿼리로 확장. 각 쿼리는 20~120자. JSON 리스트만 반환."
    }
    try:
        rsp = llm.invoke([{"role":"system","content":sys},
                          {"role":"user","content":json.dumps(user, ensure_ascii=False)}])
        qs = json.loads(rsp.content)
        if isinstance(qs, list) and qs:
            out, seen = [], set()
            for q in qs:
                q = (q or "").strip()
                if not q: continue
                k = q.lower()
                if k in seen: continue
                seen.add(k); out.append(q)
            if out:
                return out[:n]
    except Exception:
        pass
    # 폴백(휴리스틱)
    variants = [
        base_q,
        base_q + " 핵심 지표와 체크리스트는?",
        base_q.replace("는?", "과 유사 사례는?") if "는?" in base_q else base_q + " 유사 사례는?",
        base_q + " 벤치마크 수치와 경계값은?"
    ]
    seen, out = set(), []
    for q in variants:
        k = q.lower()
        if k not in seen:
            seen.add(k); out.append(q)
    return out[:n]

# ====== Post-Retriever: Re-ranking (Cross-Encoder) ======
def _rerank(query_text: str, hits: List[Dict[str, Any]], topk: int = 4) -> List[Dict[str, Any]]:
    if not hits:
        return []
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(RERANK_MODEL)
    pairs = [(query_text, h["text"]) for h in hits]
    scores = ce.predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 페이지/소스 중복 제거
    kept, seen = [], set()
    for h in hits:
        m = h["meta"]
        key = (m.get("source"), m.get("page"))
        if key in seen:
            continue
        seen.add(key); kept.append(h)
        if len(kept) >= topk: break
    return kept

# ====== Retriever: Multi-Query → Pool → Rerank → Bullets ======
def _retrieve_evidence(name: str, tech: str, market: str, 
                       per_query_topk: int = 4, final_topk: int = 4) -> Dict[str, List[str]]:
    col = _get_chroma()
    out: Dict[str, List[str]] = {}
    for pillar, base_q in _build_queries(name, tech, market):
        mq_list = _gen_multi_queries(pillar, base_q, n=4)

        pool: List[Dict[str, Any]] = []
        for q in mq_list:
            v = _embed([q])[0].tolist()
            res = col.query(query_embeddings=[v], n_results=per_query_topk, where={"pillar": pillar})
            docs  = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            for doc, meta in zip(docs, metas):
                pool.append({"text": doc, "meta": meta, "origin_query": q})

        ranked = _rerank(base_q, pool, topk=final_topk)

        bullets = []
        for h in ranked:
            m = h["meta"]
            tag = f"{m.get('investor','?')}·{m.get('pillar','?')}·p.{m.get('page','?')}"
            bullets.append(f"- {h['text'][:280].strip()} … ({tag})")
        out[pillar] = bullets
    return out

# ====== 투자 판단 노드 ======
def investment_decision_agent(state: AgentState) -> AgentState:
    """
    종합 판단 (리스크, ROI 등)
    - Scorecard Method (간이: 기술/시장) + RAG 근거 (Multi-Query & Re-ranking)
    - 이분법: invest=1 / hold=0 + 근거 요약(decision_reason) + 상세 리포트(report)
    """
    print("=== [경남] 투자 판단 에이전트 실행 ===")

    company_name    = state["startup_name"]
    tech_summary    = state["tech_summary"]
    market_analysis = state["market_analysis"]

    # 1) RAG 근거 수집
    evidence = _retrieve_evidence(company_name, tech_summary, market_analysis, per_query_topk=4, final_topk=4)

    # 2) LLM 평가 (JSON 강제)
    schema = {
        "type": "object",
        "properties": {
            "tech_score_pct":   {"type":"number", "description":"제품/기술 점수(50~150 퍼센트)"},
            "market_score_pct": {"type":"number", "description":"시장 기회 점수(50~150 퍼센트)"},
            "rationale":        {"type":"array", "items":{"type":"string"}, "minItems":3, "description":"판단 근거 bullet"},
            "decision":         {"type":"string", "enum":["invest","hold"]}
        },
        "required": ["tech_score_pct","market_score_pct","rationale","decision"],
        "additionalProperties": False
    }

    def pack_evi(p: str) -> List[str]:
        return evidence.get(p, [])[:3]

    rag_context = {
        "PMF": pack_evi("PMF"),
        "Market": pack_evi("Market"),
        "Moat": pack_evi("Moat"),
        "Traction": pack_evi("Traction"),
        "UnitEconomics": pack_evi("UnitEconomics"),
        "Growth": pack_evi("Growth"),
        "DefaultAlive": pack_evi("DefaultAlive"),
        "Team": pack_evi("Team"),
    }

    system_prompt = (
        "너는 VC 심사역이다. 아래 회사의 기술/시장 요약과 RAG 근거를 보고 "
        "Scorecard 방식으로 간이 평가를 하라. 제품/기술은 15~20%, 시장은 20~25% 범위에서 가중치를 해석하며 "
        "각 항목 점수는 50~150% 범위에서 배점한다. 최종 결론은 invest 또는 hold 중 하나만 고른다. "
        "응답은 반드시 JSON으로만 출력한다."
    )
    user_payload = {
        "company": company_name,
        "tech_summary": tech_summary,
        "market_analysis": market_analysis,
        "rag_evidence": rag_context,
        "format": schema
    }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, model_kwargs={"response_format":{"type":"json_object"}})
    res = llm.invoke([
        {"role":"system","content": system_prompt},
        {"role":"user",  "content": "다음 JSON을 참고하여 schema에 맞춰 응답:\n" + json.dumps(user_payload, ensure_ascii=False)}
    ])

    try:
        data = json.loads(res.content)
    except Exception:
        raise ValueError(f"LLM JSON 파싱 실패: {res.content}")

    for k in ("tech_score_pct","market_score_pct","rationale","decision"):
        if k not in data:
            raise KeyError(f"LLM 응답 누락: {k}")

    decision_str = data["decision"].lower()          # 'invest' | 'hold'
    decision_int = 1 if decision_str == "invest" else 0

    # 3) decision_reason (짧은 요약)
    #    예: "결정: 투자 | 기술 120%, 시장 130% | 근거: A; B; C"
    rationale_bullets = [r for r in data["rationale"] if isinstance(r, str)]
    top_reasons = "; ".join(rationale_bullets[:3])
    decision_reason = (
        f"결정: {'투자' if decision_int==1 else '비투자'} | "
        f"기술 {data['tech_score_pct']}%, 시장 {data['market_score_pct']}% | "
        f"근거: {top_reasons}"
    )

    # 4) 상세 리포트 (LLM 근거 + RAG 인용)
    lines = [
        f"# 투자 판단 — {company_name}",
        f"- **결과**: **{ '투자' if decision_int==1 else '비투자' }** (label={decision_int})",
        f"- **Scorecard(간이)**: 기술 {data['tech_score_pct']}% · 시장 {data['market_score_pct']}%",
        "", "## 판단 근거(LLM 요약)",
    ]
    for r in rationale_bullets:
        lines.append(f"- {r}")
    lines += ["", "## RAG 인용(상위 3 / 각 피벗)"]
    for p, bullets in rag_context.items():
        if bullets:
            lines.append(f"**{p}**")
            lines.extend(bullets)

    # 5) overwrite
    state["investment_decision"] = decision_int
    state["decision_reason"] = decision_reason
    state["report"] = "\n".join(lines)
    state["iteration_count"] = int(state.get("iteration_count", 0)) + 1
    print(f"투자 판단: {state['investment_decision']} (1=투자, 0=비투자)")
    return state

# ====== 단독 테스트 ======
if __name__ == "__main__":
    state: AgentState = {
        "startup_name": "OrbitalMesh",
        "startup_info": {
            "is_startup": True,
            "weekly_growth": 8.1,
            "cash_on_hand": 1500000,
            "monthly_burn": 70000,
            "ltv": 2400, "cac": 700
        },
        "tech_summary": {'full_text': '기술 평가\n핵심 기술: 초소형 큐브샛을 위한 지상국 시스템 엔지니어링 솔루션 및 AI 기반 위성영상 분석 기술.  \n개발 단계: 상용화  \n차별화: \n1. AI 딥러닝 기반의 위성영상 품질 보정 기술 제공.\n2. 지상국 안테나 및 모뎀 제작으로 통합 솔루션 제공.\n3. UAE를 포함한 해외 시장으로의 위성통신 단말기 공급.  \n강점: \n1. 8년 만에 코스닥 상장으로 시장 신뢰도 확보.\n2. 다양한 위성 관련 제품군 보유.\n3. 우주산업 솔루션에 대한 전문성.  \n약점: \n1. 초기 자금 조달의 어려움 경험.\n2. 경쟁이 치열한 우주산업 내에서의 시장 점유율 확보 필요.  \n기술 점수: 120%\n\n경쟁사 비교\n| 항목 | 컨텍 | 경쟁사A | 경쟁사B |\n|------|------|----------|----------|\n| 크기 | 중소기업 | 대기업 | 중소기업 |\n| 성능 | AI 기반 분석 우수 | 전통적 분석 | 제한적 기능 |\n\n팀 평가\n창업자: CEO 이성희 (우주산업 경력 10년 이상), CTO [이름/경력 미제공]  \n팀 규모: 50명  \n핵심 역량: \n1. 우주산업 솔루션 개발 경험.\n2. AI 및 데이터 분석 기술.\n3. 지상국 시스템 엔지니어링 전문성.  \n산업 경험: 있음 (우주산업 및 관련 기술 분야에서의 경험)  \n팀 점수: 110%\n\n종합 리스크\n- 초기 자금 조달의 어려움으로 인한 재무적 불안정성.\n- 치열한 경쟁 속에서의 시장 점유율 확보의 어려움.', 'tech_score': 120, 'team_score': 110, 'tech_summary': '초소형 큐브샛을 위한 지상국 시스템 엔지니어링 솔루션 및 AI 기반 위성영상 분석 기술.', 'team_summary': '창업자: CEO 이성희 (우주산업 경력 10년 이상), CTO [이름/경력 미제공] 팀 규모: 50명 핵심 역량:', 'key_risks': ['리스크 정보 없음'], 'competitive_table': '| 항목 | 컨텍 | 경쟁사A | 경쟁사B |\n|------|------|----------|----------|\n| 크기 | 중소기업 | 대기업 | 중소기업 |\n| 성능 | AI 기반 분석 우수 | 전통적 분석 | 제한적 기능 |', 'sources': ['https://brunch.co.kr/@cliche-cliche/106', 'https://m.blog.naver.com/ivvlove/222445956219', 'https://www.jake-james.com/blog/what-does-the-ceo-cfo-coo-cto-and-others-do']},
        "market_analysis": "해양/원격 산업 IoT를 타깃. Why Now: 발사 단가 하락·오지 연결 수요 증가.",
        "investment_decision": 0,
        "decision_reason": "",
        "report": "",
        "iteration_count": 0
    }
    investment_decision_agent(state)


# # -*- coding: utf-8 -*-
# # 필요한 패키지:
# # pip install -U chromadb sentence-transformers langchain-openai
# # (리랭킹) pip install -U "sentence-transformers>=3.0,<3.1"
# # (호환이슈시) pip install "tokenizers==0.19.1"

# from __future__ import annotations
# from typing import TypedDict, Dict, Any, List, Tuple
# import os, json

# from langchain_openai import ChatOpenAI

# # ====== 경로/모델 설정 ======
# PERSIST_DIR = "./chroma_store"          # 너의 Chroma persist 디렉터리
# COLLECTION  = "investor_kb"             # 코어 KB 컬렉션명
# EMB_MODEL   = "all-MiniLM-L6-v2"        # SBERT 임베딩 모델
# LLM_MODEL   = "gpt-4o-mini"             # 멀티쿼리/판정 LLM
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 리랭킹용 CrossEncoder

# # ====== Chroma & Embedding ======
# def _get_chroma():
#     import chromadb
#     client = chromadb.PersistentClient(path=PERSIST_DIR)
#     return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

# def _embed(texts: List[str]):
#     # SentenceTransformer/transformers/tokenizers 버전 호환 주의
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer(EMB_MODEL)
#     return model.encode(texts, batch_size=32, normalize_embeddings=True)

# # ====== 쿼리 템플릿 ======
# def _build_queries(name: str, tech: str, market: str) -> List[Tuple[str, str]]:
#     b = name
#     return [
#         ("PMF",   f"{b}의 제품-시장 적합성 신호/체크리스트는? {tech}"),
#         ("Market",f"{b}의 시장 사이징(TAM=고객수×ARPA)과 Why Now 핵심은? {market}"),
#         ("Moat",  f"{b} 카테고리의 네트워크 효과/전환비용/규모의 경제 근거는?"),
#         ("Traction", f"{b}의 AARRR·코호트 리텐션에서 최소 봐야 할 지표와 기준은?"),
#         ("UnitEconomics", f"{b}의 LTV/CAC·Payback·Quick Ratio 해석 기준 핵심은?"),
#         ("Growth", f"초기 단계에서 주간 성장률 벤치마크는?"),
#         ("DefaultAlive", f"번레이트/현금/런웨이로 Alive/Dead 판단하는 규칙은?"),
#         ("Team",  f"핵심 인력/역할/도메인 역량 체크포인트는?")
#     ]

# # ====== Pre-Retriever: Multi-Query ======
# def _gen_multi_queries(pillar: str, base_q: str, n: int = 4) -> List[str]:
#     """
#     LLM으로 base_q를 의미 다른 3~5개 쿼리로 확장.
#     실패하면 휴리스틱 폴백(예외 숨기지 않고, 폴백만 사용).
#     """
#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0, model_kwargs={"response_format":{"type":"json_object"}})
#     sys = "너는 검색 쿼리 생성기다. 같은 정보 목적을 서로 다른 관점/용어로 분해하라. 중복/사족 금지."
#     user = {
#         "pillar": pillar,
#         "base_query": base_q,
#         "instructions": "서로 의미가 겹치지 않게 3~5개 쿼리로 확장. 각 쿼리는 20~120자. JSON 리스트만 반환."
#     }
#     try:
#         rsp = llm.invoke([{"role":"system","content":sys},
#                           {"role":"user","content":json.dumps(user, ensure_ascii=False)}])
#         qs = json.loads(rsp.content)
#         if isinstance(qs, list) and qs:
#             out = []
#             seen = set()
#             for q in qs:
#                 q = (q or "").strip()
#                 if not q: 
#                     continue
#                 k = q.lower()
#                 if k in seen: 
#                     continue
#                 seen.add(k); out.append(q)
#             if out:
#                 return out[:n]
#     except Exception:
#         pass

#     # 폴백(휴리스틱): 간단 변형으로 중복 최소화
#     variants = [
#         base_q,
#         base_q + " 핵심 지표와 체크리스트는?",
#         base_q.replace("는?", "과 유사 사례는?") if "는?" in base_q else base_q + " 유사 사례는?",
#         base_q + " 벤치마크 수치와 경계값은?"
#     ]
#     seen, out = set(), []
#     for q in variants:
#         k = q.lower()
#         if k not in seen:
#             seen.add(k); out.append(q)
#     return out[:n]

# # ====== Post-Retriever: Re-ranking ======
# def _rerank(query_text: str, hits: List[Dict[str, Any]], topk: int = 4) -> List[Dict[str, Any]]:
#     """
#     hits: [{"text": ..., "meta": {...}, "origin_query": "..."}]
#     Cross-Encoder 점수로 재정렬 → 페이지 중복 제거 → topk 반환
#     """
#     if not hits:
#         return []
#     # CrossEncoder 로드 (CPU 가능, 최초 한 번 캐시)
#     from sentence_transformers import CrossEncoder
#     ce = CrossEncoder(RERANK_MODEL)
#     pairs = [(query_text, h["text"]) for h in hits]
#     scores = ce.predict(pairs).tolist()
#     for h, s in zip(hits, scores):
#         h["rerank_score"] = float(s)
#     hits.sort(key=lambda x: x["rerank_score"], reverse=True)

#     # 같은 page/source 중복 제거 (가독성/다양성)
#     kept, seen = [], set()
#     for h in hits:
#         m = h["meta"]
#         key = (m.get("source"), m.get("page"))
#         if key in seen:
#             continue
#         seen.add(key); kept.append(h)
#         if len(kept) >= topk: 
#             break
#     return kept

# # ====== Retriever: Multi-Query → Pool → Rerank → Bullets ======
# def _retrieve_evidence(name: str, tech: str, market: str, 
#                        per_query_topk: int = 4, final_topk: int = 4) -> Dict[str, List[str]]:
#     col = _get_chroma()
#     out: Dict[str, List[str]] = {}
#     for pillar, base_q in _build_queries(name, tech, market):
#         # (A) multi-query 확장
#         mq_list = _gen_multi_queries(pillar, base_q, n=4)

#         # (B) 각 확장 쿼리로 회수 → 풀 통합
#         pool: List[Dict[str, Any]] = []
#         for q in mq_list:
#             v = _embed([q])[0].tolist()
#             res = col.query(query_embeddings=[v], n_results=per_query_topk, where={"pillar": pillar})
#             docs  = res.get("documents", [[]])[0]
#             metas = res.get("metadatas", [[]])[0]
#             for doc, meta in zip(docs, metas):
#                 pool.append({"text": doc, "meta": meta, "origin_query": q})

#         # (C) 리랭킹(Cross-Encoder) → 최종 Top-k
#         ranked = _rerank(base_q, pool, topk=final_topk)

#         # (D) 인용 문자열로 변환
#         bullets = []
#         for h in ranked:
#             m = h["meta"]
#             tag = f"{m.get('investor','?')}·{m.get('pillar','?')}·p.{m.get('page','?')}"
#             bullets.append(f"- {h['text'][:280].strip()} … ({tag})")
#         out[pillar] = bullets
#     return out

# # ====== AgentState ======
# class AgentState(TypedDict):
#     startup_name: str
#     startup_info: dict
#     tech_summary: str
#     market_analysis: str
#     investment_decision: int
#     decision_reason: str
#     report: str
#     iteration_count: int

# # ====== 투자 판단 노드 (네 그래프에 그대로 붙이면 됨) ======
# def investment_decision_agent(state: AgentState) -> AgentState:
#     """
#     종합 판단 (리스크, ROI 등)
#     - Scorecard Method (간이: 기술/시장) + RAG 근거 (Multi-Query & Re-ranking)
#     - 이분법: invest(1)/hold(0) + 근거 리포트
#     """
#     print("=== [경남] 투자 판단 에이전트 실행 ===")

#     company_name    = state["startup_name"]
#     tech_summary    = state["tech_summary"]
#     market_analysis = state["market_analysis"]

#     # 1) RAG 근거 수집 (Pre: Multi-Query, Post: Re-ranking)
#     evidence = _retrieve_evidence(company_name, tech_summary, market_analysis, per_query_topk=4, final_topk=4)

#     # 2) LLM 평가 프롬프트 (JSON 강제) — Scorecard 간이판
#     schema = {
#         "type": "object",
#         "properties": {
#             "tech_score_pct":   {"type":"number", "description":"제품/기술 점수(50~150 퍼센트)"},
#             "market_score_pct": {"type":"number", "description":"시장 기회 점수(50~150 퍼센트)"},
#             "rationale":        {"type":"array", "items":{"type":"string"}, "minItems":3, "description":"판단 근거 bullet"},
#             "decision":         {"type":"string", "enum":["invest","hold"]}
#         },
#         "required": ["tech_score_pct","market_score_pct","rationale","decision"],
#         "additionalProperties": False
#     }

#     def pack_evi(p: str) -> List[str]:
#         return evidence.get(p, [])[:3]

#     rag_context = {
#         "PMF": pack_evi("PMF"),
#         "Market": pack_evi("Market"),
#         "Moat": pack_evi("Moat"),
#         "Traction": pack_evi("Traction"),
#         "UnitEconomics": pack_evi("UnitEconomics"),
#         "Growth": pack_evi("Growth"),
#         "DefaultAlive": pack_evi("DefaultAlive"),
#         "Team": pack_evi("Team"),
#     }

#     system_prompt = (
#         "너는 VC 심사역이다. 아래 회사의 기술/시장 요약과 RAG 근거를 보고 "
#         "Scorecard 방식으로 간이 평가를 하라. 제품/기술은 15~20%, 시장은 20~25% 범위에서 가중치를 해석하며 "
#         "각 항목 점수는 50~150% 범위에서 배점한다. 최종 결론은 invest 또는 hold 중 하나만 고른다. "
#         "응답은 반드시 JSON으로만 출력한다."
#     )
#     user_payload = {
#         "company": company_name,
#         "tech_summary": tech_summary,
#         "market_analysis": market_analysis,
#         "rag_evidence": rag_context,
#         "format": schema
#     }

#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0, model_kwargs={"response_format":{"type":"json_object"}})
#     res = llm.invoke([
#         {"role":"system","content": system_prompt},
#         {"role":"user",  "content": "다음 JSON을 참고하여 schema에 맞춰 응답:\n" + json.dumps(user_payload, ensure_ascii=False)}
#     ])

#     try:
#         data = json.loads(res.content)
#     except Exception:
#         raise ValueError(f"LLM JSON 파싱 실패: {res.content}")

#     for k in ("tech_score_pct","market_score_pct","rationale","decision"):
#         if k not in data:
#             raise KeyError(f"LLM 응답 누락: {k}")

#     decision_str = data["decision"].lower()  # 'invest' | 'hold'
#     decision_num = 1 if decision_str == "invest" else 0

#     # 3) 리포트 (LLM 근거 + RAG 인용 포함)
#     lines = [
#         f"# 투자 판단 — {company_name}",
#         f"- **결과**: **{ '투자' if decision_num==1 else '보류' }** (label={decision_num})",
#         f"- **Scorecard(간이)**: 기술 {data['tech_score_pct']}% · 시장 {data['market_score_pct']}%",
#         "", "## 판단 근거(LLM 요약)",
#     ]
#     for r in data["rationale"]:
#         lines.append(f"- {r}")
#     lines += ["", "## RAG 인용(상위 3 / 각 피벗)"]
#     for p, bullets in rag_context.items():
#         if bullets:
#             lines.append(f"**{p}**")
#             lines.extend(bullets)

#     state["investment_decision"] = "invest" if decision_num==1 else "hold"
#     state["decision_num"] = decision_num  # 루프용 숫자 라벨
#     state["report"] = "\n".join(lines)
#     state["iteration_count"] = state.get("iteration_count", 0) + 1
#     print(f"투자 판단: {state['investment_decision']} (label={decision_num})")
#     return state

# # ====== 단독 테스트 ======
# if __name__ == "__main__":
#     state: AgentState = {
#         "startup_name": "OrbitalMesh",
#         "startup_info": {
#             "is_startup": True,
#             "weekly_growth": 8.1,
#             "cash_on_hand": 1500000,
#             "monthly_burn": 70000,
#             "ltv": 2400, "cac": 700
#         },
#         "tech_summary": "LEO 위성 간 레이저 링크 기반 메시 네트워크 모듈.",
#         "market_analysis": "해양/원격 산업 IoT를 타깃. Why Now: 발사 단가 하락·오지 연결 수요 증가.",
#         "investment_decision": "",
#         "decision_reason"
#         "report": "",
#         "iteration_count": 0
#     }
#     investment_decision_agent(state)