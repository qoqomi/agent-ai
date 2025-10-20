# -*- coding: utf-8 -*-
# decision_module.py
# RAG 멀티쿼리 + 리랭킹 + 자율 플래닝/리플렉션 내장 판단 노드
# report는 절대 수정하지 않고, investment_decision(int) / decision_reason(str)만 갱신

from __future__ import annotations
from typing import TypedDict, Dict, Any, List, Tuple, Optional
import json, time

from langchain_openai import ChatOpenAI

# ====== 경로/모델 설정 ======
PERSIST_DIR   = "./chroma_store"                       # Chroma persist 디렉터리
COLLECTION    = "investor_kb"                          # 코어 KB 컬렉션명
EMB_MODEL     = "all-MiniLM-L6-v2"                     # SBERT 임베딩 모델
LLM_MODEL     = "gpt-4o-mini"                          # 멀티쿼리/판정 LLM
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2" # 리랭킹용 CrossEncoder

# ====== 정책(자율 행동 제약) ======
POLICY = {
    "max_rounds": 2,             # 최대 재시도 라운드
    "time_budget_s": 45,         # 시간 예산(초)
    "min_coverage": 0.55,        # 커버리지 임계치 (타겟 피벗 중 증거 보유 비율)
    "per_query_topk": [3, 5],    # 라운드별 per-query top-k
    "final_topk":     [4, 6],    # 라운드별 최종 rerank k
    "mq_n":           [3, 5],    # 라운드별 multi-query 개수
}

# ====== AgentState (최신 스키마) ======
class AgentState(TypedDict):
    startup_name: str
    startup_info: dict
    tech_summary: str | dict
    market_analysis: str | dict
    investment_decision: int      # 투자=1, 비투자=0
    decision_reason: str          # 판단 근거 요약
    report: str                   # 이 노드에서는 수정하지 않음
    iteration_count: int

# ====== 안전 문자열 변환 ======
def _coerce_text_field(v: Any, key_order: List[str] = ["full_text","tech_summary","summary"]) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in key_order:
            if isinstance(v.get(k), str) and v[k].strip():
                return v[k]
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)

# ====== Chroma & Embedding ======
def _get_chroma():
    import chromadb, sqlite3, traceback
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
        return col
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "collections.topic" in msg:
            raise RuntimeError(
                "Chroma 스키마 불일치 감지: persist_dir이 현재 chromadb 버전과 다릅니다.\n"
                f"- persist_dir: {PERSIST_DIR}\n"
                "해결: (A) 폴더명 변경 후 재-인제스트, 또는 (B) DB 생성 당시 chromadb 버전으로 맞추세요."
            ) from e
        raise

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

# ====== Pre: Multi-Query ======
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

# ====== Post: Re-ranking (Cross-Encoder) ======
def _rerank(query_text: str, hits: List[Dict[str, Any]], topk: int = 4) -> List[Dict[str, Any]]:
    if not hits:
        return []
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(RERANK_MODEL)  # 최초 로드 후 캐시
    pairs = [(query_text, h["text"]) for h in hits]
    scores = ce.predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)

    kept, seen = [], set()
    for h in hits:
        m = h["meta"]
        key = (m.get("source"), m.get("page"))
        if key in seen:
            continue
        seen.add(key); kept.append(h)
        if len(kept) >= topk: break
    return kept

# ====== 계획 기반 RAG (특정 피벗만) ======
def _retrieve_evidence_planned(target_pillars: List[str], name: str, tech: str, market: str,
                               per_query_topk: int, final_topk: int, mq_n: int) -> Dict[str, List[str]]:
    col = _get_chroma()
    out: Dict[str, List[str]] = {}
    for pillar, base_q in _build_queries(name, tech, market):
        if pillar not in target_pillars:
            continue
        mq_list = _gen_multi_queries(pillar, base_q, n=mq_n)
        pool: List[Dict[str, Any]] = []
        for q in mq_list:
            v = _embed([q])[0].tolist()
            res = col.query(query_embeddings=[v], n_results=per_query_topk, where={"pillar": pillar})
            docs, metas = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]
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

# ====== 커버리지/밀도 측정 ======
def _evidence_metrics(evi: Dict[str, List[str]], target_pillars: List[str]) -> Dict[str, float]:
    hits = sum(1 for p in target_pillars if evi.get(p))
    total = max(1, len(target_pillars))
    coverage = hits / total
    bullets = sum(len(evi.get(p, [])) for p in target_pillars)
    density = min(1.0, bullets / float(total * 3))  # 피벗당 3개 인용 기대치 기준
    return {"coverage": coverage, "density": density, "bullets": float(bullets)}

# ====== 라운드별 타겟 피벗 계획 ======
def _plan_pillars(tech_text: str, market_text: str, startup_info: Dict[str, Any], round_idx: int) -> List[str]:
    base = ["PMF", "Market", "Moat", "Traction"]
    if any(k in startup_info for k in ("ltv", "cac")):
        base.append("UnitEconomics")
    if any(k in startup_info for k in ("cash_on_hand", "monthly_burn")):
        base.append("DefaultAlive")
    if round_idx == 0:
        return ["PMF", "Market", "Moat", "Traction"]
    # 라운드 1 이상: 정량 포함 확장
    return list(dict.fromkeys(base))

# ====== 라운드별 하이퍼 파라미터 계획 ======
def _plan_params(round_idx: int) -> Tuple[int, int, int]:
    per_q = POLICY["per_query_topk"][min(round_idx, len(POLICY["per_query_topk"])-1)]
    final_k = POLICY["final_topk"][min(round_idx, len(POLICY["final_topk"])-1)]
    mq_n   = POLICY["mq_n"][min(round_idx, len(POLICY["mq_n"])-1)]
    return per_q, final_k, mq_n

# ====== LLM 판정(JSON) ======
def _llm_scorecard_judge(company: str, tech_text: str, market_text: str, rag_context: Dict[str, List[str]]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "tech_score_pct":   {"type":"number"},
            "market_score_pct": {"type":"number"},
            "rationale":        {"type":"array", "items":{"type":"string"}, "minItems":3},
            "decision":         {"type":"string", "enum":["invest","hold"]}
        },
        "required": ["tech_score_pct","market_score_pct","rationale","decision"],
        "additionalProperties": False
    }
    sys = ("너는 VC 심사역이다. 기술/시장 요약 및 RAG 근거를 보고 Scorecard 간이평가를 수행하라. "
           "제품/기술은 15~20%, 시장은 20~25% 범위 해석, 점수는 50~150%. "
           "최종 결론은 invest 또는 hold 중 하나. 응답은 JSON만.")
    user = {
        "company": company,
        "tech_summary": tech_text,
        "market_analysis": market_text,
        "rag_evidence": {k: v[:3] for k, v in rag_context.items()},
        "format": schema
    }
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, model_kwargs={"response_format":{"type":"json_object"}})
    res = llm.invoke([{"role":"system","content":sys},
                      {"role":"user","content":"다음 JSON을 참고하여 schema로만 응답:\n"+json.dumps(user, ensure_ascii=False)}])
    try:
        data = json.loads(res.content)
    except Exception:
        raise ValueError(f"LLM JSON 파싱 실패: {res.content}")
    for k in ("tech_score_pct","market_score_pct","rationale","decision"):
        if k not in data:
            raise KeyError(f"LLM 응답 누락: {k}")
    return data

# ====== 메인 노드: 자율 플래닝/리플렉션 내장 ======
def investment_decision_agent(state: AgentState) -> AgentState:
    """
    자율 플래닝/리플렉션 내장 투자판단 노드
    - 출력: investment_decision(1/0), decision_reason(str)만 갱신
    - report는 수정하지 않음
    """
    print("=== [경남] 투자 판단 에이전트(자율형) 실행 ===")
    t0 = time.monotonic()

    name   = state["startup_name"]
    info   = state.get("startup_info", {}) or {}
    tech   = _coerce_text_field(state.get("tech_summary"))
    market = _coerce_text_field(state.get("market_analysis"), ["full_text","market_summary","summary"])

    decision_int = 0
    decision_reason = "비투자 | 사유: 초기 상태"
    rounds = POLICY["max_rounds"]

    for r in range(rounds):
        # 1) 시간 예산
        if time.monotonic() - t0 > POLICY["time_budget_s"]:
            decision_int = 0
            decision_reason = "비투자 | 사유: 시간 예산 초과"
            break

        # 2) 계획 수립(타겟 피벗/하이퍼파라미터)
        target_pillars = _plan_pillars(tech, market, info, r)
        per_q, final_k, mq_n = _plan_params(r)

        # 3) RAG (계획 기반 멀티쿼리 → 리랭킹)
        evidence = _retrieve_evidence_planned(target_pillars, name, tech, market,
                                              per_query_topk=per_q, final_topk=final_k, mq_n=mq_n)
        metrics = _evidence_metrics(evidence, target_pillars)

        # 4) LLM 판정(JSON)
        data = _llm_scorecard_judge(name, tech, market, evidence)
        decision_int = 1 if str(data["decision"]).lower() == "invest" else 0
        reasons = [rr for rr in data["rationale"] if isinstance(rr, str)]
        decision_reason = (
            f"결정: {'투자' if decision_int==1 else '비투자'} | "
            f"기술 {data['tech_score_pct']}%, 시장 {data['market_score_pct']}% | "
            f"근거: " + "; ".join(reasons[:3]) + f" | 커버리지:{metrics['coverage']:.2f}"
        )

        # 5) 리플렉션: 종료/재시도 판단
        if decision_int == 1:
            break  # 투자면 즉시 종료
        if metrics["coverage"] >= POLICY["min_coverage"] or r == rounds - 1:
            break  # 근거 충분 or 마지막 라운드면 종료
        # 아니면 다음 라운드에서 파라미터 확장하여 재시도

    # --- 오직 두 필드만 갱신 ---
    state["investment_decision"] = decision_int
    state["decision_reason"] = decision_reason
    state["iteration_count"] = int(state.get("iteration_count", 0)) + 1
    print(f"투자 판단: {state['investment_decision']} (1=투자, 0=비투자) | {state['decision_reason']}")
    return state

# ====== 단독 테스트 ======
if __name__ == "__main__":
    demo: AgentState = {
        "startup_name": "OrbitalMesh",
        "startup_info": {"is_startup": True, "weekly_growth": None, "ltv": None, "cac": None},
        "tech_summary": {"full_text": "LEO 위성 간 레이저 링크 기반 메시 네트워크 모듈."},
        "market_analysis": {"full_text": "해양/원격 산업 IoT. Why Now: 발사 단가 하락·오지 연결 수요 증가."},
        "investment_decision": 0,
        "decision_reason": "",
        "report": "[외부 리포트는 유지됩니다]",
        "iteration_count": 0
    }
    investment_decision_agent(demo)


# # -*- coding: utf-8 -*-
# # 필요한 패키지:
# # pip install -U chromadb sentence-transformers langchain-openai
# # (리랭킹) pip install -U "sentence-transformers>=3.0,<3.1"
# # (호환이슈시) pip install "tokenizers==0.19.1"

# from __future__ import annotations
# from typing import TypedDict, Dict, Any, List, Tuple
# import json

# from langchain_openai import ChatOpenAI

# # ====== 경로/모델 설정 ======
# PERSIST_DIR = "./chroma_store"          # 너의 Chroma persist 디렉터리
# COLLECTION  = "investor_kb"             # 코어 KB 컬렉션명
# EMB_MODEL   = "all-MiniLM-L6-v2"        # SBERT 임베딩 모델
# LLM_MODEL   = "gpt-4o-mini"             # 멀티쿼리/판정 LLM
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 리랭킹용 CrossEncoder

# # ====== AgentState (업데이트된 스키마) ======
# class AgentState(TypedDict):
#     startup_name: str
#     startup_info: dict
#     tech_summary: str
#     market_analysis: str
#     investment_decision: int     # 투자=1, 비투자=0
#     decision_reason: str         # 판단 근거 요약 문자열
#     report: str                  # (주의) 이 함수에서 수정하지 않음
#     iteration_count: int

# # ====== Chroma & Embedding ======
# def _get_chroma():
#     import chromadb
#     client = chromadb.PersistentClient(path=PERSIST_DIR)
#     return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

# def _embed(texts: List[str]):
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
#     실패하면 휴리스틱 폴백(예외는 숨기지 않고, 단지 폴백 사용만 함).
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
#             out, seen = [], set()
#             for q in qs:
#                 q = (q or "").strip()
#                 if not q: continue
#                 k = q.lower()
#                 if k in seen: continue
#                 seen.add(k); out.append(q)
#             if out:
#                 return out[:n]
#     except Exception:
#         pass
#     # 폴백(휴리스틱)
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

# # ====== Post-Retriever: Re-ranking (Cross-Encoder) ======
# def _rerank(query_text: str, hits: List[Dict[str, Any]], topk: int = 4) -> List[Dict[str, Any]]:
#     """
#     hits: [{"text": ..., "meta": {...}, "origin_query": "..."}]
#     Cross-Encoder 점수로 재정렬 → 페이지/소스 중복 제거 → topk 반환
#     """
#     if not hits:
#         return []
#     from sentence_transformers import CrossEncoder
#     ce = CrossEncoder(RERANK_MODEL)
#     pairs = [(query_text, h["text"]) for h in hits]
#     scores = ce.predict(pairs).tolist()
#     for h, s in zip(hits, scores):
#         h["rerank_score"] = float(s)
#     hits.sort(key=lambda x: x["rerank_score"], reverse=True)

#     kept, seen = [], set()
#     for h in hits:
#         m = h["meta"]
#         key = (m.get("source"), m.get("page"))
#         if key in seen:
#             continue
#         seen.add(key); kept.append(h)
#         if len(kept) >= topk: break
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

# # --- 헬퍼: dict/None가 와도 안전하게 문자열로 변환 ---
# def _coerce_text_field(v: Any, key_order: List[str] = ["full_text", "tech_summary", "summary"]) -> str:
#     if isinstance(v, str):
#         return v
#     if isinstance(v, dict):
#         for k in key_order:
#             if k in v and isinstance(v[k], str) and v[k].strip():
#                 return v[k]
#         return json.dumps(v, ensure_ascii=False)  # fallback: dict 전체를 문자열로
#     return "" if v is None else str(v)

# # ====== 투자 판단 노드 (report는 절대 수정하지 않음) ======
# def investment_decision_agent(state: AgentState) -> AgentState:
#     """
#     종합 판단 (리스크, ROI 등)
#     - Scorecard Method (간이: 기술/시장) + RAG 근거 (Multi-Query & Re-ranking)
#     - 출력: investment_decision(int: 투자=1/비투자=0), decision_reason(str)
#     - 주의: report는 수정하지 않음
#     """
#     print("=== [경남] 투자 판단 에이전트 실행 ===")

#     company_name    = state["startup_name"]
#     tech_summary    = _coerce_text_field(state.get("tech_summary"))
#     market_analysis = _coerce_text_field(state.get("market_analysis"), key_order=["full_text", "market_summary", "summary"])

#     # 1) RAG 근거 수집 (Pre: Multi-Query, Post: Re-ranking)
#     evidence = _retrieve_evidence(company_name, tech_summary, market_analysis, per_query_topk=4, final_topk=4)

#     # 2) LLM 평가 (JSON 강제)
#     schema = {
#         "type": "object",
#         "properties": {
#             "tech_score_pct":   {"type":"number"},
#             "market_score_pct": {"type":"number"},
#             "rationale":        {"type":"array", "items":{"type":"string"}, "minItems":3},
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

#     decision_str = str(data["decision"]).lower()          # 'invest' | 'hold'
#     decision_int = 1 if decision_str == "invest" else 0

#     # decision_reason: 1줄 요약 (보고서 수정 금지)
#     rationale_bullets = [r for r in data["rationale"] if isinstance(r, str)]
#     top_reasons = "; ".join(rationale_bullets[:3])
#     decision_reason = (
#         f"결정: {'투자' if decision_int==1 else '비투자'} | "
#         f"기술 {data['tech_score_pct']}%, 시장 {data['market_score_pct']}% | "
#         f"근거: {top_reasons}"
#     )

#     # --- overwrite: 오직 두 필드만 ---
#     state["investment_decision"] = decision_int
#     state["decision_reason"] = decision_reason
#     # state["report"] = state["report"]  # ← 건드리지 않음
#     state["iteration_count"] = int(state.get("iteration_count", 0)) + 1

#     print(f"투자 판단: {state['investment_decision']} (1=투자, 0=비투자)")
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
#         "tech_summary": {
#             'full_text': '기술 평가\n핵심 기술: 초소형 큐브샛을 위한 지상국 시스템 엔지니어링 솔루션 및 AI 기반 위성영상 분석 기술.  \n개발 단계: 상용화  \n차별화: \n1. AI 딥러닝 기반의 위성영상 품질 보정 기술 제공.\n2. 지상국 안테나 및 모뎀 제작으로 통합 솔루션 제공.\n3. UAE를 포함한 해외 시장으로의 위성통신 단말기 공급.  \n강점: \n1. 8년 만에 코스닥 상장으로 시장 신뢰도 확보.\n2. 다양한 위성 관련 제품군 보유.\n3. 우주산업 솔루션에 대한 전문성.  \n약점: \n1. 초기 자금 조달의 어려움 경험.\n2. 경쟁이 치열한 우주산업 내에서의 시장 점유율 확보 필요.  \n기술 점수: 120%\n\n경쟁사 비교\n| 항목 | 컨텍 | 경쟁사A | 경쟁사B |\n|------|------|----------|----------|\n| 크기 | 중소기업 | 대기업 | 중소기업 |\n| 성능 | AI 기반 분석 우수 | 전통적 분석 | 제한적 기능 |\n\n팀 평가\n창업자: CEO 이성희 (우주산업 경력 10년 이상), CTO [이름/경력 미제공]  \n팀 규모: 50명  \n핵심 역량: \n1. 우주산업 솔루션 개발 경험.\n2. AI 및 데이터 분석 기술.\n3. 지상국 시스템 엔지니어링 전문성.  \n산업 경험: 있음 (우주산업 및 관련 기술 분야에서의 경험)  \n팀 점수: 110%\n\n종합 리스크\n- 초기 자금 조달의 어려움으로 인한 재무적 불안정성.\n- 치열한 경쟁 속에서의 시장 점유율 확보의 어려움.', 'tech_score': 120, 'team_score': 110, 'tech_summary': '초소형 큐브샛을 위한 지상국 시스템 엔지니어링 솔루션 및 AI 기반 위성영상 분석 기술.', 'team_summary': '창업자: CEO 이성희 (우주산업 경력 10년 이상), CTO [이름/경력 미제공] 팀 규모: 50명 핵심 역량:', 'key_risks': ['리스크 정보 없음'], 'competitive_table': '| 항목 | 컨텍 | 경쟁사A | 경쟁사B |\n|------|------|----------|----------|\n| 크기 | 중소기업 | 대기업 | 중소기업 |\n| 성능 | AI 기반 분석 우수 | 전통적 분석 | 제한적 기능 |', 'sources': ['https://brunch.co.kr/@cliche-cliche/106', 'https://m.blog.naver.com/ivvlove/222445956219', 'https://www.jake-james.com/blog/what-does-the-ceo-cfo-coo-cto-and-others-do'],
#         },
#         "market_analysis": {
#             "full_text": "Why Now: 발사 단가 하락·오지 연결 수요 증가 … (생략)"
#         },
#         "investment_decision": 0,
#         "decision_reason": "",
#         "report": "",
#         "iteration_count": 0
#     }
#     investment_decision_agent(state)