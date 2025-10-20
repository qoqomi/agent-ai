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
    import chromadb, sqlite3
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
    """
    LLM으로 base_q를 의미 다른 3~5개 쿼리로 확장. 실패하면 휴리스틱 폴백.
    (LangChain ChatOpenAI는 문자열 프롬프트도 가능하므로 messages 대신 단일 프롬프트 사용)
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0,
                     model_kwargs={"response_format":{"type":"json_object"}})
    prompt = (
        "너는 검색 쿼리 생성기다. 같은 정보 목적을 **서로 다른 관점/용어**로 분해하라.\n"
        f"- PILLAR: {pillar}\n"
        f"- BASE_QUERY: {base_q}\n"
        "- 3~5개의 한국어 검색 쿼리를 JSON 배열로만 반환해라(문장부호/설명 금지).\n"
    )
    try:
        rsp = llm.invoke(prompt)
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

# ====== 근거 인용에서 (작성자/기관) 집계 ======
def _collect_sources_from_evidence(evi: Dict[str, List[str]], limit: int = 5) -> List[str]:
    """
    bullets 형식: "- ... (Investor·Pillar·p.12)"
    괄호 안 첫 토큰(Investor)을 출처로 집계한다.
    """
    sources: List[str] = []
    for bullets in evi.values():
        for b in bullets:
            i, j = b.rfind("("), b.rfind(")")
            if i != -1 and j != -1 and j > i:
                tag = b[i+1:j]  # "Investor·Pillar·p.12"
                author = tag.split("·")[0].strip()
                if author and author not in sources:
                    sources.append(author)
            if len(sources) >= limit:
                break
        if len(sources) >= limit:
            break
    return sources

# ====== LLM 판정(JSON, 한국어 강제) ======
def _llm_scorecard_judge(company: str, tech_text: str, market_text: str,
                         rag_context: Dict[str, List[str]]) -> Dict[str, Any]:
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
    sys = (
        "너는 VC 심사역이다. 기술/시장 요약 및 RAG 근거를 보고 Scorecard 간이평가를 수행하라. "
        "제품/기술은 15~20%, 시장은 20~25% 범위 해석, 점수는 50~150%. "
        "최종 결론은 invest 또는 hold 중 하나. "
        "응답은 반드시 JSON 형식으로만 출력하고, 모든 텍스트는 한국어로 작성하라."
    )
    # ChatOpenAI.invoke는 문자열 프롬프트도 허용되므로 단일 프롬프트 사용
    prompt = (
        f"{sys}\n\n"
        "다음 JSON을 참고하여 schema에 맞춰 **한국어로만** 응답:\n"
        + json.dumps({
            "company": company,
            "tech_summary": tech_text,
            "market_analysis": market_text,
            "rag_evidence": {k: v[:3] for k, v in rag_context.items()},
            "format": schema
        }, ensure_ascii=False)
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0,
                     model_kwargs={"response_format":{"type":"json_object"}})
    res = llm.invoke(prompt)
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

        # 4) LLM 판정(JSON, 한국어)
        data = _llm_scorecard_judge(name, tech, market, evidence)
        decision_int = 1 if str(data["decision"]).lower() == "invest" else 0
        reasons = [rr for rr in data["rationale"] if isinstance(rr, str)]

        # ★ 출처(작성자/기관) 집계
        sources = _collect_sources_from_evidence(evidence, limit=5)
        src_str = (", ".join(sources)) if sources else "출처 미기재"

        # ★ 한국어 이유 문자열 + 출처 포함
        decision_reason = (
            f"결정: {'투자' if decision_int==1 else '비투자'} | "
            f"기술 {data['tech_score_pct']}%, 시장 {data['market_score_pct']}% | "
            f"근거: " + "; ".join(reasons[:3]) +
            f" | 출처(작성자/기관): {src_str} | 커버리지:{metrics['coverage']:.2f}"
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

# ====== (선택) 단독 테스트 ======
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
# # decision_module.py
# # RAG 멀티쿼리 + 리랭킹 + 자율 플래닝/리플렉션 내장 판단 노드
# # report는 절대 수정하지 않고, investment_decision(int) / decision_reason(str)만 갱신

# from __future__ import annotations
# from typing import TypedDict, Dict, Any, List, Tuple, Optional
# import json, time

# from langchain_openai import ChatOpenAI

# # ====== 경로/모델 설정 ======
# PERSIST_DIR   = "./chroma_store"                       # Chroma persist 디렉터리
# COLLECTION    = "investor_kb"                          # 코어 KB 컬렉션명
# EMB_MODEL     = "all-MiniLM-L6-v2"                     # SBERT 임베딩 모델
# LLM_MODEL     = "gpt-4o-mini"                          # 멀티쿼리/판정 LLM
# RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2" # 리랭킹용 CrossEncoder

# # ====== 정책(자율 행동 제약) ======
# POLICY = {
#     "max_rounds": 2,             # 최대 재시도 라운드
#     "time_budget_s": 45,         # 시간 예산(초)
#     "min_coverage": 0.55,        # 커버리지 임계치 (타겟 피벗 중 증거 보유 비율)
#     "per_query_topk": [3, 5],    # 라운드별 per-query top-k
#     "final_topk":     [4, 6],    # 라운드별 최종 rerank k
#     "mq_n":           [3, 5],    # 라운드별 multi-query 개수
# }

# # ====== AgentState (최신 스키마) ======
# class AgentState(TypedDict):
#     startup_name: str
#     startup_info: dict
#     tech_summary: str | dict
#     market_analysis: str | dict
#     investment_decision: int      # 투자=1, 비투자=0
#     decision_reason: str          # 판단 근거 요약
#     report: str                   # 이 노드에서는 수정하지 않음
#     iteration_count: int

# # ====== 안전 문자열 변환 ======
# def _coerce_text_field(v: Any, key_order: List[str] = ["full_text","tech_summary","summary"]) -> str:
#     if isinstance(v, str):
#         return v
#     if isinstance(v, dict):
#         for k in key_order:
#             if isinstance(v.get(k), str) and v[k].strip():
#                 return v[k]
#         return json.dumps(v, ensure_ascii=False)
#     return "" if v is None else str(v)

# # ====== Chroma & Embedding ======
# def _get_chroma():
#     import chromadb, sqlite3, traceback
#     try:
#         client = chromadb.PersistentClient(path=PERSIST_DIR)
#         col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
#         return col
#     except sqlite3.OperationalError as e:
#         msg = str(e)
#         if "collections.topic" in msg:
#             raise RuntimeError(
#                 "Chroma 스키마 불일치 감지: persist_dir이 현재 chromadb 버전과 다릅니다.\n"
#                 f"- persist_dir: {PERSIST_DIR}\n"
#                 "해결: (A) 폴더명 변경 후 재-인제스트, 또는 (B) DB 생성 당시 chromadb 버전으로 맞추세요."
#             ) from e
#         raise

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

# # ====== Pre: Multi-Query ======
# def _gen_multi_queries(pillar: str, base_q: str, n: int = 4) -> List[str]:
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

# # ====== Post: Re-ranking (Cross-Encoder) ======
# def _rerank(query_text: str, hits: List[Dict[str, Any]], topk: int = 4) -> List[Dict[str, Any]]:
#     if not hits:
#         return []
#     from sentence_transformers import CrossEncoder
#     ce = CrossEncoder(RERANK_MODEL)  # 최초 로드 후 캐시
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

# # ====== 계획 기반 RAG (특정 피벗만) ======
# def _retrieve_evidence_planned(target_pillars: List[str], name: str, tech: str, market: str,
#                                per_query_topk: int, final_topk: int, mq_n: int) -> Dict[str, List[str]]:
#     col = _get_chroma()
#     out: Dict[str, List[str]] = {}
#     for pillar, base_q in _build_queries(name, tech, market):
#         if pillar not in target_pillars:
#             continue
#         mq_list = _gen_multi_queries(pillar, base_q, n=mq_n)
#         pool: List[Dict[str, Any]] = []
#         for q in mq_list:
#             v = _embed([q])[0].tolist()
#             res = col.query(query_embeddings=[v], n_results=per_query_topk, where={"pillar": pillar})
#             docs, metas = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]
#             for doc, meta in zip(docs, metas):
#                 pool.append({"text": doc, "meta": meta, "origin_query": q})
#         ranked = _rerank(base_q, pool, topk=final_topk)
#         bullets = []
#         for h in ranked:
#             m = h["meta"]
#             tag = f"{m.get('investor','?')}·{m.get('pillar','?')}·p.{m.get('page','?')}"
#             bullets.append(f"- {h['text'][:280].strip()} … ({tag})")
#         out[pillar] = bullets
#     return out

# # ====== 커버리지/밀도 측정 ======
# def _evidence_metrics(evi: Dict[str, List[str]], target_pillars: List[str]) -> Dict[str, float]:
#     hits = sum(1 for p in target_pillars if evi.get(p))
#     total = max(1, len(target_pillars))
#     coverage = hits / total
#     bullets = sum(len(evi.get(p, [])) for p in target_pillars)
#     density = min(1.0, bullets / float(total * 3))  # 피벗당 3개 인용 기대치 기준
#     return {"coverage": coverage, "density": density, "bullets": float(bullets)}

# # ====== 라운드별 타겟 피벗 계획 ======
# def _plan_pillars(tech_text: str, market_text: str, startup_info: Dict[str, Any], round_idx: int) -> List[str]:
#     base = ["PMF", "Market", "Moat", "Traction"]
#     if any(k in startup_info for k in ("ltv", "cac")):
#         base.append("UnitEconomics")
#     if any(k in startup_info for k in ("cash_on_hand", "monthly_burn")):
#         base.append("DefaultAlive")
#     if round_idx == 0:
#         return ["PMF", "Market", "Moat", "Traction"]
#     # 라운드 1 이상: 정량 포함 확장
#     return list(dict.fromkeys(base))

# # ====== 라운드별 하이퍼 파라미터 계획 ======
# def _plan_params(round_idx: int) -> Tuple[int, int, int]:
#     per_q = POLICY["per_query_topk"][min(round_idx, len(POLICY["per_query_topk"])-1)]
#     final_k = POLICY["final_topk"][min(round_idx, len(POLICY["final_topk"])-1)]
#     mq_n   = POLICY["mq_n"][min(round_idx, len(POLICY["mq_n"])-1)]
#     return per_q, final_k, mq_n

# # ====== LLM 판정(JSON) ======
# def _llm_scorecard_judge(company: str, tech_text: str, market_text: str, rag_context: Dict[str, List[str]]) -> Dict[str, Any]:
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
#     sys = ("너는 VC 심사역이다. 기술/시장 요약 및 RAG 근거를 보고 Scorecard 간이평가를 수행하라. "
#            "제품/기술은 15~20%, 시장은 20~25% 범위 해석, 점수는 50~150%. "
#            "최종 결론은 invest 또는 hold 중 하나. 응답은 JSON만.")
#     user = {
#         "company": company,
#         "tech_summary": tech_text,
#         "market_analysis": market_text,
#         "rag_evidence": {k: v[:3] for k, v in rag_context.items()},
#         "format": schema
#     }
#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0, model_kwargs={"response_format":{"type":"json_object"}})
#     res = llm.invoke([{"role":"system","content":sys},
#                       {"role":"user","content":"다음 JSON을 참고하여 schema로만 응답:\n"+json.dumps(user, ensure_ascii=False)}])
#     try:
#         data = json.loads(res.content)
#     except Exception:
#         raise ValueError(f"LLM JSON 파싱 실패: {res.content}")
#     for k in ("tech_score_pct","market_score_pct","rationale","decision"):
#         if k not in data:
#             raise KeyError(f"LLM 응답 누락: {k}")
#     return data

# # ====== 메인 노드: 자율 플래닝/리플렉션 내장 ======
# def investment_decision_agent(state: AgentState) -> AgentState:
#     """
#     자율 플래닝/리플렉션 내장 투자판단 노드
#     - 출력: investment_decision(1/0), decision_reason(str)만 갱신
#     - report는 수정하지 않음
#     """
#     print("=== [경남] 투자 판단 에이전트(자율형) 실행 ===")
#     t0 = time.monotonic()

#     name   = state["startup_name"]
#     info   = state.get("startup_info", {}) or {}
#     tech   = _coerce_text_field(state.get("tech_summary"))
#     market = _coerce_text_field(state.get("market_analysis"), ["full_text","market_summary","summary"])

#     decision_int = 0
#     decision_reason = "비투자 | 사유: 초기 상태"
#     rounds = POLICY["max_rounds"]

#     for r in range(rounds):
#         # 1) 시간 예산
#         if time.monotonic() - t0 > POLICY["time_budget_s"]:
#             decision_int = 0
#             decision_reason = "비투자 | 사유: 시간 예산 초과"
#             break

#         # 2) 계획 수립(타겟 피벗/하이퍼파라미터)
#         target_pillars = _plan_pillars(tech, market, info, r)
#         per_q, final_k, mq_n = _plan_params(r)

#         # 3) RAG (계획 기반 멀티쿼리 → 리랭킹)
#         evidence = _retrieve_evidence_planned(target_pillars, name, tech, market,
#                                               per_query_topk=per_q, final_topk=final_k, mq_n=mq_n)
#         metrics = _evidence_metrics(evidence, target_pillars)

#         # 4) LLM 판정(JSON)
#         data = _llm_scorecard_judge(name, tech, market, evidence)
#         decision_int = 1 if str(data["decision"]).lower() == "invest" else 0
#         reasons = [rr for rr in data["rationale"] if isinstance(rr, str)]
#         decision_reason = (
#             f"결정: {'투자' if decision_int==1 else '비투자'} | "
#             f"기술 {data['tech_score_pct']}%, 시장 {data['market_score_pct']}% | "
#             f"근거: " + "; ".join(reasons[:3]) + f" | 커버리지:{metrics['coverage']:.2f}"
#         )

#         # 5) 리플렉션: 종료/재시도 판단
#         if decision_int == 1:
#             break  # 투자면 즉시 종료
#         if metrics["coverage"] >= POLICY["min_coverage"] or r == rounds - 1:
#             break  # 근거 충분 or 마지막 라운드면 종료
#         # 아니면 다음 라운드에서 파라미터 확장하여 재시도

#     # --- 오직 두 필드만 갱신 ---
#     state["investment_decision"] = decision_int
#     state["decision_reason"] = decision_reason
#     state["iteration_count"] = int(state.get("iteration_count", 0)) + 1
#     print(f"투자 판단: {state['investment_decision']} (1=투자, 0=비투자) | {state['decision_reason']}")
#     return state

# # ====== 단독 테스트 ======
# if __name__ == "__main__":
#     demo: AgentState = {
#         "startup_name": "OrbitalMesh",
#         "startup_info": {"is_startup": True, "weekly_growth": None, "ltv": None, "cac": None},
#         "tech_summary": {"full_text": "LEO 위성 간 레이저 링크 기반 메시 네트워크 모듈."},
#         "market_analysis": {"full_text": "해양/원격 산업 IoT. Why Now: 발사 단가 하락·오지 연결 수요 증가."},
#         "investment_decision": 0,
#         "decision_reason": "",
#         "report": "[외부 리포트는 유지됩니다]",
#         "iteration_count": 0
#     }
#     investment_decision_agent(demo)