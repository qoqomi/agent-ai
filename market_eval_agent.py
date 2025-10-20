"""
[역할]
- 한국 우주산업 스타트업/세그먼트의 '시장성'을 점수화하고 밴드(Invest/Watch/Hold)로 분류하는 에이전트
- 내부 PDF 2종을 RAG로 검색(근거 수집) + (옵션) Tavily 웹 서칭 병행 → LLM에 근거를 주입 → '점수카드(JSON)' 생성
- 추가: state["startup_info"]의 정량 신호(weekly_growth, cash_on_hand, monthly_burn, ltv, cac)를 보조 근거로 활용

[입력]
- state["startup_name"] : 분석 대상(세그먼트/회사명) 예) "저궤도 위성", "광학 위성", "한글스타트업"
- state["startup_info"] : dict (예: {"weekly_growth": 8.1, "cash_on_hand": 1500000, "monthly_burn": 70000, "ltv": 2400, "cac": 700})
- state["geo"]          : 지리 범위(기본 "KR")

[출력]
- state["market_analysis"] : 점수카드 JSON 문자열

[필수 파일]
- data/2024년 우주산업실태조사 보고서(최종본).pdf
- data/우주청_2025년_예산_편성안.pdf

[환경 변수]
- OPENAI_API_KEY : OpenAI 키 (.env에 "OPENAI_API_KEY=..." 형태로 저장)
- (선택) MARKET_EVAL_PDF1, MARKET_EVAL_PDF2 : 기본 PDF 경로 커스터마이즈
- (선택) EMBEDDING_MODEL      : 임베딩 모델명(기본 intfloat/multilingual-e5-small)
- (선택) USE_HYBRID_WEB       : "true"/"false" (웹 검색 병행 여부, 기본 true)
- (선택) TAVILY_API_KEY       : 설정 시 Tavily 웹 서칭 사용
"""

from __future__ import annotations
from typing import Dict, Any, List
import os
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- LangChain 관련 모듈 로드 ---
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Tavily (웹 검색, 선택) ---
# pip install tavily-python
try:
    from tavily import TavilyClient

    _TAVILY_AVAILABLE = True
except Exception:
    _TAVILY_AVAILABLE = False

# =========================
# 1) 점수 가중치 정의 (총 100점)
# -------------------------
# - 평가 항목은 '시장성' 판단에 핵심이 되는 7개 지표로 구성
# - 각 항목은 0~100점으로 LLM이 제안하고, 아래 가중치로 가중합(총점) 계산
# =========================
WEIGHTS: Dict[str, int] = {
    "market_size": 20,  # 시장규모(크기/확장성): TAM/SAM, 수요 규모
    "growth": 15,  # 성장률: 산업 성장 속도, CAGR, 정책 드라이브
    "demand_signals": 15,  # 수요 신호: 정부 조달/민간 상용 수요의 가시성
    "entry_barriers": 10,  # 진입장벽: 규제/CapEx/기술난이도(낮을수록 유리)
    "policy_budget_tailwind": 15,  # 정책/예산 순풍: 정부 예산 확대, 기구 신설 등
    "competition_intensity": 10,  # 경쟁 강도: 경쟁 심하면 불리(낮을수록 유리)
    "gtm_feasibility": 15,  # 시장진입 용이성: 채널/GTM/파트너십/PoC 가능성
}

# 검색 쿼리에 붙일 힌트(한국 우주 스타트업 맥락 강화)
SECTOR_HINTS = [
    "우주 스타트업",
    "민간",
    "상업",
    "조달",
    "VC",
    "투자",
    "위성 소형화",
    "저궤도 위성",
    "광학 위성",
    "LEO",
    "EO",
    "payload",
    "지상국",
    "발사체 보급",
]

# PDF 경로 (환경변수로 오버라이드 가능)
_DEFAULT_PDF1 = os.getenv(
    "MARKET_EVAL_PDF1", "data/2024년 우주산업실태조사 보고서(최종본).pdf"
)
_DEFAULT_PDF2 = os.getenv("MARKET_EVAL_PDF2", "data/우주청_2025년_예산_편성안.pdf")

# 임베딩 모델(한국어 품질 강화 기본값)
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# 웹 검색 병행 여부 (기본 True)
_USE_HYBRID_WEB = os.getenv("USE_HYBRID_WEB", "true").lower() == "true"

# 모듈 전역 retriever 캐시
_RETRIEVER = None


# ==========================================================
# 2) FAISS 인덱스: 자동 저장 + 재사용
# ----------------------------------------------------------
# - 최초 실행 시: PDF 로드 → 청크 분할 → 임베딩 → FAISS 인덱스 생성/저장
# - 이후 실행 시: 디스크에서 로드(빠름)
# - 저장 위치: data/faiss_market_eval/{index.faiss, index.pkl}
# ==========================================================
def _ensure_retriever(k: int = 8):
    global _RETRIEVER
    if _RETRIEVER is not None:
        return _RETRIEVER

    save_dir = Path("data/faiss_market_eval")
    os.makedirs(save_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)

    # 인덱스가 있으면 로드, 없으면 생성
    if (save_dir / "index.faiss").exists():
        vectordb = FAISS.load_local(
            str(save_dir),
            embeddings,
            allow_dangerous_deserialization=True,  # 신뢰 경로에서만 사용
        )
        print(f"[FAISS] 기존 인덱스 로드 완료 → {save_dir}")
    else:
        # --- 2-1) PDF 존재 확인 ---
        pdf_paths = [_DEFAULT_PDF1, _DEFAULT_PDF2]
        for p in pdf_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"PDF not found: {p}")

        # --- 2-2) 로드 + 청크 분할 ---
        # 문서 길이를 1000자 단위로 분할(150자 중첩) → 검색 품질 및 문맥성 유지
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs_all: List[Document] = []
        for p in pdf_paths:
            loader = PyPDFLoader(p)
            pages = loader.load()  # 페이지별 Document 리스트
            for d in pages:
                d.metadata["source"] = Path(
                    p
                ).name  # 출처 파일명 유지(후속 근거 표시에 사용)
                # 페이지 번호 보존(뷰어/근거 추적 편의)
                if "page" not in d.metadata:
                    d.metadata["page"] = d.metadata.get("page", None)
            docs_all.extend(splitter.split_documents(pages))

        # --- 2-3) 임베딩 → FAISS 인덱스 생성 ---
        vectordb = FAISS.from_documents(docs_all, embeddings)

        # --- 2-4) 디스크에 저장(영속화) ---
        vectordb.save_local(str(save_dir))
        print(f"[FAISS] 새 인덱스 생성 및 저장 완료 → {save_dir}")

    # retriever: Top-k 유사 문서 검색 인터페이스 (가능하면 MMR)
    try:
        _RETRIEVER = vectordb.as_retriever(
            search_kwargs={
                "k": k,
                "search_type": "mmr",
                "fetch_k": max(k * 3, 20),
                "lambda_mult": 0.7,
            }
        )
    except Exception:
        _RETRIEVER = vectordb.as_retriever(search_kwargs={"k": k})
    return _RETRIEVER


# ==========================================================
# 3) 총점/밴드 계산 유틸
# ----------------------------------------------------------
# - entry_barriers, competition_intensity는 '낮을수록 좋음'이므로
#   점수를 (100 - 원점수)로 반전하여 가중합에 반영
# - 점수 누락 시 55점(중립)으로 보정하여 보수적/중립적 평가 유지
# - 밴드 기준:
#   * Invest : 총점 ≥ 80 (즉시 투자 고려)
#   * Watch  : 60~79      (긍정적이나 추적/확증 필요)
#   * Hold   : < 60       (보류/추가 근거 필요)
# ==========================================================
def _total_score(scores: Dict[str, float]) -> float:
    adj = {k: float(scores.get(k, 55)) for k in WEIGHTS}
    adj["entry_barriers"] = 100 - adj["entry_barriers"]  # 낮을수록 유리 → 반전
    adj["competition_intensity"] = (
        100 - adj["competition_intensity"]
    )  # 낮을수록 유리 → 반전
    return round(sum(adj[k] * WEIGHTS[k] / 100 for k in WEIGHTS), 1)


def _band(total: float) -> str:
    if total >= 80:
        return "Invest"  # 투자 유망: 근거 충분, 규모/성장/정책 드라이브 등 우수
    elif total >= 60:
        return "Watch"  # 긍정적이지만 리스크/불확실성 존재 → 모니터링
    return "Hold"  # 보류: 근거 부족/경쟁과열/시장 미성숙 등


# ==========================================================
# 4) LLM JSON 파싱 유틸
# ----------------------------------------------------------
# - 코드펜스(````json ... ````)를 포함/미포함 모두 대응
# - 최상위 { ... } 블록만 추출하여 json.loads 시도
# - 실패 시 {} 반환 → 이후 기본값 보정 로직이 동작
# ==========================================================
def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    b = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if b:
        s = b.group(0)
    try:
        return json.loads(s)
    except Exception:
        return {}


# ==========================================================
# 5) 컨텍스트 수집 (RAG 핵심)
# ----------------------------------------------------------
# - 평가 대상(target)과 geo를 포함한 4가지 쿼리로 PDF/웹에서 근거 검색
# - 쿼리 의도: (1)시장규모/성장성 (2)정부 예산/정책 (3)수요/상용 (4)경쟁/진입장벽
# - 검색된 문단의 '출처 파일명(+페이지)/웹 제목+URL'을 LLM에 그대로 주입
# - 추가: state["startup_info"]의 정량 신호를 [INTERNAL] 블록으로 함께 주입
# ==========================================================
def _gather_context_pdf(target: str, geo: str = "KR", topk_each: int = 6) -> str:
    retriever = _ensure_retriever()
    hints = " ".join(SECTOR_HINTS)
    base = f"{geo} {target}".strip()

    queries = [
        f"{base} {hints} 시장 규모 성장률 활동금액",
        f"{base} {hints} 정부 예산 신규 사업 우주청 2025 계획",
        f"{base} {hints} 수요 조달 상용 활용서비스",
        f"{base} {hints} 경쟁 진입장벽 국내 기업 동향 투자",
    ]

    docs: List[Document] = []
    for q in queries:
        try:
            docs.extend(retriever.get_relevant_documents(q))
        except Exception:
            continue

    # (간단한) 중복 제거: (source 파일명 + 앞부분 해시) 기준
    seen, uniq = set(), []
    for d in docs:
        src = d.metadata.get("source") or ""
        key = (src, hash(d.page_content[:200]))
        if key not in seen:
            uniq.append(d)
            seen.add(key)

    # 과도한 컨텍스트 길이 방지
    uniq = uniq[:topk_each]

    # LLM에 바로 주입할 문자열 형태로 포맷팅
    blocks: List[str] = []
    for d in uniq:
        src = d.metadata.get("source") or ""
        page = d.metadata.get("page")
        page_str = f" p.{page}" if isinstance(page, int) else ""
        text = d.page_content.strip().replace("\n", " ")
        blocks.append(f"[PDF:{src}{page_str}] {text}")
    return "\n\n".join(blocks)


def _web_search_context(target: str, geo: str = "KR", max_results: int = 4) -> str:
    """Tavily가 활성화되면 최신 웹 근거를 수집한다."""
    if not _TAVILY_AVAILABLE:
        return ""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ""

    tavily = TavilyClient(api_key=api_key)

    queries = [
        f"{geo} {target} 시장 규모 성장률 2024 2025",
        f"{geo} {target} 정부 예산 정책 우주청 2025",
        f"{geo} {target} 수요 상용화 조달 고객사",
        f"{geo} {target} 경쟁사 투자 유치 VC 라운드",
    ]

    blocks, seen_urls = [], set()
    for q in queries:
        try:
            r = tavily.search(
                query=q,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=True,
                include_domains=None,
            )
            for item in r.get("results", []):
                url = (item.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = (item.get("title") or "").strip()
                content = (item.get("raw_content") or item.get("content") or "").strip()
                published = (item.get("published_date") or "").strip()
                snippet = (content[:800] + "…") if len(content) > 800 else content
                date_str = f" ({published})" if published else ""
                blocks.append(f"[WEB:{title}{date_str}] {snippet}\n(source: {url})")
        except Exception:
            continue

    return "\n\n".join(blocks)


def _format_internal_signals(startup_info: Dict[str, Any]) -> str:
    """state['startup_info']에서 보조 근거 신호를 계산/서식화."""
    if not isinstance(startup_info, dict) or not startup_info:
        return ""

    def _num(x):
        try:
            return float(x)
        except Exception:
            return None

    is_startup = startup_info.get("is_startup")
    weekly_growth = _num(startup_info.get("weekly_growth"))
    cash = _num(startup_info.get("cash_on_hand"))
    burn = _num(startup_info.get("monthly_burn"))
    ltv = _num(startup_info.get("ltv"))
    cac = _num(startup_info.get("cac"))

    runway_months = None
    if cash is not None and burn and burn > 0:
        runway_months = round(cash / burn, 2)

    ltv_cac = None
    if ltv is not None and cac and cac > 0:
        ltv_cac = round(ltv / cac, 2)

    lines = ["[INTERNAL] 내부 정량 신호 (출처: 입력 startup_info)"]
    if is_startup is not None:
        lines.append(f"- is_startup: {bool(is_startup)}")
    if weekly_growth is not None:
        lines.append(f"- weekly_growth(%): {weekly_growth}")
    if cash is not None:
        lines.append(f"- cash_on_hand: {cash}")
    if burn is not None:
        lines.append(f"- monthly_burn: {burn}")
    if runway_months is not None:
        lines.append(f"- runway_months: {runway_months}")
    if ltv is not None:
        lines.append(f"- LTV: {ltv}")
    if cac is not None:
        lines.append(f"- CAC: {cac}")
    if ltv_cac is not None:
        lines.append(f"- LTV/CAC: {ltv_cac}")

    # 간단 해석 힌트(LLM 가이드용)
    hints = []
    if weekly_growth is not None:
        hints.append("weekly_growth가 높을수록 growth/demand_signals에 우호적")
    if runway_months is not None:
        hints.append("runway_months<6이면 gtm_feasibility/수요 실현 리스크 고려")
    if ltv_cac is not None:
        hints.append("LTV/CAC>1.5면 gtm_feasibility/demand_signals 가점, <1이면 감점")
    if hints:
        lines.append("- 해석 가이드: " + "; ".join(hints))

    return "\n".join(lines)


def _gather_context(
    target: str,
    geo: str = "KR",
    topk_each: int = 6,
    startup_info: Dict[str, Any] | None = None,
) -> str:
    """PDF + (옵션)웹 + (옵션)내부 정량 신호를 하나의 컨텍스트로 합친다."""
    pdf_ctx = _gather_context_pdf(target=target, geo=geo, topk_each=topk_each)
    ctx_parts = [pdf_ctx] if pdf_ctx else []

    if _USE_HYBRID_WEB:
        web_ctx = _web_search_context(target=target, geo=geo, max_results=4)
        if web_ctx:
            ctx_parts.append(web_ctx)

    internal_ctx = _format_internal_signals(startup_info or {})
    if internal_ctx:
        ctx_parts.append(internal_ctx)

    return "\n\n".join([p for p in ctx_parts if p])


# ==========================================================
# 6) 메인 에이전트
# ----------------------------------------------------------
# - 컨텍스트(RAG) + 시스템/유저 프롬프트로 LLM 호출
# - LLM이 제안한 7개 지표 점수/근거를 JSON으로 파싱
# - 점수 누락 보정 → 총점/밴드 계산 → state에 최종 JSON 저장
# ==========================================================
def market_eval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    target = state.get("startup_name") or "저궤도 위성"
    geo = state.get("geo", "KR")
    startup_info = state.get("startup_info") or {}

    # 1) (하이브리드) 근거 수집 + 내부 정량 신호 포함
    context = _gather_context(target=target, geo=geo, startup_info=startup_info)

    # 2) 평가 원칙(시스템 프롬프트)
    system_prompt = (
        "당신은 한국 우주산업 스타트업의 시장성 평가를 수행하는 전문 애널리스트입니다. "
        "정부·산업 리포트(PDF)와 신뢰 가능한 웹 자료, 그리고 제공된 내부 정량 신호를 근거로 "
        "‘시장성 평가 점수카드(JSON)’를 작성하십시오.\n\n"
        "평가 원칙:\n"
        "1. 근거 기반 평가(출처 파일명/URL/날짜/내부신호 여부를 최대한 명시)\n"
        "2. 수치/경향/정책 예산 등은 가능한 한 구체적으로 기술\n"
        "3. 항목별 점수는 0~100 범위로 부여\n"
        "4. 최종 출력은 완전한 JSON 하나만 포함"
    )

    # 3) 사용자 프롬프트(컨텍스트 + 출력 스키마 예시)
    user_prompt = f"""
[분석 대상]
세그먼트/스타트업: {target}
지리범위: {geo}

[컨텍스트(근거)]
{context}

[출력 형식(JSON 예시)]
{{
  "target": "{target}",
  "geo": "{geo}",
  "scores": {{
    "market_size": 0,
    "growth": 0,
    "demand_signals": 0,
    "entry_barriers": 0,
    "policy_budget_tailwind": 0,
    "competition_intensity": 0,
    "gtm_feasibility": 0
  }},
  "total": 0,
  "band": "Invest|Watch|Hold",
  "rationale": "핵심 요약",
  "key_evidence": [
    {{"claim": "string", "source": "filename or URL or INTERNAL", "date": "YYYY-MM-DD"}}
  ],
  "risks": ["string"],
  "assumptions": ["string"],
  "data_freshness": "as of YYYY-MM-DD"
}}
※ 반드시 유효한 JSON만 반환하십시오.
"""

    # 4) LLM 호출 (온도 0: 결정적/일관된 출력)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # 5) JSON 파싱(+안전한 폴백)
    data = _extract_json(getattr(result, "content", ""))
    if not isinstance(data, dict) or "scores" not in data:
        # 파싱 실패 시 기본값(중립 55점)으로 안전하게 구성
        data = {
            "target": target,
            "geo": geo,
            "scores": {k: 55 for k in WEIGHTS},
            "rationale": "기본값 적용(LLM 파싱 실패 또는 근거 부족)",
            "key_evidence": [],
            "risks": [],
            "assumptions": [],
        }

    # 6) 점수 누락 보정 → 총점/밴드/날짜 계산
    for k in WEIGHTS:
        try:
            v = float(data["scores"].get(k, 55))
        except Exception:
            v = 55.0
        data["scores"][k] = max(0.0, min(100.0, v))  # 0~100 클램핑

    total = _total_score(data["scores"])
    data["total"] = total
    data["band"] = _band(total)
    data["data_freshness"] = f"as of {datetime.now().date()}"

    # 7) 최종 결과 JSON 문자열을 state에 저장
    state["market_analysis"] = json.dumps(data, ensure_ascii=False, indent=2)
    return state


# ==========================================================
# 7) 단독 실행 테스트
# ----------------------------------------------------------
# - 바로 실행 시, '저궤도 위성'에 대한 점수카드 JSON을 콘솔에 출력
# - 최초 1회 실행 때 FAISS 인덱스를 생성/저장 후, 이후에는 재사용
# ==========================================================
if __name__ == "__main__":
    test_state = {
        "startup_name": "저궤도 위성",
        "geo": "KR",
        "startup_info": {
            "is_startup": True,
            "weekly_growth": 8.1,
            "cash_on_hand": 1500000,
            "monthly_burn": 70000,
            "ltv": 2400,
            "cac": 700,
        },
        "tech_summary": "",
        "market_analysis": "",
        "investment_decision": "",
        "report": "",
        "iteration_count": 0,
    }
    out = market_eval_agent(test_state)
    print(out["market_analysis"])

# """
# [역할]
# - 한국 우주산업 스타트업/세그먼트의 '시장성'을 점수화하고 밴드(Invest/Watch/Hold)로 분류하는 에이전트
# - 내부 PDF 2종을 RAG로 검색(근거 수집) → LLM에 근거를 주입 → '점수카드(JSON)' 생성

# [입력]
# - state["startup_name"] : 분석 대상(세그먼트/회사명) 예) "저궤도 위성", "광학 위성", "한글스타트업"
# - state["geo"]          : 지리 범위(기본 "KR")

# [출력]
# - state["market_analysis"] : 점수카드 JSON 문자열

# [필수 파일]
# - data/2024년 우주산업실태조사 보고서(최종본).pdf
# - data/우주청_2025년_예산_편성안.pdf

# [환경 변수]
# - OPENAI_API_KEY : OpenAI 키 (.env에 "OPENAI_API_KEY=..." 형태로 저장)
# - (선택) MARKET_EVAL_PDF1, MARKET_EVAL_PDF2 : 기본 PDF 경로 커스터마이즈
# """

# from __future__ import annotations
# from typing import Dict, Any, List
# import os
# import json
# import re
# from datetime import datetime
# from pathlib import Path

# from dotenv import load_dotenv

# load_dotenv()

# # --- LangChain 관련 모듈 로드 ---
# from langchain_openai import ChatOpenAI
# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# # =========================
# # 1) 점수 가중치 정의 (총 100점)
# # -------------------------
# # - 평가 항목은 '시장성' 판단에 핵심이 되는 7개 지표로 구성
# # - 각 항목은 0~100점으로 LLM이 제안하고, 아래 가중치로 가중합(총점) 계산
# # =========================
# WEIGHTS: Dict[str, int] = {
#     "market_size": 20,  # 시장규모(크기/확장성): TAM/SAM, 수요 규모
#     "growth": 15,  # 성장률: 산업 성장 속도, CAGR, 정책 드라이브
#     "demand_signals": 15,  # 수요 신호: 정부 조달/민간 상용 수요의 가시성
#     "entry_barriers": 10,  # 진입장벽: 규제/CapEx/기술난이도(낮을수록 유리)
#     "policy_budget_tailwind": 15,  # 정책/예산 순풍: 정부 예산 확대, 기구 신설 등
#     "competition_intensity": 10,  # 경쟁 강도: 경쟁 심하면 불리(낮을수록 유리)
#     "gtm_feasibility": 15,  # 시장진입 용이성: 채널/GTM/파트너십/PoC 가능성
# }

# # 검색 쿼리에 붙일 힌트(한국 우주 스타트업 맥락 강화)
# SECTOR_HINTS = [
#     "우주 스타트업",
#     "민간",
#     "상업",
#     "조달",
#     "VC",
#     "투자",
#     "위성 소형화",
#     "저궤도 위성",
#     "광학 위성",
#     "LEO",
#     "EO",
#     "payload",
#     "지상국",
#     "발사체 보급",
# ]

# # PDF 경로 (환경변수로 오버라이드 가능)
# _DEFAULT_PDF1 = os.getenv(
#     "MARKET_EVAL_PDF1", "data/2024년 우주산업실태조사 보고서(최종본).pdf"
# )
# _DEFAULT_PDF2 = os.getenv("MARKET_EVAL_PDF2", "data/우주청_2025년_예산_편성안.pdf")

# # 모듈 전역 retriever 캐시
# _RETRIEVER = None


# # ==========================================================
# # 2) FAISS 인덱스: 자동 저장 + 재사용
# # ----------------------------------------------------------
# # - 최초 실행 시: PDF 로드 → 청크 분할 → 임베딩 → FAISS 인덱스 생성/저장
# # - 이후 실행 시: 디스크에서 로드(빠름)
# # - 저장 위치: data/faiss_market_eval/{index.faiss, index.pkl}
# # ==========================================================
# def _ensure_retriever(k: int = 8):
#     global _RETRIEVER
#     if _RETRIEVER is not None:
#         return _RETRIEVER

#     save_dir = Path("data/faiss_market_eval")
#     os.makedirs(save_dir, exist_ok=True)

#     # (권장) 경량 범용 문서 임베딩 모델
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     # 인덱스가 있으면 로드, 없으면 생성
#     if (save_dir / "index.faiss").exists():
#         vectordb = FAISS.load_local(
#             str(save_dir),
#             embeddings,
#             allow_dangerous_deserialization=True,  # 신뢰 경로에서만 사용
#         )
#         print(f"[FAISS] 기존 인덱스 로드 완료 → {save_dir}")
#     else:
#         # --- 2-1) PDF 존재 확인 ---
#         pdf_paths = [_DEFAULT_PDF1, _DEFAULT_PDF2]
#         for p in pdf_paths:
#             if not os.path.exists(p):
#                 raise FileNotFoundError(f"PDF not found: {p}")

#         # --- 2-2) 로드 + 청크 분할 ---
#         # 문서 길이를 1000자 단위로 분할(150자 중첩) → 검색 품질 및 문맥성 유지
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         docs_all: List[Document] = []
#         for p in pdf_paths:
#             loader = PyPDFLoader(p)
#             pages = loader.load()  # 페이지별 Document 리스트
#             for d in pages:
#                 d.metadata["source"] = Path(
#                     p
#                 ).name  # 출처 파일명 유지(후속 근거 표시에 사용)
#             docs_all.extend(splitter.split_documents(pages))

#         # --- 2-3) 임베딩 → FAISS 인덱스 생성 ---
#         vectordb = FAISS.from_documents(docs_all, embeddings)

#         # --- 2-4) 디스크에 저장(영속화) ---
#         vectordb.save_local(str(save_dir))
#         print(f"[FAISS] 새 인덱스 생성 및 저장 완료 → {save_dir}")

#     # retriever: Top-k 유사 문서 검색 인터페이스
#     _RETRIEVER = vectordb.as_retriever(search_kwargs={"k": k})
#     return _RETRIEVER


# # ==========================================================
# # 3) 총점/밴드 계산 유틸
# # ----------------------------------------------------------
# # - entry_barriers, competition_intensity는 '낮을수록 좋음'이므로
# #   점수를 (100 - 원점수)로 반전하여 가중합에 반영
# # - 점수 누락 시 55점(중립)으로 보정하여 보수적/중립적 평가 유지
# # - 밴드 기준:
# #   * Invest : 총점 ≥ 80 (즉시 투자 고려)
# #   * Watch  : 60~79      (긍정적이나 추적/확증 필요)
# #   * Hold   : < 60       (보류/추가 근거 필요)
# # ==========================================================
# def _total_score(scores: Dict[str, float]) -> float:
#     adj = {k: float(scores.get(k, 55)) for k in WEIGHTS}
#     adj["entry_barriers"] = 100 - adj["entry_barriers"]  # 낮을수록 유리 → 반전
#     adj["competition_intensity"] = (
#         100 - adj["competition_intensity"]
#     )  # 낮을수록 유리 → 반전
#     return round(sum(adj[k] * WEIGHTS[k] / 100 for k in WEIGHTS), 1)


# def _band(total: float) -> str:
#     if total >= 80:
#         return "Invest"  # 투자 유망: 근거 충분, 규모/성장/정책 드라이브 등 우수
#     elif total >= 60:
#         return "Watch"  # 긍정적이지만 리스크/불확실성 존재 → 모니터링
#     return "Hold"  # 보류: 근거 부족/경쟁과열/시장 미성숙 등


# # ==========================================================
# # 4) LLM JSON 파싱 유틸
# # ----------------------------------------------------------
# # - 코드펜스(```json ... ```)를 포함/미포함 모두 대응
# # - 최상위 { ... } 블록만 추출하여 json.loads 시도
# # - 실패 시 {} 반환 → 이후 기본값 보정 로직이 동작
# # ==========================================================
# def _extract_json(text: str) -> Dict[str, Any]:
#     if not text:
#         return {}
#     s = text.strip()
#     m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
#     if m:
#         s = m.group(1).strip()
#     b = re.search(r"\{.*\}", s, flags=re.DOTALL)
#     if b:
#         s = b.group(0)
#     try:
#         return json.loads(s)
#     except Exception:
#         return {}


# # ==========================================================
# # 5) 컨텍스트 수집 (RAG 핵심)
# # ----------------------------------------------------------
# # - 평가 대상(target)과 geo를 포함한 4가지 쿼리로 PDF에서 Top-k 근거 검색
# # - 쿼리 의도: (1)시장규모/성장성 (2)정부 예산/정책 (3)수요/상용 (4)경쟁/진입장벽
# # - 검색된 문단의 '출처 파일명 + 본문 요약'을 LLM에 그대로 주입
# # ==========================================================
# def _gather_context(target: str, geo: str = "KR", topk_each: int = 6) -> str:
#     retriever = _ensure_retriever()
#     hints = " ".join(SECTOR_HINTS)
#     base = f"{geo} {target}".strip()

#     queries = [
#         f"{base} {hints} 시장 규모 성장률 활동금액",
#         f"{base} {hints} 정부 예산 신규 사업 우주청 2025 계획",
#         f"{base} {hints} 수요 조달 상용 활용서비스",
#         f"{base} {hints} 경쟁 진입장벽 국내 기업 동향 투자",
#     ]

#     docs: List[Document] = []
#     for q in queries:
#         try:
#             docs.extend(retriever.get_relevant_documents(q))
#         except Exception:
#             # retriever가 일시 실패해도 전체 플로우는 계속(부분 강건성)
#             continue

#     # (간단한) 중복 제거: (source 파일명 + 앞부분 해시) 기준
#     seen, uniq = set(), []
#     for d in docs:
#         src = d.metadata.get("source") or ""
#         key = (src, hash(d.page_content[:200]))
#         if key not in seen:
#             uniq.append(d)
#             seen.add(key)

#     # 과도한 컨텍스트 길이 방지
#     uniq = uniq[:topk_each]

#     # LLM에 바로 주입할 문자열 형태로 포맷팅
#     blocks: List[str] = []
#     for d in uniq:
#         src = d.metadata.get("source") or ""
#         text = d.page_content.strip().replace("\n", " ")
#         blocks.append(f"[{src}] {text}")
#     return "\n\n".join(blocks)


# # ==========================================================
# # 6) 메인 에이전트
# # ----------------------------------------------------------
# # - 컨텍스트(RAG) + 시스템/유저 프롬프트로 LLM 호출
# # - LLM이 제안한 7개 지표 점수/근거를 JSON으로 파싱
# # - 점수 누락 보정 → 총점/밴드 계산 → state에 최종 JSON 저장
# # ==========================================================
# def market_eval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
#     target = state.get("startup_name") or "저궤도 위성"
#     geo = state.get("geo", "KR")

#     # 1) PDF에서 근거 수집
#     context = _gather_context(target=target, geo=geo)

#     # 2) 평가 원칙을 명확히 고정(시스템 프롬프트)
#     system_prompt = (
#         "당신은 한국 우주산업 스타트업의 시장성 평가를 수행하는 전문 애널리스트입니다. "
#         "정부·산업 리포트를 근거로 ‘시장성 평가 점수카드(JSON)’를 작성하십시오.\n\n"
#         "평가 원칙:\n"
#         "1. 근거 기반 평가(출처 파일명을 명시)\n"
#         "2. 수치/경향/정책 예산 등은 가능한 한 구체적으로 기술\n"
#         "3. 항목별 점수는 0~100 범위로 부여\n"
#         "4. 최종 출력은 완전한 JSON 하나만 포함"
#     )

#     # 3) 사용자 프롬프트(컨텍스트 + 출력 스키마 예시 제공)
#     user_prompt = f"""
# [분석 대상]
# 세그먼트/스타트업: {target}
# 지리범위: {geo}

# [컨텍스트(근거)]
# {context}

# [출력 형식(JSON 예시)]
# {{
#   "target": "{target}",
#   "geo": "{geo}",
#   "scores": {{
#     "market_size": 0-100,
#     "growth": 0-100,
#     "demand_signals": 0-100,
#     "entry_barriers": 0-100,
#     "policy_budget_tailwind": 0-100,
#     "competition_intensity": 0-100,
#     "gtm_feasibility": 0-100
#   }},
#   "total": 0-100,
#   "band": "Invest|Watch|Hold",
#   "rationale": "핵심 요약",
#   "key_evidence": [{{"claim": "string", "source": "filename"}}],
#   "risks": ["string"],
#   "assumptions": ["string"],
#   "data_freshness": "as of YYYY-MM-DD"
# }}
# ※ 반드시 유효한 JSON만 반환하십시오.
# """

#     # 4) LLM 호출 (온도 0: 결정적/일관된 출력)
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     result = llm.invoke(
#         [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ]
#     )

#     # 5) JSON 파싱(+안전한 폴백)
#     data = _extract_json(getattr(result, "content", ""))
#     if not isinstance(data, dict) or "scores" not in data:
#         # 파싱 실패 시 기본값(중립 55점)으로 안전하게 구성
#         data = {
#             "target": target,
#             "geo": geo,
#             "scores": {k: 55 for k in WEIGHTS},
#             "rationale": "기본값 적용(LLM 파싱 실패 또는 근거 부족)",
#             "key_evidence": [],
#             "risks": [],
#             "assumptions": [],
#         }

#     # 6) 점수 누락 보정 → 총점/밴드/날짜 계산
#     for k in WEIGHTS:
#         data["scores"][k] = float(data["scores"].get(k, 55))

#     total = _total_score(data["scores"])
#     data["total"] = total
#     data["band"] = _band(total)  # ← 밴드 기준 설명은 위 _band() 주석 참고
#     data["data_freshness"] = f"as of {datetime.now().date()}"

#     # 7) 최종 결과 JSON 문자열을 state에 저장
#     state["market_analysis"] = json.dumps(data, ensure_ascii=False, indent=2)
#     return state