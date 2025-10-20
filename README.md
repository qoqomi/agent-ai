# 위성 산업 스타트업 탐색 에이전트

LangGraph 기반 Multi-Agent 시스템의 스타트업 탐색 노드

## 주요 기능

- 웹 검색: 국내 위성 스타트업 키워드 기반 검색
- RAG 검색: 한글 PDF 문서 기반 심층 분석 (4개 문서, 98페이지)
- 기본값 출력: 테스트 모드 (국내 대표 스타트업 선정)
- JSON 출력: 팀원 공유용 표준화된 데이터 형식 (startup_name, startup_info)

## 빠른 시작
```bash
# 1. 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. API 키 설정
echo "HUGGINGFACE_API_TOKEN=your-token" > .env

# 4. 실행
python3 start_agent.py
```

## 출력 예시
```json
{
  "startup_name": "컨텍",
  "startup_info": {
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
}
```

## 문서

- [통합 가이드](./INTEGRATION.md)
- [requirements.txt](./requirements.txt)

## 기술 스택

- LangChain 1.0.0
- LangGraph 1.0.0
- HuggingFace Embeddings (jhgan/ko-sroberta-multitask)
- FAISS (벡터 스토어)
- PyPDF (한글 문서 처리)
- 기본값 모드 (테스트용, LLM 미사용)

## 팀

**스타트업 탐색 담당:** [조현준]

## 라이센스

MIT License