# Multimodal Analysis Agent — 프로젝트 컨텍스트

## 목표
LangGraph 기반 멀티모달 분석 에이전트.
이미지/영상 입력 → 다중 에이전트 협력 → 구조화된 분석 리포트 출력.

## 사용자 배경
- LLM/RAG/PEFT 경험 (강점)
- CV 경험: YOLO/SAM/DepthAnything, 자율주행 파이프라인 구축
- FastAPI + Docker 경험
- 목표: 에이전트 프레임워크 실전 경험 추가

## 기술 스택
- **LangGraph** — 에이전트 상태머신 (orcherstration)
- **Anthropic Claude API** — LLM 백본
- **ChromaDB** — 벡터 스토어 (RAG)
- **CV 툴** — YOLO/SAM/DepthAnything (기존 모델 재활용)
- **FastAPI + Docker** — 서빙

## 4단계 로드맵

### Notebook 01 — LangGraph 기초
- StateGraph, Node, Edge 직접 구현
- 단일 에이전트 → 멀티 에이전트로 확장
- Human-in-the-loop 패턴
- 핵심 논문/개념: ReAct (2022), LangGraph 공식 docs

### Notebook 02 — Vision Tools
- YOLO / SAM / DepthAnything을 LangGraph Tool로 래핑
- Tool call → 결과 파싱 → 다음 에이전트로 전달
- 이미지/영상 입력 처리 파이프라인

### Notebook 03 — RAG Pipeline
- ChromaDB 벡터 스토어 구축
- 문서 ingestion → chunking → embedding → retrieval
- RAG Agent 구현 (질문 → 검색 → 컨텍스트 주입)

### Notebook 04 — Full Agent Integration
- Orchestrator → Vision Analyst → RAG Retriever → Report Writer
- 전체 파이프라인 통합 테스트
- Streaming 응답

## 에이전트 구조

```
사용자 입력 (이미지 + 질문)
        ↓
  Orchestrator Agent  ← LangGraph StateGraph
   ↙        ↘
Vision      RAG
Analyst     Retriever  ← ChromaDB
Agent       Agent
   ↘        ↙
  Report Writer Agent
        ↓
  구조화된 분석 리포트
```

## 환경
- Python: Anaconda (autonomous_cv 커널 또는 새 커널)
- API 키: .env 파일 관리 (절대 코드에 직접 입력 금지)
- 주요 패키지: langgraph, langchain, anthropic, chromadb, fastapi

## 노트북 작성 규칙
- 한글 폰트 설정 (첫 셀):
  ```python
  import matplotlib
  matplotlib.rcParams['font.family'] = 'Malgun Gothic'
  matplotlib.rcParams['axes.unicode_minus'] = False
  ```
- 커널: `autonomous_cv` 또는 신규 `agent_env` 커널
- 각 셀마다 목적 주석 포함

## 문제 해결 로그
- `TROUBLESHOOTING.md` 유지 — 에러 발생 시 즉시 기록

## 진행 상황
- [x] 프로젝트 구조 생성
- [ ] Notebook 01 — LangGraph 기초
- [ ] Notebook 02 — Vision Tools
- [ ] Notebook 03 — RAG Pipeline
- [ ] Notebook 04 — Full Agent Integration
- [ ] FastAPI 서빙
- [ ] Docker 배포
