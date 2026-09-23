# Multimodal Analysis Agent

LangGraph 기반 멀티모달 분석 파이프라인.  
이미지 + 질문을 입력하면 Vision AI(YOLO/SAM/DepthAnything) + RAG 검색이 협력하여 구조화된 분석 리포트를 반환합니다.

---

## Architecture

```
User Input (image + question)
          │
          ▼
  ┌───────────────────┐
  │  Orchestrator     │  ← LangGraph StateGraph
  │  (routing 결정)   │
  └──────┬────────────┘
         │
    ┌────┴─────┐
    ▼           ▼
┌────────┐  ┌──────────┐
│ Vision │  │   RAG    │
│Analyst │  │Retriever │ ← ChromaDB
│YOLO    │  └────┬─────┘
│SAM     │       │
│Depth   │       │
└────┬───┘       │
     └─────┬─────┘
           ▼
   ┌───────────────┐
   │ Report Writer │  ← LLM (Ollama / Claude)
   └───────┬───────┘
           ▼
   Structured Report
```

### Agent 역할

| Agent | 역할 | 핵심 도구 |
|-------|------|----------|
| Orchestrator | 질문/이미지 분석 → 실행 경로 결정 | LangGraph conditional edge |
| Vision Analyst | 이미지 → 객체탐지/세그멘테이션/깊이추정 | YOLO / SAM / DepthAnything |
| RAG Retriever | 질문 + Vision 결과 → 지식 검색 | ChromaDB similarity search |
| Report Writer | 전체 결과 통합 → 구조화된 리포트 생성 | LLM (Ollama llama3.2 / Claude) |

---

## Performance

로컬 환경(Ollama llama3.2) 기준 측정값:

| 시나리오 | 라우팅 | 응답시간 |
|---------|--------|---------|
| 이미지 + 질문 (Vision → Report) | `vision_only` | 4.3s |
| 텍스트 질문 (RAG → Report) | `rag_only` | 3.5s |
| 이미지 + 지식 검색 (Vision → RAG → Report) | `both` | ~6s |

---

## 라우팅 벤치마크 — 고정 체인과 비교하고, 폴백을 교체한 기록

이 프로젝트의 주장은 "고정 체인이 아니라 조건부 엣지로 경로를 정한다"였는데, 그 라우팅이
실제로 맞게 가는지는 재 본 적이 없었다. 라벨을 붙인 질의 90건(vision_only / rag_only / both
각 30건)에 **실제 파이프라인 코드**를 태워 재고, 그 결과로 폴백 로직을 교체했다.

```bash
python tools/bench_routing.py     # reports/routing_bench.{json,png}
```

경로 비용은 가정하지 않고 이 머신에서 직접 쟀다 — vision 경로(YOLO+SAM+DepthAnything 실제 추론)
**0.503s**, rag 경로(ChromaDB 유사도 검색) **0.002s**.

### 결과 (test 60건, 키워드를 손대지 않은 분할)

| 라우터 | 정확도 | 총 처리시간 | 불필요한 RAG 호출 | 무관 컨텍스트 주입 |
|---|---:|---:|---:|---:|
| 고정 체인 (항상 both) | 33.3% | 30.29s | 20건 | 4,913자 |
| 폴백 — 교체 전 (이미지 있으면 both) | 66.7% | 20.22s | 20건 | 4,913자 |
| **폴백 — 교체 후 (지식 필요할 때만 both)** | **98.3%** | 20.19s | **1건** | **243자** |

- 고정 체인 대비 **총 처리시간 33.3% 절감**. 이 절감은 텍스트 질문에서 비전 3종을 안 태워서 나온다.
- **교체의 실체는 지연이 아니다.** 교체 전후 시간 차는 0.03초뿐이다 — 이 환경의 RAG 검색이 2ms라
  아낄 게 없기 때문이다. 교체로 실제 좋아진 것은 **라우팅 정확도(66.7%→98.3%)와, 리포트 프롬프트에
  섞여 들어가던 무관 컨텍스트(4,913자→243자)**다.

### 벤치마크가 찾아낸 결함과 교체

교체 전 폴백은 `이미지가 있으면 무조건 both`였다. 그래서 "이 이미지에 뭐가 보여?"처럼 이미지만
보면 되는 질문에도 지식 검색을 한 번 더 태웠고, **vision_only 클래스 재현율이 0.0**이었다(20건 전부 놓침).
질문에 지식 힌트(원리·차이·개념·모델명 등)가 있을 때만 `both`로 보내는 `heuristic_rule_routing`으로
교체했다. 교체 전 함수(`legacy_rule_routing`)는 지우지 않고 남겨 두어 벤치마크가 전후를 다시 잰다.

### 남은 한계

- 질의 라벨은 **손으로 붙인 합성 라벨**이다. 실제 사용자 로그가 아니다.
- 질의와 휴리스틱 키워드를 같은 사람이 썼다. 그래서 키워드는 dev 30건만 보고 정하고 test 60건으로
  보고한다(dev 96.7% / test 98.3%).
- 남은 오답 2건은 전부 같은 원인이다 — **"설명해줘", "배경"처럼 지식 질문과 시각 질문에서 같이 쓰이는
  낱말**("화면 오른쪽을 설명해줘", "배경에 뭐가 있는지 알려줘")이 both로 새어 나간다. 키워드 방식의
  한계이고, test로 튜닝하지 않기 위해 고치지 않고 남겼다.
- 이 벤치는 **LLM 라우터를 호출하지 않는다.** 측정 대상은 LLM이 없거나 실패했을 때 파이프라인이
  의존하는 폴백 경로다(유료 호출 방지 목적도 있다).

---

## Tech Stack

- **LangGraph** — 멀티에이전트 StateGraph 오케스트레이션
- **Anthropic Claude / Ollama** — LLM 백본 (자동 폴백)
- **ChromaDB** — 벡터 스토어 (RAG)
- **YOLO / SAM / DepthAnything** — 컴퓨터 비전 툴
- **FastAPI + Uvicorn** — REST API 서빙
- **Docker** — 컨테이너화

---

## Quick Start

### 1. 환경 설정

```bash
git clone https://github.com/MJHolics/multimodal-analysis-agent.git
cd multimodal-analysis-agent

pip install -r requirements.txt
```

`.env` 파일 생성:

```bash
ANTHROPIC_API_KEY=your_key_here   # Claude API (선택)
# Ollama를 사용하면 API 키 없이도 동작
```

### 2. 로컬 실행

```bash
python run_server.py
```

→ http://localhost:8000/docs (Swagger UI)

### 3. Docker 실행

```bash
docker-compose up --build
```

### 4. LLM 설정 (선택)

**Ollama (무료, 로컬):**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

**Claude API:**
`.env`에 `ANTHROPIC_API_KEY` 설정 시 자동 사용

---

## API

### `POST /api/v1/analyze`

이미지 + 질문을 받아 멀티에이전트 파이프라인을 실행합니다.

**Request:**
```json
{
  "question": "이 이미지에서 탐지된 객체와 깊이 정보를 분석해줘",
  "image_b64": "<base64 encoded image>"
}
```

**Response:**
```json
{
  "report": "## 분석 요약\n- 이미지에서 차량 2대, 신호등 1개 탐지...",
  "routing": "both",
  "task_plan": "Vision 분석 후 RAG 검색으로 배경 지식 보완",
  "rag_sources": ["yolo_overview.txt", "depth_estimation.txt"],
  "step_log": [
    "[Orchestrator] → both",
    "[Vision/YOLO] 4개 객체: car, traffic light, road",
    "[Vision/SAM] 5개 마스크, 커버리지 82%",
    "[Vision/Depth] 평균 깊이: 14.7m",
    "[RAG] 3개 청크 검색",
    "[Report] 917자 생성"
  ],
  "elapsed_seconds": 4.27
}
```

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "llm_backend": "ollama/llama3.2",
  "vectorstore_docs": 5
}
```

---

## Project Structure

```
multimodal-analysis-agent/
├── api/
│   ├── main.py              # FastAPI 앱
│   ├── models.py            # Pydantic 스키마
│   ├── pipeline.py          # LangGraph 파이프라인 (StateGraph 정의)
│   └── routers/
│       └── analyze.py       # /analyze, /analyze/stream 엔드포인트
├── tools/
│   └── cv_tools.py          # YOLO / SAM / DepthAnything LangGraph Tool 래퍼
├── data/
│   ├── chroma_db/           # ChromaDB 벡터스토어 (영속)
│   └── knowledge_base/      # RAG 지식 문서 (YOLO, SAM, Depth, LangGraph, RAG)
├── notebooks/
│   ├── 01_langgraph_basics.ipynb       # StateGraph 기초, Human-in-the-loop
│   ├── 02_vision_tools.ipynb           # CV Tool 래핑 및 파이프라인
│   ├── 03_rag_pipeline.ipynb           # ChromaDB 구축 및 RAG Agent
│   └── 04_full_agent_integration.ipynb # 전체 파이프라인 통합 테스트
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run_server.py
```

---

## Roadmap

- [x] Notebook 01 — LangGraph StateGraph 기초 및 Human-in-the-loop
- [x] Notebook 02 — Vision Tools (YOLO/SAM/Depth) LangGraph Tool 래핑
- [x] Notebook 03 — RAG Pipeline (ChromaDB 벡터스토어 구축)
- [x] Notebook 04 — Full Agent Integration (전체 파이프라인 통합 테스트)
- [x] FastAPI REST API 서빙 (`/analyze`)
- [x] Streaming 응답 (`/analyze/stream` — SSE)
- [x] 실제 CV 모델 연동 (YOLO/SAM/Depth, Mock 폴백 포함)
- [x] Docker 컨테이너화
- [ ] LangSmith 트레이싱 연동
- [ ] 멀티모달 입력 확장 (영상 스트림)

---

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models (2022)](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [Segment Anything Model (Meta AI)](https://github.com/facebookresearch/segment-anything)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
