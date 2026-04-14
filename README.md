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
git clone https://github.com/<your-id>/ai_agent_project.git
cd ai_agent_project

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
ai_agent_project/
├── api/
│   ├── main.py              # FastAPI 앱
│   ├── models.py            # Pydantic 스키마
│   ├── pipeline.py          # LangGraph 파이프라인
│   └── routers/
│       └── analyze.py       # /analyze 엔드포인트
├── data/
│   ├── chroma_db/           # 벡터스토어 (영속)
│   └── knowledge_base/      # RAG 지식 문서
├── notebooks/
│   ├── 01_langgraph_basics.ipynb
│   ├── 02_vision_tools.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_full_agent_integration.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run_server.py
```

---

## Roadmap

- [x] Notebook 01 — LangGraph StateGraph 기초
- [x] Notebook 02 — Vision Tools (YOLO/SAM/Depth) LangGraph 래핑
- [x] Notebook 03 — RAG Pipeline (ChromaDB)
- [x] Notebook 04 — Full Agent Integration
- [x] FastAPI REST API 서빙
- [x] Docker 컨테이너화
- [ ] 실제 CV 모델 연동 (현재 Mock)
- [ ] Streaming 응답 (`/analyze/stream`)
- [ ] LangSmith 트레이싱 연동

---

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models (2022)](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [Segment Anything Model (Meta AI)](https://github.com/facebookresearch/segment-anything)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
