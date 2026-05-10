import time
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.models import AnalyzeRequest, AnalyzeResponse, HealthResponse
from api.pipeline import pipeline, vectorstore, LLM_BACKEND

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="서버 상태 확인")
def health():
    try:
        doc_count = vectorstore._collection.count()
    except Exception:
        doc_count = -1
    return HealthResponse(
        status="ok",
        llm_backend=LLM_BACKEND,
        vectorstore_docs=doc_count,
    )


@router.post("/analyze", response_model=AnalyzeResponse, summary="멀티모달 분석 실행")
def analyze(req: AnalyzeRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question은 비워둘 수 없습니다.")

    initial_state = {
        "question":      req.question,
        "image_b64":     req.image_b64,
        "routing":       "",
        "task_plan":     "",
        "yolo_result":   {},
        "sam_result":    {},
        "depth_result":  {},
        "vision_summary": "",
        "rag_context":   "",
        "rag_sources":   [],
        "report":        "",
        "step_log":      [],
    }

    start = time.time()
    try:
        result = pipeline.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파이프라인 오류: {str(e)}")

    return AnalyzeResponse(
        report=result.get("report", ""),
        routing=result.get("routing", ""),
        task_plan=result.get("task_plan", ""),
        rag_sources=result.get("rag_sources", []),
        step_log=result.get("step_log", []),
        elapsed_seconds=round(time.time() - start, 2),
    )


@router.post("/analyze/stream", summary="멀티모달 분석 — 스트리밍 (SSE)")
def analyze_stream(req: AnalyzeRequest):
    """
    Server-Sent Events 스트리밍 응답.

    각 에이전트 노드가 완료될 때마다 이벤트를 전송합니다.

    이벤트 형식:
      data: {"type": "step",  "node": "orchestrator", "log": "..."}
      data: {"type": "step",  "node": "vision_analyst", "log": "..."}
      data: {"type": "step",  "node": "rag_retriever",  "log": "..."}
      data: {"type": "done",  "report": "...", "routing": "...",
                              "rag_sources": [...], "step_log": [...]}
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question은 비워둘 수 없습니다.")

    initial_state = {
        "question":       req.question,
        "image_b64":      req.image_b64,
        "routing":        "",
        "task_plan":      "",
        "yolo_result":    {},
        "sam_result":     {},
        "depth_result":   {},
        "vision_summary": "",
        "rag_context":    "",
        "rag_sources":    [],
        "report":         "",
        "step_log":       [],
    }

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        accumulated: dict = {}
        for step in pipeline.stream(initial_state):
            node_name = next(iter(step))
            node_output = step[node_name]
            accumulated.update(node_output)

            logs = node_output.get("step_log", [])
            latest_log = logs[-1] if logs else ""

            yield _sse({
                "type": "step",
                "node": node_name,
                "log":  latest_log,
            })

        yield _sse({
            "type":       "done",
            "report":     accumulated.get("report", ""),
            "routing":    accumulated.get("routing", ""),
            "task_plan":  accumulated.get("task_plan", ""),
            "rag_sources": accumulated.get("rag_sources", []),
            "step_log":   accumulated.get("step_log", []),
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
