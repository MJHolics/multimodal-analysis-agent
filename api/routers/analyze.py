import time
from fastapi import APIRouter, HTTPException
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
