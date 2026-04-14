from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.analyze import router

app = FastAPI(
    title="Multimodal Analysis Agent API",
    description="LangGraph 기반 멀티모달 분석 파이프라인 — Vision(YOLO/SAM/Depth) + RAG + LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["analyze"])


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Multimodal Analysis Agent API", "docs": "/docs"}
