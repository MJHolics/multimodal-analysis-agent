from pydantic import BaseModel, Field
from typing import Optional, List


class AnalyzeRequest(BaseModel):
    question: str = Field(..., description="분석 질문")
    image_b64: Optional[str] = Field(None, description="base64 인코딩 이미지 (없으면 RAG만 실행)")


class AnalyzeResponse(BaseModel):
    report: str = Field(..., description="최종 분석 리포트")
    routing: str = Field(..., description="실행 경로: vision_only | rag_only | both")
    task_plan: str = Field(..., description="Orchestrator의 작업 계획")
    rag_sources: List[str] = Field(default_factory=list, description="참조한 문서 목록")
    step_log: List[str] = Field(default_factory=list, description="에이전트 실행 로그")
    elapsed_seconds: float = Field(..., description="총 소요 시간(초)")


class HealthResponse(BaseModel):
    status: str
    llm_backend: str
    vectorstore_docs: int
