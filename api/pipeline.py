"""
멀티모달 분석 파이프라인
Notebook 04 핵심 로직을 FastAPI 서빙용으로 추출
"""
import os
import json
import re
import requests
from pathlib import Path
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = str(BASE_DIR / "data" / "chroma_db")
KB_DIR = BASE_DIR / "data" / "knowledge_base"
COLLECTION_NAME = "cv_knowledge_base"


# ══════════════════════════════════════════════════════════════
# 1. LLM 초기화 (Claude 우선, Ollama 폴백)
# ══════════════════════════════════════════════════════════════

def _init_llm():
    # Ollama 우선 확인
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if any("llama" in m for m in models):
                from langchain_ollama import ChatOllama
                return ChatOllama(model="llama3.2"), "ollama/llama3.2"
    except Exception:
        pass

    # Claude API 폴백
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=anthropic_key,
            max_tokens=2048,
        )
        return llm, "claude-haiku-4-5-20251001"

    return None, "mock"


LLM, LLM_BACKEND = _init_llm()


# ══════════════════════════════════════════════════════════════
# 2. CV 래퍼 (실제 모델 우선, 없으면 Mock 폴백)
# ══════════════════════════════════════════════════════════════

from tools.cv_tools import YOLOWrapper, SAMWrapper, DepthWrapper

yolo  = YOLOWrapper()
sam   = SAMWrapper()
depth = DepthWrapper()


# ══════════════════════════════════════════════════════════════
# 3. RAG 벡터스토어 초기화
# ══════════════════════════════════════════════════════════════

def _init_vectorstore():
    ollama_ok = False
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            ollama_ok = any("nomic-embed-text" in m for m in models)
    except Exception:
        pass

    if ollama_ok:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    else:
        from langchain_core.embeddings.fake import FakeEmbeddings
        embeddings = FakeEmbeddings(size=768)

    if os.path.exists(CHROMA_DIR):
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        if vs._collection.count() > 0:
            return vs
        # 디렉토리는 있지만 비어있으면 아래에서 재구축

    # 벡터스토어 없으면 샘플 문서로 재구축
    KB_DIR.mkdir(parents=True, exist_ok=True)
    sample_docs = {
        "yolo_overview.txt": "YOLO는 실시간 객체 탐지 모델입니다. YOLOv8은 anchor-free 방식으로 차량, 보행자, 신호등을 탐지합니다.",
        "sam_overview.txt": "SAM(Segment Anything Model)은 포인트·박스 프롬프트로 객체를 세그멘테이션합니다.",
        "depth_estimation.txt": "DepthAnything은 단안 RGB 이미지에서 픽셀별 깊이값을 추정하는 Foundation Model입니다.",
        "rag_concepts.txt": "RAG는 외부 지식베이스를 검색해 LLM 답변 정확도를 높이는 패턴입니다.",
        "langgraph_concepts.txt": "LangGraph는 에이전트를 StateGraph로 정의하는 오케스트레이션 프레임워크입니다.",
    }
    for fname, content in sample_docs.items():
        (KB_DIR / fname).write_text(content, encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    loader = DirectoryLoader(str(KB_DIR), glob="*.txt",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    chunks = splitter.split_documents(loader.load())
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vs


vectorstore = _init_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# ══════════════════════════════════════════════════════════════
# 4. LangGraph State
# ══════════════════════════════════════════════════════════════

class PipelineState(TypedDict):
    question: str
    image_b64: Optional[str]
    routing: str
    task_plan: str
    yolo_result: dict
    sam_result: dict
    depth_result: dict
    vision_summary: str
    rag_context: str
    rag_sources: List[str]
    report: str
    step_log: List[str]


# ══════════════════════════════════════════════════════════════
# 5. 에이전트 노드
# ══════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM = """당신은 멀티모달 분석 시스템의 오케스트레이터입니다.
사용자 요청을 분석하여 어떤 에이전트를 실행할지 결정합니다.

결정 규칙:
- 이미지가 있고 이미지 분석이 필요하면: routing = "both" 또는 "vision_only"
- 이미지가 없거나 지식 검색만 필요하면: routing = "rag_only"
- 이미지 분석 + 배경 지식이 모두 필요하면: routing = "both"

반드시 JSON 형식으로만 응답하세요:
{"routing": "both|vision_only|rag_only", "task_plan": "작업 계획 (한국어)"}"""


# ── 규칙 라우팅 (LLM이 없거나 실패했을 때 쓰는 폴백 경로) ──────────
#
# legacy_rule_routing 은 2026-08-09 이전에 쓰던 폴백이다.
# 이미지가 있으면 무조건 both 로 보내서, 이미지만 보면 되는 질문에도 RAG 검색을
# 한 번 더 태웠다. tools/bench_routing.py 로 재 보니 vision_only 질의를 전부
# 놓쳤고(해당 클래스 재현율 0), 그래서 heuristic_rule_routing 으로 교체했다.
# 교체 전후를 다시 잴 수 있도록 옛 함수도 남겨 둔다.

KNOWLEDGE_HINTS = (
    "원리", "개념", "설명", "차이", "비교", "어떻게 동작", "동작 방식", "구조",
    "알고리즘", "이론", "논문", "배경", "장단점", "언제 쓰", "왜 쓰", "정의",
    "yolo", "sam", "segment anything", "depth", "depthanything", "rag",
    "langgraph", "임베딩", "벡터", "트랜스포머",
)


def legacy_rule_routing(question: str, has_image: bool) -> str:
    """교체 전 폴백 — 이미지가 있으면 항상 both."""
    return "both" if has_image else "rag_only"


def heuristic_rule_routing(question: str, has_image: bool) -> str:
    """교체 후 폴백 — 이미지가 있어도 지식이 필요 없으면 vision_only 로 보낸다."""
    if not has_image:
        return "rag_only"
    q = question.lower()
    return "both" if any(h in q for h in KNOWLEDGE_HINTS) else "vision_only"


def orchestrator_node(state: PipelineState) -> dict:
    question = state["question"]
    has_image = bool(state.get("image_b64"))
    log = list(state.get("step_log", []))
    log.append(f'[Orchestrator] 질문: "{question}" | 이미지: {has_image}')

    if LLM:
        try:
            user_msg = f"질문: {question}\n이미지 존재: {has_image}\n\n라우팅을 JSON으로 결정해주세요."
            raw = LLM.invoke([SystemMessage(content=ORCHESTRATOR_SYSTEM),
                              HumanMessage(content=user_msg)]).content.strip()
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}
            routing = parsed.get("routing", heuristic_rule_routing(question, has_image))
            task_plan = parsed.get("task_plan", "자동 계획")
        except Exception:
            routing = heuristic_rule_routing(question, has_image)
            task_plan = "규칙 기반 라우팅 (LLM 오류)"
    else:
        routing = heuristic_rule_routing(question, has_image)
        task_plan = "규칙 기반 라우팅 (Mock 모드)"

    if not has_image and routing != "rag_only":
        routing = "rag_only"

    log.append(f"[Orchestrator] → {routing}: {task_plan}")
    return {"routing": routing, "task_plan": task_plan, "step_log": log}


def orchestrator_router(state: PipelineState) -> str:
    r = state.get("routing", "both")
    if r == "vision_only":
        return "vision"
    elif r == "rag_only":
        return "rag"
    return "vision"


def vision_analyst_node(state: PipelineState) -> dict:
    image_b64 = state.get("image_b64", "")
    log = list(state.get("step_log", []))

    if not image_b64:
        log.append("[Vision] 이미지 없음 — 건너뜀")
        return {"vision_summary": "이미지 없음", "step_log": log}

    yolo_result = yolo.detect(image_b64)
    log.append(f'[Vision/YOLO] {yolo_result["count"]}개 객체: {yolo_result["classes"]}')

    sam_result = sam.segment(image_b64)
    log.append(f'[Vision/SAM] {sam_result["mask_count"]}개 마스크, 커버리지 {sam_result["coverage_ratio"]}')

    depth_result = depth.estimate(image_b64, yolo_objects=yolo_result.get("objects", []))
    log.append(f'[Vision/Depth] 평균 깊이: {depth_result["mean_depth"]}m')

    classes_str = ", ".join(yolo_result["classes"])
    near_str = ", ".join(depth_result["near_objects"])

    vision_summary = f"""## Vision 분석 결과

### 객체 탐지 (YOLO)
- 탐지된 객체: {yolo_result['count']}개
- 클래스: {classes_str}
- 세부 목록:
{chr(10).join(f"  * {o['class']} (신뢰도 {o['confidence']:.0%})" for o in yolo_result['objects'])}

### 세그멘테이션 (SAM)
- 분할된 영역: {sam_result['mask_count']}개
- 전체 커버리지: {sam_result['coverage_ratio']:.0%}
- 가장 큰 영역: {sam_result['largest_segment']['class']} ({sam_result['largest_segment']['area_ratio']:.0%})

### 깊이 추정 (DepthAnything)
- 평균 깊이: {depth_result['mean_depth']}m
- 깊이 범위: {depth_result['depth_range']['min']}m ~ {depth_result['depth_range']['max']}m
- 근거리 객체: {near_str}
- 원거리 객체: {', '.join(depth_result['far_objects'])}
"""
    return {
        "yolo_result": yolo_result,
        "sam_result": sam_result,
        "depth_result": depth_result,
        "vision_summary": vision_summary,
        "step_log": log,
    }


def _format_rag_context(docs: list, max_chars: int = 2000) -> str:
    parts, total = [], 0
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
        text = f"[문서 {i}: {src}]\n{doc.page_content}"
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def rag_retriever_node(state: PipelineState) -> dict:
    question = state["question"]
    yolo_result = state.get("yolo_result", {})
    log = list(state.get("step_log", []))

    enhanced_query = question
    if yolo_result.get("classes"):
        enhanced_query = f'{question} {" ".join(yolo_result["classes"])}'

    docs = retriever.invoke(enhanced_query)
    context = _format_rag_context(docs)
    sources = list({doc.metadata.get("source", "").split("\\")[-1].split("/")[-1] for doc in docs})

    log.append(f'[RAG] "{enhanced_query[:50]}..." → {len(docs)}개 청크 | 출처: {sources}')
    return {"rag_context": context, "rag_sources": sources, "step_log": log}


def _mock_report(question: str, vision_summary: str, rag_context: str) -> str:
    """LLM 없을 때 사용하는 규칙 기반 Mock 리포트"""
    parts = [
        "## 분석 요약",
        f"- 질문 \"{question}\"에 대해 파이프라인이 정상 실행되었습니다.",
    ]
    if vision_summary and vision_summary != "이미지 없음":
        parts.append("- Vision 분석(YOLO/SAM/Depth) 완료: 객체 탐지 및 공간 분석 수행됨")
    if rag_context:
        parts.append("- RAG 검색 완료: 관련 지식 문서가 컨텍스트에 포함됨")
    parts += [
        "",
        "## 상세 분석",
        vision_summary if vision_summary and vision_summary != "이미지 없음" else "이미지 분석 없음 (텍스트 전용 모드)",
        "",
        "## 배경 지식 연계",
        rag_context[:800] if rag_context else "검색 결과 없음",
        "",
        "## 결론",
        "⚠️ LLM 백엔드가 연결되지 않아 Mock 리포트를 반환합니다.",
        "  Ollama(llama3.2)를 실행하거나 ANTHROPIC_API_KEY를 설정하면 실제 분석 리포트가 생성됩니다.",
    ]
    return "\n".join(parts)


REPORT_SYSTEM = """당신은 컴퓨터 비전 분석 전문 리포터입니다.
Vision 분석 결과와 RAG 컨텍스트를 통합하여 구조화된 분석 리포트를 작성합니다.

리포트 형식:
## 분석 요약
- 핵심 발견 사항 2~3가지

## 상세 분석
- 이미지에서 발견된 객체와 의미
- 공간 구성 (깊이/세그멘테이션 기반)

## 배경 지식 연계
- 검색된 지식 베이스 내용과의 연계

## 결론 및 제안
- 사용자 질문에 대한 직접 답변

한국어로 작성하세요."""


def report_writer_node(state: PipelineState) -> dict:
    question = state["question"]
    vision_summary = state.get("vision_summary", "")
    rag_context = state.get("rag_context", "")
    log = list(state.get("step_log", []))

    sections = [f"## 사용자 질문\n{question}"]
    if vision_summary and vision_summary != "이미지 없음":
        sections.append(f"## Vision 분석 결과\n{vision_summary}")
    if rag_context:
        sections.append(f"## 관련 지식 (RAG)\n{rag_context[:1500]}")
    combined = "\n\n".join(sections)

    if LLM:
        try:
            report = LLM.invoke([SystemMessage(content=REPORT_SYSTEM),
                                  HumanMessage(content=combined)]).content
        except Exception as e:
            report = _mock_report(question, state.get("vision_summary", ""), state.get("rag_context", ""))
            log.append(f"[Report] LLM 오류 → Mock 리포트 ({e})")
    else:
        report = _mock_report(question, state.get("vision_summary", ""), state.get("rag_context", ""))

    log.append(f"[Report] {len(report)}자 생성")
    return {"report": report, "step_log": log}


# ══════════════════════════════════════════════════════════════
# 6. 파이프라인 그래프 조립
# ══════════════════════════════════════════════════════════════

def _after_vision_router(state: PipelineState) -> str:
    return "report" if state.get("routing") == "vision_only" else "rag"


def build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("orchestrator",   orchestrator_node)
    g.add_node("vision_analyst", vision_analyst_node)
    g.add_node("rag_retriever",  rag_retriever_node)
    g.add_node("report_writer",  report_writer_node)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", orchestrator_router,
                             {"vision": "vision_analyst", "rag": "rag_retriever"})
    g.add_conditional_edges("vision_analyst", _after_vision_router,
                             {"rag": "rag_retriever", "report": "report_writer"})
    g.add_edge("rag_retriever", "report_writer")
    g.add_edge("report_writer", END)
    return g.compile()


pipeline = build_pipeline()
