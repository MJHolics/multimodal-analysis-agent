"""
라우팅 결정 계층 벤치마크
=========================

이 프로젝트의 주장은 "고정 체인이 아니라 조건부 엣지로 경로를 정한다"였는데,
그 라우팅이 실제로 맞게 가는지, 고정 체인 대비 무엇을 아끼는지는 재 본 적이 없었다.
그래서 라벨을 붙인 질의 세트에 **실제 파이프라인 코드**를 태워 세 가지 라우터를 비교한다.

  A. always_both      — 고정 체인. 모든 요청에 Vision + RAG 를 전부 실행 (이 구조 이전의 방식)
  B. legacy_rule      — 교체 전 폴백. 이미지가 있으면 무조건 both
  C. heuristic_rule   — 교체 후 폴백. 이미지가 있어도 지식이 필요 없으면 vision_only

B와 C는 api/pipeline.py 에 실제로 들어 있는 함수를 import 해서 쓴다(별도 재구현 아님).
LLM 라우터는 이 벤치에서 호출하지 않는다 — 유료 API 호출이 나가고, 여기서 재려는 것은
LLM 이 없거나 실패했을 때 파이프라인이 의존하는 폴백 경로이기 때문이다.

경로 비용(지연)은 가정하지 않고 이 머신에서 직접 잰다:
  vision 경로 = YOLO + SAM + DepthAnything 실제 추론
  rag 경로    = ChromaDB 유사도 검색

한계
----
- 질의 라벨은 손으로 붙인 합성 라벨이다. 실제 사용자 로그가 아니다.
- 질의와 휴리스틱 키워드를 같은 사람이 썼다. 그래서 키워드는 dev 30건만 보고 정하고,
  성능은 손대지 않은 test 60건으로 보고한다. 두 수치를 모두 남긴다.

사용:  python tools/bench_routing.py
출력:  reports/routing_bench.json, reports/routing_bench.png
"""
from __future__ import annotations

import base64
import json
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

LABELS = ("vision_only", "rag_only", "both")


# ══════════════════════════════════════════════════════════════
# 1. 라벨 질의 세트 (손으로 작성 · dev 30 / test 60)
# ══════════════════════════════════════════════════════════════

VISION_ONLY = [
    "이 이미지에 뭐가 보여?",
    "사진 속 객체를 세어줘",
    "여기 사람 몇 명이야?",
    "화면에서 가장 큰 물체는?",
    "이 장면을 묘사해줘",
    "앞쪽에 있는 물체가 뭐야?",
    "배경에 뭐가 있는지 알려줘",
    "이 사진은 실내야 실외야?",
    "가까이 있는 대상만 알려줘",
    "이미지에서 차량을 찾아줘",
    "사람과 물건을 구분해서 알려줘",
    "이 그림에서 눈에 띄는 게 뭐야?",
    "중앙에 있는 대상이 뭔지 알려줘",
    "이 이미지에 위험해 보이는 게 있어?",
    "물체들이 얼마나 떨어져 있어?",
    "왼쪽에 있는 게 뭐야?",
    "이 이미지 요약해줘",
    "화면에 글자가 보여?",
    "사진 속 인원수를 알려줘",
    "이 장면에서 가장 앞에 있는 게 뭐야?",
    "이미지에 동물이 있어?",
    "여기서 탈것을 찾아줘",
    "이 사진의 주요 피사체는?",
    "화면 오른쪽을 설명해줘",
    "이 이미지에 몇 종류의 물체가 있어?",
    "사진에서 가장 먼 대상은?",
    "보이는 것만 정리해줘",
    "이 사진 안전해 보여?",
    "화면 아래쪽에 뭐가 있어?",
    "찍힌 대상들을 나열해줘",
]

BOTH = [
    "이 이미지를 감지하고 YOLO가 어떻게 동작하는지도 설명해줘",
    "사진 속 객체를 찾고 SAM 세그멘테이션 원리도 알려줘",
    "이 장면을 분석하고 깊이 추정 알고리즘의 개념도 설명해줘",
    "감지 결과를 보여주고 YOLO와 SAM의 차이도 알려줘",
    "이 이미지를 분석하고 관련 이론 배경도 덧붙여줘",
    "객체를 세고 그 방식의 장단점도 설명해줘",
    "사진을 해석하고 RAG가 뭔지도 알려줘",
    "이 이미지 분석 결과와 LangGraph 구조 설명을 같이 줘",
    "탐지 결과와 DepthAnything 논문 배경을 함께 알려줘",
    "이 장면 분석하고 왜 그 모델을 쓰는지도 설명해줘",
    "객체 검출하고 임베딩 개념도 정리해줘",
    "사진 보고 판단한 뒤 알고리즘 동작 방식도 알려줘",
    "감지된 것들을 설명하고 벡터 검색 원리도 알려줘",
    "이 사진 분석하고 트랜스포머 구조도 설명해줘",
    "결과를 주고 언제 쓰는 방법인지도 알려줘",
    "이미지를 보고 세그멘테이션 정의도 함께 설명해줘",
    "사진 판독하고 배경 지식도 덧붙여줘",
    "객체 인식 결과와 그 원리를 같이 설명해줘",
    "이 이미지 결과와 두 방법의 비교도 알려줘",
    "탐지하고 depth 개념도 정리해줘",
    "장면 분석과 함께 rag 파이프라인 설명도 줘",
    "이 사진 해석하고 관련 논문도 짚어줘",
    "객체를 찾고 그 모델 구조도 알려줘",
    "결과 정리하고 이론적 정의도 붙여줘",
    "이미지 분석하고 sam 장단점도 설명해줘",
    "탐지 결과와 yolo 알고리즘 배경을 같이 줘",
    "이 장면 보고 왜 그렇게 판단했는지 원리로 설명해줘",
    "감지하고 langgraph 개념도 알려줘",
    "사진 분석 후 벡터 임베딩 설명도 해줘",
    "이 이미지 결과와 동작 방식 설명을 함께 줘",
]

RAG_ONLY = [
    "YOLO가 뭐야?",
    "SAM과 YOLO의 차이는?",
    "RAG 파이프라인을 설명해줘",
    "LangGraph는 어떤 프레임워크야?",
    "깊이 추정이 뭐야?",
    "벡터 데이터베이스가 뭐지?",
    "임베딩 개념을 알려줘",
    "세그멘테이션과 검출의 차이는?",
    "ChromaDB의 특징을 알려줘",
    "StateGraph가 뭐야?",
    "객체 검출 모델은 어떻게 학습해?",
    "Segment Anything은 왜 강력해?",
    "DepthAnything의 원리는?",
    "리트리버는 무슨 역할이야?",
    "청킹은 왜 필요해?",
    "조건부 엣지가 뭐야?",
    "멀티에이전트 구조의 장점은?",
    "유사도 검색은 어떻게 동작해?",
    "YOLO의 버전별 차이가 뭐야?",
    "RAG와 파인튜닝 중 뭘 써야 해?",
    "트랜스포머 구조를 설명해줘",
    "에이전트와 체인의 차이는?",
    "벡터 인덱스는 왜 쓰는 거야?",
    "프롬프트 엔지니어링이 뭐야?",
    "모델 경량화 방법에는 뭐가 있어?",
    "시맨틱 검색이란?",
    "툴 콜링은 어떻게 동작해?",
    "컨텍스트 윈도우가 뭐야?",
    "온디바이스 추론의 장점은?",
    "파인튜닝과 프롬프트의 차이는?",
]


def build_dataset() -> list[dict]:
    """각 클래스 30건 중 앞 10건을 dev, 나머지 20건을 test 로 쓴다."""
    rows = []
    for label, qs, has_image in (
        ("vision_only", VISION_ONLY, True),
        ("both", BOTH, True),
        ("rag_only", RAG_ONLY, False),
    ):
        for i, q in enumerate(qs):
            rows.append({
                "question": q,
                "has_image": has_image,
                "label": label,
                "split": "dev" if i < 10 else "test",
            })
    return rows


# ══════════════════════════════════════════════════════════════
# 2. 라우터 3종
# ══════════════════════════════════════════════════════════════

def make_routers(pipeline):
    def always_both(question: str, has_image: bool) -> str:
        return "both"

    def legacy_rule(question: str, has_image: bool) -> str:
        return pipeline.legacy_rule_routing(question, has_image)

    def heuristic_rule(question: str, has_image: bool) -> str:
        return pipeline.heuristic_rule_routing(question, has_image)

    return [
        ("always_both", "고정 체인 (Vision+RAG 항상 실행)", always_both),
        ("legacy_rule", "교체 전 폴백 (이미지 있으면 both)", legacy_rule),
        ("heuristic_rule", "교체 후 폴백 (지식 필요할 때만 both)", heuristic_rule),
    ]


# ══════════════════════════════════════════════════════════════
# 3. 경로 비용 실측
# ══════════════════════════════════════════════════════════════

def measure_path_costs(pipeline, repeat: int = 3) -> dict:
    """vision 경로와 rag 경로의 실제 지연을 이 머신에서 잰다."""
    img_dir = BASE / "data" / "test_images"
    images = sorted([p for p in img_dir.glob("*.jpg")])
    if not images:
        raise SystemExit(f"테스트 이미지가 없다: {img_dir}")

    b64s = [base64.b64encode(p.read_bytes()).decode() for p in images]

    # 워밍업 (첫 호출의 CUDA 초기화 비용을 측정에서 뺀다)
    pipeline.vision_analyst_node({"question": "warmup", "image_b64": b64s[0], "step_log": []})
    pipeline.rag_retriever_node({"question": "warmup", "step_log": []})

    vision_t, rag_t = [], []
    for _ in range(repeat):
        for b in b64s:
            t = time.perf_counter()
            pipeline.vision_analyst_node({"question": "이 이미지 분석", "image_b64": b, "step_log": []})
            vision_t.append(time.perf_counter() - t)
    for _ in range(repeat * len(b64s)):
        t = time.perf_counter()
        pipeline.rag_retriever_node({"question": "YOLO가 뭐야?", "step_log": []})
        rag_t.append(time.perf_counter() - t)

    vision = statistics.median(vision_t)
    rag = statistics.median(rag_t)
    return {
        "vision_s": round(vision, 4),
        "rag_s": round(rag, 4),
        "vision_samples": len(vision_t),
        "rag_samples": len(rag_t),
        "images": [p.name for p in images],
        "path_cost_s": {"vision_only": round(vision, 4),
                        "rag_only": round(rag, 4),
                        "both": round(vision + rag, 4)},
    }


# ══════════════════════════════════════════════════════════════
# 4. 평가
# ══════════════════════════════════════════════════════════════

def evaluate(router, rows: list[dict], path_cost: dict, ctx_chars: dict | None = None) -> dict:
    correct = 0
    per_class = {c: {"n": 0, "hit": 0} for c in LABELS}
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    total_cost = 0.0
    vision_calls = wasted_vision = 0
    rag_calls = wasted_rag = 0
    injected_chars = wasted_chars = 0
    misrouted = []

    for r in rows:
        pred = router(r["question"], r["has_image"])
        gold = r["label"]
        per_class[gold]["n"] += 1
        confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
            per_class[gold]["hit"] += 1
        else:
            misrouted.append({"question": r["question"], "gold": gold, "pred": pred})

        total_cost += path_cost[pred]
        if pred in ("vision_only", "both"):
            vision_calls += 1
            if gold == "rag_only":
                wasted_vision += 1          # 이미지가 없는데 비전 경로로 보냄
        if pred in ("rag_only", "both"):
            rag_calls += 1
            c = (ctx_chars or {}).get(r["question"], 0)
            injected_chars += c
            if gold == "vision_only":
                wasted_rag += 1             # 지식이 필요 없는데 검색을 태움
                wasted_chars += c           # 그 결과 리포트 프롬프트에 섞여 들어가는 무관 컨텍스트

    n = len(rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "per_class_recall": {c: round(v["hit"] / v["n"], 4) if v["n"] else None
                             for c, v in per_class.items()},
        "confusion": confusion,
        "total_cost_s": round(total_cost, 2),
        "avg_cost_s": round(total_cost / n, 4),
        "vision_calls": vision_calls,
        "rag_calls": rag_calls,
        "wasted_vision_calls": wasted_vision,
        "wasted_rag_calls": wasted_rag,
        "injected_context_chars": injected_chars,
        "wasted_context_chars": wasted_chars,
        "misrouted": misrouted,
    }


# ══════════════════════════════════════════════════════════════
# 5. 그래프
# ══════════════════════════════════════════════════════════════

def plot(results: dict, out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["font.family"] = "Malgun Gothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
    except Exception as e:      # 그래프는 부가물이라 실패해도 벤치는 유효하다
        print(f"[plot] 건너뜀: {e}")
        return

    names = [r["name"] for r in results["routers"]]
    test = [r["test"] for r in results["routers"]]
    acc = [t["accuracy"] * 100 for t in test]
    cost = [t["total_cost_s"] for t in test]
    wasted = [t["wasted_context_chars"] for t in test]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = ["#9e9e9e", "#c66b3d", "#3d7cc6"]

    for ax, vals, title, unit in (
        (axes[0], acc, "라우팅 정확도 (test 60건)", "%"),
        (axes[1], cost, "총 처리시간 (실측 경로비용 합)", "초"),
        (axes[2], wasted, "무관 컨텍스트 주입량 (불필요 RAG)", "자"),
    ):
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) else 1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}" if unit == "자" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10)
        ax.tick_params(axis="x", labelrotation=12, labelsize=9)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("멀티모달 에이전트 라우팅 — 고정 체인 vs 규칙 폴백 (교체 전/후)", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"[plot] 저장: {out_png}")


# ══════════════════════════════════════════════════════════════
# 6. main
# ══════════════════════════════════════════════════════════════

def main():
    print("파이프라인 로딩 (CV 모델 3종 + ChromaDB)...")
    import api.pipeline as pipeline

    # LLM 라우터는 이 벤치에서 쓰지 않는다 — 유료 호출 방지 + 폴백 경로가 측정 대상
    pipeline.LLM = None
    print(f"  LLM 백엔드: {pipeline.LLM_BACKEND} → 벤치에서는 비활성화")
    print(f"  CV 모드: YOLO={pipeline.yolo.mode} SAM={pipeline.sam.mode} Depth={pipeline.depth.mode}")

    print("경로 비용 측정 중...")
    costs = measure_path_costs(pipeline)
    print(f"  vision {costs['vision_s']:.3f}s · rag {costs['rag_s']:.3f}s · both {costs['path_cost_s']['both']:.3f}s")

    rows = build_dataset()
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]

    # 질의별로 RAG 가 실제로 끌어오는 컨텍스트 길이를 미리 재 둔다.
    # 불필요한 검색의 대가는 지연(2ms)보다 리포트 프롬프트에 섞여 들어가는 무관 컨텍스트다.
    print("질의별 RAG 컨텍스트 길이 측정 중...")
    ctx_chars = {}
    for r in rows:
        out = pipeline.rag_retriever_node({"question": r["question"], "step_log": []})
        ctx_chars[r["question"]] = len(out.get("rag_context", ""))
    print(f"  질의당 평균 {sum(ctx_chars.values()) / len(ctx_chars):.0f}자")

    results = {
        "meta": {
            "dataset": {"total": len(rows), "dev": len(dev), "test": len(test),
                        "labels": {c: sum(1 for r in rows if r["label"] == c) for c in LABELS}},
            "cv_mode": {"yolo": pipeline.yolo.mode, "sam": pipeline.sam.mode,
                        "depth": pipeline.depth.mode},
            "llm_router": "벤치에서 비활성화 (폴백 경로를 측정 대상으로 삼음)",
            "path_costs": costs,
            "note": "라벨은 손으로 붙인 합성 라벨. 키워드는 dev 30건만 보고 정하고 test 60건으로 보고한다.",
        },
        "routers": [],
    }

    for key, desc, fn in make_routers(pipeline):
        r = {"name": key, "desc": desc,
             "dev": evaluate(fn, dev, costs["path_cost_s"], ctx_chars),
             "test": evaluate(fn, test, costs["path_cost_s"], ctx_chars),
             "all": evaluate(fn, rows, costs["path_cost_s"], ctx_chars)}
        results["routers"].append(r)
        t = r["test"]
        print(f"\n[{key}] {desc}")
        print(f"  test 정확도 {t['accuracy']:.1%} | 총 {t['total_cost_s']}s "
              f"| 낭비 호출 vision {t['wasted_vision_calls']} / rag {t['wasted_rag_calls']}")
        print(f"  클래스별 재현율 {t['per_class_recall']}")

    base_line = next(r for r in results["routers"] if r["name"] == "always_both")["test"]
    after = next(r for r in results["routers"] if r["name"] == "heuristic_rule")["test"]
    before = next(r for r in results["routers"] if r["name"] == "legacy_rule")["test"]
    results["summary"] = {
        "정확도_고정체인": base_line["accuracy"],
        "정확도_교체전": before["accuracy"],
        "정확도_교체후": after["accuracy"],
        "총시간_고정체인_s": base_line["total_cost_s"],
        "총시간_교체후_s": after["total_cost_s"],
        "시간절감_대비_고정체인": round(1 - after["total_cost_s"] / base_line["total_cost_s"], 4),
        "시간절감_대비_교체전": round(1 - after["total_cost_s"] / before["total_cost_s"], 4),
        "불필요_rag호출_교체전": before["wasted_rag_calls"],
        "불필요_rag호출_교체후": after["wasted_rag_calls"],
        "무관컨텍스트_교체전_자": before["wasted_context_chars"],
        "무관컨텍스트_교체후_자": after["wasted_context_chars"],
    }

    out_json = BASE / "reports" / "routing_bench.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] 저장: {out_json}")
    plot(results, BASE / "reports" / "routing_bench.png")

    s = results["summary"]
    print("\n── 요약 (test 60건) ──")
    print(f"정확도: 고정 체인 {s['정확도_고정체인']:.1%} · 교체 전 {s['정확도_교체전']:.1%} "
          f"· 교체 후 {s['정확도_교체후']:.1%}")
    print(f"총 처리시간: {s['총시간_고정체인_s']}s → {s['총시간_교체후_s']}s "
          f"({s['시간절감_대비_고정체인']:.1%} 절감)")
    print(f"불필요한 RAG 호출: {s['불필요_rag호출_교체전']}건 → {s['불필요_rag호출_교체후']}건")
    print(f"리포트 프롬프트에 섞이는 무관 컨텍스트: {s['무관컨텍스트_교체전_자']:,}자 → {s['무관컨텍스트_교체후_자']:,}자")


if __name__ == "__main__":
    main()
