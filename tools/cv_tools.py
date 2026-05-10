"""
CV 도구 래퍼 — 실제 모델 우선, 설치 안 된 경우 Mock 폴백.

설치 방법 (선택):
  pip install ultralytics                    # YOLO
  pip install segment-anything               # SAM  (체크포인트 별도 필요)
  pip install transformers torch             # DepthAnything
"""
import base64
import io
import random
from pathlib import Path


# ── 공통 유틸 ─────────────────────────────────────────────────

def _b64_to_pil(image_b64: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _b64_to_numpy(image_b64: str):
    import numpy as np
    return np.array(_b64_to_pil(image_b64))


# ══════════════════════════════════════════════════════════════
# YOLO — ultralytics (yolov8n.pt 자동 다운로드 ~6MB)
# ══════════════════════════════════════════════════════════════

class YOLOWrapper:
    def __init__(self):
        self._model = None
        self.mode = "mock"
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
            self.mode = "real"
            print("[CV] YOLO: ultralytics yolov8n 로드 완료")
        except Exception as e:
            print(f"[CV] YOLO: 실제 모델 없음 → Mock 모드 ({e})")

    def detect(self, image_b64: str) -> dict:
        if self._model and image_b64:
            try:
                return self._real_detect(image_b64)
            except Exception as e:
                print(f"[CV/YOLO] 추론 실패 → Mock 폴백: {e}")
        return self._mock_detect()

    def _real_detect(self, image_b64: str) -> dict:
        img = _b64_to_numpy(image_b64)
        results = self._model(img, verbose=False)[0]
        objects = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [round(float(x), 1) for x in box.xyxy[0].tolist()]
            label = results.names[cls_id]
            objects.append({"class": label, "confidence": round(conf, 3), "bbox": xyxy})
        return {
            "objects": objects,
            "count": len(objects),
            "classes": list({o["class"] for o in objects}),
            "model": "yolov8n",
        }

    def _mock_detect(self) -> dict:
        objects = [
            {"class": "car",           "confidence": 0.92, "bbox": [100, 200, 200, 280]},
            {"class": "car",           "confidence": 0.87, "bbox": [400, 210, 520, 270]},
            {"class": "traffic light", "confidence": 0.78, "bbox": [280, 160, 340, 240]},
            {"class": "road",          "confidence": 0.95, "bbox": [0,   240, 640, 480]},
        ]
        return {
            "objects": objects,
            "count": len(objects),
            "classes": list({o["class"] for o in objects}),
            "model": "yolov8n-mock",
        }


# ══════════════════════════════════════════════════════════════
# SAM — segment-anything (체크포인트 자동 다운로드 ~375MB)
# ══════════════════════════════════════════════════════════════

SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CKPT_PATH = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"


class SAMWrapper:
    def __init__(self):
        self._generator = None
        self.mode = "mock"
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

            if not SAM_CKPT_PATH.exists():
                import urllib.request
                print(f"[CV] SAM: 체크포인트 다운로드 중 (~375MB) → {SAM_CKPT_PATH}")
                SAM_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(SAM_CKPT_URL, SAM_CKPT_PATH)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam_model = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT_PATH))
            sam_model.to(device)
            self._generator = SamAutomaticMaskGenerator(sam_model, points_per_side=16)
            self.mode = "real"
            print(f"[CV] SAM: sam-vit-b 로드 완료 (device={device})")
        except Exception as e:
            print(f"[CV] SAM: 실제 모델 없음 → Mock 모드 ({e})")

    def segment(self, image_b64: str) -> dict:
        if self._generator and image_b64:
            try:
                return self._real_segment(image_b64)
            except Exception as e:
                print(f"[CV/SAM] 추론 실패 → Mock 폴백: {e}")
        return self._mock_segment()

    def _real_segment(self, image_b64: str) -> dict:
        import numpy as np
        img = _b64_to_numpy(image_b64)
        masks = self._generator.generate(img)

        h, w = img.shape[:2]
        total_area = h * w
        areas = [m["area"] for m in masks]
        coverage = min(sum(areas) / total_area, 1.0) if total_area else 0
        largest_area = max(areas) if areas else 0

        return {
            "mask_count": len(masks),
            "coverage_ratio": round(coverage, 3),
            "largest_segment": {
                "class": "unknown",
                "area_ratio": round(largest_area / total_area, 3) if total_area else 0,
            },
            "model": "sam-vit-b",
        }

    def _mock_segment(self) -> dict:
        return {
            "mask_count": 5,
            "coverage_ratio": 0.82,
            "largest_segment": {"class": "road", "area_ratio": 0.45},
            "model": "sam-vit-b-mock",
        }


# ══════════════════════════════════════════════════════════════
# DepthAnything — transformers HuggingFace (~100MB 자동 다운로드)
# ══════════════════════════════════════════════════════════════

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthWrapper:
    def __init__(self):
        self._pipe = None
        self.mode = "mock"
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=_DEPTH_MODEL_ID,
            )
            self.mode = "real"
            print(f"[CV] Depth: {_DEPTH_MODEL_ID} 로드 완료")
        except Exception as e:
            print(f"[CV] Depth: 실제 모델 없음 → Mock 모드 ({e})")

    def estimate(self, image_b64: str, yolo_objects: list | None = None) -> dict:
        if self._pipe and image_b64:
            try:
                return self._real_estimate(image_b64, yolo_objects or [])
            except Exception as e:
                print(f"[CV/Depth] 추론 실패 → Mock 폴백: {e}")
        return self._mock_estimate()

    def _real_estimate(self, image_b64: str, yolo_objects: list) -> dict:
        import numpy as np
        pil_img = _b64_to_pil(image_b64)
        result = self._pipe(pil_img)
        depth_map = np.array(result["depth"], dtype=float)

        mean_d = float(depth_map.mean())
        min_d  = float(depth_map.min())
        max_d  = float(depth_map.max())
        q25    = float(np.percentile(depth_map, 25))
        q75    = float(np.percentile(depth_map, 75))

        # YOLO 탐지 객체가 있으면 bbox 중심 깊이로 near/far 분류
        near_objs, far_objs = [], []
        if yolo_objects:
            h, w = depth_map.shape[:2]
            for obj in yolo_objects:
                x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
                cx = min(max((x1 + x2) // 2, 0), w - 1)
                cy = min(max((y1 + y2) // 2, 0), h - 1)
                obj_depth = float(depth_map[cy, cx])
                if obj_depth <= q25:
                    near_objs.append(obj["class"])
                elif obj_depth >= q75:
                    far_objs.append(obj["class"])

        if not near_objs:
            near_objs = ["foreground"]
        if not far_objs:
            far_objs = ["background"]

        return {
            "mean_depth": round(mean_d, 2),
            "depth_range": {"min": round(min_d, 2), "max": round(max_d, 2)},
            "near_objects": list(dict.fromkeys(near_objs)),
            "far_objects":  list(dict.fromkeys(far_objs)),
            "model": "depth-anything-v2-small",
        }

    def _mock_estimate(self) -> dict:
        return {
            "mean_depth": round(random.uniform(8, 15), 1),
            "depth_range": {"min": 2.1, "max": 45.0},
            "near_objects": ["car", "traffic light"],
            "far_objects":  ["road"],
            "model": "depth-anything-v2-mock",
        }
