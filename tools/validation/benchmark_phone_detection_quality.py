"""Reproducible phone detector/association benchmark for local, consented assets.

This tool never records a camera. It evaluates an existing image and optional
COCO-format ground-truth phone box at several inference sizes, then reports the
detector output, localization IoU, association result, latency, and VRAM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detection.object_detector import YOLOObjectDetector
from src.detection.types import BBox
from src.orchestration.phone_associator import PhoneAssociator
from src.tracking.tracker import Track


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def iou(a: BBox, b: BBox) -> float:
    return a.iou(b)


def parse_gt(raw: str | None) -> BBox | None:
    if not raw:
        return None
    x, y, w, h = (float(part) for part in raw.split(","))
    return BBox(x, y, x + w, y + h)


def evaluate(frame: np.ndarray, sizes: list[int], gt: BBox | None, repeats: int) -> list[dict]:
    rows = []
    height, width = frame.shape[:2]
    for image_size in sizes:
        detector = YOLOObjectDetector(
            model_path="models/trained/yolo26m.pt",
            image_size=image_size,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            target_classes=["person", "cell phone"],
            confidence=0.35,
            person_confidence=0.35,
            phone_candidate_confidence=0.20,
            phone_strong_confidence=0.35,
        )
        for _ in range(3):
            detector.detect(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        timings = []
        detections = []
        for _ in range(repeats):
            start = time.perf_counter()
            detections = detector.detect(frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

        phones = [d for d in detections if d.class_name.lower().strip() in ("cell phone", "phone")]
        people = [d for d in detections if d.class_name.lower().strip() == "person"]
        tracks = [
            Track(index + 1, person.bbox, person.confidence, 1.0)
            for index, person in enumerate(people)
        ]
        associator = PhoneAssociator(phone_strong_confidence=0.35)
        associations = associator.associate(tracks, phones, timestamp_sec=1.0, frame_shape=(height, width))
        best_phone = max(phones, key=lambda detection: detection.confidence, default=None)

        rows.append({
            "image_size": image_size,
            "frame_size": [width, height],
            "latency_ms": {
                "mean": round(statistics.mean(timings), 3),
                "p50": round(statistics.median(timings), 3),
                "p95": round(percentile(timings, 0.95), 3),
            },
            "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2) if torch.cuda.is_available() else 0.0,
            "phone_count": len(phones),
            "best_phone_confidence": round(best_phone.confidence, 5) if best_phone else None,
            "best_phone_bbox": list(best_phone.bbox.as_tuple()) if best_phone else None,
            "best_phone_iou": round(iou(best_phone.bbox, gt), 5) if best_phone and gt else None,
            "legacy_absolute_filter_would_accept": bool(
                best_phone
                and best_phone.bbox.width <= 350.0
                and best_phone.bbox.height <= 420.0
                and best_phone.bbox.area <= 90000.0
            ),
            "association_filter": associator.last_filter_diagnostics,
            "associations": {
                str(track_id): {
                    "status": association.status,
                    "detected": association.detected,
                    "candidate_only": association.is_candidate_only,
                    "confidence": association.confidence,
                }
                for track_id, association in associations.items()
            },
            "unassociated_phones": associator.get_unassociated_phones(),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--gt", help="Ground-truth phone bbox as x,y,width,height")
    parser.add_argument("--sizes", default="640,960,1280")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--synthetic-patch-widths", help="Comma-separated source-image widths embedded on a neutral 4K canvas")
    parser.add_argument("--canvas", default="3840x2160")
    args = parser.parse_args()

    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Cannot read image: {image_path}")
    sizes = [int(value) for value in args.sizes.split(",")]
    ground_truth = parse_gt(args.gt)
    payload = {
        "image": str(image_path),
        "ground_truth": list(ground_truth.as_tuple()) if ground_truth else None,
        "scope": "single-image pipeline regression; not a dataset-level accuracy claim",
        "results": evaluate(frame, sizes, ground_truth, max(1, args.repeats)),
    }
    if args.synthetic_patch_widths:
        if ground_truth is None:
            raise SystemExit("--synthetic-patch-widths requires --gt")
        canvas_width, canvas_height = (int(value) for value in args.canvas.lower().split("x"))
        sweeps = []
        for patch_width in (int(value) for value in args.synthetic_patch_widths.split(",")):
            patch_height = round(frame.shape[0] * patch_width / frame.shape[1])
            patch = cv2.resize(frame, (patch_width, patch_height))
            canvas = np.full((canvas_height, canvas_width, 3), 114, dtype=np.uint8)
            offset_x = (canvas_width - patch_width) // 2
            offset_y = (canvas_height - patch_height) // 2
            canvas[offset_y:offset_y + patch_height, offset_x:offset_x + patch_width] = patch
            scale = patch_width / frame.shape[1]
            transformed_gt = BBox(
                offset_x + ground_truth.x1 * scale,
                offset_y + ground_truth.y1 * scale,
                offset_x + ground_truth.x2 * scale,
                offset_y + ground_truth.y2 * scale,
            )
            sweeps.append({
                "patch_width": patch_width,
                "transformed_phone_size": [round(transformed_gt.width, 2), round(transformed_gt.height, 2)],
                "results": evaluate(canvas, sizes, transformed_gt, max(1, args.repeats)),
            })
        payload["synthetic_scale_sweep"] = {
            "scope": "deterministic scale sensitivity check on a neutral canvas; not real-scene accuracy",
            "canvas": [canvas_width, canvas_height],
            "cases": sweeps,
        }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
