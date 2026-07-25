"""
YOLOv8 (Ultralytics) ONNX inference wrapper for the pothole detector.

Model: best.onnx
  input  "images": float32 [1, 3, 640, 640]  (RGB, 0-1 normalized, CHW)
  output "output0": float32 [1, 5, 8400]     (cx, cy, w, h, pothole_conf) x 8400 anchors
                                              boxes are already decoded to pixel
                                              coords in the 640x640 input space;
                                              the class score already has sigmoid
                                              applied (verified empirically).

Only one class is trained: 0 -> "pothole".
"""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).parent / "model" / "best.onnx"
INPUT_SIZE = 640
CLASS_NAMES = {0: "pothole"}

CONF_THRES = 0.35
IOU_THRES = 0.45


class HazardDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize (stretch, no letterbox) to 640x640, BGR->RGB, CHW, /255."""
        resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(chw, axis=0)

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list:
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            rest = idxs[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-9)
            idxs = rest[iou <= iou_thres]
        return keep

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Run detection on a BGR frame (as read by cv2.imdecode).

        Returns every detection that clears CONF_THRES and survives NMS (not
        just the single best one) — a road photo can easily show several
        potholes at once, and the caller uses the full list to draw every
        box and to scale severity with how many were found in this one frame.

        {"detected": False} if nothing clears CONF_THRES.

        Box coordinates are returned as fractions (0-1) of the 640x640 input
        space, which map 1:1 back onto the *original* frame's axes independently
        (since we stretch-resize per-axis), so the frontend can draw the overlay
        directly by multiplying by the original frame's width/height.
        """
        t0 = time.time()
        inp = self.preprocess(frame_bgr)
        raw = self.session.run(None, {self.input_name: inp})[0]  # [1,5,8400]
        raw = raw[0].T  # -> [8400, 5]

        boxes_px = raw[:, :4]
        scores = raw[:, 4]

        mask = scores >= CONF_THRES
        if not mask.any():
            return {"detected": False, "inference_ms": round((time.time() - t0) * 1000, 1)}

        boxes_px = boxes_px[mask]
        scores = scores[mask]
        boxes_xyxy = self._xywh_to_xyxy(boxes_px)

        keep = self._nms(boxes_xyxy, scores, IOU_THRES)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]

        # Sort by confidence descending so "detections[0]" is always the
        # most confident one (used for the headline / representative box).
        order = np.argsort(scores)[::-1]

        detections = []
        for idx in order:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            x1, x2 = np.clip([x1, x2], 0, INPUT_SIZE)
            y1, y2 = np.clip([y1, y2], 0, INPUT_SIZE)
            # np.clip returns numpy.float32 scalars, not plain Python floats.
            # FastAPI's JSON encoder can't serialize numpy types, which is
            # what threw "'numpy.float32' object is not iterable" — cast
            # explicitly so everything downstream is a native Python float.
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            confidence = float(scores[idx])

            box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            defect_area_ratio = float(box_area / (INPUT_SIZE * INPUT_SIZE))

            detections.append(
                {
                    "confidence": round(confidence, 4),
                    "defect_area_ratio": round(defect_area_ratio, 4),
                    "bbox_frac": [
                        round(x1 / INPUT_SIZE, 4),
                        round(y1 / INPUT_SIZE, 4),
                        round(x2 / INPUT_SIZE, 4),
                        round(y2 / INPUT_SIZE, 4),
                    ],
                }
            )

        # Aggregate signal for severity scoring: total damaged fraction of
        # the frame (sum of each box's area, capped at 1.0) and the highest
        # single confidence — both feed compute_severity() in severity.py.
        total_area_ratio = round(min(1.0, sum(d["defect_area_ratio"] for d in detections)), 4)
        top_confidence = detections[0]["confidence"]

        return {
            "detected": True,
            "type": CLASS_NAMES[0],
            "count": len(detections),
            "detections": detections,
            "confidence": top_confidence,
            "defect_area_ratio": total_area_ratio,
            "bbox_frac": detections[0]["bbox_frac"],  # kept for backward compat
            "inference_ms": round((time.time() - t0) * 1000, 1),
        }


detector = HazardDetector()
