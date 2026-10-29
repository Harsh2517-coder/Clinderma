import os
import requests
import io
import time
import base64
from typing import Dict, Any, Optional
from PIL import Image

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "T8q7J07ioYGmmvTOzzp5")
ROBOFLOW_MODEL_ID = "tanayaaa/acne-detection-v1-tjjbs-2-rfdetr-small-t1"
# Default to 0.15 for high sensitivity detection of lesions
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("ACNE_CONFIDENCE_THRESHOLD", 0.15))

def compute_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    ix1 = max(box1["x1"], box2["x1"])
    iy1 = max(box1["y1"], box2["y1"])
    ix2 = min(box1["x2"], box2["x2"])
    iy2 = min(box2["y2"], box2["y2"])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area1 = max(1.0, (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"]))
    area2 = max(1.0, (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"]))
    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union

def apply_nms(
    detections: list[Dict[str, Any]],
    cross_class_iou: float = 0.30,
    same_class_iou: float = 0.55,
) -> list[Dict[str, Any]]:
    """Applies cross-class Non-Maximum Suppression (IoU 0.30 across different class labels)
    per Clinderma standard, while preserving adjacent clustered lesions of the same class.
    """
    if not detections:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept: list[Dict[str, Any]] = []

    for det in sorted_dets:
        box = det["bbox"]
        cls = det.get("class", "")
        suppress = False
        for k in kept:
            iou = compute_iou(box, k["bbox"])
            if cls != k.get("class", ""):
                # Duplicate box across different class labels: suppress if IoU >= 0.30
                if iou >= cross_class_iou:
                    suppress = True
                    break
            else:
                # Same-class lesion: only suppress if nearly identical duplicate box
                if iou >= same_class_iou:
                    suppress = True
                    break
        if not suppress:
            kept.append(det)

    return kept


class AcneDetector:
    def __init__(self):
        self.api_key = ROBOFLOW_API_KEY
        self.model_id = ROBOFLOW_MODEL_ID
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    def is_model_available(self) -> bool:
        """Check if the API key is set."""
        return bool(self.api_key)
        
    def load_model(self):
        """Not required for hosted API, prints config."""
        if not self.is_model_available():
            print("WARNING: ROBOFLOW_API_KEY is not set in the environment.")
            
        print("====================================")
        print("ACNE MODEL (ROBOFLOW API)")
        print("====================================")
        print(f"Model ID: {self.model_id}")
        print(f"API Key Set: {self.is_model_available()}")
        print(f"Default Confidence: {self.confidence_threshold}")
        print("====================================")

    def predict(
        self,
        image_bytes: bytes,
        confidence: Optional[float] = None,
        iou_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """Runs inference via Roboflow Hosted API, applies NMS deduplication, and returns detections."""
        if not self.is_model_available():
            raise ValueError("ROBOFLOW_API_KEY is missing. Cannot call inference API.")
            
        start_time = time.time()
        conf_to_use = confidence if confidence is not None else self.confidence_threshold
        conf_pct = int(round(conf_to_use * 100))
        conf_pct = max(5, min(95, conf_pct))
            
        # 1. Decode image just to get original dimensions for response
        image = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = image.size
        
        # 2. Call Roboflow API
        url = f"https://detect.roboflow.com/{self.model_id}?api_key={self.api_key}&confidence={conf_pct}"
        
        # Base64 encode the image for Roboflow API
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        res = requests.post(
            url,
            data=image_b64,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            },
            timeout=30
        )
        
        if not res.ok:
            raise Exception(f"Roboflow API error: {res.status_code} - {res.text}")
            
        result_json = res.json()
        raw_predictions = result_json.get("predictions", [])
        
        # 3. Post-processing with strict image coordinate clipping
        raw_detections = []
        
        for p in raw_predictions:
            cls = str(p["class"])
            x_center = float(p["x"])
            y_center = float(p["y"])
            width = float(p["width"])
            height = float(p["height"])
            
            # Center coords to bounding box corners
            x1 = x_center - (width / 2.0)
            y1 = y_center - (height / 2.0)
            x2 = x_center + (width / 2.0)
            y2 = y_center + (height / 2.0)
            
            # Clip strictly within image boundaries
            x1 = max(0.0, min(x1, float(img_w - 1)))
            y1 = max(0.0, min(y1, float(img_h - 1)))
            x2 = max(0.0, min(x2, float(img_w)))
            y2 = max(0.0, min(y2, float(img_h)))
            
            # Discard degenerate zero-size boxes
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue
            
            raw_detections.append({
                "class": cls,
                "confidence": float(p["confidence"]),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

        # 4. Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        detections = apply_nms(raw_detections, cross_class_iou=iou_threshold)


        # Count lesions after NMS
        lesions_summary = {}
        for d in detections:
            cls = d["class"]
            lesions_summary[cls] = lesions_summary.get(cls, 0) + 1
                
        inference_time_ms = int((time.time() - start_time) * 1000)
        print(f"Acne inference API call (conf={conf_pct}%, nms={iou_threshold}): {inference_time_ms} ms, raw: {len(raw_detections)}, post-nms: {len(detections)}")

        return {
            "image_width": img_w,
            "image_height": img_h,
            "detections": detections,
            "total_detections": len(detections),
            "lesions_summary": lesions_summary,
            "confidence_threshold": conf_to_use
        }
