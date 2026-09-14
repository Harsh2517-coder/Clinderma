"""Face cropping pipeline: detect a face (front, left, or right profile) in an
image and return just the cropped face region. Uses a YOLOv8 model trained on
WIDERFace (akanametov/yolo-face), which -- unlike a plain frontal Haar
cascade -- was trained on profile faces too, so it. handles all three angles
the way a skin-analysis capture flow needs.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8n-face.pt")
MARGIN = 0.25  # expand the tight face box so chin/forehead/ears aren't clipped

_model = YOLO(WEIGHTS)


def detect_faces(img_bgr, conf=0.3):
    """Return all detected face boxes as (x0, y0, x1, y1, confidence), largest first."""
    h, w = img_bgr.shape[:2]
    r = _model.predict(img_bgr, conf=conf, verbose=False)[0]
    boxes = []
    for box, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        x0, y0, x1, y1 = box
        boxes.append((x0, y0, x1, y1, c))
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def crop_face(img_bgr, conf=0.3, margin=MARGIN):
    """Detect the largest face and return the cropped region with a margin.
    Returns None if no face is found."""
    h, w = img_bgr.shape[:2]
    boxes = detect_faces(img_bgr, conf=conf)
    if not boxes:
        return None
    x0, y0, x1, y1, _ = boxes[0]
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * margin, bh * margin
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    return img_bgr[y0:y1, x0:x1]


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    crop = crop_face(img)
    if crop is None:
        print("no face detected")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else "cropped.jpg"
        cv2.imwrite(out, crop)
        print("saved", out, crop.shape)
