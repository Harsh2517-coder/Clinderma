"""Detect acne on a photo with a model trained by train.py. The photo goes through the SAME preprocessing as the training data
(CLAHE -> face crop -> face-only cutout) before the detector sees it; boxes are mapped back onto the original photo.
usage: python3 detect.py --weights runs/<name>/weights/best.pt --image face.jpg [--out result.jpg] [--conf 0.25] [--imgsz 1280]
As a library:  from detect import detect_acne;  boxes = detect_acne(img_bgr, YOLO(weights))   # [(x0,y0,x1,y1,conf,class_id)]
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from preprocess import face_cutout, to_input_coords  # noqa: E402


def detect_acne(img_bgr, model, imgsz=1280, conf=0.25):
    """Returns [] if no face is found (check with face_cutout first if you need to tell the two cases apart)."""
    cut = face_cutout(img_bgr)
    if cut is None:
        return []
    r = model.predict(cut["image"], imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = [(*b, c, int(k)) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    return to_input_coords(boxes, cut["origin"])


if __name__ == "__main__":
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="result.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    model = YOLO(a.weights)
    img = cv2.imread(a.image)
    boxes = detect_acne(img, model, a.imgsz, a.conf)
    if not boxes and face_cutout(img) is None:
        sys.exit("no face detected")
    t = max(2, min(img.shape[:2]) // 300)
    for x0, y0, x1, y1, c, k in boxes:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), t)
        cv2.putText(img, f"{model.names[k]} {c:.2f}", (int(x0), max(12, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX, t / 3, (0, 255, 0), max(1, t // 2))
    cv2.imwrite(a.out, img)
    print(f"{len(boxes)} detections -> {a.out}")
    for x0, y0, x1, y1, c, k in boxes:
        print(f"  {model.names[k]:16} conf {c:.2f}  box ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")
