"""Train a YOLOv8 acne detector on a dataset made by prepare_dataset.py (CLAHE face cutouts).
usage: python3 train.py --data <out>/data.yaml [--weights yolov8s.pt] [--imgsz 1280] [--batch 4] [--epochs 300] [--patience 100] [--name run1]
results land in training/runs/<name>/weights/{best,last}.pt
resume an interrupted run:  python3 train.py --resume runs/<name>/weights/last.pt
"""
import argparse
import os

import torch
from ultralytics import YOLO


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ap = argparse.ArgumentParser()
ap.add_argument("--data")
ap.add_argument("--weights", default="yolov8s.pt", help="start point: yolov8s.pt (COCO) or an earlier acne checkpoint")
ap.add_argument("--imgsz", type=int, default=1280, help="face cutouts are ~2000px; lesions are tiny, so go as high as memory allows (1280-1920)")
ap.add_argument("--batch", type=int, default=4, help="16GB Apple/laptop GPU: 4 @1280; a 24GB+ CUDA card can take 8-16 (or -1 = auto on CUDA)")
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--patience", type=int, default=100)
ap.add_argument("--name", default="acne_cutout")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--resume", help="path to last.pt of an interrupted run")
a = ap.parse_args()

if a.resume:
    YOLO(a.resume).train(resume=True)
else:
    YOLO(a.weights).train(data=a.data, imgsz=a.imgsz, batch=a.batch, epochs=a.epochs, patience=a.patience, device=pick_device(),
                          workers=a.workers, project=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"), name=a.name, plots=False)
