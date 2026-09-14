#!/usr/bin/env bash
# Fetches the pretrained weights V_3 needs. Not committed to the repo directly
# (kept out of git on purpose -- redistributing third-party binaries through
# git bloats the repo forever; a fetch script is the standard alternative).
set -e
cd "$(dirname "$0")"

# YOLOv8-face (akanametov/yolo-face, GPL-3.0) -- face detection, front/left/right robust
[ -f yolov8n-face.pt ] || curl -sL -o yolov8n-face.pt \
  "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"

# BiSeNet face parsing (yakhyo/face-parsing, MIT) -- pixel-level face segmentation
[ -f face_parsing_resnet18.onnx ] || curl -sL -o face_parsing_resnet18.onnx \
  "https://github.com/yakhyo/face-parsing/releases/download/v0.0.2/resnet18.onnx"

echo "Models ready in $(pwd)"
