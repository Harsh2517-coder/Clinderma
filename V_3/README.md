# V_3 -- Face crop, segmentation & spot detection

Pipeline: detect a face (front/left/right profile) -> cut out just the face
outline (background, hair, neck, clothing removed; eyes/eyebrows/mouth also
removed as holes) -> box every skin spot (acne, pigmentation, or anything
else that isn't smooth regular skin -- not classified yet, just localized).

## Setup

```bash
pip install ultralytics opencv-python onnxruntime gradio numpy
bash models/download_models.sh   # fetches the two pretrained models (not committed to git)
```

## Run

```bash
python3 app.py
```

Opens a Gradio app on port 7861 with webcam capture + file upload.

## Files

- `face_crop.py` -- face detection (YOLOv8-face, WIDERFace-trained)
- `face_segment.py` -- pixel-level face parsing/cutout (BiSeNet, CelebAMask-HQ)
- `spot_detect.py` -- class-agnostic skin spot detection (classical CV, no model/training)
- `app.py` -- the Gradio frontend wiring it all together
- `models/download_models.sh` -- fetches the two pretrained weight files this depends on
