# V_3 -- CLAHE-first face preprocessing, high-recall spot candidates, and detector training

Pipeline: (1) exposure fix + CLAHE on the whole photo -> (2) detect the face (front/left/right) and parse it pixel-by-pixel ->
(3) remove background, hair, neck, eyes, eyebrows, lips, nostrils (skin around them is kept) -> (4) find candidate skin spots
(multi-scale, four colour signals, absolute contrast floors) -> (5) one padded box per spot. Candidates are *proposals*: acne vs
pigmentation vs "nothing" is decided downstream.

## Setup

```bash
pip install -r requirements.txt
bash models/download_models.sh   # fetches the two pretrained models (not committed to git)
```

## Run

```bash
python3 app.py                          # Gradio app on port 7861: webcam/upload, shows all 5 pipeline stages
python3 preprocess.py photo.jpg out/    # CLI: saves original / CLAHE / face+skin mask / candidate mask / final boxes
python3 run_tests.py                    # 18-case qualitative matrix -> test_outputs/ (set CLINDERMA_DATA=<data root>)
```

## Training a detector on top of this pipeline -> see `training/README.md`

## Tuning

All thresholds are in `config.py` (CLAHE, exposure, mask margins, contrast floors, box size, merge/NMS, padding). The key knob is
`DEVIATION_FLOOR` / `CHANNEL_FLOOR`: raise for fewer boxes, lower for more recall (a measured table is in the comments).
`eval_candidates.py` measures candidate recall against labelled data (`python3 eval_candidates.py . out.json 1 NAME=value ...`).

## Files

- `config.py` -- every tunable value
- `enhance.py` -- stage 1: gated exposure correction + CLAHE (LAB L channel)
- `face_crop.py` -- face detection (YOLOv8-face, WIDERFace-trained)
- `face_segment.py` -- face parsing / outline (BiSeNet, CelebAMask-HQ); `analyze_face` runs CLAHE then the face parse
- `spot_detect.py` -- skin mask + multi-scale candidate generation (classical CV, no training)
- `preprocess.py` -- `preprocess_image` (candidates), `face_cutout` (what the detector sees), debug renderers
- `app.py` -- the Gradio frontend
- `training/` -- dataset preparation, YOLO training, inference using the same preprocessing
- `eval_candidates.py`, `run_tests.py`, `test_outputs/` -- evaluation tools and the last results
- `models/download_models.sh` -- fetches the two pretrained weight files this depends on
