# Training the acne detector on the preprocessed (CLAHE + face-only) images

The detector must be trained on the same kind of image it will see at inference: the **face cutout** produced by V_3 --
CLAHE-enhanced, cropped to the face, with background / hair / neck / eyes / eyebrows / lips blacked out. `prepare_dataset.py`
builds that dataset from any YOLO-format folder; `detect.py` applies the identical step to a new photo before detecting.
Both call `face_cutout()` in `../preprocess.py`, so training and inference cannot drift apart.

```
photo -> exposure fix + CLAHE -> face detect/crop -> face parsing -> blank non-face pixels -> tight crop  = cutout
              train:  cutout + remapped labels -> YOLOv8        infer:  cutout -> YOLOv8 -> boxes mapped back to the photo
```

## Steps

1. **Environment** (Python 3.10-3.13)
   ```bash
   cd V_3
   pip install -r requirements.txt
   bash models/download_models.sh     # face detector + face parser (not stored in git)
   ```
2. **Get the labelled data** as a folder with `images/`, `labels/` (YOLO txt, one per image, same stem) and `data.yaml`
   (`names: [...]`). Either
   - `DATASET_COMBINED/acne` (in this repo: the V1 + V2 images, 7 classes), or
   - the unzipped `ACNE_COMBINED_DATA_CLINDERMA` (594 images from V1+V2+V3, 8 classes incl. milia) -- not in git (1 GB), ask for the zip.
3. **Build the preprocessed dataset** (~4 min for 600 images; writes ~2 GB of JPEGs)
   ```bash
   python3 training/prepare_dataset.py --src ../DATASET_COMBINED/acne --out ../acne_cutouts
   ```
   It prints how many images had no detectable face (skipped), how many boxes fell outside the face skin (dropped), and the boxes
   per class in each split. Read that report: tiny classes (e.g. milia, cyst) will have almost no validation boxes.
   The 80/10/10 split is grouped by person (phone-number prefix for v1, timestamp sessions for v2/v3) so a face never appears in
   both train and val.
4. **Train**
   ```bash
   python3 training/train.py --data ../acne_cutouts/data.yaml --weights yolov8s.pt --imgsz 1280 --batch 4 --epochs 300 --patience 100
   ```
   - Device is picked automatically (CUDA > Apple MPS > CPU).
   - `--imgsz`: lesions are tiny, so higher is better: 1280 is the safe default, 1536-1920 if memory allows.
   - `--batch`: 4 @ 1280 fits a 16 GB Mac. On an MPS out-of-memory error, lower `--batch` or `--imgsz`. A 24 GB+ CUDA card can use 8-16.
   - Expect ~5 s per batch of 4 at 1536 on an M-series Mac (hours-to-days for hundreds of epochs); a CUDA GPU is much faster.
   - Interrupted? `python3 training/train.py --resume training/runs/acne_cutout/weights/last.pt`
   - Results: `training/runs/<name>/weights/best.pt`, metrics in `results.csv`.
   - Keep the Mac awake for long runs: `caffeinate -dimsu python3 training/train.py ...`
5. **Detect on a new photo** (preprocessing is applied automatically)
   ```bash
   python3 training/detect.py --weights training/runs/acne_cutout/weights/best.pt --image face.jpg --out result.jpg --imgsz 1280
   ```
   In code: `from detect import detect_acne; boxes = detect_acne(img_bgr, YOLO(weights))` -> `[(x0, y0, x1, y1, conf, class_id)]` in the
   original photo's pixels.

## Things to know

- **Do not mix**: a model trained on raw photos will perform worse on cutouts and vice versa. Always train and infer through `face_cutout()`.
- If you change `config.py` (CLAHE strength, exposure normalisation, mask classes), regenerate the dataset and retrain.
- Class imbalance is heavy (papule >> milia/cyst/open comedone). Judge by per-class results, not just overall mAP50.
- Labels are circle annotations converted to boxes, so mAP50 has a ceiling well below 1.0; compare runs against each other.
- To check the preprocessing itself (not the detector), run `python3 run_tests.py` -- it writes five-stage montages to `test_outputs/`.
