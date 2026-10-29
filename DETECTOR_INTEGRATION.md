# Plugging in the Roboflow-trained acne detector

## Status
- **29 Oct 2026: a locally retrained RF-DETR small (4 classes: comedone, nodule, papule, pustule) exists.** See `training/README.md`; the ONNX (123 MB, too big for GitHub) is sent directly by Harsh; check the sha256 in `training/notes.txt`. Settings: `DETECTOR=onnx`, `DETECTOR_CLASSES=comedone,nodule,papule,pustule`, `DETECTOR_INPUT=full`, `DETECTOR_CONF=0.30`, `DETECTOR_SIZE=512`.

## Older notes (what we knew before retraining)
- Model: **RF-DETR small** (Roboflow id `tanayaaa/acne-detection-v1-tjjbs-2-rfdetr-small-t1`), trained on **plain photos**, so use `DETECTOR_INPUT=full`.
- Classes (from the friend's notes, to be confirmed in index order): comedone, papule, pustule, nodule, cyst.
- **The model file is not available yet.** On GitHub, `tanaya1806/clinderma_acne` has only the `main` branch (no model file, release, fork or LFS), and her backend still calls the hosted Roboflow API.
  Her new branch has not been pushed, or is in a repo we cannot see. The file must come from her directly (branch push, or a file transfer).
- Still needed: the model file, the exact class order, the input size (512 expected for RF-DETR small), and which photos were in Roboflow's **test split** (so accuracy is not measured on training photos).
- We do not use the Roboflow API key: it would upload patient photos to a third party.


The platform runs the detector **locally**, so patient photos never leave the machine. That is why the hosted Roboflow API is not used: it would upload every patient's face to a third party.
The detector tells the platform where the acne-type spots are; the platform then counts them per face zone.

## What your friend should send (best first)

| # | Send | How it is used | Notes |
|---|------|----------------|-------|
| 1 | **ONNX file** (`inference_model.onnx`) + **class list** | `DETECTOR=onnx` | Easiest and fastest to plug in. From an RF-DETR checkpoint: `pip install "rfdetr[onnx]"`, then `RFDETRSmall(pretrain_weights="checkpoint.pth").export()` (the docs name the file `inference_model.onnx`; the extra is `rfdetr[onnx]` or `rfdetr[onnxexport]` depending on version). Outputs are `dets` and `labels`. |
| 2 | **`.pth` checkpoint** + model variant (nano/small/base) + class list | `DETECTOR=rfdetr`, `DETECTOR_VARIANT=small` | Needs `pip install rfdetr` on this machine. |
| 3 | Ultralytics **`.pt`** (YOLO) + class list | `DETECTOR=yolo` | Works immediately (ultralytics is installed). |
| 4 | **Dataset export only** (YOLO or COCO format from Roboflow) | I train a detector locally | Takes a couple of hours on this Mac; no model file needed. |

Roboflow's hosted models are often not downloadable as weights on every plan. If "download weights" is not offered for the model `acne-detection-v1-...-rfdetr-small`, the friend can export the **dataset** (option 4)
or retrain with the open-source `rfdetr` package on that dataset, which gives a `.pth` and an ONNX file they own.

## Also send (these decide whether the numbers are right)
1. **Class names in the model's order**, for example `comedone, papule, pustule, nodule, cyst` (`DETECTOR_CLASSES`).
2. **What the model was trained on**: face cutouts (CLAHE, background/hair/eyes blacked out: `DETECTOR_INPUT=cutout`, the default), the plain face crop (`crop`) or full photos (`full`). Getting this wrong hurts accuracy a lot.
3. **Training/inference resolution** (for example 512) if the ONNX input shape is dynamic (`DETECTOR_SIZE`).
4. **The test split**: the photos the model never saw, exported with labels in YOLO format. Accuracy must be measured on these, not on training photos.
5. **Do not send API keys.** They are not needed and should not be shared.

## Plug it in
Put the file in `models/` (for example `models/acne_detector.onnx`) and add to `.env`:
```
DETECTOR=onnx
DETECTOR_PATH=models/acne_detector.onnx
DETECTOR_CLASSES=comedone,papule,pustule,nodule,cyst
DETECTOR_CONF=0.15
DETECTOR_INPUT=cutout
```
Then measure it on the held-out photos (this writes `models/detector_metrics.json`, which the app shows next to every scan):
```
python3 eval_detector.py path/to/test/images path/to/test/labels
```
Until that file exists the app hides the accuracy box instead of showing another model's numbers. Restart the app after changing `.env`.

## What I could not verify yet
The ONNX and RF-DETR parsing is unit-tested against hand-made outputs and the documented output names (`dets`, `labels`), but not yet against your friend's real model.
The first real run should be checked once: confirm the class order, the class index offset (`DETECTOR_CLASS_OFFSET`, 0 or 1) and the input mode by looking at a few annotated previews.
