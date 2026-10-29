# UPDATE (29 Oct 2026): the model has been retrained locally, read this first

Roboflow would not let the model weights be downloaded, so Harsh trained a new RF-DETR small on the same dataset (Roboflow `tanayaaa/acne-detection-v1-tjjbs` v2) on his Mac.
Everything is in [`training/`](training/README.md): scripts, class list, accuracy numbers, training curve.
The trained model (`rfdetr-small.onnx`, `acne_rfdetr_small.pth`) is over GitHub's 100 MB limit, so it is not in the repo: Harsh sends it directly (folder `FaceVeda_acne_detector`).

- Classes: comedone, nodule, papule, pustule (4; the dataset has no cyst). Accuracy on 71 never-seen photos: about 42% of detections correct, about 39% of lesions found (confidence 0.30).
- Not needed any more from section 2 below: the model file. Still useful from Tanaya: her **5-class dataset** (`ACNE_COMBINED_DATASET`, with cyst) to train a better model, and a GPU machine for more epochs.
- Your next steps: see "What Tanaya can do next" in `training/README.md`. The rest of this file is the older plan for getting the Roboflow model and is kept for reference.

---

# Hand-off: get the RF-DETR acne model into FaceVeda (to be done on Tanaya's computer)

**Goal.** FaceVeda (this repo) needs the acne detector that was trained on Roboflow (RF-DETR small, id `tanayaaa/acne-detection-v1-tjjbs-2-rfdetr-small-t1`) as a **local model file**, so patient photos never leave the machine.
The detector code is already here (`detectors.py`, `eval_detector.py`); only the model file and a few facts are missing. The model file was not on GitHub (`tanaya1806/clinderma_acne` has only `main`, no model), so it has to come from Tanaya's computer or her Roboflow account.

## 1. Get the project (5 minutes)
```bash
git clone -b face-wellness-platform https://github.com/Harsh2517-coder/Clinderma.git face-wellness-platform
cd face-wellness-platform
pip install fastapi uvicorn opencv-python numpy scikit-learn scipy joblib onnxruntime pillow
python3 -m pytest -q test_core.py      # optional: 31 tests; some need the Clinderma V_3 models, see README
```
You do not need to run the platform to finish this task.

## 2. What to collect (the checklist)
1. **The model file.** In order of preference: `.onnx`; a `.pth` checkpoint plus the model size (`small`); a YOLO `.pt`.
   - Look in the `clinderma_acne` project on this computer (all branches, including any unpushed local branch: `git branch -a`) and in Downloads.
   - Run `python3 tools/find_model_files.py <folder-of-your-project> ~/Downloads` to list candidates and any class lists. It only reads file names, sizes and hashes; it never runs a model.
   - If the model only exists inside Roboflow: open the project in Roboflow, the trained model version, and look for a *download weights* option (availability depends on the plan). If there is none, export the **dataset** instead (YOLO format, with the train/valid/test split).
2. **Class names in index order** (expected: 0 comedone, 1 papule, 2 pustule, 3 nodule, 4 cyst). Confirm from the Roboflow dataset's class list.
3. **Input size** the model was trained at (RF-DETR small is normally 512).
4. **What it was trained on**: plain full photos (as believed), plain face crops, or CLAHE face cutouts.
5. **The test split**: the images the model never saw, with labels, exported in **YOLO format** (`images/` + `labels/`). Roboflow: dataset version, export, *YOLOv8* format, then keep the `test` folder.
6. If you have a `.pth`, make an ONNX file too: `pip install "rfdetr[onnx]"` then
   ```python
   from rfdetr import RFDETRSmall
   RFDETRSmall(pretrain_weights="checkpoint.pth").export()   # writes inference_model.onnx
   ```
   (the extra is `rfdetr[onnx]` or `rfdetr[onnxexport]` depending on the version).

## 3. Do not
- Do **not** paste, print, commit or send the Roboflow API key (it is in `backend/app/acne_detector.py` of `clinderma_acne`). It is not needed.
- Do **not** commit patient photos, labels of patient photos, `.env`, or `data/` to GitHub. The model *weights* may be shared; the test photos should be sent privately (zip, Drive), not pushed to a public repo.

## 4. Hand back
Put these in one folder, zip it and send it to Harsh: the model file(s), `classes.txt` (one name per line, in index order), the test split (`images/`, `labels/`), and a short `notes.txt` with:
```
model file:        <name>   sha256: <first 16 chars>
architecture:      RF-DETR small
class order:       comedone, papule, pustule, nodule, cyst   (confirm)
input size:        512 x 512
trained on:        full photos / face crops / CLAHE cutouts
default confidence: 0.15
test split:        <number of images> exported from Roboflow version <n>
```
Weights alone (not photos) can also go on a new branch of her repo (for example `rfdetr-weights`), pushed to GitHub, if they are under the 100 MB file limit (RF-DETR small should be well under).

## 5. How Harsh plugs it in afterwards (for reference)
Copy the model to `models/`, then in `.env`:
```
DETECTOR=onnx                         # or rfdetr for a .pth (pip install rfdetr)
DETECTOR_PATH=models/acne_rfdetr_small.onnx
DETECTOR_CLASSES=comedone,papule,pustule,nodule,cyst
DETECTOR_INPUT=full
DETECTOR_CONF=0.15
```
Then measure it on photos it never saw: `python3 eval_detector.py <test/images> <test/labels>`. See `DETECTOR_INTEGRATION.md`.

---

## Prompt to paste into Claude on Tanaya's computer
```
I am helping integrate an acne detector into the FaceVeda platform. Repo: https://github.com/Harsh2517-coder/Clinderma, branch face-wellness-platform.
Please clone it, read HANDOFF_TANAYA.md and DETECTOR_INTEGRATION.md, then do section 2 of the hand-off for me:
find the trained RF-DETR small acne model on this computer (check every git branch of my clinderma_acne project and my Downloads folder; use tools/find_model_files.py),
confirm the class order, the input size and what the model was trained on, and help me export the test split in YOLO format from Roboflow.
If you can, export the model to ONNX. Collect everything into one folder with a notes.txt as described in section 4.
Rules: never print, copy or commit the Roboflow API key; never push patient photos to GitHub; ask me before downloading or uploading anything.
```
