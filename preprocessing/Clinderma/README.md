# Clinderma Preprocessing

This repository currently provides the **image preprocessing / facial region layer** for Clinderma. Acne detection, pigmentation detection, GAGS scoring, and disease-specific downstream models are separate modules that should consume this preprocessing output.

The current implementation preserves the tested v2 pipeline from the image-processing and geometry workstream. It processes one image at a time and generates five Clinderma facial zones.

## Current Preprocessing Scope

Input image -> face detection / face parsing -> MediaPipe facial geometry + orientation -> semantic facial skin parsing -> anatomical subdivision -> five Clinderma zones.

The implementation does not train a model, does not manually annotate images, does not include ACNE04 annotations, and does not batch-process the ACNE04 dataset.

## Five Clinderma Facial Regions

1. `forehead`
2. `left_cheek`
3. `right_cheek`
4. `nose`
5. `chin`

The v2 implementation preserves orientation-aware behavior: `FRONT`, `LEFT_PROFILE`, and `RIGHT_PROFILE`.

For side-profile images, only the anatomically visible cheek is active and the opposite cheek is empty. For frontal images, left/right cheek subdivision uses the facial midline.

## Pipeline Architecture

The recommended entry point is:

```text
preprocessing/generate_face_regions_v2.py
```

The v2 pipeline combines MediaPipe Face Mesh for facial landmarks and orientation, a pretrained Hugging Face semantic face-parsing model for facial classes, and deterministic anatomical subdivision logic to derive the five Clinderma masks.

The historical v1 script is retained at `preprocessing/generate_face_regions.py` only for reference. It is not the recommended preprocessing pipeline.

## Technologies Used

- Python 3.10 recommended
- OpenCV
- NumPy
- Pillow
- MediaPipe Face Mesh
- PyTorch
- Hugging Face Transformers
- Pretrained model: [`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing)

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pretrained face-parsing model is not committed to this repository. It downloads and caches automatically on first execution through Hugging Face Transformers.

## Run One Image

```bash
python preprocessing/generate_face_regions_v2.py examples/input/levle1_426.jpg
```

For another image:

```bash
python preprocessing/generate_face_regions_v2.py path/to/image.jpg
```

## Expected Output Files

For an input named `example.jpg`, the script writes outputs under `preprocessing/outputs/`:

- `example_v2_face_regions.jpg`: colored region overlay.
- `example_v2_debug_boundaries.jpg`: landmark and boundary debug visualization.
- `example_v2_view_report.json`: detected view, orientation metrics, mask pixel counts, and output paths.
- `example_v2_region_masks/example_forehead_mask.png`
- `example_v2_region_masks/example_left_cheek_mask.png`
- `example_v2_region_masks/example_right_cheek_mask.png`
- `example_v2_region_masks/example_nose_mask.png`
- `example_v2_region_masks/example_chin_mask.png`

## Sample Outputs

Representative ACNE04 input examples are copied into `examples/input/`. Representative output visualizations and view reports are copied into `examples/outputs/`. A small demonstration set of region masks is included under `examples/outputs/masks/`.

These examples are for reproducibility and handoff review only. The full ACNE04 dataset is not included.

## Downstream Contract For Acne Detection

Downstream acne detection should receive:

- Original or normalized image
- Five binary Clinderma region masks
- View classification
- Face geometry metadata

The acne detector can run lesion detection independently, then assign each detected lesion to a Clinderma region by intersecting lesion coordinates or lesion masks with the five preprocessing masks.

This repository does not implement acne detection.

## Downstream Contract For Pigmentation

Pigmentation models should operate on the standardized face and region outputs from this preprocessing layer. The pigmentation team should not need to build a separate face detection, orientation, or facial-region preprocessing pipeline unless this shared layer fails a documented validation case.

This repository does not implement pigmentation detection.

## Current Limitations

- The model is not trained on ACNE04.
- Region quality depends on MediaPipe landmark detection and pretrained face-parsing performance.
- The output is intended for anatomical validation and downstream integration, not final clinical validation.
- Hairline, occlusion, extreme pose, lighting, and accessories can affect semantic parsing and forehead/chin boundaries.
- View classification thresholds may need calibration on a larger validation split.

## What Is Not Included Yet

- Acne detection
- Pigmentation detection
- GAGS scoring
- Model training
- Manual annotations
- Full ACNE04 dataset
- ACNE04 annotation files
- YOLO training, evaluation, or prediction scripts

## Future Work

- Validate region masks across a broader pose and skin-tone set.
- Add automated quality-control flags for low-confidence landmark or parsing output.
- Define a stable JSON schema for downstream teams.
- Add tests around orientation classification and output-file contracts.
- Package the preprocessing module as an installable Python package.
- Add optional batch processing only after single-image validation is complete.

## Citation And Model Information

The semantic face-parsing model reference is [`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing). The Hugging Face model card describes it as a SegFormer-based semantic segmentation model fine-tuned on CelebAMask-HQ for face parsing. It should not be described as trained on ACNE04.

This repository is a handoff artifact for Clinderma preprocessing and should be credited separately from downstream acne and pigmentation modules.
