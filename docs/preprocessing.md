# Clinderma Preprocessing Technical Notes

This document describes the current image preprocessing layer used to create five Clinderma anatomical facial-region masks from a single image.

## Output Contract

For one input image, the preprocessing layer produces:

- `forehead` mask
- `left_cheek` mask
- `right_cheek` mask
- `nose` mask
- `chin` mask
- Region overlay visualization
- Debug visualization
- View report JSON

Downstream teams should treat these outputs as the shared face-region contract. Acne and pigmentation models should consume these outputs instead of creating independent face preprocessing pipelines.

## MediaPipe Geometry

The v2 implementation uses MediaPipe Face Mesh in static-image mode with one detected face. Landmarks are converted from normalized coordinates to image pixel coordinates.

MediaPipe is used for face landmark detection, face oval geometry, nose/mouth/brow/jaw/facial-midline boundary support, orientation metrics, and debug visualization.

MediaPipe landmarks are not treated as semantic segmentation. They provide geometry used to guide and subdivide the pretrained semantic face parsing output.

## Orientation Classification

The implementation classifies each image as one of `FRONT`, `LEFT_PROFILE`, or `RIGHT_PROFILE`.

The classifier uses two geometric signals:

- `nose_shift_ratio`: the horizontal displacement of the nose tip from the visible face center, normalized by visible face width.
- `eye_width_ratio`: the ratio of the smaller projected eye width to the larger projected eye width.

Large nose shift and strong eye-width asymmetry indicate a profile view. Near-balanced eye widths indicate a frontal or near-frontal view. The JSON report stores the final view and raw metrics so downstream teams can audit threshold behavior.

## Semantic Face Parsing

The v2 implementation uses the pretrained Hugging Face model [`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing).

The Hugging Face model card describes this as a SegFormer-based semantic segmentation model fine-tuned on CelebAMask-HQ for face parsing. It provides labels such as skin, nose, eyes, eyebrows, lips, hair, neck, and background.

The model is not trained on ACNE04 and model weights are not committed to this repository. Transformers downloads and caches the model automatically on first execution.

## Anatomical Region Generation

The v2 implementation combines semantic masks and landmarks:

- Visible facial skin is built from the parser's `skin` label, clipped to the MediaPipe face oval, and cleaned.
- Exclusion masks remove parser-supported non-skin regions such as eyes, eyebrows, lips, mouth, hair, hat, and eyeglasses.
- The nose mask combines the parser's `nose` label with landmark-guided nose support to improve side-profile coverage.
- The forehead mask uses parsed visible skin above the eyebrow/glabella boundary while excluding hair and brows.
- The chin mask uses parsed visible skin below the lower-mouth/lip boundary and extends to the visible chin/jaw region.
- Cheek masks are generated from remaining visible mid-face skin after removing forehead, nose, chin, and mouth support.

For frontal images, cheek masks are separated using the anatomical facial midline through glabella, nose, philtrum, lower lip, and chin. For profile images, only the anatomically visible cheek is active and the opposite cheek is empty.

## Region Masks

Each mask is a binary PNG with foreground pixels set to `255` and background pixels set to `0`.

Downstream lesion assignment can use pixel intersection:

1. Run acne lesion detection on the original or normalized image.
2. Convert each lesion to a point, bounding box, or binary lesion mask.
3. Intersect lesion geometry with the five Clinderma masks.
4. Assign the lesion to the region with the largest overlap.

For profile images, one cheek mask may intentionally contain zero pixels because the opposite cheek is not anatomically visible.

## Reproducibility

Recommended environment:

- Python 3.10
- Dependencies from `requirements.txt`
- Network access on first run to download `jonathandinu/face-parsing`

Example:

```bash
python preprocessing/generate_face_regions_v2.py examples/input/levle1_426.jpg
```

The first run may be slower while the pretrained model downloads and caches.

## Not Implemented Here

This preprocessing export does not implement acne detection, pigmentation detection, GAGS scoring, YOLO training or inference, dataset-wide batch processing, model training, or ACNE04 annotation handling.
