# Clinderma Preprocessing Architecture

## Flow

```text
Input Image
    |
    v
Face Detection / Parsing
    |
    v
MediaPipe Geometry
    |
    v
Orientation Classification
    |
    v
Semantic Face Parsing
    |
    v
Anatomical Region Generation
    |
    v
Five Clinderma Zones
    |
    v
Downstream Acne / Pigmentation Models
```

## Components

`preprocessing/generate_face_regions_v2.py` is the current recommended preprocessing implementation.

The pipeline detects facial landmarks with MediaPipe Face Mesh, classifies view orientation as `FRONT`, `LEFT_PROFILE`, or `RIGHT_PROFILE`, runs pretrained semantic face parsing with `jonathandinu/face-parsing`, and generates five binary Clinderma zone masks.

The five output zones are:

- `forehead`
- `left_cheek`
- `right_cheek`
- `nose`
- `chin`

For side-profile images, only the anatomically visible cheek is active. For frontal images, cheek subdivision uses the facial midline.

Downstream acne and pigmentation modules should consume the generated masks, view report, and original or normalized image. Acne detection, pigmentation detection, GAGS scoring, and model training are outside the scope of this repository.
