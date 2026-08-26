# Preprocessing Module

This folder contains the Clinderma facial-region preprocessing scripts.

## Install From Repository Root

Python 3.10 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment first with:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Recommended Script

Use the current v2 implementation:

```bash
python preprocessing/generate_face_regions_v2.py examples/input/levle1_426.jpg
```

`generate_face_regions_v2.py` is the working implementation. It combines MediaPipe Face Mesh, pretrained semantic face parsing, view classification, and anatomical subdivision into five Clinderma zones.

The first run downloads and caches the pretrained Hugging Face model `jonathandinu/face-parsing`.

## Historical Reference

`generate_face_regions.py` is retained only as the earlier v1 reference implementation. It is not the recommended preprocessing pipeline.

## Outputs

For each input image, v2 saves:

- Five binary PNG masks: `forehead`, `left_cheek`, `right_cheek`, `nose`, and `chin`.
- A colored overlay visualization.
- A debug visualization with landmarks and region boundaries.
- A JSON view report with orientation metrics and mask pixel counts.

Generated outputs are written to:

```text
preprocessing/outputs/
```

Runtime outputs are ignored by Git.

## Independent Developer Check

From the repository root, run:

```bash
python preprocessing/generate_face_regions_v2.py examples/input/levle1_426.jpg
```

Confirm that `preprocessing/outputs/` contains:

- A region overlay image.
- A debug boundary image.
- A view report JSON file.
- A mask folder containing `forehead`, `left_cheek`, `right_cheek`, `nose`, and `chin` PNG masks.
