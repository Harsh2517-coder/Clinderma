# Acne Module Placeholder

This repository currently contains the Clinderma image preprocessing and facial-region layer only.

The acne detection module is intentionally not implemented here. The acne team should consume the preprocessing output contract:

- Original or normalized image
- Five Clinderma region masks
- View classification
- Face geometry metadata

The acne detector can assign detected lesions to `forehead`, `left_cheek`, `right_cheek`, `nose`, or `chin` by intersecting each lesion with the region masks.

Do not add ACNE04 annotations, YOLO training scripts, or acne model weights to this preprocessing handoff unless the repository scope changes.
