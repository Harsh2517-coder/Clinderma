"""Candidate-generation pipeline (everything before the acne/pigmentation models):
  input -> 1. CLAHE -> 2. face detect/crop/parse (unchanged) -> 3. skin mask (eyes/brows/lips/nostrils/background out,
  surrounding skin kept) -> 4. multi-scale candidate detection -> 5. recall-first filtering -> 6. padded boxes.
CLI: python3 preprocess.py <image> [out_dir]   saves the five debug stages."""
import os
import sys

import cv2
import numpy as np

import config as cfg
from face_segment import analyze_face
from spot_detect import find_candidates, rescue_mask, skin_only_mask


def preprocess_image(img_bgr):
    """Returns None if no face is found, else a dict:
      boxes          -- [(x0, y0, x1, y1)] in crop coordinates (same contract as the old find_spots output)
      candidates     -- richer per-box dicts (box, core, score, method, scale); nothing is dropped on `score`
      boxes_full     -- the boxes mapped back to the input image's coordinates
      crop           -- CLAHE-enhanced face crop (what detection ran on; feed this to the downstream model)
      crop_original  -- the untouched face crop (use for display)
      crop_box       -- (x0, y0, x1, y1) of the crop inside the input image
      clahe_image    -- full CLAHE-enhanced input
      skin_fallback  -- True if too little skin was parsed and the whole crop was searched instead
      class_mask / face_mask / skin_mask / candidate_mask -- parsing result, face outline, searched skin, union of anomalies"""
    a = analyze_face(img_bgr, conf=cfg.FACE_CONF, margin=cfg.FACE_CROP_MARGIN)
    if a is None:
        return None
    skin_mask = skin_only_mask(a["class_mask"], a["crop"], a["face_mask"])
    skin_fallback = (skin_mask > 0).mean() < cfg.MIN_SKIN_FRAC
    if skin_fallback:  # recall-first: an unparseable close-up must not silently yield zero candidates -- search everything
        b = max(1, int(cfg.BORDER_FRAC * min(skin_mask.shape)))
        skin_mask = np.zeros_like(skin_mask)
        skin_mask[b:-b, b:-b] = 255
    cands, cand_mask = find_candidates(a["crop"], skin_mask)
    ox, oy = a["crop_box"][:2]
    boxes = [c["box"] for c in cands]
    return {**a, "skin_mask": skin_mask, "candidate_mask": cand_mask, "candidates": cands, "boxes": boxes, "skin_fallback": bool(skin_fallback),
            "boxes_full": [(x0 + ox, y0 + oy, x1 + ox, y1 + oy) for x0, y0, x1, y1 in boxes]}


def face_cutout(img_bgr):
    """The image a detector should see, for BOTH training and inference: CLAHE-enhanced face crop with everything that isn't face
    skin (background, hair, neck, eyes, brows, lips) blacked out, cropped tight to the face outline.
    Returns None if no face is found, else {image, origin, mask}: origin = (x, y) of the cutout's top-left corner in the INPUT
    image, so boxes map back with `to_input_coords` (and labels map in with `to_cutout_coords`)."""
    a = analyze_face(img_bgr, conf=cfg.FACE_CONF, margin=cfg.FACE_CROP_MARGIN)
    if a is None:
        return None
    mask = (a["face_mask"] > 0) | rescue_mask(a["class_mask"], a["face_mask"])
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = a["crop"].copy()
    crop[~mask] = 0
    ox, oy = a["crop_box"][0] + int(x0), a["crop_box"][1] + int(y0)
    return {"image": crop[y0:y1, x0:x1], "origin": (ox, oy), "mask": mask[y0:y1, x0:x1]}


def to_input_coords(boxes, origin):
    """[(x0, y0, x1, y1, ...)] in cutout pixels -> input-image pixels (extra tuple fields, e.g. conf/class, are kept)."""
    ox, oy = origin
    return [(b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, *b[4:]) for b in boxes]


def to_cutout_coords(boxes, origin):
    ox, oy = origin
    return [(b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy, *b[4:]) for b in boxes]


def _overlay(img, mask, color, alpha=0.5):
    out = img.copy()
    out[mask > 0] = (img[mask > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def render_debug(result, original_bgr, thickness=None):
    """The five inspection stages (all BGR): original, clahe, face_skin_mask, candidate_mask, final_boxes. All but the first two
    are the face crop; boxes are drawn on the *original* pixels, never the CLAHE'd ones."""
    co = result["crop_original"]
    x0, y0, x1, y1 = result["crop_box"]
    t = thickness or max(1, min(co.shape[:2]) // 400)
    face = _overlay(co, result["face_mask"], (255, 160, 0))                 # blue-ish: face outline the parser found
    face = _overlay(face, result["skin_mask"], (0, 200, 0), 0.45)           # green: skin actually searched
    boxed = co.copy()
    for bx0, by0, bx1, by1 in result["boxes"]:
        cv2.rectangle(boxed, (bx0, by0), (bx1, by1), (0, 255, 0), t)
    return {"1_original": original_bgr, "2_clahe": result["clahe_image"], "3_face_skin_mask": face,
            "4_candidate_mask": _overlay(co, result["candidate_mask"], (0, 0, 255), 0.6), "5_final_boxes": boxed}


def save_debug(result, original_bgr, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for name, im in render_debug(result, original_bgr).items():
        cv2.imwrite(f"{out_dir}/{stem}_{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    r = preprocess_image(img)
    if r is None:
        sys.exit("no face detected")
    stem = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    save_debug(r, img, sys.argv[2] if len(sys.argv) > 2 else "debug_out", stem)
    print(f"{len(r['boxes'])} candidate boxes; debug images saved")
