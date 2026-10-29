"""Face scan: photo -> spots per face zone, using the Clinderma V_3 pipeline (CLAHE, YOLOv8-face crop, BiSeNet face parsing, skin-spot candidates).

Zones come from the face-parsing labels, no landmarks needed:
  forehead = skin above the eyebrows | nose = nose class (slightly dilated) | chin = skin below the lower lip
  cheeks   = the remaining skin between brows and lip, split at the nose's x position (image-left side of the photo = the person's RIGHT cheek).
V_3 candidates are recall-first proposals (pores, specks, spots, acne alike); this module only counts them per zone, it does not classify them.
Photo is processed in memory, never stored.  Set CLINDERMA_V3 to the V_3 folder if it is not at the default path.
"""
import os, sys

import json

import cv2
import numpy as np

import detectors
import spot_features as SF

V3 = os.environ.get("CLINDERMA_V3", "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/Clinderma/V_3")
BROWS, EYES, NOSE, LIPS, SKIN = (2, 3), (4, 5), 10, (12, 13), 1      # CelebAMask-HQ label ids
_v3 = None
HERE = os.path.dirname(os.path.abspath(__file__)); _filter = None


def spot_filter():
    """The trained filter (models/spot_filter.joblib) or None. Without it every proposed spot is counted, which is much less accurate."""
    global _filter
    if _filter is None:
        p = os.path.join(HERE, "models", "spot_filter.joblib")
        try: import joblib; _filter = joblib.load(p) if os.path.exists(p) else False
        except Exception: _filter = False
    return _filter or None


def reliability():
    """Measured accuracy of the ACTIVE detector per zone, shown to users next to each result. The built-in filter's numbers come from person-grouped cross-validation;
    a plugged-in detector has its own models/detector_metrics.json (create it with eval_detector.py on photos the detector was not trained on). None if not measured."""
    name = "detector_metrics.json" if detectors.get() is not None else "spot_filter_metrics.json"; p = os.path.join(HERE, "models", name)
    return json.load(open(p)) if os.path.exists(p) else None


def _load():
    global _v3
    if _v3 is None:
        if V3 not in sys.path: sys.path.insert(0, V3)
        cwd = os.getcwd(); os.chdir(V3)                                  # V_3 loads its models relative to its own folder at import time
        try:
            import preprocess; _v3 = preprocess
        finally: os.chdir(cwd)
    return _v3


def zone_masks(class_mask, face_mask):
    """Five boolean masks (crop coordinates) from the parser output; None if the face cannot be split."""
    skin = (face_mask > 0) & np.isin(class_mask, (SKIN, NOSE)); h, w = class_mask.shape
    ys, xs = np.nonzero(face_mask > 0)
    if len(ys) < 500: return None
    fy0, fy1 = ys.min(), ys.max()
    brow = np.nonzero(np.isin(class_mask, BROWS))[0]; eye = np.nonzero(np.isin(class_mask, EYES))[0]; lip = np.nonzero(np.isin(class_mask, LIPS))[0]
    nose_ys, nose_xs = np.nonzero(class_mask == NOSE)
    brow_top = brow.min() if len(brow) else (eye.min() - 0.06 * (fy1 - fy0) if len(eye) else fy0 + 0.30 * (fy1 - fy0))
    lip_bot = lip.max() if len(lip) else fy0 + 0.80 * (fy1 - fy0)
    mid_x = float(np.median(nose_xs)) if len(nose_xs) > 50 else float(np.median(xs))
    yy, xx = np.mgrid[0:h, 0:w]
    nose = cv2.dilate((class_mask == NOSE).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    nose &= skin | (class_mask == NOSE)
    forehead = skin & (yy < brow_top) & ~nose
    chin = skin & (yy > lip_bot) & ~nose
    cheeks = skin & ~forehead & ~chin & ~nose
    return {"forehead": forehead, "nose": nose, "chin": chin, "right_cheek": cheeks & (xx < mid_x), "left_cheek": cheeks & (xx >= mid_x)}


def find_spots(r, img_bgr, zm0):
    """Which proposed spots count. Uses the plugged-in detector (detectors.py: ONNX / YOLO / RF-DETR) if one is configured, otherwise the built-in spot finder plus the trained filter.
    Returns (candidates [{core, box, ...}] in face-crop coordinates, number proposed, method name). zm0: zone masks before any profile folding."""
    det = detectors.get()
    if det is not None:
        mode = detectors.input_mode(); ox, oy = r["crop_box"][:2]; h, w = r["crop"].shape[:2]
        if mode == "full": inp = img_bgr
        elif mode == "crop": inp = r["crop_original"]
        else:                                                         # cutout: CLAHE face crop with everything that is not face skin blacked out (the way V_3 trains detectors)
            mask = (r["face_mask"] > 0) | _load().rescue_mask(r["class_mask"], r["face_mask"]); inp = r["crop"].copy(); inp[~mask] = 0
        sx, sy = (ox, oy) if mode == "full" else (0, 0); out = []
        for d in det.detect(inp):
            x0, y0, x1, y1 = [int(v) for v in (d["box"][0] - sx, d["box"][1] - sy, d["box"][2] - sx, d["box"][3] - sy)]
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h: continue            # detection lies outside the face crop
            b = (max(0, x0), max(0, y0), min(w, x1), min(h, y1)); out.append({"core": b, "box": b, "conf": d["conf"], "cls": d["cls"]})
        return out, len(out), det.name
    cands = r["candidates"]; filt = spot_filter(); proposed = len(cands)
    if filt is None or not cands: return cands, proposed, "built-in spot finder (no trained filter found)"
    P = SF.prepare(r["crop_original"], r["skin_mask"]); ctr = np.array([((c["core"][0] + c["core"][2]) / 2, (c["core"][1] + c["core"][3]) / 2) for c in cands]); X = []
    for i, c in enumerate(cands):
        sz = max(c["core"][2] - c["core"][0], c["core"][3] - c["core"][1]); nb = int((np.hypot(*(ctr - ctr[i]).T) < 3 * sz).sum() - 1)
        z0 = next((z for z, m in zm0.items() if 0 <= int(ctr[i][1]) < m.shape[0] and 0 <= int(ctr[i][0]) < m.shape[1] and m[int(ctr[i][1]), int(ctr[i][0])]), None)
        X.append(SF.features(P, c, SF.ZONES.index(z0) if z0 else -1, nb))
    keep = filt["model"].predict_proba(np.array(X, float))[:, 1] >= filt["threshold"]
    return [c for c, k in zip(cands, keep) if k], proposed, "built-in spot finder + trained filter"


def detect_view(zm0):
    """front / left / right from how the cheek areas split around the nose: a profile shows one cheek far larger than the other.
    left = the person's LEFT cheek is the visible one (image-right of the nose), matching the zone names used everywhere else."""
    l, r = int(zm0["left_cheek"].sum()), int(zm0["right_cheek"].sum())
    return "front" if max(l, r) < 2.5 * max(1, min(l, r)) else ("left" if l > r else "right")


def scan_image(img_bgr, side=None):
    """side=None for a front photo; "left"/"right" for a profile photo: the visible cheek is the whole cheek area of that photo.
    Returns {"status": "ok", zone_counts, zone_area, total_spots, preview(bgr crop with zones+spots)} or {"status": "no_face"|"no_zones", message}."""
    v3 = _load(); r = v3.preprocess_image(img_bgr)
    if r is None: return {"status": "no_face", "message": "No face was found. Please use a clear, well-lit photo with your whole face visible."}
    zm = zone_masks(r["class_mask"], r["face_mask"])
    if zm is None: return {"status": "no_zones", "message": "The face could not be mapped. Try a straight-on photo with the whole face in view."}
    zm0 = {z: m.copy() for z, m in zm.items()}
    view = detect_view(zm0)
    if side == "auto": side = None if view == "front" else view          # auto: decide from the face layout
    if side in ("left", "right"):                                   # profile: the far-side sliver is not a cheek, fold both halves into the visible cheek
        keep, other = f"{side}_cheek", ("right_cheek" if side == "left" else "left_cheek"); zm[keep] = zm[keep] | zm[other]; zm[other] = np.zeros_like(zm[keep])
    cands, proposed, method = find_spots(r, img_bgr, zm0)
    counts = {z: 0 for z in zm}; area = {z: int(m.sum()) for z, m in zm.items()}; prev = r["crop_original"].copy()
    colors = {"forehead": (214, 160, 50), "nose": (60, 120, 230), "chin": (80, 170, 80), "right_cheek": (170, 90, 190), "left_cheek": (60, 190, 220)}
    for z, m in zm.items(): prev[m] = (prev[m] * 0.78 + np.array(colors[z]) * 0.22).astype(np.uint8)
    t = max(1, min(prev.shape[:2]) // 300)
    for c in cands:
        x0, y0, x1, y1 = c["box"]; cx, cy = (c["core"][0] + c["core"][2]) // 2, (c["core"][1] + c["core"][3]) // 2
        z = next((z for z, m in zm.items() if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]), None)
        if z: counts[z] += 1; cv2.rectangle(prev, (x0, y0), (x1, y1), (255, 255, 255), t + 1); cv2.rectangle(prev, (x0, y0), (x1, y1), colors[z], t)
    return {"status": "ok", "view": view, "zone_counts": counts, "zone_area": area, "total_spots": sum(counts.values()), "skin_fallback": bool(r.get("skin_fallback")), "method": method, "filtered": method.startswith("built-in"), "proposed": proposed, "preview": prev}


if __name__ == "__main__":      # python3 scan.py photo.jpg [out.jpg]  -> prints zone counts, writes the zone/spot preview
    img = cv2.imread(sys.argv[1]); s = scan_image(img)
    print({k: v for k, v in s.items() if k != "preview"})
    if s["status"] == "ok" and len(sys.argv) > 2: cv2.imwrite(sys.argv[2], s["preview"])
