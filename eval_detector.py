"""Measures the ACTIVE detector (see detectors.py) per face zone and writes models/detector_metrics.json, which the app then shows next to every scan.
    DETECTOR=onnx DETECTOR_PATH=models/acne_detector.onnx python3 eval_detector.py <images_dir> <labels_dir> [max_images]
<images_dir>/<labels_dir>: YOLO-format labels (class cx cy w h, normalised), same file stems.  IMPORTANT: use photos the detector was NOT trained on (for a Roboflow model: its test split,
exported as YOLO format). Evaluating on training photos gives flattering numbers that will not hold for new patients."""
import glob, json, os, sys
import cv2, numpy as np
from scipy.stats import spearmanr
import detectors, scan
from zone_match import touches, zone_of

ZONES = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]
imgs_dir, labels_dir = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
if detectors.get() is None: sys.exit("No detector configured: set DETECTOR and DETECTOR_PATH (see detectors.py).")
v3 = scan._load(); per = {z: {"gt": 0, "hit": 0, "k": 0, "m": 0} for z in ZONES}; agree, top2, kept_tot, gt_tot, n_ok = [], [], [], [], 0
for f in sorted(glob.glob(f"{imgs_dir}/*"))[:limit]:
    stem = os.path.splitext(os.path.basename(f))[0]; lab = f"{labels_dir}/{stem}.txt"; img = cv2.imread(f)
    if img is None or not os.path.exists(lab): continue
    H, W = img.shape[:2]; gt = [(float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H) for p in (l.split() for l in open(lab) if l.strip())]
    gt = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for cx, cy, w, h in gt]; r = v3.preprocess_image(img)
    if r is None: continue
    zm = scan.zone_masks(r["class_mask"], r["face_mask"])
    if zm is None: continue
    ox, oy = r["crop_box"][:2]; g = [(a - ox, b - oy, c - ox, d - oy) for a, b, c, d in gt]; cands, _, _ = scan.find_spots(r, img, zm); n_ok += 1
    area = {z: int(m.sum()) for z, m in zm.items()}; gc, kc = np.zeros(5), np.zeros(5)
    for b in g:
        z = zone_of((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, zm)
        if z: per[z]["gt"] += 1; per[z]["hit"] += any(touches(b, c["box"]) for c in cands); gc[ZONES.index(z)] += 1
    for c in cands:
        z = zone_of((c["core"][0] + c["core"][2]) / 2, (c["core"][1] + c["core"][3]) / 2, zm)
        if z: per[z]["k"] += 1; per[z]["m"] += any(touches(c["box"], b) for b in g); kc[ZONES.index(z)] += 1
    kept_tot.append(kc.sum()); gt_tot.append(gc.sum()); tot = sum(area.values()) or 1; elig = [k for k, z in enumerate(ZONES) if area[z] / tot >= 0.015]
    if gc.sum() >= 3 and len(elig) >= 3 and kc.sum() > 0:
        dg = {k: gc[k] / area[ZONES[k]] for k in elig}; dc = {k: kc[k] / area[ZONES[k]] for k in elig}; best = max(dg, key=dg.get)
        agree.append(best == max(dc, key=dc.get)); top2.append(best in sorted(dc, key=dc.get, reverse=True)[:2])
    print(stem, "gt", len(g), "detected", len(cands), flush=True)
G = sum(v["gt"] for v in per.values()); H_ = sum(v["hit"] for v in per.values()); K = sum(v["k"] for v in per.values()); M = sum(v["m"] for v in per.values())
m = {"detector": detectors.get().name, "validation": f"held-out photos supplied by the user ({n_ok} photos, {G} labelled lesions)", "photos": n_ok,
     "filtered": {"recall": round(H_ / max(1, G), 3), "precision": round(M / max(1, K), 3), "agree": round(float(np.mean(agree)) if agree else 0, 3), "top2": round(float(np.mean(top2)) if top2 else 0, 3), "kept_per_photo": round(float(np.mean(kept_tot)), 1)},
     "zones": {z: {"recall": round(per[z]["hit"] / max(1, per[z]["gt"]), 3), "precision": round(per[z]["m"] / max(1, per[z]["k"]), 3), "lesions": per[z]["gt"]} for z in ZONES}, "count_spearman": round(float(spearmanr(kept_tot, gt_tot)[0]), 3)}
os.makedirs("models", exist_ok=True); json.dump(m, open("models/detector_metrics.json", "w"), indent=1); print(json.dumps(m, indent=1))
