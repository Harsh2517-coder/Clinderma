"""Builds the candidate dataset for the spot filter from the labelled images (READ-ONLY).  python3 build_spot_dataset.py <shard> <n_shards>  -> results/spots_shard_<i>.pkl
One row per proposed spot: features, label (touches an expert-labelled lesion), zone, which labelled lesions it touches; plus per-image labelled-lesion zone counts and zone areas."""
import glob, os, pickle, sys
import cv2, numpy as np
import scan, spot_features as SF
from zone_match import touches, zone_of

DATA = os.environ.get("EVAL_DATA", "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/DATASET/ACNE_COMBINED_DATA_CLINDERMA")
shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
files = sorted(glob.glob(f"{DATA}/images/*"))[shard::n_shards]; v3 = scan._load(); rows, imgs = [], []
for f in files:
    stem = os.path.splitext(os.path.basename(f))[0]; lab = f"{DATA}/labels/{stem}.txt"; img = cv2.imread(f)
    if img is None or not os.path.exists(lab): continue
    H, W = img.shape[:2]; gt = [(float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H) for p in (l.split() for l in open(lab) if l.strip())]
    gt = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for cx, cy, w, h in gt]
    try: r = v3.preprocess_image(img)
    except Exception: continue
    if r is None: continue
    zm = scan.zone_masks(r["class_mask"], r["face_mask"])
    if zm is None: continue
    ox, oy = r["crop_box"][:2]; g = [(a - ox, b - oy, c - ox, d - oy) for a, b, c, d in gt]; P = SF.prepare(r["crop_original"], r["skin_mask"]); cands = r["candidates"]
    centres = np.array([((c["core"][0] + c["core"][2]) / 2, (c["core"][1] + c["core"][3]) / 2) for c in cands]) if cands else np.zeros((0, 2)); img_id = len(imgs)
    gz = [zone_of((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, zm) for b in g]
    imgs.append({"image": stem, "group": stem.rsplit("_", 1)[0], "area": {z: int(m.sum()) for z, m in zm.items()}, "gt_zone": gz})
    for i, c in enumerate(cands):
        sz = max(c["core"][2] - c["core"][0], c["core"][3] - c["core"][1]); nb = int((np.hypot(*(centres - centres[i]).T) < 3 * sz).sum() - 1)
        z = zone_of(centres[i][0], centres[i][1], zm); zi = SF.ZONES.index(z) if z else -1; m = [j for j, b in enumerate(g) if touches(c["box"], b)]
        rows.append({"img": img_id, "zone": zi, "label": int(bool(m)), "matches": m, "x": SF.features(P, c, zi, nb)})
    print(stem, len(cands), "cands", len(g), "gt", flush=True)
pickle.dump({"images": imgs, "rows": rows, "names": SF.NAMES}, open(f"results/spots_shard_{shard}.pkl", "wb")); print("done", len(imgs), "images", len(rows), "candidates")
