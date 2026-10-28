"""How accurate is the scan, by face region?  Compares the spots the scan finds with expert-labelled acne lesions (YOLO boxes), per zone.  READ-ONLY on the data.
Usage: python3 eval_accuracy.py <shard> <n_shards> [n_images]   -> results/eval_shard_<shard>.json ; then python3 eval_report.py
Per zone and image: labelled lesions (GT), GT lesions touched by a scan box (recall), scan boxes touching a GT lesion (precision).
Also: does the zone the scan calls 'most affected' (by spot density) match the zone with the most labelled lesions (by density)?"""
import glob, json, os, random, sys
import cv2, numpy as np
import scan

DATA = os.environ.get("EVAL_DATA", "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/DATASET/ACNE_COMBINED_DATA_CLINDERMA")
shard, n_shards = int(sys.argv[1]), int(sys.argv[2]); n_images = int(sys.argv[3]) if len(sys.argv) > 3 else 150
ZONES = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]


from zone_match import touches, zone_of


files = sorted(glob.glob(f"{DATA}/images/*")); random.Random(0).shuffle(files); files = files[:n_images][shard::n_shards]
v3 = scan._load(); out = []
for f in files:
    stem = os.path.splitext(os.path.basename(f))[0]; lab = f"{DATA}/labels/{stem}.txt"
    img = cv2.imread(f)
    if img is None or not os.path.exists(lab): continue
    H, W = img.shape[:2]; gt = [(float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H) for p in (l.split() for l in open(lab) if l.strip())]
    gt = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for cx, cy, w, h in gt]
    try: r = v3.preprocess_image(img)
    except Exception as e: out.append({"image": stem, "status": "error", "error": str(e)[:100]}); continue
    if r is None: out.append({"image": stem, "status": "no_face", "n_gt": len(gt)}); continue
    zm = scan.zone_masks(r["class_mask"], r["face_mask"])
    if zm is None: out.append({"image": stem, "status": "no_zones", "n_gt": len(gt)}); continue
    ox, oy = r["crop_box"][:2]; g = [(a - ox, b - oy, c - ox, d - oy) for a, b, c, d in gt]; cands = list(r["boxes"]); area = {z: int(m.sum()) for z, m in zm.items()}
    rec = {z: {"gt": 0, "gt_hit": 0, "cand": 0, "cand_match": 0} for z in ZONES}
    for b in g:
        z = zone_of((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, zm)
        if z: rec[z]["gt"] += 1; rec[z]["gt_hit"] += any(touches(b, c) for c in cands)
    for c in cands:
        z = zone_of((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, zm)
        if z: rec[z]["cand"] += 1; rec[z]["cand_match"] += any(touches(c, b) for b in g)
    out.append({"image": stem, "status": "ok", "zones": rec, "area": area, "n_gt": len(gt), "n_cand": len(cands), "gt_outside_zones": len(g) - sum(rec[z]["gt"] for z in ZONES)})
    print(stem, "gt", len(gt), "cand", len(cands), flush=True)
json.dump(out, open(f"results/eval_shard_{shard}.json", "w")); print("done", len(out))
