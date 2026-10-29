"""Measure candidate-box recall against the Clinderma lesion labels (ground truth).
usage: eval_candidates.py <v3_dir> <out.json> [k:n | stride] [CONFIG_NAME=value ...]
Works on both the old (analyze_face + find_spots) and new (preprocess_image) pipelines, so they can be compared.
For every labelled lesion, records whether its centre lies in a candidate box / the searched skin mask, plus box counts and
how much of the face the boxes blanket (so a trivially huge box can't fake recall)."""
import glob, json, os, sys, time
import cv2, numpy as np

import ast
v3, out = sys.argv[1], sys.argv[2]
k, n = map(int, (sys.argv[3] if len(sys.argv) > 3 else "1").split(":")) if ":" in (sys.argv[3] if len(sys.argv) > 3 else "") else (0, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
sys.path.insert(0, v3)
if os.path.exists(f"{v3}/config.py"):
    import config
    for kv in sys.argv[4:]:  # NAME=value overrides for tuning sweeps, e.g. DEVIATION_PERCENTILE=80
        name, val = kv.split("=", 1)
        setattr(config, name, eval(val, {}, {}))
_D = os.environ.get("CLINDERMA_DATA", "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/DATASET")  # folder holding ACNE_COMBINED_DATA_CLINDERMA/ and DATASET_COMBINED/
SRC = {"acne": f"{_D}/ACNE_COMBINED_DATA_CLINDERMA", "pig": f"{_D}/DATASET_COMBINED/pigmentation"}[os.environ.get("EVAL_SET", "acne")]
NEW = os.path.exists(f"{v3}/preprocess.py")
if NEW:
    from preprocess import preprocess_image
else:
    from face_crop import detect_faces
    from face_segment import analyze_face
    from spot_detect import find_spots, skin_only_mask


def run(img):
    """-> (boxes in crop coords, crop-origin (x, y), search mask, face mask) or None"""
    if NEW:
        r = preprocess_image(img)
        return None if r is None else (r["boxes"], r["crop_box"][:2], r["skin_mask"], r["face_mask"])
    a = analyze_face(img)
    if a is None:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1, _ = detect_faces(img, conf=0.3)[0]
    mx, my = (x1 - x0) * 0.3, (y1 - y0) * 0.3
    sm = skin_only_mask(a["class_mask"], a["crop"])
    return find_spots(a["crop"], sm), (max(0, int(x0 - mx)), max(0, int(y0 - my))), sm, a["face_mask"]


def area_cov(cov, g, poly):
    """Fraction of the labelled lesion's area (polygon, else its box) lying inside some candidate box."""
    m = np.zeros_like(cov)
    if poly is not None:
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    else:
        m[max(0, int(g[1])):max(0, int(g[3])), max(0, int(g[0])):max(0, int(g[2]))] = 1
    return float(cov[m > 0].mean()) if (m > 0).any() else 0.0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


import re


def label_path(f):
    """Image path -> label path. Pigmentation labels are named v1_pigdata_<id> while the images are v1_{melasma_tan,pih,pie}_<id>."""
    stem = os.path.splitext(os.path.basename(f))[0]
    m = re.match(r"v1_(?:melasma_tan|pih|pie)_(.*)", stem)
    return f"{SRC}/labels/{'v1_pigdata_' + m.group(1) if m and SRC.endswith('pigmentation') else stem}.txt"


files = [f for f in sorted(glob.glob(f"{SRC}/images/*")) if os.path.exists(label_path(f))][k::n]
rows, imgs, t0 = [], [], time.time()
for i, f in enumerate(files):
    stem = os.path.splitext(os.path.basename(f))[0]
    img = cv2.imread(f)
    r = run(img)
    if r is None:
        imgs.append({"stem": stem, "face": False})
        continue
    boxes, (ox, oy), search, face = r
    h, w = img.shape[:2]
    ch, cw = search.shape
    cand = [(x0, y0, x1, y1) for x0, y0, x1, y1 in boxes]
    cov = np.zeros((ch, cw), np.uint8)
    for x0, y0, x1, y1 in cand:
        cov[y0:y1, x0:x1] = 1
    near = cv2.dilate(face, np.ones((int(0.05 * min(ch, cw)) * 2 + 1,) * 2, np.uint8)) > 0  # background check: box centres off the face
    out_face = float(np.mean([not near[min(ch - 1, (b[1] + b[3]) // 2), min(cw - 1, (b[0] + b[2]) // 2)] for b in cand])) if cand else 0.0
    imgs.append({"stem": stem, "face": True, "n_boxes": len(cand), "out_face": out_face, "face_cover": float(cov[face > 0].mean()) if (face > 0).any() else 0.0,
                 "median_box_frac": float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in cand]) / (face > 0).sum()) if cand else 0.0})
    for l in open(label_path(f)).read().splitlines():
        if not l.strip():
            continue
        t = l.split()
        if len(t) == 5:  # YOLO box
            _, bx, by, bw, bh = t
            g = [(float(bx) - float(bw) / 2) * w - ox, (float(by) - float(bh) / 2) * h - oy, (float(bx) + float(bw) / 2) * w - ox, (float(by) + float(bh) / 2) * h - oy]
            poly = None
        else:  # YOLO polygon (pigmentation): box = its extent, area coverage measured on the polygon itself
            pts = np.array([[float(x) * w - ox, float(y) * h - oy] for x, y in zip(t[1::2], t[2::2])], np.float32)
            g = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
            poly = pts
        c = t[0]
        gcx, gcy = int((g[0] + g[2]) / 2), int((g[1] + g[3]) / 2)
        in_crop = 0 <= gcx < cw and 0 <= gcy < ch
        rows.append({"stem": stem, "cls": int(c), "in_crop": in_crop,
                     "in_face": bool(in_crop and face[gcy, gcx] > 0), "in_search": bool(in_crop and search[gcy, gcx] > 0),
                     "center": bool(any(b[0] <= gcx < b[2] and b[1] <= gcy < b[3] for b in cand)),
                     "iou20": bool(any(iou(g, b) >= 0.2 for b in cand)), "area_cov": area_cov(cov, g, poly)})
    if i % 50 == 0:
        print(i, len(files), f"{time.time() - t0:.0f}s", flush=True)
json.dump({"gt": rows, "imgs": imgs}, open(out, "w"))
print("done", len(rows), "GT boxes", sum(i["face"] for i in imgs), "faces")
