"""Qualitative + quantitative test matrix for the candidate-generation pipeline, one case per requirement category.
usage: python3 run_tests.py [out_dir]     (default test_outputs/)
Each case saves a 5-panel montage: original | CLAHE | face(blue)+searched skin(green) | candidate mask | final boxes (green) with the
labelled lesions drawn in red where ground truth exists -- a red outline with no green box over it is a false negative.
Lighting/resolution cases are synthetic variants of a real acne image (labels are normalised, so they stay valid)."""
import glob, os, re, sys, time
import cv2
import numpy as np

from preprocess import preprocess_image, render_debug

D = os.environ.get("CLINDERMA_DATA", "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/DATASET")  # folder holding ACNE_COMBINED_DATA_CLINDERMA/, DATASET_COMBINED/, DATASET_V2/
ACNE, PIG = f"{D}/ACNE_COMBINED_DATA_CLINDERMA", f"{D}/DATASET_COMBINED/pigmentation"
OUT = sys.argv[1] if len(sys.argv) > 1 else "test_outputs"
os.makedirs(OUT, exist_ok=True)


def read_labels(path):
    """-> list of ('box'|'poly', normalised points Nx2)"""
    out = []
    for l in open(path).read().splitlines():
        t = l.split()
        if len(t) == 5:
            _, x, y, w, h = map(float, t)
            out.append(("box", np.array([[x - w / 2, y - h / 2], [x + w / 2, y + h / 2]])))
        elif len(t) > 5:
            out.append(("poly", np.array(list(map(float, t[1:]))).reshape(-1, 2)))
    return out


def pig_label_path(img_path):
    m = re.match(r"v1_(?:melasma_tan|pih|pie)_(.*)", os.path.splitext(os.path.basename(img_path))[0])
    return f"{PIG}/labels/v1_pigdata_{m.group(1)}.txt"


def thumb_gray(path):
    return cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)


# ---- pick cases from the real data -------------------------------------------------------------------------
acne = []
for f in sorted(glob.glob(f"{ACNE}/images/*")):
    lab = read_labels(f"{ACNE}/labels/{os.path.splitext(os.path.basename(f))[0]}.txt")
    sizes = [(p[1] - p[0]).mean() for k, p in lab]
    acne.append({"path": f, "n": len(lab), "size": float(np.median(sizes)) if sizes else 0, "lab": lab})
for a in acne:
    a["bright"] = float(thumb_gray(a["path"]).mean())
rich = [a for a in acne if a["n"] >= 3]

pig = []
for f in sorted(glob.glob(f"{PIG}/images/*")):
    lp = pig_label_path(f)
    if not os.path.exists(lp):
        continue
    lab = read_labels(lp)
    im = cv2.imread(f, cv2.IMREAD_REDUCED_COLOR_8)
    h, w = im.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for k, p in lab:
        cv2.fillPoly(m, [(p * [w, h]).astype(np.int32)], 1)
    if m.sum() < 5 or (m == 0).sum() < 5:
        continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    pig.append({"path": f, "n": len(lab), "area": m.mean(), "contrast": g[m == 0].mean() - g[m == 1].mean(), "lab": lab})


def first(seq, key, reverse=False, skip=()):
    """Candidates best-first; the runner takes the first one in which a face is found."""
    return [x for x in sorted(seq, key=key, reverse=reverse) if x["path"] not in skip]


base = first([a for a in rich if 4 <= a["n"] <= 8], key=lambda a: abs(a["bright"] - 110))  # mid-exposure mild cases
cases = [
    ("01_clear_skin", first([a for a in acne if a["n"] <= 1], key=lambda a: a["n"]), None),
    ("02_mild_acne", base, None),
    ("03_severe_acne", first(acne, key=lambda a: a["n"], reverse=True), None),
    ("04_small_lesions", first([a for a in rich if a["n"] >= 5], key=lambda a: a["size"]), None),
    ("05_dark_pigmentation", first(pig, key=lambda p: p["contrast"], reverse=True), None),
    ("06_subtle_pigmentation", first(pig, key=lambda p: abs(p["contrast"])), None),
    ("07_large_pigment_patch", first(pig, key=lambda p: p["area"], reverse=True), None),
    ("08a_dark_skin_tone", first(rich, key=lambda a: a["bright"]), None),
    ("08b_light_skin_tone", first(rich, key=lambda a: a["bright"], reverse=True), None),
    ("09_bright_lighting", base, "bright"),
    ("10_low_lighting", base, "low"),
    ("11_uneven_lighting", base, "uneven"),
    ("12_frontal", [{"path": f"{D}/DATASET_V2/downloads/batch_1/front_original.jpg", "lab": []}], None),
    ("13_left_facing", [{"path": f"{D}/DATASET_V2/downloads/batch_1/left_original.jpg", "lab": []}], None),
    ("14_right_facing", [{"path": f"{D}/DATASET_V2/downloads/batch_1/right_original.jpg", "lab": []}], None),
    ("15a_res_long4000", first(acne, key=lambda a: a["n"], reverse=True), 4000),
    ("15b_res_long1000", first(acne, key=lambda a: a["n"], reverse=True), 1000),
    ("15c_res_long400", first(acne, key=lambda a: a["n"], reverse=True), 400),
]


def variant(img, how):
    if how == "bright":
        return np.clip(255 * (img / 255.0) ** 0.45 * 1.15, 0, 255).astype(np.uint8)
    if how == "low":
        return np.clip(255 * (img / 255.0) ** 2.2 * 0.8, 0, 255).astype(np.uint8)
    if how == "uneven":  # strong left-to-right light fall-off
        ramp = np.linspace(1.0, 0.25, img.shape[1], dtype=np.float32)[None, :, None]
        return np.clip(img * ramp, 0, 255).astype(np.uint8)
    if isinstance(how, int):
        s = how / max(img.shape[:2])
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def draw_gt(panel, result, lab, shape):
    """Draw labelled lesions (red) onto a crop-sized panel."""
    h, w = shape
    x0, y0 = result["crop_box"][:2]
    t = max(1, min(panel.shape[:2]) // 300)
    for kind, p in lab:
        pts = (p * [w, h] - [x0, y0]).astype(np.int32)
        if kind == "box":
            cv2.rectangle(panel, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), t)
        else:
            cv2.polylines(panel, [pts], True, (0, 0, 255), t)


def gt_metrics(result, lab, shape):
    h, w = shape
    x0, y0 = result["crop_box"][:2]
    ch, cw = result["skin_mask"].shape
    cov = np.zeros((ch, cw), np.uint8)
    for a, b, c, d in result["boxes"]:
        cov[b:d, a:c] = 1
    hit, area = [], []
    for kind, p in lab:
        pts = (p * [w, h] - [x0, y0])
        m = np.zeros((ch, cw), np.uint8)
        if kind == "box":
            m[max(0, int(pts[0, 1])):max(0, int(pts[1, 1])), max(0, int(pts[0, 0])):max(0, int(pts[1, 0]))] = 1
        else:
            cv2.fillPoly(m, [pts.astype(np.int32)], 1)
        if not m.any():
            continue  # outside the face crop
        area.append(cov[m > 0].mean())
        c = pts.mean(0) if kind == "poly" else pts.mean(0)
        hit.append(any(a <= c[0] < c2 and b <= c[1] < d for a, b, c2, d in result["boxes"]))
    return len(area), (np.mean(hit) if hit else float("nan")), (np.mean(area) if area else float("nan"))


def panel(im, title, height=620):
    s = height / im.shape[0]
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), height), interpolation=cv2.INTER_AREA)
    cv2.putText(im, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return im


rows = []
used = set()
for name, cands, how in cases:
    r = None
    for case in cands:
        if case["path"] in used and name[:2] in ("06", "07"):
            continue  # distinct pigmentation images per case
        img = variant(cv2.imread(case["path"]), how)
        t0 = time.time()
        r = preprocess_image(img)
        dt = time.time() - t0
        if r is not None:
            break
    if r is None:
        rows.append((name, "-", "-", "-", "NO FACE in any candidate", 0, "", "", "", ""))
        print(name, "NO FACE")
        continue
    if name[:2] in ("05", "06", "07"):
        used.add(case["path"])
    st = render_debug(r, img)
    lab = case["lab"] if "lab" in case else []
    orig_shape = None
    if lab:
        orig_shape = img.shape[:2]
        draw_gt(st["5_final_boxes"], r, lab, orig_shape)
    n_gt, rc, ac = gt_metrics(r, lab, img.shape[:2]) if lab else (0, float("nan"), float("nan"))
    tiles = [panel(st["1_original"], "original"), panel(st["2_clahe"], "CLAHE")] + [
        panel(st[k], t) for k, t in (("3_face_skin_mask", "face+skin mask"), ("4_candidate_mask", "candidates"), ("5_final_boxes", f"{len(r['boxes'])} boxes" + (" (red=GT)" if lab else "")))]
    cv2.imwrite(f"{OUT}/{name}.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rows.append((name, os.path.basename(case["path"])[:28], img.shape[1], img.shape[0], "ok", len(r["boxes"]), n_gt or "", "" if not n_gt else f"{rc:.2f}", "" if not n_gt else f"{ac:.2f}", f"{dt:.1f}"))
    print(rows[-1], flush=True)

hdr = "| case | image | W | H | face | boxes | labelled lesions in crop | centre recall | area coverage | sec |\n|---|---|---|---|---|---|---|---|---|---|\n"
table = hdr + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
open(f"{OUT}/summary.md", "w").write(table + "\n")
print(table)
