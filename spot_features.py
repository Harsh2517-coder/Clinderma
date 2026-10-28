"""Per-candidate features for the spot filter: which of the scan's proposed spots are real acne-type lesions?
Computed from the UNTOUCHED face crop (colour contrast against the surrounding skin), the candidate's own shape/score, and where it sits on the face."""
import cv2, numpy as np

ZONES = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]
METHODS = ["deviation", "redness", "darkness", "pigment", "color", "red", "dark"]      # mapped by position; unknown names get the last slot
NAMES = ["w", "h", "aspect", "area", "score", "strength", "scale", "method", "L_in", "a_in", "b_in", "L_ring", "a_ring", "b_ring", "dL", "da", "db", "dE", "a_p90", "L_std", "lap", "skin_ratio",
         "x_rel", "y_rel", "edge_dist", "neighbors"] + [f"zone_{z}" for z in ZONES]


def prepare(crop_orig_bgr, skin_mask):
    lab = cv2.cvtColor(crop_orig_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB); gray = cv2.cvtColor(crop_orig_bgr, cv2.COLOR_BGR2GRAY)
    skin = (skin_mask > 0).astype(np.uint8); dist = cv2.distanceTransform(skin, cv2.DIST_L2, 3); ys, xs = np.nonzero(skin)
    face = (xs.min(), ys.min(), max(1, xs.max() - xs.min()), max(1, ys.max() - ys.min())) if len(xs) else (0, 0, skin.shape[1], skin.shape[0])
    return {"lab": lab, "gray": gray, "skin": skin, "dist": dist, "face": face, "dim": float(min(skin.shape))}


def method_index(m): return METHODS.index(m) if m in METHODS else len(METHODS)


def features(P, cand, zone_idx, neighbors):
    lab, gray, skin, dist, (fx, fy, fw, fh), dim = P["lab"], P["gray"], P["skin"], P["dist"], P["face"], P["dim"]; H, W = skin.shape
    x0, y0, x1, y1 = [int(v) for v in cand["core"]]; x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, max(x1, x0 + 1)), min(H, max(y1, y0 + 1)); w, h = x1 - x0, y1 - y0
    inside = lab[y0:y1, x0:x1].reshape(-1, 3); pad = max(w, h); rx0, ry0, rx1, ry1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    ring_m = skin[ry0:ry1, rx0:rx1] > 0; ring_m = ring_m.copy(); ring_m[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = False; ring = lab[ry0:ry1, rx0:rx1][ring_m]
    if len(ring) < 20: ring = inside
    mi, mr = np.median(inside, axis=0), np.median(ring, axis=0); d = mi - mr; g = gray[y0:y1, x0:x1]; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var()) if g.size >= 9 else 0.0
    f = [w / dim, h / dim, w / max(1, h), w * h / dim ** 2, cand.get("score", 0), cand.get("strength", 0), cand.get("scale", 0), method_index(cand.get("method", "")), *mi, *mr, *d, float(np.linalg.norm(d)),
         float(np.percentile(inside[:, 1], 90) - mr[1]), float(inside[:, 0].std()), lap, float(skin[y0:y1, x0:x1].mean()), (cx - fx) / fw, (cy - fy) / fh, float(dist[min(cy, H - 1), min(cx, W - 1)] / dim), neighbors]
    return f + [1.0 if zone_idx == i else 0.0 for i in range(len(ZONES))]
