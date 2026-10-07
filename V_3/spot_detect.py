"""High-recall candidate generation for facial skin spots -- acne, pigmentation, or anything else that isn't smooth regular
skin. No trained classifier: a candidate is just "skin that looks different from its own smoothed-out local neighborhood".
Whether a candidate is acne / pigmentation / nothing is decided downstream, so this stage keeps weak, small, oddly-shaped
and low-contrast regions on purpose: a false positive is rejected later, a missed lesion is gone for good.

Detection runs several colour signals (LAB distance = the original method, redness, darkness, brownness) at several
baseline scales (small lesions -> large patches), takes the union of all of them, merges only blobs that are essentially
touching, and collapses only near-duplicate boxes. All thresholds live in config.py.

Restricted to the CelebAMask-HQ skin/nose/ear classes from face_segment's parsing pass, so eyes, eyebrows, lips etc. are
never searched; the skin directly *around* them is.
"""
import cv2
import numpy as np
from scipy import ndimage

import config as cfg

# Searchable parser classes live in config.SEARCH_CLASSES (skin, ears, nose -- real skin the parser gives its own labels) and
# config.RESCUE_CLASSES ('hat', which it wrongly assigns to some cheeks; only searched near the face).
# eyebrows, eyes, glasses, earring, mouth, lips -- genuinely not skin (earring(9) is jewelry, not ear skin). Only these
# pixels (plus a thin rim, cfg.FEATURE_BUFFER_FRAC) are removed -- not the surrounding zones.
NON_SKIN_FEATURE_CLASSES = {2, 3, 4, 5, 6, 9, 11, 12, 13}
NOSE_CLASS = 10
EYE_CLASSES = {2, 3, 4, 5, 6}  # brows, eyes, glasses


def rescue_mask(class_mask, face_mask):
    """Pixels the parser mislabelled (config.RESCUE_CLASSES, i.e. 'hat') that lie within RESCUE_DILATE_FRAC of the parsed face:
    in practice sun-lit / oily / acne-covered cheeks. Bool array."""
    r = int(cfg.RESCUE_DILATE_FRAC * min(class_mask.shape)) * 2 + 1
    near_face = cv2.dilate(face_mask, np.ones((r, r), np.uint8)) > 0
    return np.isin(class_mask, list(cfg.RESCUE_CLASSES)) & near_face


def skin_only_mask(class_mask, img_bgr=None, face_mask=None):
    """The valid search area: skin/nose/ear pixels, minus the actual eye/brow/glasses/lip pixels (with a thin rim), minus the
    crop border, minus the nostril cavities. Skin immediately around the eyes and mouth stays in -- periorbital and
    upper-lip pigmentation is legitimate. The cost is that eyelid/lip edges can now yield false-positive candidates, which
    is acceptable at this stage (the downstream model rejects them)."""
    skin = np.isin(class_mask, list(cfg.SEARCH_CLASSES)).astype(np.uint8) * 255
    if face_mask is not None:
        skin[rescue_mask(class_mask, face_mask)] = 255
    face_h, face_w = class_mask.shape
    dilated = np.zeros((face_h, face_w), np.uint8)
    # eyes/brows/glasses get a wider margin than the mouth: lids and eye-socket shading light up as big false blobs
    for classes, frac in ((EYE_CLASSES, cfg.EYE_BUFFER_FRAC), (NON_SKIN_FEATURE_CLASSES - EYE_CLASSES, cfg.FEATURE_BUFFER_FRAC)):
        k = max(3, int(min(face_h, face_w) * frac) | 1)
        dilated |= cv2.dilate(np.isin(class_mask, list(classes)).astype(np.uint8) * 255, np.ones((k, k), np.uint8))
    features_dilated = dilated

    border = max(1, int(cfg.BORDER_FRAC * min(face_h, face_w)))
    border_mask = np.zeros((face_h, face_w), np.uint8)
    border_mask[border:-border, border:-border] = 255

    result = cv2.bitwise_and(cv2.bitwise_and(skin, cv2.bitwise_not(features_dilated)), border_mask)

    # the nostril openings are actual cavities, not skin -- the parser has no separate class for them so they get lumped
    # into "nose", and being far darker than surrounding nose skin they'd otherwise register as a huge false "spot".
    # Cut the darkest fraction of the nose region (relative to that nose's own brightness, so it works across lighting).
    if img_bgr is not None:
        nose_region = class_mask == NOSE_CLASS
        if nose_region.any():
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            dark_cutoff = np.percentile(gray[nose_region], cfg.NOSTRIL_DARK_PERCENTILE)
            nostril = (nose_region & (gray < dark_cutoff)).astype(np.uint8) * 255
            nostril = cv2.dilate(nostril, np.ones((cfg.NOSTRIL_DILATE_PX,) * 2, np.uint8))
            result = cv2.bitwise_and(result, cv2.bitwise_not(nostril))

    return result


def _baseline_diff(lab, skin_mask, k):
    """lab minus a *masked* local average (float32 HxWx3). The baseline is built only from nearby skin pixels -- a plain blur
    leaks in eyebrow/eye/hair/background colour at every mask edge and falsely flags the whole boundary. The baseline is
    low-frequency, so it's computed on a downscaled copy (fast at any resolution) and upsampled."""
    h, w = skin_mask.shape
    f = min(1.0, 600.0 / max(h, w))
    sw, sh = max(1, int(w * f)), max(1, int(h * f))
    ks = max(3, int(k * f) | 1)
    sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
    small = cv2.resize(lab, (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize((skin_mask > 0).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    base = cv2.GaussianBlur(small * m[:, :, None], (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma)[:, :, None] + 1e-6)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    return lab - base


def _deviation_map(crop_bgr, skin_mask):
    """Original single-scale LAB-distance deviation map (kept for compatibility)."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    face_h, face_w = skin_mask.shape
    dev = np.linalg.norm(_baseline_diff(lab, skin_mask, max(9, (min(face_h, face_w) // 8) | 1)), axis=2)
    dev[skin_mask == 0] = 0
    return dev


def _signal(name, diff):
    dl, da, db = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
    if name == "deviation":
        return np.linalg.norm(diff, axis=2)
    if name == "redness":
        return np.maximum(da, 0)
    if name == "darkness":
        return np.maximum(-dl, 0)
    if name == "brownness":
        return np.hypot(np.maximum(db, 0), np.maximum(-dl, 0))
    raise ValueError(name)


def _extract(mask, sig, min_area, max_area, max_side, min_side, merge_px, depth, o):
    """Binary anomaly mask -> [(x0, y0, x1, y1, score)]. Blobs within merge_px of each other count as one lesion (the box is
    still drawn around the real anomaly pixels, not the dilated blob). Oversize blobs (too big for the baseline scale that
    found them) are re-thresholded higher and split, so a patch of unrelated lesions doesn't become one huge box; whatever
    is still oversize after that is a feature for a larger scale to pick up."""
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    if merge_px > 1:
        grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1,) * 2))
    else:
        grouped = mask
    n, lab = cv2.connectedComponents(grouped, connectivity=8)
    lab = lab * (mask > 0)
    areas = np.bincount(lab.ravel(), minlength=n)
    sums = np.bincount(lab.ravel(), weights=sig.ravel(), minlength=n)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or areas[i] < min_area:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if areas[i] > max_area or max(w, h) > max_side:
            if depth > 0:
                region = lab[sl] == i
                sub = (region & (sig[sl] > np.percentile(sig[sl][region], cfg.OVERSIZE_RESPLIT_PERCENTILE))).astype(np.uint8)
                out += [(a + x0, b + y0, c + x0, d + y0, s) for a, b, c, d, s in
                        _extract(sub, sig[sl], min_area, max_area, max_side, min_side, merge_px, depth - 1, o)]
            continue
        if max(w, h) < min_side:  # specks smaller than any lesion
            continue
        # reject scattered/elongated detections (stray hair strands merged by closing; skin-fold shadows) -- loosely
        if areas[i] / (w * h) < o("min_fill") or max(w, h) / max(1, min(w, h)) > o("max_aspect"):
            continue
        out.append((x0, y0, x1, y1, sums[i] / areas[i]))
    return out


def _dedupe(cands, iou_thr, contain_thr):
    """One box per lesion: strongest candidate first (strength = signal / the cutoff it had to clear, comparable across signals
    and scales); a weaker box is dropped if it overlaps a kept one by IoU >= iou_thr, or lies >= contain_thr inside it."""
    if not cands:
        return cands
    cands = sorted(cands, key=lambda c: c["strength"], reverse=True)
    b = np.array([c["box"] for c in cands], np.float32)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = []
    for i in range(len(cands)):
        if keep:
            kb = b[keep]
            iw = np.clip(np.minimum(kb[:, 2], b[i, 2]) - np.maximum(kb[:, 0], b[i, 0]), 0, None)
            ih = np.clip(np.minimum(kb[:, 3], b[i, 3]) - np.maximum(kb[:, 1], b[i, 1]), 0, None)
            inter = iw * ih
            if (inter / (area[keep] + area[i] - inter + 1e-9) >= iou_thr).any() or (inter / (area[i] + 1e-9) >= contain_thr).any():
                continue
        keep.append(i)
    return [cands[i] for i in keep]


def _pad_inside(box, valid, ratio):
    """Grow a box by ratio of its size on each side, but per side only as far as the new strip is still mostly valid skin
    (so padding never bleeds into background, eyes, lips)."""
    x0, y0, x1, y1 = box
    h, w = valid.shape
    px, py = max(1, int(round((x1 - x0) * ratio))), max(1, int(round((y1 - y0) * ratio)))

    def ok(strip):
        return strip.size > 0 and (strip > 0).mean() >= 0.5

    def grow(p, strip_at):
        while p > 0 and not ok(strip_at(p)):
            p //= 2
        return p

    nx0 = x0 - grow(min(px, x0), lambda p: valid[y0:y1, x0 - p:x0])
    nx1 = x1 + grow(min(px, w - x1), lambda p: valid[y0:y1, x1:x1 + p])
    ny0 = y0 - grow(min(py, y0), lambda p: valid[y0 - p:y0, nx0:nx1])
    ny1 = y1 + grow(min(py, h - y1), lambda p: valid[y1:y1 + p, nx0:nx1])
    return int(nx0), int(ny0), int(nx1), int(ny1)


def find_candidates(crop_bgr, skin_mask, **opts):
    """Returns (candidates, candidate_mask). candidates: list of dicts {box (padded, crop coords), core (tight box), score
    (mean signal strength -- informational, nothing is dropped on it), method, scale}. candidate_mask: union of every
    anomaly mask. `opts` overrides config values (lowercase names, e.g. min_fill=0.1)."""
    o = lambda name: opts.get(name.lower(), getattr(cfg, name.upper()))
    full_h, full_w = skin_mask.shape

    # detect at a capped working resolution: much faster on 3000px+ crops, and area-averaging the downscale suppresses
    # per-pixel sensor noise that otherwise fragments every blob. Boxes are mapped back to full resolution afterwards.
    f = min(1.0, o("work_max_side") / max(full_h, full_w))
    if f < 1.0:
        crop_bgr = cv2.resize(crop_bgr, (int(full_w * f), int(full_h * f)), interpolation=cv2.INTER_AREA)
        small_skin = cv2.resize(skin_mask, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        small_skin = skin_mask
    face_h, face_w = small_skin.shape
    dim = min(face_h, face_w)

    # evaluate only *interior* skin: right at class edges the masked baseline has too few real skin samples nearby to be
    # reliable and produces false "deviation" along the boundary.
    erode_px = max(2, int(dim * o("interior_erode_frac")))
    interior = cv2.erode(small_skin, np.ones((erode_px * 2 + 1,) * 2, np.uint8)) > 0
    if not interior.any():
        return [], np.zeros_like(skin_mask)
    interior_f = interior.astype(np.float32)

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    face_area = int((small_skin > 0).sum())
    min_area = max(o("min_candidate_area_px"), int(o("min_candidate_area_frac") * face_area))
    max_area = int(o("max_candidate_area_frac") * face_area)
    merge_px = max(1, int(o("merge_distance_frac") * dim / 2))

    cands, cand_mask = [], np.zeros((face_h, face_w), np.uint8)
    for scale in o("baseline_scales"):
        k = max(9, int(dim * scale) | 1)
        diff = _baseline_diff(lab, small_skin, k)
        sigma = max(0.8, o("smooth_frac_of_scale") * k)  # lesion-sized smoothing: blobs, not per-pixel noise
        norm = cv2.GaussianBlur(interior_f, (0, 0), sigma) + 1e-6
        for name in o("signals"):
            sig = _signal(name, diff) * interior_f
            sig = cv2.GaussianBlur(sig, (0, 0), sigma) / norm * interior_f
            floor = (o("deviation_floor") if name == "deviation" else o("channel_floor")) * (o("coarse_floor_mult") if scale >= o("coarse_scale_from") else 1.0)
            cutoff = max(np.percentile(sig[interior], o(name + "_percentile")), floor)
            mask = (sig > cutoff).astype(np.uint8)
            cand_mask |= mask
            for x0, y0, x1, y1, score in _extract(mask, sig, min_area, max_area, o("max_box_side_mult") * k, o("min_box_side_frac") * dim, merge_px,
                                                  o("oversize_resplit_depth"), o):
                cands.append({"core": (x0, y0, x1, y1), "box": (x0, y0, x1, y1), "score": float(score), "strength": float(score / cutoff), "method": name, "scale": scale})

    kept = _dedupe(cands, o("nms_iou_threshold"), o("nms_containment"))
    if f < 1.0:
        cand_mask = cv2.resize(cand_mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    for c in kept:
        x0, y0, x1, y1 = c["core"]
        if f < 1.0:
            x0, y0, x1, y1 = int(x0 / f), int(y0 / f), min(full_w, int(np.ceil(x1 / f))), min(full_h, int(np.ceil(y1 / f)))
        c["core"] = (x0, y0, x1, y1)
        # padding is for the downstream model's context (lesion + surrounding skin), not part of the evidence
        c["box"] = _pad_inside(c["core"], skin_mask, o("box_padding_ratio"))
    return kept, cand_mask * 255


def find_spots(crop_bgr, skin_mask, **opts):
    """Original interface: list of (x0, y0, x1, y1) boxes in crop_bgr's coordinate frame. Legacy keyword names
    (min_area_frac, max_area_frac, percentile) still work."""
    legacy = {"min_area_frac": "min_candidate_area_frac", "max_area_frac": "max_candidate_area_frac", "percentile": "deviation_percentile"}
    opts = {legacy.get(k, k): v for k, v in opts.items()}
    return [c["box"] for c in find_candidates(crop_bgr, skin_mask, **opts)[0]]


def draw_spots(crop_bgr, boxes, color=(0, 255, 0), thickness=2):
    out = crop_bgr.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


if __name__ == "__main__":
    import sys
    from face_segment import analyze_face

    img = cv2.imread(sys.argv[1])
    analysis = analyze_face(img)
    if analysis is None:
        print("no face detected")
        sys.exit()

    boxes = find_spots(analysis["crop"], skin_only_mask(analysis["class_mask"], analysis["crop"]))
    print(f"found {len(boxes)} spots")
    out = draw_spots(analysis["crop_original"], boxes)
    out_path = sys.argv[2] if len(sys.argv) > 2 else "spots.jpg"
    cv2.imwrite(out_path, out)
    print("saved", out_path)
