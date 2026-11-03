"""Stitches several photos of one face into one unwrapped texture (u = angle around the head, v = height).
Quality: each photo is cropped to the face and used at (nearly) full resolution, warped with bicubic sampling, and the photos are blended with a strong preference for the one facing the
camera best (so overlapping photos do not average into a blur); brightness differences between photos are evened out before blending."""
import cv2, numpy as np
from . import fuse as F

MAX_CROP = 4000          # longest side (px) of the face crop used from a photo; larger photos are scaled down to this
GAIN = (0.7, 1.4)        # limits of the brightness correction between photos


def _prepare(v, img, m):
    """Crop the photo to the face (all visible mesh points + margin) and scale it down only if it is huge. Returns (crop, scale, offset)."""
    P = v.P[:, :2]; x0, y0 = np.floor(P.min(0) - 0.08 * np.ptp(P, 0)).astype(int); x1, y1 = np.ceil(P.max(0) + 0.08 * np.ptp(P, 0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0]); crop = img[y0:y1, x0:x1]; s = min(1.0, MAX_CROP / max(crop.shape[:2]))
    return (cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop), s, np.array([x0, y0], float)


def stitch(m, views, images, size=2048, power=4.0):
    """views: fitted View list; images: {view.name: BGR}. Returns (atlas BGR uint8, total weight map float, per-view weight dict)."""
    uv = m["uv"] * (size - 1); T = m["tris"]; ref = F._signed_area(m["C"][T][:, :, :2]); layers = {}; weights = {}; feather = max(5.0, size / 170)
    for v in views:
        crop, s, off = _prepare(v, images[v.name], m); A = (v.P[T][:, :, :2] - off) * s; Afull = v.P[T][:, :, :2]
        ok = (v.vis >= F.VIS_MIN) & (np.sign(F._signed_area(Afull)) == np.sign(ref))                  # all triangles, including those closing eyes and mouth
        layer = np.zeros((size, size, 3), np.float32); wl = np.zeros((size, size), np.float32)
        for ti in np.nonzero(ok)[0]:
            src = A[ti].astype(np.float32); dst = uv[T[ti]].astype(np.float32); x, y, w, h = cv2.boundingRect(np.round(dst).astype(np.int32)); x0, y0 = max(x - 1, 0), max(y - 1, 0)
            x1, y1 = min(x + w + 2, size), min(y + h + 2, size)
            if x1 <= x0 or y1 <= y0: continue
            Mx = cv2.getAffineTransform(src, dst - np.array([x0, y0], np.float32)); patch = cv2.warpAffine(crop, Mx, (x1 - x0, y1 - y0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillConvexPoly(mask, np.round(dst - np.array([x0, y0])).astype(np.int32), 1, lineType=cv2.LINE_AA); mk = mask.astype(bool)
            layer[y0:y1, x0:x1][mk] = patch[mk]; wl[y0:y1, x0:x1][mk] = v.vis[ti] ** power
        cov = (wl > 0).astype(np.float32); wl = cv2.GaussianBlur(wl, (0, 0), feather / 2) * (cv2.GaussianBlur(cov, (0, 0), feather / 2) > 0.5)
        layers[v.name], weights[v.name] = layer, wl
    names = [v.name for v in views]
    if len(names) > 1:                                                                  # even out brightness between photos using their overlap, relative to the photo that covers most
        base = max(names, key=lambda n: weights[n].sum()); both_min = 0.15
        for n in names:
            if n == base: continue
            ov = (weights[n] > both_min * weights[n].max()) & (weights[base] > both_min * weights[base].max())
            if ov.sum() > 2000: layers[n] *= np.clip(layers[base][ov].mean(0) / np.maximum(layers[n][ov].mean(0), 1e-3), *GAIN)
    acc = sum(layers[n] * weights[n][..., None] for n in names); wsum = sum(weights[n] for n in names); have = (wsum > 1e-6).astype(np.float32); atlas = acc / np.maximum(wsum[..., None], 1e-6)
    if have.any():                                                      # parts no photo covers: smooth skin-tone fill from the surrounding texture (invented, not measured)
        k = max(31, size // 6) | 1; fill = cv2.GaussianBlur(atlas * have[..., None], (0, 0), k / 3) / np.maximum(cv2.GaussianBlur(have, (0, 0), k / 3)[..., None], 1e-3)
        atlas = np.where(have[..., None] > 0, atlas, fill)
    return np.clip(atlas, 0, 255).astype(np.uint8), wsum, weights
