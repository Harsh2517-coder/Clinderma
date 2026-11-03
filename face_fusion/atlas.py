"""Stitches several photos of one face into one unwrapped texture (u = angle around the head, v = height), blending each photo where its surface faces the camera."""
import cv2, numpy as np
from . import fuse as F

MAX_SIDE = 1800


def stitch(m, views, images, size=1024, feather=21):
    """views: fitted View list; images: {view.name: BGR}. Returns (atlas BGR uint8, weight map float, per-view weight dict)."""
    uv = m["uv"] * (size - 1); T = m["tris"]; ref = F._signed_area(m["C"][T][:, :, :2]); acc = np.zeros((size, size, 3), np.float32); wsum = np.zeros((size, size), np.float32); per = {}
    for v in views:
        img = images[v.name]; s = min(1.0, MAX_SIDE / max(img.shape[:2])); small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        A = v.P[T][:, :, :2] * s; ok = (v.vis >= F.VIS_MIN) & (np.sign(F._signed_area(A)) == np.sign(ref)) & m["tri_skin"]
        layer = np.zeros((size, size, 3), np.float32); wl = np.zeros((size, size), np.float32)
        for ti in np.nonzero(ok)[0]:
            src = A[ti].astype(np.float32); dst = uv[T[ti]].astype(np.float32); x, y, w, h = cv2.boundingRect(np.round(dst).astype(np.int32)); x0, y0 = max(x - 1, 0), max(y - 1, 0)
            x1, y1 = min(x + w + 2, size), min(y + h + 2, size)
            if x1 <= x0 or y1 <= y0: continue
            Mx = cv2.getAffineTransform(src, dst - np.array([x0, y0], np.float32)); patch = cv2.warpAffine(small, Mx, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillConvexPoly(mask, np.round(dst - np.array([x0, y0])).astype(np.int32), 1, lineType=cv2.LINE_AA); mk = mask.astype(bool)
            layer[y0:y1, x0:x1][mk] = patch[mk]; wl[y0:y1, x0:x1][mk] = v.vis[ti] ** 2
        wl = cv2.GaussianBlur(wl, (0, 0), feather / 3) * (cv2.GaussianBlur((wl > 0).astype(np.float32), (0, 0), feather / 3) > 0.5)          # feathered edges
        acc += layer * wl[..., None]; wsum += wl; per[v.name] = wl
    have = (wsum > 1e-4).astype(np.float32); atlas = acc / np.maximum(wsum[..., None], 1e-4)
    if have.any():                                                      # parts no photo covers: smooth skin-tone fill from the surrounding texture, so the 3D map has no black holes (invented, not measured)
        k = max(31, size // 6) | 1; fill = cv2.GaussianBlur(atlas * have[..., None], (0, 0), k / 3) / np.maximum(cv2.GaussianBlur(have, (0, 0), k / 3)[..., None], 1e-3)
        atlas = np.where(have[..., None] > 0, atlas, fill)
    return np.clip(atlas, 0, 255).astype(np.uint8), wsum, per
