"""One call: photos + the spots found in each -> fused result. `spots[name]` is a list of dicts with x, y (photo pixels) and anything else you want to keep (box, conf, ...)."""
from . import model as M, fuse as F, landmarks as LM


def fuse_photos(images, spots, radius=F.MATCH_RADIUS):
    """images {name: BGR}, spots {name: [{x, y, ...}]}. Photos whose face mesh cannot be fitted are left out and listed in `skipped` (their spots stay as per-photo counts)."""
    m = M.load(); lms = LM.detect(images, m); views, skipped = [], []
    for name, img in images.items():
        if lms[name] is None: skipped.append(name); continue
        v = F.fit_view(m, name, lms[name], img.shape[1], img.shape[0]); v.spots = spots.get(name, []); views.append(v)
    res = F.fuse(m, views, radius) if views else None
    return {"result": res, "skipped": skipped, "views": views, "model": m}
