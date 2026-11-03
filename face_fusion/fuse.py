"""Puts the spots found in several photos of one face onto the shared face model (face_fusion.model) and merges the ones that are the same spot.
Pure numpy/opencv; works with any spot detector (give it spot centres in photo pixels)."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import model as M

VIS_MIN = 0.25            # a triangle counts as visible in a photo when its surface faces the camera at least this much (cos of the angle)
MATCH_RADIUS = 0.04       # two spots from different photos are the same spot when closer than this on the face model (face width = 1, so about 5-6 mm); chosen on labelled lesions, see eval_labels.py


@dataclass
class View:
    name: str; W: int; H: int; P: np.ndarray; R: np.ndarray; s: float; t: np.ndarray; yaw: float; vis: np.ndarray; resid: float
    spots: list = field(default_factory=list)


def tri_normals(m):
    C, T = m["C"], m["tris"]; n = np.cross(C[T[:, 1]] - C[T[:, 0]], C[T[:, 2]] - C[T[:, 0]]); return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def fit_view(m, name, lm, W, H):
    """Pose of the canonical face in one photo (similarity fit of the canonical landmarks to the photo's), and how well each triangle faces the camera."""
    P = M.points(lm, W, H); w = np.ones(len(P))
    for _ in range(3):                                                           # robust: down-weight landmarks that disagree (the hidden half of a turned face is guessed)
        s, R, t = M.similarity(m["C"], P, w); r = np.linalg.norm(s * m["C"] @ R.T + t - P, axis=1); w = 1 / (1 + (r / (np.median(r) + 1e-9)) ** 2)
    ncam = tri_normals(m) @ R.T; vis = np.clip(-ncam[:, 2], 0, 1)               # camera looks along +z; a surface facing it has its outward normal along -z
    nose_dir = R @ np.array([0, 0, -1.0]); yaw = float(np.degrees(np.arctan2(nose_dir[0], -nose_dir[2])))
    return View(name, W, H, P, R, float(s), t, yaw, vis, float(np.median(r) / np.linalg.norm(P[M.EAR_L, :2] - P[M.EAR_R, :2])))


def _signed_area(xy): return (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])


def locate(m, v, xy):
    """Photo points (n, 2) -> position on the face model. Returns dict of arrays: tri (-1 if none), bary, pos (n,3), uv (n,2), vis (n), extrapolated (n)."""
    xy = np.asarray(xy, float).reshape(-1, 2); T = m["tris"]; A = v.P[T][:, :, :2]                              # (t,3,2) triangle corners in the photo
    ref = _signed_area(m["C"][T][:, :, :2]); ok = (v.vis >= VIS_MIN) & (np.sign(_signed_area(A)) == np.sign(ref)) & m["tri_skin"]   # facing us AND not folded over
    a, b, c = A[:, 0], A[:, 1], A[:, 2]; den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]); den = np.where(abs(den) < 1e-9, 1e-9, den)
    px, py = xy[:, 0][:, None], xy[:, 1][:, None]
    l1 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / den; l2 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / den; l3 = 1 - l1 - l2
    B = np.stack([l1, l2, l3], -1)                                                                           # (n,t,3)
    inside = (B.min(-1) >= -1e-6) & ok[None, :]; score = np.where(inside, v.vis[None, :], -1.0); best = score.argmax(1); found = score.max(1) >= 0
    tri = np.where(found, best, -1); extrap = np.zeros(len(xy), bool)
    for i in np.nonzero(~found)[0]:                                                                          # outside the mesh (hairline, ear): extend the nearest facing triangle
        cand = np.nonzero(ok)[0]
        if not len(cand): continue
        d = np.linalg.norm(A[cand].mean(1) - xy[i], axis=1); j = cand[d.argmin()]
        if B[i, j].min() > -1.5: tri[i] = j; extrap[i] = True
    bary = np.where((tri >= 0)[:, None], B[np.arange(len(xy)), np.maximum(tri, 0)], np.nan)
    C, uv = m["C"], m["uv"]; pos = np.full((len(xy), 3), np.nan); uvp = np.full((len(xy), 2), np.nan); g = tri >= 0
    pos[g] = np.einsum("ni,nij->nj", bary[g], C[T[tri[g]]]); uvp[g] = np.einsum("ni,nij->nj", bary[g], uv[T[tri[g]]])
    return {"tri": tri, "bary": bary, "pos": pos, "uv": uvp, "vis": np.where(g, v.vis[np.maximum(tri, 0)], 0.0), "extrapolated": extrap}


def fuse(m, views, radius=MATCH_RADIUS):
    """views: list of View, each with .spots = [{x, y, ...}] in photo pixels. Returns the unique spots and the per-zone result."""
    L = []
    for v in views:
        loc = locate(m, v, [(s["x"], s["y"]) for s in v.spots]) if v.spots else None; L.append(loc)
    nodes = []                                                                       # (view index, spot index)
    for vi, v in enumerate(views): nodes += [(vi, si) for si in range(len(v.spots))]
    parent = {n: n for n in nodes}
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    pairs = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            if L[i] is None or L[j] is None: pairs[f"{views[i].name}|{views[j].name}"] = {"matched": 0, "spots_in_overlap_a": 0, "spots_in_overlap_b": 0}; continue
            pi, pj = L[i]["pos"], L[j]["pos"]; gi, gj = np.nonzero(~np.isnan(pi[:, 0]))[0], np.nonzero(~np.isnan(pj[:, 0]))[0]; matched = 0
            if len(gi) and len(gj):
                d = np.linalg.norm(pi[gi][:, None] - pj[gj][None], axis=2); r, c = linear_sum_assignment(np.where(d < radius, d, 1e3))
                for a, b in zip(r, c):
                    if d[a, b] < radius: parent[find((i, int(gi[a])))] = find((j, int(gj[b]))); matched += 1
            overlap_i = int(sum(1 for k in gi if views[j].vis[L[i]["tri"][k]] >= VIS_MIN)); overlap_j = int(sum(1 for k in gj if views[i].vis[L[j]["tri"][k]] >= VIS_MIN))
            pairs[f"{views[i].name}|{views[j].name}"] = {"matched": matched, "spots_in_overlap_a": overlap_i, "spots_in_overlap_b": overlap_j}
    groups = {}
    for n in nodes: groups.setdefault(find(n), []).append(n)
    spots = []
    for g in groups.values():
        pts = [(L[vi]["pos"][si], L[vi]["vis"][si], vi, si) for vi, si in g if not np.isnan(L[vi]["pos"][si][0])]
        if not pts:                                                                  # spot outside the mapped face: keep it, unplaced
            vi, si = g[0]; spots.append({"zone": None, "orig_zone": views[vi].spots[si].get("zone"), "pos": None, "uv": None, "n_views": len(g), "views": {views[a].name: b for a, b in g}}); continue
        w = np.array([p[1] + 1e-3 for p in pts]); pos = (w[:, None] * np.array([p[0] for p in pts])).sum(0) / w.sum(); best = max(pts, key=lambda p: p[1]); vi, si = best[2], best[3]
        zone = str(m["tri_zone"][L[vi]["tri"][si]]); uv = (w[:, None] * np.array([L[a]["uv"][b] for _, _, a, b in pts])).sum(0) / w.sum()
        spots.append({"zone": zone, "orig_zone": views[vi].spots[si].get("zone"), "pos": pos.tolist(), "uv": uv.tolist(), "n_views": len(g), "views": {views[a].name: b for a, b in g}, "extrapolated": bool(any(L[a]["extrapolated"][b] for a, b in g))})
    zc = {z: 0 for z in M.ZONES}
    for s in spots:
        if s["zone"]: zc[s["zone"]] += 1
    skin = m["tri_skin"]; area = {z: float(m["tri_area"][(m["tri_zone"] == z) & skin].sum()) for z in M.ZONES}
    cov1 = {z: 0.0 for z in M.ZONES}; cov2 = {z: 0.0 for z in M.ZONES}
    if views:
        V = np.stack([v.vis >= VIS_MIN for v in views]); n_seen = V.sum(0)
        for z in M.ZONES:
            msk = (m["tri_zone"] == z) & skin; tot = m["tri_area"][msk].sum(); cov1[z] = float(m["tri_area"][msk & (n_seen >= 1)].sum() / tot); cov2[z] = float(m["tri_area"][msk & (n_seen >= 2)].sum() / tot)
    raw = sum(len(v.spots) for v in views)
    return {"spots": spots, "unique_total": len(spots), "raw_total": raw, "merged": raw - len(spots), "zone_counts": zc, "zone_area": area, "coverage_one_view": cov1, "coverage_two_views": cov2, "pairs": pairs,
            "views": [{"name": v.name, "yaw": round(v.yaw, 1), "fit_error": round(v.resid, 3)} for v in views]}
