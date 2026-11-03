"""The shared face model every photo is mapped onto: canonical 3D landmark positions (478 x [x, y, z], face width = 1, x to the person's image-left, y down, z away from the camera),
the mesh triangles, the left/right mirror partner of every landmark, the zone and area of every triangle, and the cylinder unwrap (u, v) used for the stitched texture."""
import itertools, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PATH = os.path.join(HERE, "..", "models", "face_canonical.npz")
EAR_L, EAR_R = 234, 454            # face-oval landmarks beside the ears (width reference)
ZONES = ("forehead", "nose", "chin", "right_cheek", "left_cheek")


def similarity(src, dst, weights=None):
    """Least-squares scale s, rotation R, translation t with dst ~ s * src @ R.T + t (Umeyama). src, dst: (n, 3)."""
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float); w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0); S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd((w[:, None] * D).T @ S); d = np.sign(np.linalg.det(U @ Vt)); E = np.diag([1, 1, d])
    R = U @ E @ Vt; s = (sig * np.diag(E)).sum() / (w[:, None] * S ** 2).sum(); return s, R, md - s * ms @ R.T


def points(lm, w, h):
    """normalised landmarks (478, 3) -> pixel-like 3D points of the photo (z uses the same scale as x, as MediaPipe defines it)."""
    p = np.asarray(lm, float).copy(); p[:, 0] *= w; p[:, 1] *= h; p[:, 2] *= w; return p


def edges_to_triangles(edges):
    """Triangles of the MediaPipe tessellation: every 3-clique of its edges (this gives 854, two are bridges removed later in build())."""
    import collections
    adj = collections.defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    return sorted({tuple(sorted((a, b, k))) for a, b in edges for k in adj[a] & adj[b]})


def build(samples, tris, regions, iters=5):
    """samples: list of (n,3) frontal-ish pixel-like landmark sets. Generalised Procrustes -> mean shape, scaled so the ear-to-ear width is 1.
    regions: dict name -> set of landmark ids (eyes, lips, nose, brows). Returns a dict of arrays (see module docstring)."""
    ref = samples[0] - samples[0].mean(0); aligned = [s for s in samples]
    for _ in range(iters):
        aligned = []
        for s in samples:
            sc, R, t = similarity(s, ref); aligned.append(sc * s @ R.T + t)
        new = np.mean(aligned, 0); new -= new.mean(0); ref = new
    C = ref.copy(); C /= np.linalg.norm(C[EAR_L] - C[EAR_R])
    # orient: x towards the person's image-left? keep the photo convention: x right in the image, y down, z away from the camera (frontal face looks towards -z)
    if C[EAR_L, 0] > C[EAR_R, 0]: C[:, 0] *= -1; C[:, 2] *= -1                       # make landmark 234 the image-left ear
    if C[1, 2] > C[EAR_L, 2]: C[:, 2] *= -1                                          # nose tip (1) must be closer to the camera than the ears
    C[:, 0] -= (C[EAR_L, 0] + C[EAR_R, 0]) / 2
    # mirror partner of each landmark: nearest landmark to its mirror image
    from scipy.optimize import linear_sum_assignment
    M = C * np.array([-1, 1, 1]); d = ((M[:, None, :] - C[None, :, :]) ** 2).sum(-1); perm = linear_sum_assignment(d)[1]       # one-to-one mirror pairing
    involution = float((perm[perm] == np.arange(len(C))).mean())
    C = (C + (C * np.array([-1, 1, 1]))[perm]) / 2; C[:, 0] = (C[:, 0] - C[perm, 0]) / 2                          # exactly left/right symmetric
    # outward-facing triangles (away from a point inside the head) and removal of the two bridge triangles
    head = np.array([0.0, C[:, 1].mean(), C[:, 2].max() + 0.1 + 0.25]); tri = np.array(tris)
    n = np.cross(C[tri[:, 1]] - C[tri[:, 0]], C[tri[:, 2]] - C[tri[:, 0]]); cen = C[tri].mean(1)
    flip = (n * (cen - head)).sum(1) < 0; tri[flip] = tri[flip][:, [0, 2, 1]]
    use = {}
    for ti, t in enumerate(tri):
        for e in itertools.combinations(sorted(t), 2): use.setdefault(e, []).append(ti)
    drop = set()
    for e, ts in use.items():
        if len(ts) > 2:                                                     # three triangles on one edge: drop the one whose far vertex lies farthest from the edge
            far = [np.linalg.norm(C[[v for v in tri[ti] if v not in e][0]] - (C[e[0]] + C[e[1]]) / 2) for ti in ts]; drop.add(ts[int(np.argmax(far))])
    keep = [i for i in range(len(tri)) if i not in drop]; tri = tri[keep]
    return {"C": C, "tris": tri, "perm": perm, "involution": involution}


def cylinder(C):
    """(u, v) in 0..1 for every landmark: angle around a vertical axis behind the face (ear to ear = +-80 deg) and height."""
    z_axis = C[[EAR_L, EAR_R], 2].mean() + 0.5 / np.tan(np.radians(80))
    theta = np.arctan2(C[:, 0], z_axis - C[:, 2]); u = (theta + np.radians(100)) / np.radians(200)
    y0, y1 = C[:, 1].min() - 0.02, C[:, 1].max() + 0.02; return np.stack([u, (C[:, 1] - y0) / (y1 - y0)], 1), (z_axis, y0, y1)


def zones_of_triangles(C, tris, regions):
    """Zone of every triangle (forehead above the brows, chin below the lips, nose = a polygon over the nose bridge and wings, cheeks left/right of the middle) and whether it is
    skin (not eye opening / lip). 'left_cheek' is the person's LEFT cheek, i.e. image-right in a frontal photo, matching scan.py."""
    import cv2
    cen = C[tris].mean(1); brow_y = np.mean([C[i, 1] for i in regions["brows"]]); lip_y = max(C[i, 1] for i in regions["lips"])
    top, base, wing = C[168], C[2], max(abs(C[129, 0]), abs(C[358, 0])) + 0.025
    poly = np.array([[-0.05, top[1]], [0.05, top[1]], [wing, base[1] - 0.04], [wing, base[1] + 0.03], [-wing, base[1] + 0.03], [-wing, base[1] - 0.04]], np.float32) * 1000
    in_nose = np.array([cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(c[0] * 1000), float(c[1] * 1000)), False) >= 0 for c in cen])
    eyes_lips = set(regions["eyes"]) | set(regions["lips"]) | set(regions["irises"])
    skin = ~np.array([all(v in eyes_lips for v in t) for t in tris])
    zone = np.where(in_nose, "nose", np.where(cen[:, 1] < brow_y, "forehead", np.where(cen[:, 1] > lip_y, "chin", np.where(cen[:, 0] < 0, "right_cheek", "left_cheek"))))
    area = 0.5 * np.linalg.norm(np.cross(C[tris[:, 1]] - C[tris[:, 0]], C[tris[:, 2]] - C[tris[:, 0]]), axis=1)
    return zone, skin, area


def save(d, path=PATH): np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})


def load(path=PATH):
    z = np.load(path, allow_pickle=False); d = {k: z[k] for k in z.files}; d["uv"], d["cyl"] = cylinder(d["C"]); return d
