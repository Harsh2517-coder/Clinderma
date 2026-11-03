"""One-off: builds models/face_canonical.npz from frontal photos' landmarks.   <python-with-mediapipe> build_canonical.py landmarks.json photos_list.txt
(needs mediapipe only for the mesh topology; the result is a plain .npz that the rest of face_fusion loads with numpy)."""
import json, sys
import cv2, numpy as np
from mediapipe.python.solutions import face_mesh_connections as c
sys.path.insert(0, __file__.rsplit("/", 2)[0]); from face_fusion import model as M

res = json.load(open(sys.argv[1])); samples = []; meta = []
def ids(s): return sorted({i for e in s for i in e})
for r in res:
    if not r["ok"] or r["flipped"]: continue
    m = np.array(r["pose"])[:3, :3]; m = m / np.linalg.norm(m, axis=0); yaw = np.degrees(np.arctan2(-m[2, 0], np.hypot(m[2, 1], m[2, 2])))
    if abs(yaw) > 10: continue
    im = cv2.imread(r["path"]); h, w = im.shape[:2]; samples.append(M.points(r["lm"], w, h)); meta.append(r["path"])
print("frontal samples:", len(samples))
tris = M.edges_to_triangles(set(tuple(sorted(e)) for e in c.FACEMESH_TESSELATION))
regions = {"eyes": ids(c.FACEMESH_LEFT_EYE) + ids(c.FACEMESH_RIGHT_EYE), "lips": ids(c.FACEMESH_LIPS), "irises": ids(c.FACEMESH_IRISES),
           "brows": ids(c.FACEMESH_LEFT_EYEBROW) + ids(c.FACEMESH_RIGHT_EYEBROW), "nose": ids(c.FACEMESH_NOSE)}
d = M.build(samples, tris, regions); d["regions"] = np.array(json.dumps(regions))
zone, skin, area = M.zones_of_triangles(d["C"], d["tris"], regions, int(d["n_mesh_tris"])); d["tri_zone"] = zone; d["tri_skin"] = skin; d["tri_area"] = area
print("triangles:", len(d["tris"]), "(of which", len(d["tris"]) - int(d["n_mesh_tris"]), "close the eyes/mouth/nostrils)", "| mirror partner is an exact involution for", round(d["involution"] * 100, 1), "% of landmarks")
print("zone surface share (skin only):", {z: round(float(area[(zone == z) & skin].sum() / area[skin].sum()), 3) for z in M.ZONES})
M.save(d); print("saved", M.PATH)
