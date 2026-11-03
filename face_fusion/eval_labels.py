"""Same test as eval_fusion.py but with HUMAN-LABELLED lesions (YOLO labels) instead of detector output: photos of one patient taken within one visit, lesion centres from the labels.
The share of labelled lesions in the overlap of two angles that have a labelled partner at the same place on the face model, against the chance level (different patients).
    python3 -m face_fusion.eval_labels <clean_dataset_dir>   (images/ + labels/, file names <patient>_<hhmmss>.jpg)"""
import collections, itertools, os, random, re, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_fusion import landmarks as LM, model as M, fuse as F

root = sys.argv[1]; m = M.load(); random.seed(5); grp = lambda f: re.sub(r"_\d{6}\.\w+$", "", f); G = collections.defaultdict(list)
for f in sorted(os.listdir(f"{root}/images")): G[grp(f)].append(f)
G = {k: v for k, v in G.items() if 2 <= len(v) <= 8 and not k.startswith("v3_batch")}          # one patient, one visit
views = {}
for k, fs in G.items():
    imgs = {f: cv2.imread(f"{root}/images/{f}") for f in fs}; lms = LM.detect(imgs, m); vs = []
    for f, im in imgs.items():
        if lms[f] is None: continue
        lab = [list(map(float, l.split()))[1:3] for l in open(f"{root}/labels/{os.path.splitext(f)[0]}.txt") if l.strip()]
        if not lab: continue
        v = F.fit_view(m, f"{k}:{f}", lms[f], im.shape[1], im.shape[0]); v.spots = [{"x": x * im.shape[1], "y": y * im.shape[0]} for x, y in lab]; vs.append(v)
    if len(vs) >= 2: views[k] = vs
print(len(views), "patients with 2+ usable labelled photos;", sum(len(v.spots) for vs in views.values() for v in vs), "labelled lesions")
real = [(a, b) for vs in views.values() for a, b in itertools.combinations(vs, 2) if abs(a.yaw - b.yaw) >= 20]
null = []
for a, b in real:
    for _ in range(5):
        k2 = random.choice(list(views)); 
        if k2 == b.name.split(":")[0] or k2 == a.name.split(":")[0]: continue
        null.append((a, min(views[k2], key=lambda v: abs(v.yaw - b.yaw))))
def stats(pairs, r):
    mt = ov = 0
    for a, b in pairs:
        res = F.fuse(m, [a, b], r)["pairs"][f"{a.name}|{b.name}"]; mt += res["matched"]; ov += min(res["spots_in_overlap_a"], res["spots_in_overlap_b"])
    return mt, ov
print(len(real), "pairs with >=20 degrees head-angle difference;", len(null), "different-patient pairs")
print("radius   same patient: matched / lesions in overlap     chance (different patients)")
for r in (0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
    a1, b1 = stats(real, r); a0, b0 = stats(null, r); print(f"{r:5.2f}    {a1:4d} / {b1:4d} = {100 * a1 / max(b1, 1):5.1f}%                 {a0:4d} / {b0:4d} = {100 * a0 / max(b0, 1):5.1f}%")
