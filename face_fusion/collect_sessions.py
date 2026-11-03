"""Evaluation helper: for every photo session (folder of photos of one patient on one day) pick up to 3 photos with different head angles, run the platform's spot scan on each and
save the spots + landmarks fit to data/fusion_work/sessions.json (so evaluation can be repeated without rescanning).   python3 -m face_fusion.collect_sessions <root> [<root> ...]"""
import collections, glob, json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_fusion import landmarks as LM, model as M, fuse as F
import scan

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fusion_work", "sessions.json")
sess = collections.defaultdict(list)
for root in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "*.jpg"))): sess["/".join(f.split("/")[-3:-1])].append(f)
m = M.load(); done = json.load(open(OUT)) if os.path.exists(OUT) else {}; t0 = time.time()
for k, files in sorted(sess.items()):
    if k in done: continue
    imgs = {os.path.basename(f): cv2.imread(f) for f in files}; imgs = {n: i for n, i in imgs.items() if i is not None}
    lms = LM.detect(imgs, m); fits = {n: F.fit_view(m, n, lms[n], i.shape[1], i.shape[0]) for n, i in imgs.items() if lms[n] is not None}
    # one photo per head-angle bin (best fit in each 20-degree bin), then the three most different
    best = {}
    for n, v in fits.items():
        b = int(round(v.yaw / 20))
        if b not in best or v.resid < fits[best[b]].resid: best[b] = n
    pick = sorted(best.values(), key=lambda n: fits[n].yaw)
    if len(pick) > 3: pick = [pick[0], min(pick, key=lambda n: abs(fits[n].yaw)), pick[-1]]
    if len(pick) < 2 or max(fits[n].yaw for n in pick) - min(fits[n].yaw for n in pick) < 25: done[k] = {"skip": f"{len(fits)} usable photos, no angle spread"}; json.dump(done, open(OUT, "w")); continue
    ph = []
    for n in pick:
        s = scan.scan_image(imgs[n])
        if s["status"] != "ok": continue
        ph.append({"photo": n, "yaw": round(fits[n].yaw, 1), "fit_error": round(fits[n].resid, 3), "W": imgs[n].shape[1], "H": imgs[n].shape[0], "lm": np.asarray(lms[n]).round(5).tolist(),
                   "spots": s["spots"], "zone_counts": s["zone_counts"]})
    done[k] = {"photos": ph}; json.dump(done, open(OUT, "w")); print(k, [(p["yaw"], len(p["spots"])) for p in ph], "%.0fs" % (time.time() - t0), flush=True)
