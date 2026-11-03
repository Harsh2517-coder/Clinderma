"""Does matching spots across photos on the shared face model work?  Uses data/fusion_work/sessions.json (from collect_sessions.py).
For photo pairs of the SAME patient the share of spots in the overlap that find a partner is compared with the same measure for photos of DIFFERENT patients (the chance level).
    python3 -m face_fusion.eval_fusion"""
import itertools, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_fusion import model as M, fuse as F

D = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fusion_work", "sessions.json"))); m = M.load(); random.seed(3)


def view(p, name=None):
    v = F.fit_view(m, name or p["photo"], np.array(p["lm"]), p["W"], p["H"]); v.spots = [{"x": s["x"], "y": s["y"]} for s in p["spots"]]; return v


sess = {k: v["photos"] for k, v in D.items() if "photos" in v and len(v["photos"]) >= 2}
print(len(sess), "sessions with 2-3 usable angles;", sum(len(p["spots"]) for ps in sess.values() for p in ps), "spots in total")


def pair_stats(pairs, radius):
    matched = overl = 0; dists = []
    for a, b in pairs:
        va, vb = a, b; r = F.fuse(m, [va, vb], radius); st = r["pairs"][f"{va.name}|{vb.name}"]
        matched += st["matched"]; overl += min(st["spots_in_overlap_a"], st["spots_in_overlap_b"])
    return matched, overl


real = []
for k, ps in sess.items():
    vs = [view(p, f"{k}:{p['photo']}") for p in ps]
    for a, b in itertools.combinations(vs, 2): real.append((a, b))
null = []                                      # same angle types, different patients: view a of one patient with view b (similar head angle) of another
keys = list(sess)
for a, b in real:
    for _ in range(5):
        k2 = random.choice(keys)
        if k2 == a.name.split(":")[0]: continue
        cand = [p for p in sess[k2]]; yaw_b = b.yaw; p2 = min(cand, key=lambda p: abs(view(p, "x").yaw - yaw_b)); vb2 = view(p2, f"{k2}:{p2['photo']}"); null.append((a, vb2))
print(len(real), "real pairs;", len(null), "different-patient pairs (chance level)")
print("radius   same patient: matched / spots in overlap       different patients (chance)")
for r in (0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
    m1, o1 = pair_stats(real, r); m0, o0 = pair_stats(null, r)
    print(f"{r:5.2f}    {m1:4d} / {o1:4d} = {100 * m1 / max(o1, 1):5.1f}%                   {m0:4d} / {o0:4d} = {100 * m0 / max(o0, 1):5.1f}%")
tot_raw = tot_u = 0
for k, ps in sess.items():
    r = F.fuse(m, [view(p, f"{k}:{p['photo']}") for p in ps]); tot_raw += r["raw_total"]; tot_u += r["unique_total"]
print(f"\nspots summed over photos: {tot_raw}  ->  unique spots after merging (radius {F.MATCH_RADIUS}): {tot_u}  ({100 * (tot_raw - tot_u) / max(tot_raw, 1):.1f}% were duplicates)")
