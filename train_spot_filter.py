"""Trains the spot filter: keeps only proposed spots that look like real acne-type lesions.  python3 train_spot_filter.py
Validation is person-grouped 5-fold CV (a person's photos never sit on both sides).  Reports, per threshold: lesion recall, spot precision, and how often the zone the scan calls
'most affected' (by spot density) matches the zone with the most expert-labelled lesions.  Saves models/spot_filter.joblib and results/spot_filter_cv.json."""
import glob, json, os, pickle
import joblib, numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ZONES = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]
images, rows = [], []
for f in sorted(glob.glob("results/spots_shard_*.pkl")):
    sh = pickle.load(open(f, "rb")); off = len(images); images += sh["images"]; names = sh["names"]
    rows += [{**r, "img": r["img"] + off} for r in sh["rows"]]
X = np.array([r["x"] for r in rows], float); y = np.array([r["label"] for r in rows]); img = np.array([r["img"] for r in rows]); zone = np.array([r["zone"] for r in rows])
gid = {g: i for i, g in enumerate(sorted({im["group"] for im in images}))}; groups = np.array([gid[images[i]["group"]] for i in img])
print(f"{len(images)} photos, {len(rows)} proposed spots, {y.mean():.1%} touch a labelled lesion, {len(gid)} persons")
mk = lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0, random_state=0)
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, groups): oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
print("CV AUC %.3f | average precision %.3f (random = %.3f)" % (roc_auc_score(y, oof), average_precision_score(y, y * 0 + oof), y.mean()))


def evaluate(keep):
    """keep: boolean per proposed spot. Returns overall + per-zone recall/precision and the most-affected-zone agreement."""
    per = {z: {"gt": 0, "hit": 0, "k": 0, "m": 0} for z in ZONES}; hit_sets = {}
    for i, (r, k) in enumerate(zip(rows, keep)):
        if not k: continue
        z = r["zone"]
        if z >= 0: per[ZONES[z]]["k"] += 1; per[ZONES[z]]["m"] += r["label"]
        hit_sets.setdefault(r["img"], set()).update(r["matches"])
    agree, top2, kept_tot, gt_tot = [], [], [], []; k_img = {}; 
    for i, r in enumerate(rows):
        if keep[i] and r["zone"] >= 0: k_img.setdefault(r["img"], np.zeros(5))[r["zone"]] += 1
    for ii, im in enumerate(images):
        gz = im["gt_zone"]; hs = hit_sets.get(ii, set())
        for j, z in enumerate(gz):
            if z: per[z]["gt"] += 1; per[z]["hit"] += j in hs
        kc = k_img.get(ii, np.zeros(5)); tot = sum(im["area"].values()) or 1; elig = [k for k, z in enumerate(ZONES) if im["area"][z] / tot >= 0.015]
        gc = np.array([sum(1 for z in gz if z == zz) for zz in ZONES], float); kept_tot.append(kc.sum()); gt_tot.append(gc.sum())
        if gc.sum() >= 3 and len(elig) >= 3 and kc.sum() > 0:
            dg = {k: gc[k] / im["area"][ZONES[k]] for k in elig}; dc = {k: kc[k] / im["area"][ZONES[k]] for k in elig}; best = max(dg, key=dg.get)
            agree.append(best == max(dc, key=dc.get)); top2.append(best in sorted(dc, key=dc.get, reverse=True)[:2])
    G = sum(v["gt"] for v in per.values()); H = sum(v["hit"] for v in per.values()); K = sum(v["k"] for v in per.values()); M = sum(v["m"] for v in per.values())
    return {"recall": H / G, "precision": M / max(1, K), "kept_per_photo": float(np.mean(kept_tot)), "agree": float(np.mean(agree)), "top2": float(np.mean(top2)), "n_agree": len(agree), "rho": float(spearmanr(kept_tot, gt_tot)[0]),
            "zones": {z: {"recall": per[z]["hit"] / max(1, per[z]["gt"]), "precision": per[z]["m"] / max(1, per[z]["k"])} for z in ZONES}}


base = evaluate(np.ones(len(y), bool)); res = {"unfiltered": base, "thresholds": {}}
print(f"{'threshold':>10s} {'spots/photo':>12s} {'lesion recall':>14s} {'precision':>10s} {'zone exact':>11s} {'in top-2':>9s} {'count rho':>10s}")
print(f"{'none':>10s} {base['kept_per_photo']:12.1f} {base['recall']:14.1%} {base['precision']:10.1%} {base['agree']:11.1%} {base['top2']:9.1%} {base['rho']:10.2f}")
for t in (0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5):
    e = evaluate(oof >= t); res["thresholds"][str(t)] = e
    print(f"{t:10.2f} {e['kept_per_photo']:12.1f} {e['recall']:14.1%} {e['precision']:10.1%} {e['agree']:11.1%} {e['top2']:9.1%} {e['rho']:10.2f}")
ok = {t: e for t, e in res["thresholds"].items() if e["recall"] >= 0.45}; best = max(ok, key=lambda t: (ok[t]["agree"], ok[t]["precision"])) if ok else "0.2"
print("chosen threshold:", best, "(highest zone agreement with lesion recall >= 45%)"); res["chosen_threshold"] = float(best); res["cv_auc"] = float(roc_auc_score(y, oof)); res["photos"] = len(images); res["spots"] = len(rows)
json.dump(res, open("results/spot_filter_cv.json", "w"), indent=1)
os.makedirs("models", exist_ok=True); final = mk().fit(X, y)
joblib.dump({"model": final, "threshold": float(best), "names": names, "cv_auc": res["cv_auc"], "trained_on_photos": len(images)}, "models/spot_filter.joblib"); print("saved models/spot_filter.joblib")

m = {"threshold": res["chosen_threshold"], "cv_auc": round(res["cv_auc"], 3), "photos": res["photos"], "validation": "person-grouped 5-fold cross-validation on expert-labelled photos",
     "unfiltered": {k: round(res["unfiltered"][k], 3) for k in ("recall", "precision", "agree", "top2")}, "filtered": {k: round(res["thresholds"][best][k], 3) for k in ("recall", "precision", "agree", "top2", "kept_per_photo")},
     "zones": {z: {"recall": round(v["recall"], 3), "precision": round(v["precision"], 3)} for z, v in res["thresholds"][best]["zones"].items()}}
json.dump(m, open("models/spot_filter_metrics.json", "w"), indent=1); print("saved models/spot_filter_metrics.json")
