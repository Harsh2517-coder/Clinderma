"""Synthetic-cohort SIMULATION of future-condition risk from face-zone spot patterns + intake information.

THIS IS A SIMULATION, NOT CLINICAL EVIDENCE.  No dataset links face zones to the conditions people later develop, so the cohort is generated from the assumptions written below.
The key assumption is `link` (lambda): how strongly a person's hidden imbalance in a body system raises spot density in the face zones traditionally tied to that system.
lambda = 0 means the face map has no real effect.  Results therefore describe what a method could detect IF an effect of a given size existed, never whether it exists.
Two uses:  (1) a demonstrator model for the app (trained at an assumed moderate lambda, shown as a qualitative tier and clearly labelled simulated);
           (2) a power study: how many people a real study would need to detect the link at each assumed strength.
Scanner noise is taken from the MEASURED accuracy of the scan (models/spot_filter_metrics.json), so the synthetic spot counts are as noisy as real scans.
Run:  python3 risk_sim.py   -> results/simulation_results.json, models/risk_sim.joblib, SIMULATION.md
"""
import json, os, sys

import numpy as np

import insights

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = insights.ZONES
SYS = list(insights.SYSTEMS)                                              # digestive, endocrine, respiratory, liver, cardiac, reproductive, kidney, stress_sleep
AREA = np.array([250000, 600000, 600000, 150000, 200000], float)          # typical zone skin areas in scan pixels: forehead, left cheek, right cheek, nose, chin
W = np.zeros((len(ZONES), len(SYS)))                                      # the traditional map: each zone shares weight equally among the systems it is linked to
for zi, z in enumerate(ZONES):
    for s in insights.KB[z]["systems"]: W[zi, SYS.index(s)] = 1 / len(insights.KB[z]["systems"])
BASE_LOGIT, SELF_REPORT_LOGIT, SELF_REPORT_BETA, LATENT_BETA, REPORTED_BETA = -2.8, -2.0, 0.9, 0.8, 1.2
SCENARIOS = {"null": 0.0, "weak": 0.25, "moderate": 0.5, "strong": 0.8}
PRODUCT_SCENARIO = "moderate"
sig = lambda x: 1 / (1 + np.exp(-x))


def scanner_noise():
    """Recall and precision per zone of the real scan (cross-validated); fallback to typical values if the metrics file is missing."""
    p = os.path.join(HERE, "models", "spot_filter_metrics.json")
    if os.path.exists(p):
        z = json.load(open(p))["zones"]; return np.array([z[k]["recall"] for k in ZONES]), np.array([max(z[k]["precision"], 0.05) for k in ZONES])
    return np.full(5, 0.5), np.full(5, 0.3)


def generate(n, link, seed):
    rng = np.random.default_rng(seed); g = lambda *a, **k: rng.normal(*a, **k)
    age = rng.integers(15, 61, n).astype(float); female = rng.random(n) < 0.55; bmi = np.clip(g(24.5, 4.5, n) + 0.04 * (age - 30), 15, 45)
    stress = rng.choice([0, 1, 2], n, p=[.4, .4, .2]).astype(float); sleep = np.clip(g(6.8, 1.1, n), 3, 10)
    smoke, alc = rng.random(n) < .10, rng.random(n) < .12; fe, fc, fd = rng.random(n) < .20, rng.random(n) < .15, rng.random(n) < .12
    Z = np.column_stack([                                                     # hidden imbalance per system: observable risk factors + unobserved noise
        .5 * (stress == 2) + .3 * (stress == 1) - .25 * (sleep - 7) + .6 * fd + g(0, 1, n),                       # digestive
        .05 * (bmi - 25) + .7 * fe + g(0, 1, n),                                                                    # endocrine
        1.0 * smoke + .2 * (age > 45) + g(0, 1, n),                                                                 # respiratory
        .9 * alc + .04 * (bmi - 25) + g(0, 1, n),                                                                   # liver
        .03 * (age - 35) + .04 * (bmi - 25) + .6 * smoke + .6 * fc + g(0, 1, n),                                    # cardiac
        .03 * (bmi - 25) + .5 * fe * female - 3.0 * (~female) + g(0, 1, n),                                         # reproductive (almost only in women)
        .02 * (age - 35) + .3 * fe + g(0, 1, n),                                                                    # kidney
        .8 * (stress == 2) + .4 * (stress == 1) + .3 * (6.5 - sleep) + g(0, 1, n)])                                 # stress and sleep
    reported = rng.random((n, len(SYS))) < sig(SELF_REPORT_LOGIT + SELF_REPORT_BETA * Z)
    future = rng.random((n, len(SYS))) < sig(BASE_LOGIT + LATENT_BETA * Z + REPORTED_BETA * reported)               # a condition within 5 years
    acne_tendency = g(0, .5, n) + .5 * (age < 25)
    rate = AREA[None, :] / 1e5 * 0.3 * np.exp(acne_tendency[:, None] + link * (Z @ W.T))                            # expected true spots per zone
    rec, prec = scanner_noise(); tp = rng.binomial(rng.poisson(rate), rec[None, :]); fp = rng.poisson(rate * rec[None, :] * (1 - prec[None, :]) / prec[None, :])
    return {"age": age, "female": female.astype(float), "bmi": bmi, "stress": stress, "sleep": sleep, "smoke": smoke.astype(float), "alc": alc.astype(float), "fe": fe.astype(float), "fc": fc.astype(float),
            "fd": fd.astype(float), "reported": reported.astype(float), "obs": (tp + fp).astype(float), "future": future.astype(int)}


def features(d, with_zones=True, area=None):
    """One code path for synthetic cohorts and for real patients (arrays of length n)."""
    obs = d["obs"]; ar = AREA[None, :] if area is None else area; tot = obs.sum(1); dens = obs / ar; share = dens / np.maximum(dens.sum(1, keepdims=True), 1e-9); share = np.where(tot[:, None] >= insights.MIN_SPOTS, share, 0.2)
    cols = [d[k] for k in ("age", "female", "bmi", "stress", "sleep", "smoke", "alc", "fe", "fc", "fd")] + list(d["reported"].T)
    if with_zones: cols += [np.log1p(tot)] + list(share.T)
    return np.column_stack(cols)


def patient_arrays(intake, zone_counts, zone_area=None):
    """The same variables from a patient's intake form and scan report."""
    it = intake or {}; L = it.get("lifestyle") or {}; fh = (it.get("family_history") or "").lower(); conds = set(it.get("conditions") or [])
    num = lambda x, dflt: x if isinstance(x, (int, float)) else dflt
    d = {"age": [num(it.get("age"), 30.0)], "female": [1.0 if it.get("sex") == "Female" else 0.0], "bmi": [num(it.get("bmi"), 24.5)], "stress": [{"Low": 0, "Moderate": 1, "High": 2}.get(L.get("stress"), 1)],
         "sleep": [num(L.get("sleep_hours"), 7.0)], "smoke": [1.0 if L.get("smoking") == "Regularly" else 0.0], "alc": [1.0 if L.get("alcohol") == "Regularly" else 0.0],
         "fe": [1.0 if any(k in fh for k in ("diabet", "thyroid", "pcos", "pcod")) else 0.0], "fc": [1.0 if "heart" in fh else 0.0], "fd": [1.0 if any(k in fh for k in ("gastr", "ulcer", "ibs", "bowel", "colitis")) else 0.0],
         "reported": np.array([[1.0 if s in conds else 0.0 for s in SYS]]), "obs": np.array([[float(zone_counts.get(z, 0)) for z in ZONES]])}
    d = {k: (np.asarray(v, float) if k not in ("reported", "obs") else v) for k, v in d.items()}
    area = np.array([[float((zone_area or {}).get(z, AREA[i])) for i, z in enumerate(ZONES)]]); return d, area


def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0, random_state=0)


def auc_table(link, n_train=30000, n_test=20000, seed=0, boots=150):
    from sklearn.metrics import roc_auc_score
    tr, te = generate(n_train, link, seed), generate(n_test, link, seed + 1); Xa, Xb = features(tr, False), features(tr, True); Ta, Tb = features(te, False), features(te, True); rng = np.random.default_rng(seed + 2); out = {}
    for si, s in enumerate(SYS):
        y, yt = tr["future"][:, si], te["future"][:, si]; pa = _hgb().fit(Xa, y).predict_proba(Ta)[:, 1]; pb = _hgb().fit(Xb, y).predict_proba(Tb)[:, 1]
        d = []
        for _ in range(boots):
            i = rng.integers(0, len(yt), len(yt)); d.append(roc_auc_score(yt[i], pb[i]) - roc_auc_score(yt[i], pa[i]))
        out[s] = {"prevalence": round(float(yt.mean()), 3), "auc_intake_only": round(roc_auc_score(yt, pa), 3), "auc_with_zones": round(roc_auc_score(yt, pb), 3),
                  "gain": round(roc_auc_score(yt, pb) - roc_auc_score(yt, pa), 3), "gain_ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]}
    return out


def power(link, n, system, reps=40, seed=100):
    """Share of simulated studies of n people in which adding the face-zone variables significantly (p<0.05, likelihood-ratio test) improves prediction of one system's future condition."""
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler
    si, hits = SYS.index(system), 0
    for r in range(reps):
        d = generate(n, link, seed + r); y = d["future"][:, si]
        if y.min() == y.max(): continue
        F = features(d, True); A = features(d, False); B = np.column_stack([A, F[:, -6:-1]])                                  # + log total spots + 4 zone shares (df=5)
        A, B = StandardScaler().fit_transform(A), StandardScaler().fit_transform(B)
        ll = lambda X: -log_loss(y, LogisticRegression(C=1e6, max_iter=3000).fit(X, y).predict_proba(X)[:, 1], normalize=False)
        hits += chi2.sf(2 * (ll(B) - ll(A)), B.shape[1] - A.shape[1]) < 0.05
    return hits / reps


def train_product_model(link=SCENARIOS[PRODUCT_SCENARIO], n=40000, seed=7):
    import joblib
    from sklearn.metrics import roc_auc_score
    d = generate(n, link, seed); X = features(d, True); te = generate(15000, link, seed + 1); Xt = features(te, True); models, cuts, aucs = [], [], []
    for si in range(len(SYS)):
        m = _hgb().fit(X, d["future"][:, si]); p = m.predict_proba(X)[:, 1]; models.append(m); cuts.append([float(np.quantile(p, .40)), float(np.quantile(p, .80))]); aucs.append(round(roc_auc_score(te["future"][:, si], m.predict_proba(Xt)[:, 1]), 3))
    meta = {"scenario": PRODUCT_SCENARIO, "link": link, "n_train": n, "systems": SYS, "tier_cutoffs": cuts, "synthetic_test_auc": dict(zip(SYS, aucs)),
            "note": "Trained on a SYNTHETIC cohort with an ASSUMED face-map link. Qualitative tiers only; not clinical evidence."}
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True); joblib.dump({"models": models, **meta}, os.path.join(HERE, "models", "risk_sim.joblib")); return meta


_loaded = None


def simulated_outlook(intake, zone_counts, zone_area=None):
    """Qualitative attention tier per body system from the demonstrator model, or None if it has not been trained. Tiers are relative to the synthetic cohort, never absolute probabilities."""
    global _loaded
    p = os.path.join(HERE, "models", "risk_sim.joblib")
    if _loaded is None:
        import joblib; _loaded = joblib.load(p) if os.path.exists(p) else False
    if not _loaded: return None
    d, area = patient_arrays(intake, zone_counts, zone_area); X = features(d, True, area); out = []
    for si, s in enumerate(SYS):
        pr = float(_loaded["models"][si].predict_proba(X)[:, 1][0]); lo, hi = _loaded["tier_cutoffs"][si]
        out.append({"system": s, "label": insights.SYSTEMS[s], "tier": "higher" if pr >= hi else "lower" if pr < lo else "typical"})
    order = {"higher": 0, "typical": 1, "lower": 2}; return {"simulated": True, "scenario": _loaded["scenario"], "note": _loaded["note"], "tiers": sorted(out, key=lambda o: order[o["tier"]])}


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); res = {"assumptions": {"base_logit": BASE_LOGIT, "latent_beta": LATENT_BETA, "reported_beta": REPORTED_BETA, "scenarios": SCENARIOS,
                                                                                    "scanner_recall_precision": [list(map(float, a)) for a in scanner_noise()]}, "auc": {}, "power": {}}
    for name, lam in SCENARIOS.items():
        res["auc"][name] = auc_table(lam); g = [v["gain"] for v in res["auc"][name].values()]; print(f"{name:9s} (link={lam}): mean AUC gain from face zones = {np.mean(g):+.3f}  (largest {max(g):+.3f})", flush=True)
    for system in ("digestive", "stress_sleep", "endocrine"):
        for lam in (0.25, 0.5):
            res["power"][f"{system}@{lam}"] = {str(n): power(lam, n, system) for n in (300, 1000, 3000, 10000)}; print("power", system, lam, res["power"][f"{system}@{lam}"], flush=True)
    res["product_model"] = train_product_model(); json.dump(res, open(os.path.join(HERE, "results", "simulation_results.json"), "w"), indent=1); print("saved results/simulation_results.json and models/risk_sim.joblib")
