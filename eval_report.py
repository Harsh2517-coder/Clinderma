"""Aggregates results/eval_shard_*.json into results/accuracy_by_zone.json and prints the table."""
import glob, json
import numpy as np
ZONES = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]
recs = [r for f in sorted(glob.glob("results/eval_shard_*.json")) for r in json.load(open(f))]
ok = [r for r in recs if r["status"] == "ok"]; print("photos", len(recs), "| scanned ok", len(ok), "| no face", sum(r["status"] == "no_face" for r in recs), "| no zones", sum(r["status"] == "no_zones" for r in recs), "| errors", sum(r["status"] == "error" for r in recs))
out = {"photos": len(recs), "scanned": len(ok), "zones": {}}
print(f"{'zone':12s} {'GT lesions':>10s} {'found (recall)':>15s} {'scan boxes':>11s} {'real (precision)':>17s}")
for z in ZONES:
    g = sum(r["zones"][z]["gt"] for r in ok); h = sum(r["zones"][z]["gt_hit"] for r in ok); c = sum(r["zones"][z]["cand"] for r in ok); m = sum(r["zones"][z]["cand_match"] for r in ok)
    out["zones"][z] = {"gt": g, "recall": round(h / g, 3) if g else None, "scan_boxes": c, "precision": round(m / c, 3) if c else None}
    print(f"{z:12s} {g:10d} {h / g if g else float('nan'):15.1%} {c:11d} {m / c if c else float('nan'):17.1%}")
G = sum(r["zones"][z]["gt"] for r in ok for z in ZONES); H = sum(r["zones"][z]["gt_hit"] for r in ok for z in ZONES); C = sum(r["zones"][z]["cand"] for r in ok for z in ZONES); M = sum(r["zones"][z]["cand_match"] for r in ok for z in ZONES)
out["overall"] = {"gt": G, "recall": round(H / G, 3), "scan_boxes": C, "precision": round(M / C, 3), "lesions_outside_any_zone": sum(r["gt_outside_zones"] for r in ok)}; print("overall recall %.1f%%  precision %.1f%%  | labelled lesions outside any zone: %d of %d" % (100 * H / G, 100 * M / C, out["overall"]["lesions_outside_any_zone"], sum(r["n_gt"] for r in ok)))
# does the zone the scan calls 'most affected' (density) match the zone with most labelled lesions (density)?
agree = []; top2 = []
for r in ok:
    ar = r["area"]; tot = sum(ar.values()) or 1; elig = [z for z in ZONES if ar[z] / tot >= 0.015]
    if sum(r["zones"][z]["gt"] for z in ZONES) < 3 or len(elig) < 3: continue
    dg = {z: r["zones"][z]["gt"] / ar[z] for z in elig}; dc = {z: r["zones"][z]["cand"] / ar[z] for z in elig}
    agree.append(max(dg, key=dg.get) == max(dc, key=dc.get)); top2.append(max(dg, key=dg.get) in sorted(dc, key=dc.get, reverse=True)[:2])
out["most_affected_zone"] = {"images": len(agree), "exact_match": round(float(np.mean(agree)), 3), "within_scan_top2": round(float(np.mean(top2)), 3), "chance_exact": 0.2}
print("most-affected zone: scan matches the labelled zone in %.0f%% of %d photos (chance 20%%); labelled zone is in the scan's top 2 in %.0f%%" % (100 * np.mean(agree), len(agree), 100 * np.mean(top2)))
cnt_s = np.array([sum(r["zones"][z]["cand"] for z in ZONES) for r in ok]); cnt_g = np.array([sum(r["zones"][z]["gt"] for z in ZONES) for r in ok])
from scipy.stats import spearmanr; rho = spearmanr(cnt_s, cnt_g)[0]; out["total_count_spearman"] = round(float(rho), 3); print("photo-level: scan spot count vs labelled lesion count, Spearman rho = %.2f" % rho)
json.dump(out, open("results/accuracy_by_zone.json", "w"), indent=1)
