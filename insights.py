"""Turn per-zone spot counts + the patient's intake form into wellness indications.

Every indication is labelled with where it comes from. The face-zone -> body-system links are TRADITIONAL (Traditional Chinese Medicine face
mapping, Ayurveda mukha pariksha); only the chin/jaw hormonal-metabolic link has partial modern support. Nothing here is a diagnosis: output is
"may relate to ... discuss with your doctor". Sources and the evidence review are in README.md.
"""
ZONES = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]          # same five zones as the Clinderma pipeline
DISCLAIMER = ("These are traditional-medicine indications (Ayurveda and Traditional Chinese Medicine), not a medical diagnosis. Most are not "
              "confirmed by clinical studies. Please discuss any concern with your doctor.")

# systems a patient can report on the intake form; zone links reference these keys
SYSTEMS = {"digestive": "digestive system / intestines", "endocrine": "hormonal / metabolic health (thyroid, diabetes, PCOS)",
           "respiratory": "respiratory system", "liver": "liver", "cardiac": "heart / circulation", "reproductive": "reproductive health",
           "kidney": "kidneys", "stress_sleep": "stress and sleep"}

KB = {   # zone -> traditions; `systems` is what we match against the intake form; `evidence` is shown to the user
    "forehead": {"label": "Forehead", "systems": ["digestive", "stress_sleep"], "evidence": "traditional",
                 "tcm": "Small intestine and digestive function; the lower forehead is also linked to stress and sleep.",
                 "ayurveda": "Nervous and digestive systems (Vata).",
                 "modern": "Not supported by studies. Common ordinary causes: hair products (pomade), oily hair, hats and helmets."},
    "nose": {"label": "Nose", "systems": ["cardiac", "digestive"], "evidence": "traditional",
             "tcm": "Heart, or 'stomach heat' depending on the school.",
             "ayurveda": "Small intestine and digestive fire (Pitta).",
             "modern": "A sebaceous-gland-rich area; no organ link is supported by studies."},
    "left_cheek": {"label": "Left cheek", "systems": ["liver", "respiratory", "digestive"], "evidence": "traditional",
                   "tcm": "Liver in some schools, respiratory health in others (sources disagree).",
                   "ayurveda": "Small intestine and digestive fire (Pitta).",
                   "modern": "No organ link supported. Phone, pillow and hand-to-face contact and hormones are common ordinary factors."},
    "right_cheek": {"label": "Right cheek", "systems": ["respiratory", "digestive", "stress_sleep"], "evidence": "traditional",
                    "tcm": "Lungs in some schools; stomach and stress in others (sources disagree).",
                    "ayurveda": "Small intestine and digestive fire (Pitta).",
                    "modern": "No organ link supported. Phone, pillow and hand-to-face contact and hormones are common ordinary factors."},
    "chin": {"label": "Chin and jaw", "systems": ["endocrine", "reproductive", "kidney"], "evidence": "traditional + partial modern",
             "tcm": "Hormonal and reproductive health (kidney system in some schools).",
             "ayurveda": "Mouth, chin and neck region: chest and stomach (Kapha).",
             "modern": "Partly supported: moderate-to-severe acne is more common with hyperandrogenism, PCOS and insulin resistance, especially in adult "
                       "women. The pattern is not specific, since most adults with acne have it in several zones."},
}
MIN_SPOTS = 3          # below this there is no meaningful pattern
DOMINANT_SHARE = 0.30  # ponytail: raw share of spots; zones differ in area, so weight by zone area if this proves too cheek-heavy


# ---- "if this is not addressed": what doctors watch for.  Two tiers, always labelled:
#  * SKIN_OUTLOOK: consequences of untreated acne-type spots themselves (well established in dermatology)
#  * OUTLOOK[system]: conditions of the body system a zone is traditionally linked to.  These are real conditions of that system, listed so a doctor can check;
#    the link from facial spots to them is traditional/unvalidated (partly supported for the endocrine system), and the UI says so.
SKIN_OUTLOOK = {"evidence": "established",
                "watch_for": ["Scarring and dark marks (post-inflammatory hyperpigmentation) that can last months to years after spots heal",
                              "Skin infection or deeper, painful lesions, especially if spots are picked or squeezed",
                              "Lowered mood, anxiety or reduced confidence in people with persistent acne"]}
OUTLOOK = {
    "digestive": {"evidence": "traditional", "watch_for": ["Gastritis or acid reflux (GERD)", "Irritable bowel-type symptoms or long-standing constipation", "Poor absorption of nutrients, for example low iron or vitamin B12"]},
    "stress_sleep": {"evidence": "traditional", "watch_for": ["Chronic insomnia or poor sleep quality", "Persistent anxiety or low mood", "Stress-related high blood pressure and digestive upset"]},
    "endocrine": {"evidence": "traditional + partial modern", "watch_for": ["Insulin resistance progressing to type 2 diabetes", "Polycystic ovary syndrome (PCOS) and, over time, metabolic syndrome",
                                                                          "Thyroid disorders, which can also show as skin and hair changes"]},
    "reproductive": {"evidence": "traditional + partial modern", "watch_for": ["Irregular or missed periods", "Reduced fertility if PCOS or another hormonal disorder is present", "Endometrial or ovarian problems in long-standing hormonal imbalance"]},
    "respiratory": {"evidence": "traditional", "watch_for": ["Allergic rhinitis or recurring sinus problems", "Asthma or chronic bronchitis", "Frequent chest infections"]},
    "liver": {"evidence": "traditional", "watch_for": ["Fatty liver", "Raised liver enzymes, often silent until checked by a blood test", "Poor tolerance of fatty foods and alcohol"]},
    "cardiac": {"evidence": "traditional", "watch_for": ["High blood pressure", "High cholesterol", "Heart rhythm or circulation problems"]},
    "kidney": {"evidence": "traditional", "watch_for": ["Recurrent urinary tract infections", "Kidney stones", "Reduced kidney function, which a simple blood and urine test can check"]},
}


def risk_factors(system, intake):
    """Risk factors the patient themselves reported that make this system worth checking earlier. Simple, well-known factors only."""
    it = intake or {}; L = it.get("lifestyle") or {}; fh = (it.get("family_history") or "").lower(); out = []
    num = lambda x: x if isinstance(x, (int, float)) else None
    if system in ("endocrine", "reproductive", "cardiac"):
        if num(it.get("bmi")) and it["bmi"] >= 25: out.append(f"BMI {it['bmi']} (above 25)")
        if any(k in fh for k in ("diabet", "thyroid", "pcos", "pcod")): out.append("family history of diabetes, thyroid disease or PCOS")
    if system in ("digestive", "stress_sleep"):
        if L.get("stress") == "High": out.append("high stress level")
        if num(L.get("sleep_hours")) and L["sleep_hours"] < 6: out.append(f"sleeping only {L['sleep_hours']} hours")
    if system in ("liver", "respiratory", "cardiac"):
        if L.get("smoking") in ("Regularly",): out.append("regular smoking")
        if system in ("liver", "cardiac") and L.get("alcohol") == "Regularly": out.append("regular alcohol use")
    if system == "cardiac" and "heart" in fh: out.append("family history of heart disease")
    if system == "digestive" and any(k in fh for k in ("gastr", "ulcer", "ibs", "bowel", "colitis")): out.append("family history of digestive disease")
    return out


def outlook_for(system_keys, intake, matched_keys):
    """Up to two systems per finding, those the patient already reported first."""
    keys = sorted(system_keys, key=lambda k: (k not in matched_keys, system_keys.index(k)))[:2]
    return [{"system": SYSTEMS[k], "evidence": OUTLOOK[k]["evidence"], "watch_for": OUTLOOK[k]["watch_for"], "risk_factors": risk_factors(k, intake), "already_reported": k in matched_keys} for k in keys]


def analyse_zones(zone_counts, intake=None, zone_area=None):
    """zone_counts: {zone: spots}.  intake: the patient's intake form dict (may be None).  zone_area: {zone: skin pixels}; when given, zones are compared by
    spot DENSITY (spots per area) so large zones such as the cheeks do not win just by being large.  Returns a JSON-able report."""
    counts = {z: int(zone_counts.get(z, 0)) for z in ZONES}; total = sum(counts.values())
    conditions = set((intake or {}).get("conditions", []))                 # keys of SYSTEMS the patient reported an issue in
    rep = {"zone_counts": counts, "total_spots": total, "disclaimer": DISCLAIMER, "findings": [], "suggest_appointment": False, "summary": ""}
    if total < MIN_SPOTS:
        rep["summary"] = "Very few spots were found, so no regional pattern stands out."; return rep
    if zone_area:    # density share; zones with under 1.5% of the measured skin area are too small to judge
        tot_area = sum(zone_area.get(z, 0) for z in ZONES) or 1
        dens = {z: counts[z] / zone_area[z] if zone_area.get(z, 0) / tot_area >= 0.015 else 0.0 for z in ZONES}; dsum = sum(dens.values()) or 1
        shares = {z: dens[z] / dsum for z in ZONES}
    else: shares = {z: counts[z] / total for z in ZONES}
    rep["zone_share"] = {z: round(v, 2) for z, v in shares.items()}
    for z in sorted(ZONES, key=lambda k: -shares[k]):
        share = shares[z]
        if share < DOMINANT_SHARE: continue
        k = KB[z]; known = [s for s in k["systems"] if s in conditions]
        if known:
            names = " and ".join(SYSTEMS[s] for s in known)
            note = (f"You already told us about an issue with your {names}. Spots concentrated in the {k['label'].lower()} are traditionally linked to the same area, "
                    f"so the two may be related. This is a reason to discuss it with your doctor, not a diagnosis.")
        else:
            names = " or ".join(SYSTEMS[s] for s in k["systems"][:2])
            note = (f"You did not report any issue in this area. Traditionally, spots here may relate to your {names}. "
                    f"If this keeps up, consider a check-up.")
        rep["findings"].append({"zone": z, "label": k["label"], "spots": counts[z], "share": round(share, 2), "evidence": k["evidence"],
                                "tcm": k["tcm"], "ayurveda": k["ayurveda"], "modern": k["modern"], "systems": [SYSTEMS[s] for s in k["systems"]], "system_keys": k["systems"],
                                "matches_reported_conditions": [SYSTEMS[s] for s in known], "personal_note": note, "outlook": outlook_for(k["systems"], intake, known)})
    if not rep["findings"]:      # spots are spread evenly: no zone reaches the dominance threshold
        rep["summary"] = "Spots are spread fairly evenly across the face, so no single region stands out."; return rep
    top = rep["findings"][0]
    rep["suggest_appointment"] = bool(top["matches_reported_conditions"]) or total >= 15
    rep["skin_outlook"] = SKIN_OUTLOOK
    rep["summary"] = (f"Spots are most concentrated in the {top['label'].lower()} ({top['spots']} of {total} found). " +
                      ("This matches something you reported, so a doctor review is suggested." if top["matches_reported_conditions"]
                       else "No matching condition was reported; consider booking a check-up if it persists."))
    return rep


def recommend_doctors(doctors, report, intake=None):
    """doctors: [{id, name, specialization, focus:[system keys]}].  Ranks them for a scan report + intake form.
    Weighting: systems of the strongest finding x2, other findings x1, systems the patient reported x2.  Doctors with no focus are generalists:
    a small baseline score puts them after matching specialists but before non-matching ones.  Returns the full list (the patient may pick any doctor) with `score`, `recommended` and `why`."""
    want = {}
    for rank, f in enumerate(report.get("findings", [])):
        for s in f.get("system_keys", []): want[s] = want.get(s, 0) + (2 if rank == 0 else 1)
    for s in (intake or {}).get("conditions", []): want[s] = want.get(s, 0) + 2
    out = []
    for d in doctors:
        hit = [s for s in d["focus"] if s in want]; score = sum(want[s] for s in hit) or (0.5 if not d["focus"] else 0)   # generalists: below a matching specialist, above a non-matching one
        out.append({**d, "score": score, "recommended": score > 0 or not d["focus"], "generalist": not d["focus"],
                    "why": [SYSTEMS[s] for s in sorted(hit, key=lambda s: -want[s])]})
    return sorted(out, key=lambda d: (-d["score"], d["generalist"], d["name"]))


def bmi(height_cm, weight_kg):
    return round(weight_kg / ((height_cm / 100) ** 2), 1) if height_cm and weight_kg else None


SCAN_EVERY_DAYS = 7


def progress(scans, now=None):
    """scans: [{"created": iso, "report": {...}}] in any order. Weekly-scan status plus the trend over time.
    Spot counts depend on lighting, distance and camera, so changes are indicative only (the UI says so)."""
    from datetime import datetime, timedelta
    now = now or datetime.now(); rows = sorted(scans, key=lambda s: s["created"])
    series = [{"date": s["created"], "total": s["report"].get("total_spots", 0), "zones": s["report"].get("zone_counts", {}), "share": s["report"].get("zone_share", {})} for s in rows]
    out = {"n": len(series), "series": series, "due": False, "days_since": None, "next_due": None, "status": "no_scan",
           "change_vs_first_pct": None, "change_vs_previous_pct": None, "streak_weeks": 0, "adherence": {"weeks": 8, "weeks_with_scan": 0}}
    if not series: return out
    pct = lambda a, b: None if not b else round(100 * (a - b) / b)
    out["change_vs_first_pct"] = pct(series[-1]["total"], series[0]["total"]) if len(series) > 1 else None
    out["change_vs_previous_pct"] = pct(series[-1]["total"], series[-2]["total"]) if len(series) > 1 else None
    last = datetime.fromisoformat(series[-1]["date"]); days = (now - last).days
    out["days_since"] = days; out["next_due"] = (last + timedelta(days=SCAN_EVERY_DAYS)).isoformat(timespec="seconds"); out["due"] = days >= SCAN_EVERY_DAYS
    out["status"] = "overdue" if days >= SCAN_EVERY_DAYS + 3 else "due" if out["due"] else "on_track"
    weeks = {datetime.fromisoformat(s["date"]).isocalendar()[:2] for s in series}; cur = now.isocalendar()[:2]
    wk = lambda back: (now - timedelta(weeks=back)).isocalendar()[:2]
    out["adherence"]["weeks_with_scan"] = sum(1 for b in range(8) if wk(b) in weeks)
    b = 0 if cur in weeks else 1                    # a week still in progress does not break the streak
    while wk(b) in weeks: out["streak_weeks"] += 1; b += 1
    return out
