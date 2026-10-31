"""GAGS (Global Acne Grading System) from the per-zone lesion types the acne detector finds.

Each zone is graded by its worst lesion (comedone 1, papule 2, pustule 3, nodule/cyst 4) and multiplied by the zone factor
(forehead, left cheek, right cheek x2; nose, chin x1). Facial maximum is 32. 1-18 mild, 19-30 moderate, 31-38 severe.
GAGS only looks at the worst lesion per zone, so widespread acne made of milder lesions can still read low. The severity shown is therefore the higher
of the GAGS band and a lesion-count band (Hayashi: per half face <=5 mild, 6-20 moderate, 21-50 severe, >50 very severe).
Same rules as the Clinderma acne prototype (backend/app/pipeline.py), so both screens agree.
"""
ZONE_FACTORS = {"forehead": 2, "right_cheek": 2, "left_cheek": 2, "nose": 1, "chin": 1}
GRADES = {"comedone": 1, "papule": 2, "pustule": 3, "nodule": 4, "cyst": 4}
ORDER = ["Clear", "Mild", "Moderate", "Severe", "Very Severe"]
INFO = {
    "comedone": "Blackhead or whitehead without inflammation (grade 1)",
    "papule": "Small inflamed red bump without pus (grade 2)",
    "pustule": "Inflamed bump with visible pus (grade 3)",
    "nodule": "Large, deep, painful inflamed lump (grade 4)",
    "cyst": "Deep pus-filled lesion, higher scarring risk (grade 4)",
}
OVERCOUNT = 0.85        # at confidence 0.15 the detector over-counts by about 15% on labelled photos


def grade(cls): return GRADES.get(str(cls).lower(), 1)


def gags_band(score): return "Clear" if score == 0 else "Mild" if score <= 18 else "Moderate" if score <= 30 else "Severe" if score <= 38 else "Very Severe"


def count_band(half_face_lesions):
    n = half_face_lesions * OVERCOUNT
    return "Clear" if half_face_lesions == 0 else "Mild" if n <= 5 else "Moderate" if n <= 20 else "Severe" if n <= 50 else "Very Severe"


def compute(views):
    """views: [{"view": "front"|"left"|"right", "zone_types": {zone: {class: count}}}]. Returns the GAGS block for a scan report, or None if no per-type data."""
    if not views or not any(v.get("zone_types") for v in views): return None
    zone = {z: {"max_grade": 0, "types": {}} for z in ZONE_FACTORS}
    for v in views:                                                            # worst lesion per zone over all photos, type counts added up
        for z, types in (v.get("zone_types") or {}).items():
            if z not in zone: continue
            for cls, n in types.items():
                if n: zone[z]["types"][cls] = zone[z]["types"].get(cls, 0) + n; zone[z]["max_grade"] = max(zone[z]["max_grade"], grade(cls))
    for z, d in zone.items(): d["factor"] = ZONE_FACTORS[z]; d["score"] = d["max_grade"] * ZONE_FACTORS[z]; d["count"] = sum(d["types"].values())
    total = sum(d["score"] for d in zone.values()); by = {v["view"]: sum(sum(t.values()) for t in (v.get("zone_types") or {}).values()) for v in views}
    half = (by["left"] + by["right"]) / 2 if "left" in by and "right" in by else by.get("front", max(by.values())) / 2
    gb, cb = gags_band(total), count_band(half); sev = max(gb, cb, key=ORDER.index)
    types = {}
    for d in zone.values():
        for cls, n in d["types"].items(): types[cls] = types.get(cls, 0) + n
    n_all = max(sum(types.values()), 1)
    return {"total_score": total, "max_score": 32, "severity": sev, "gags_severity": gb, "count_severity": cb, "severity_basis": "gags" if sev == gb else "lesion_count",
            "zones": zone, "acne_types": [{"type": c, "count": n, "percent": round(100 * n / n_all, 1), "grade": grade(c), "description": INFO.get(c, "")} for c, n in sorted(types.items(), key=lambda kv: -kv[1])],
            "note": "Computed from the lesion types the detector found. Moderate accuracy (about 4 in 10 lesions found); nodules are often missed, so severe acne can be under-graded. Not a diagnosis."}
