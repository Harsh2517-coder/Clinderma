"""Demo data from real photos: scans every photo once (cached), then builds a large demo (many doctors, patients, scans with stored face photos, a full calendar).
  python3 demo_data.py build <photo_dir> [<photo_dir> ...]    # slow, once: runs the scan on every photo and caches the results under data/demo_cache
Then `python3 seed.py` builds the demo database from the cache.  The photos stay on this machine (data/ is git-ignored)."""
import glob, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "data", "demo_cache")


def _key(path): return hashlib.sha1(f"{path}|{os.path.getmtime(path)}".encode()).hexdigest()[:20]


def scan_one(path):
    """Runs the real scan on one photo (view detected automatically) and caches the result + the original and annotated JPEGs."""
    import cv2
    import scan as scanner
    k = _key(path); meta = os.path.join(CACHE, k + ".json")
    if os.path.exists(meta): return json.load(open(meta))
    rec = {"path": path, "key": k, "patient_dir": os.path.basename(os.path.dirname(os.path.dirname(path))), "date": os.path.basename(os.path.dirname(path)), "status": "error"}
    try:
        img = cv2.imread(path)
        if img is None:
            from PIL import Image; import numpy as np
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        s = scanner.scan_image(img, "auto")
        if s["status"] == "ok":
            def jpg(im, m, q):
                h, w = im.shape[:2]; f = m / max(h, w); im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else im
                return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
            open(os.path.join(CACHE, k + "_orig.jpg"), "wb").write(jpg(img, 1400, 85)); open(os.path.join(CACHE, k + "_ann.jpg"), "wb").write(jpg(s["preview"], 1000, 80))
            rec.update(status="ok", view=s["view"], zone_counts=s["zone_counts"], zone_area=s["zone_area"], total_spots=s["total_spots"])
        else: rec["status"] = s["status"]
    except Exception as e: rec["error"] = f"{type(e).__name__}: {e}"[:200]
    json.dump(rec, open(meta, "w")); return rec


def build(dirs, workers=4):
    os.makedirs(CACHE, exist_ok=True); files = sorted(f for d in dirs for f in glob.glob(os.path.join(os.path.expanduser(d), "*", "*", "*")) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    print(len(files), "photos", flush=True); done = 0
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(scan_one, files, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(files): print(done, "/", len(files), flush=True)
    recs = load_cache(); ok = [r for r in recs if r["status"] == "ok"]; print("ok", len(ok), "| no face", sum(r["status"] == "no_face" for r in recs), "| other", len(recs) - len(ok) - sum(r["status"] == "no_face" for r in recs))
    from collections import Counter; print("views detected:", dict(Counter(r["view"] for r in ok)))


def load_cache():
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(CACHE, "*.json")))]


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "build": build(sys.argv[2:])
    else: print(__doc__)


# ---------------------------------------------------------------- the large demo ----------------------------------------------------------------
DOCTORS = [  # name, specialization, focus systems (empty = general), weekly windows {weekday: [(start, end)]}, slot minutes
    ("Dr. Asha Rao", "Ayurveda: digestive and stress", ["digestive", "stress_sleep"], {**{d: [("09:00", "13:00")] for d in range(5)}, 1: [("09:00", "13:00"), ("15:00", "18:00")], 3: [("09:00", "13:00"), ("15:00", "18:00")]}),
    ("Dr. Wei Chen", "Integrative medicine (TCM): hormonal health", ["endocrine", "reproductive"], {0: [("09:00", "13:00")], 2: [("09:00", "13:00")], 4: [("09:00", "13:00")], 1: [("14:00", "18:00")], 3: [("14:00", "18:00")]}),
    ("Dr. Meera Iyer", "General Ayurveda physician", [], {d: [("11:00", "17:00")] for d in range(5)}),
    ("Dr. Rohan Deshmukh", "Respiratory and allergy wellness", ["respiratory"], {d: [("10:00", "14:00")] for d in range(5)}),
    ("Dr. Priya Nair", "Liver and metabolic health", ["liver", "endocrine"], {d: [("09:00", "15:00")] for d in (0, 1, 3, 4)}),
    ("Dr. Imran Sheikh", "Heart and circulation wellness", ["cardiac", "stress_sleep"], {d: [("12:00", "18:00")] for d in range(5)}),
    ("Dr. Lakshmi Menon", "Women's health", ["reproductive", "endocrine"], {**{d: [("09:30", "13:30")] for d in range(5)}, 5: [("09:00", "12:00")]}),
    ("Dr. Arvind Kulkarni", "Kidney and urinary wellness", ["kidney", "digestive"], {d: [("10:00", "16:00")] for d in range(4)}),
]
PATIENT_NAMES = ["Riya Shah", "Arjun Mehta", "Li Na", "Kavya Reddy", "Rahul Verma", "Sneha Patil", "Aditya Joshi", "Pooja Nair", "Vikram Singh", "Ananya Das", "Karan Malhotra", "Divya Pillai", "Rohit Kulkarni",
                 "Neha Gupta", "Sanjay Bose", "Meghna Roy", "Tarun Khanna", "Ishita Sen", "Manish Agarwal", "Shruti Menon", "Harsh Vyas", "Nandini Rao", "Yash Thakur", "Tanvi Jain", "Abhishek Pandey",
                 "Swati Mishra", "Gaurav Saxena", "Pallavi Kapoor", "Siddharth Rao", "Aarti Bhatt", "Mohit Chawla", "Jyoti Dubey", "Varun Sethi", "Rekha Iyer", "Nikhil Bhatia", "Simran Kaur"]
REASONS = ["Weekly review", "Follow-up", "Scan results discussion", "Diet and lifestyle advice", "New spots", "Skin care routine", "Progress check", "Stress and sleep check-in"]
FAMILY = ["", "", "Father has type 2 diabetes", "Mother has thyroid disease", "Mother had a stomach ulcer", "Father has heart disease", "Sister has PCOS", "No known conditions"]
NOTES = {
    "digestive": ["Reviewed the weekly scan with the patient. Advised regular meal timings, a light dinner and warm water through the day. Review in one week.", "Spots remain mostly on the forehead. Suggested reducing fried food and eating at fixed times. Continue the current routine.",
                  "Patient reports less bloating this week. Continue the same diet; bring the next scan to the review."],
    "stress_sleep": ["Discussed sleep: aim for 7 to 8 hours and a fixed bedtime. Gentle breathing practice twice a day. Review next week.", "Stress is still high. Suggested short walks after meals and limiting screens at night."],
    "endocrine": ["Reviewed the scan and the health form. Advised regular exercise and balanced meals; blood sugar and thyroid tests suggested if symptoms continue.", "Weight is stable. Continue the current plan and repeat the scan next week."],
    "reproductive": ["Cycle diary discussed. Continue regular meals and sleep; review the scan trend at the next visit.", "Advised stress reduction and regular exercise. No change to the plan."],
    "respiratory": ["Discussed allergens and dust exposure at home. Steam inhalation as needed. Review next week.", "Breathing comfort is better this week. Continue the routine."],
    "liver": ["Advised limiting fried and oily food and alcohol. Plenty of water; liver function test suggested at the next visit.", "Diet is improving. Keep a light dinner and repeat the scan next week."],
    "cardiac": ["Blood pressure noted at the visit. Advised daily walking and less salt. Review in two weeks.", "Discussed stress and sleep. Continue light exercise."],
    "kidney": ["Advised steady water intake through the day and less salt. Review next week.", "No new complaints. Continue the routine and repeat the scan."],
    "general": ["Reviewed the weekly scan. Skin care routine unchanged: gentle cleansing and sunscreen. Review in one week.", "Spots are stable. Continue the current routine and keep scans weekly.", "Discussed hair products and pillow hygiene. No other changes."],
}


TRENDS = ["improving", "improving", "improving", "worsening", "worsening", "hill", "hill", "valley", "wavy", "wavy"]      # mix across patients


def trend_totals(kind, n, base_mean, rng):
    """Total spots per weekly scan following a shape, never flat: always a clear range and never two equal values in a row."""
    import math
    t = [i / max(1, n - 1) for i in range(n)]; ph = rng.uniform(0, 6.28)
    f = {"improving": [1.55 - 0.95 * x for x in t], "worsening": [0.6 + 0.95 * x for x in t], "hill": [0.65 + 0.95 * math.sin(math.pi * x) for x in t],
         "valley": [1.5 - 0.9 * math.sin(math.pi * x) for x in t], "wavy": [1 + 0.38 * math.sin(2 * math.pi * 1.6 * x + ph) + 0.1 * (x - 0.5) for x in t]}[kind]
    T = [max(1, round(base_mean * v * rng.uniform(0.94, 1.06))) for v in f]
    for i in range(1, n):                                                       # no two consecutive scans with the same total
        if T[i] == T[i - 1]: T[i] += 1 if (kind in ("worsening",) or (kind in ("hill", "wavy") and i < n // 2)) else -1; T[i] = max(1, T[i])
    for i in range(1, n):                                                       # (the nudge can create a new tie: sweep again)
        if T[i] == T[i - 1]: T[i] += 1
    if max(T) - min(T) < max(4, 0.3 * (sum(T) / n)):                            # make sure the range is clearly visible on a chart
        c = sum(T) / n; T = [max(1, round(c + (v - c) * 1.8 + (2 if v >= c else -2))) for v in T]
    return T


def rescale(counts, total, fallback):
    """Distributes `total` spots over the zones in proportion to `counts` (largest remainder); falls back to the patient's usual distribution if the scan had none."""
    w = {z: counts[z] for z in counts}; s = sum(w.values())
    if s == 0: w = dict(fallback); s = sum(w.values()) or 1
    raw = {z: total * v / s for z, v in w.items()}; out = {z: int(raw[z]) for z in raw}
    for z in sorted(raw, key=lambda z: raw[z] - out[z], reverse=True)[: total - sum(out.values())]: out[z] += 1
    return out


def is_flat(totals):
    n = len(totals); mean = sum(totals) / n
    return n < 3 or any(a == b for a, b in zip(totals, totals[1:])) or (max(totals) - min(totals)) < max(4, 0.3 * mean)


def _intake(rng, insights):
    female = rng.random() < 0.6; h, w = rng.randint(152, 185), rng.randint(46, 98); conds = rng.sample(list(insights.SYSTEMS), rng.choice([0, 0, 1, 1, 2]))
    if not female and "reproductive" in conds: conds.remove("reproductive")
    return {"height_cm": h, "weight_kg": w, "age": rng.randint(18, 58), "sex": "Female" if female else "Male", "conditions": conds, "condition_notes": "", "medications": rng.choice(["", "", "Multivitamin", "Thyroid tablet"]),
            "allergies": rng.choice(["", "", "Dust", "Pollen"]), "family_history": rng.choice(FAMILY), "skin_notes": rng.choice(["", "Uses a face wash twice a day", "Oil-based hair products"]),
            "lifestyle": {"sleep_hours": rng.choice([5, 6, 6.5, 7, 7.5, 8]), "diet": rng.choice(["Vegetarian", "Mixed", "Non-vegetarian", "Vegan"]), "exercise": rng.choice(["None", "1-2 days a week", "3-5 days a week", "Daily"]),
                          "stress": rng.choice(["Low", "Moderate", "Moderate", "High"]), "smoking": rng.choice(["No", "No", "No", "Occasionally", "Regularly"]), "alcohol": rng.choice(["No", "No", "Occasionally", "Occasionally", "Regularly"])},
            "bmi": insights.bmi(h, w)}


def build_big_demo(A, insights, reminders, n_patients=36, seed=7, weeks_back=8, weeks_forward=4):
    """Builds the large demo in the database A.connect() points at. Needs the photo cache (python3 demo_data.py build ...)."""
    import random, shutil
    from datetime import datetime, timedelta, time as dtime
    rng = random.Random(seed); recs = [r for r in load_cache() if r["status"] == "ok"]
    if not recs: raise SystemExit("No cached photo scans found. Run: python3 demo_data.py build ~/Downloads/patients_part1 (and part2)")
    by = {}
    for r in sorted(recs, key=lambda r: (r["date"], r["path"])): by.setdefault(r["patient_dir"], {"front": [], "left": [], "right": []})[r["view"]].append(r)
    full = sorted(((p, v) for p, v in by.items() if all(v[k] for k in ("front", "left", "right"))), key=lambda t: -sum(len(x) for x in t[1].values()))[:n_patients]
    n = len(full); print(f"{n} patients have all three views (wanted {n_patients})")
    if os.path.exists(A.DB_PATH): os.remove(A.DB_PATH)
    shutil.rmtree(A.PHOTO_DIR, ignore_errors=True); A.init_db(); db = A.connect(); now = datetime.now(); iso = lambda d: d.isoformat(timespec="seconds"); PW = "Demo-pass-2026"
    def user(role, name, email): return db.execute("INSERT INTO users(role,name,email,pw_hash,created) VALUES(?,?,?,?,?)", (role, name, email, A.hash_pw(PW), iso(now - timedelta(days=rng.randint(60, 200))))).lastrowid
    user("admin", "Platform Admin", "admin@demo.local"); docs = []
    for i, (name, spec, focus, chart) in enumerate(DOCTORS, 1):
        d = user("doctor", name, f"doctor{i}@demo.local"); db.execute("INSERT INTO doctors(user_id, specialization, focus) VALUES(?,?,?)", (d, spec, json.dumps(focus)))
        for wd, wins in chart.items():
            for a, b in wins: db.execute("INSERT INTO availability(doctor_id,weekday,start,end,slot_minutes) VALUES(?,?,?,?,20)", (d, wd, a, b))
        docs.append({"id": d, "name": name, "specialization": spec, "focus": focus, "chart": chart})
    def slots_on(doc, day):
        out = []
        for a, b in doc["chart"].get(day.weekday(), []):
            t = datetime.combine(day, dtime(*map(int, a.split(":")))); end = datetime.combine(day, dtime(*map(int, b.split(":"))))
            while t + timedelta(minutes=20) <= end: out.append(t); t += timedelta(minutes=20)
        return out
    trends = [TRENDS[i % len(TRENDS)] for i in range(n)]; rng.shuffle(trends)
    people = []; load = {d["id"]: 0 for d in docs}                         # one entry per patient with their assignment history; load = patients per doctor
    for i, (pdir, v) in enumerate(full, 1):
        name = PATIENT_NAMES[i - 1] if i <= len(PATIENT_NAMES) else f"Patient {i}"; pid = user("patient", name, f"patient{i}@demo.local"); intake = _intake(rng, insights); intake["saved"] = iso(now - timedelta(days=60))
        db.execute("INSERT INTO patients(user_id, doctor_id, intake_json, email_reminders) VALUES(?,?,?,1)", (pid, None, json.dumps(intake)))      # row first: scans reference it
        n_scans = rng.choice([7, 8, 8, 9, 10]); last = now - timedelta(days=rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 9, 12]), hours=rng.randint(0, 6)); dates = [last - timedelta(days=7 * k) for k in range(n_scans)][::-1]
        off = {k: rng.randint(0, 5) for k in ("front", "left", "right")}; scans = []; base = []
        for si, dt in enumerate(dates):
            pick = {k: v[k][(si + off[k]) % len(v[k])] for k in ("front", "left", "right")}
            base.append((dt, pick, {z: sum(pick[k]["zone_counts"][z] for k in pick) for z in insights.ZONES}, {z: sum(pick[k]["zone_area"][z] for k in pick) for z in insights.ZONES}))
        usual = {z: sum(b_[2][z] for b_ in base) for z in insights.ZONES}                       # the patient's usual zone pattern, from their real photos
        kind = trends[i - 1]; mean0 = max(8.0, sum(sum(b_[2].values()) for b_ in base) / n_scans); totals = trend_totals(kind, n_scans, mean0, rng)
        while is_flat(totals): totals = trend_totals(kind, n_scans, mean0, rng)
        for si, (dt, pick, counts0, area) in enumerate(base):
            # The real photos repeat from scan to scan, so the spot totals follow the patient's trend while keeping the zone pattern each photo set shows.
            counts = rescale(counts0, totals[si], usual)
            rep = insights.analyse_zones(counts, intake, area); rep["views"] = ["front", "left", "right"]; rep["zone_area"] = area; A.finalize_report(rep, intake, counts, area)
            sid = db.execute("INSERT INTO scans(patient_id, created, report_json) VALUES(?,?,?)", (pid, iso(dt), json.dumps(rep))).lastrowid
            for k, r in pick.items():
                for kind_, suffix in (("original", "_orig"), ("annotated", "_ann")): A.save_photo(db, sid, k, kind_, open(os.path.join(CACHE, r["key"] + suffix + ".jpg"), "rb").read())
            scans.append((dt, rep))
        rec = insights.recommend_doctors([{"id": d["id"], "name": d["name"], "specialization": d["specialization"], "focus": d["focus"]} for d in docs], scans[0][1], intake)
        first = min(rec[:4], key=lambda r: (load[r["id"]], rng.random()))["id"]; load[first] += 1          # one of the 4 best-matching doctors, preferring the least busy
        start0 = min(dates[0], now - timedelta(weeks=9)) - timedelta(days=1)                      # patients joined their doctor before the first weekly scan
        hist = [(first, start0, None, "first choice")]
        if rng.random() < 0.2 and n_scans >= 6:                           # some patients switch doctor part-way: both doctors keep access
            sw = dates[rng.randint(3, n_scans - 2)]; other = min([d["id"] for d in docs if d["id"] != first], key=lambda i_: (load[i_], rng.random())); load[first] -= 1; load[other] += 1; hist = [(first, start0, sw, "first choice"), (other, sw, None, rng.choice(["new hormonal concern", "new digestive concern", "wanted a second opinion"]))]
        db.execute("UPDATE patients SET doctor_id=? WHERE user_id=?", (hist[-1][0], pid))
        for d_, a_, b_, why in hist: db.execute("INSERT INTO patient_doctors(patient_id,doctor_id,start_ts,end_ts,reason) VALUES(?,?,?,?,?)", (pid, d_, iso(a_), iso(b_) if b_ else None, why))
        people.append({"id": pid, "hist": hist, "scans": scans, "intake": intake})
    # ---- calendar: two visits per patient per week from 6 weeks back to 4 weeks ahead, inside each doctor's real weekly chart
    by_id = {d["id"]: d for d in docs}; taken, ptaken, n_app = set(), set(), 0; today = now.date(); monday = today - timedelta(days=today.weekday())
    for w in range(-weeks_back, weeks_forward + 1):
        for p in rng.sample(people, len(people)):
            for _ in range(2):
                day = monday + timedelta(days=7 * w + rng.randint(0, 5)); ts0 = datetime.combine(day, dtime(12))
                cur = next(((d, a, b) for d, a, b, _ in p["hist"] if a <= ts0 and (b is None or ts0 < b)), None)
                if not cur: continue
                doc = by_id[cur[0]]; free = [t for t in slots_on(doc, day) if (doc["id"], t) not in taken and (p["id"], t) not in ptaken]
                if not free: continue
                t = rng.choice(free); taken.add((doc["id"], t)); ptaken.add((p["id"], t)); past = t < now
                roll = rng.random()                                                       # past: 6% cancelled, 9% missed (never completed), the rest completed; some visits were moved
                status = ("cancelled" if roll < 0.06 else "booked" if roll < 0.15 else "completed") if past else "booked"; note = None
                if status == "completed":
                    key = rng.choice(doc["focus"]) if doc["focus"] else "general"; note = rng.choice(NOTES[key] + NOTES["general"][:1])
                moved = status in ("completed", "booked") and rng.random() < (0.07 if past else 0.09); by = rng.choice(["patient", "patient", "doctor"]) if moved else None
                old = (t - timedelta(days=rng.randint(1, 5))).replace(hour=rng.choice([9, 10, 11, 14, 15, 16]), minute=rng.choice([0, 20, 40])).strftime("%Y-%m-%d %H:%M") if moved else None
                reason = (rng.choice(["Doctor unavailable", "Emergency at the clinic", "Attending a conference", "Doctor unwell"]) if by == "doctor" else rng.choice(["Clash with work", "Travelling", "Not feeling well", "Family commitment", "Exam"])) if moved else None
                aid_ = db.execute("INSERT INTO appointments(doctor_id,patient_id,start_ts,end_ts,status,reason,note,note_ts,reschedule_count,rescheduled_from,rescheduled_by,reschedule_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                  (doc["id"], p["id"], t.strftime("%Y-%m-%d %H:%M"), (t + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M"), status, rng.choice(REASONS), note, iso(t + timedelta(minutes=25)) if note else None,
                                   1 if moved else 0, old, by, reason)).lastrowid; n_app += 1
                if moved: db.execute("INSERT INTO appointment_changes(appt_id, old_start, new_start, by_role, by_id, reason, created) VALUES(?,?,?,?,?,?,?)",
                                     (aid_, old, t.strftime("%Y-%m-%d %H:%M"), by, doc["id"] if by == "doctor" else p["id"], reason, iso(min(now, t - timedelta(days=rng.randint(0, 2), hours=rng.randint(1, 20))))))
    db.commit(); reminders.run_reminders(db); db.close(); print(f"built: {len(docs)} doctors, {n} patients, {sum(len(p['scans']) for p in people)} scans with stored photos, {n_app} appointments")
