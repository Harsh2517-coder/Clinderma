"""Creates demo data (rebuilds the demo DB every time).  python3 seed.py = large demo: 8 doctors, ~36 patients with many scans using REAL photos (stored, visible only to the patient and their doctors),
switched doctors and a full calendar. Needs the photo cache: python3 demo_data.py build ~/Downloads/patients_part1.   python3 seed.py --quick = the small synthetic demo (no photos).
DEMO LOGINS (local prototype only):   admin@demo.local / doctor1..8@demo.local / patient1..36@demo.local   password for all: Demo-pass-2026"""
import json, os, sys
from datetime import datetime, timedelta

import app as A
import insights

PW = "Demo-pass-2026"
AREA = {"forehead": 250000, "left_cheek": 600000, "right_cheek": 600000, "nose": 150000, "chin": 200000}
HISTORY = {"Riya Shah": ([57, 50, 43, 36, 29, 8], [1.7, 1.55, 1.4, 1.25, 1.1, 1.0]),      # last scan 8 days ago: weekly scan is DUE
           "Arjun Mehta": ([21, 14, 3], [1.3, 1.15, 1.0]), "Li Na": ([12, 5], [1.2, 1.0])}


def quick():
    if os.path.exists(A.DB_PATH): os.remove(A.DB_PATH)
    A.init_db(); db = A.connect(); now = datetime.now().isoformat(timespec="seconds")
    def user(role, name, email): return db.execute("INSERT INTO users(role,name,email,pw_hash,created) VALUES(?,?,?,?,?)", (role, name, email, A.hash_pw(PW), now)).lastrowid
    user("admin", "Platform Admin", "admin@demo.local")
    docs = []
    for name, em, spec, focus in (("Dr. Asha Rao", "doctor1@demo.local", "Ayurveda: digestive and stress", ["digestive", "stress_sleep"]),
                                  ("Dr. Wei Chen", "doctor2@demo.local", "Integrative medicine (TCM): hormonal health", ["endocrine", "reproductive"]),
                                  ("Dr. Meera Iyer", "doctor3@demo.local", "General Ayurveda physician", [])):
        d = user("doctor", name, em); docs.append(d); db.execute("INSERT INTO doctors(user_id, specialization, focus) VALUES(?,?,?)", (d, spec, json.dumps(focus)))
        hours = [(wd, "10:00", "13:00") for wd in range(5)] + [(wd, "15:00", "17:00") for wd in (1, 3)] if d == docs[0] else \
                [(wd, "09:00", "12:00") for wd in (0, 2, 4)] if d == docs[1] else [(wd, "11:00", "16:00") for wd in range(5)]
        db.executemany("INSERT INTO availability(doctor_id,weekday,start,end,slot_minutes) VALUES(?,?,?,?,20)", [(d, *h) for h in hours])
    pats = []
    for name, em, doc, cond, zones in (("Riya Shah", "patient1@demo.local", docs[0], ["digestive"], {"forehead": 11, "left_cheek": 2, "right_cheek": 2, "nose": 1, "chin": 1}),
                                       ("Arjun Mehta", "patient2@demo.local", docs[0], [], {"forehead": 7, "left_cheek": 1, "right_cheek": 1, "nose": 2, "chin": 0}),
                                       ("Li Na", "patient3@demo.local", docs[1], ["endocrine"], {"forehead": 1, "left_cheek": 3, "right_cheek": 3, "nose": 0, "chin": 9})):
        p = user("patient", name, em); pats.append(p)
        intake = {"height_cm": 165, "weight_kg": 62, "age": 27, "sex": "", "conditions": cond, "condition_notes": "", "medications": "", "allergies": "", "family_history": "",
                  "lifestyle": {"sleep_hours": 7}, "skin_notes": "", "bmi": insights.bmi(165, 62), "saved": now}
        db.execute("INSERT INTO patients(user_id, doctor_id, intake_json) VALUES(?,?,?)", (p, doc, json.dumps(intake)))
        if name == "Arjun Mehta":      # switched doctors after a few months: both doctors keep access to the record
            db.execute("INSERT INTO patient_doctors(patient_id,doctor_id,start_ts,end_ts,reason) VALUES(?,?,?,?,?)", (p, docs[2], (datetime.now() - timedelta(days=130)).isoformat(timespec="seconds"), (datetime.now() - timedelta(days=20)).isoformat(timespec="seconds"), "first choice"))
            db.execute("INSERT INTO patient_doctors(patient_id,doctor_id,start_ts,reason) VALUES(?,?,?,?)", (p, doc, (datetime.now() - timedelta(days=20)).isoformat(timespec="seconds"), "new digestive concern"))
        else:
            db.execute("INSERT INTO patient_doctors(patient_id,doctor_id,start_ts,reason) VALUES(?,?,?,?)", (p, doc, (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds"), "first choice"))
        days, factors = HISTORY[name]       # weekly scans, oldest first; counts shrink over time so the progress charts have a trend
        for d, f in zip(days, factors):
            z = {k: int(round(v * f)) for k, v in zones.items()}; rep = insights.analyse_zones(z, intake, AREA); rep["zone_area"] = AREA; rep["views"] = ["front", "left", "right"]; A.finalize_report(rep, intake, z, AREA)
            db.execute("INSERT INTO scans(patient_id, created, report_json) VALUES(?,?,?)", (p, (datetime.now() - timedelta(days=d)).isoformat(timespec="seconds"), json.dumps(rep)))
    t0 = datetime.now().replace(minute=0, second=0, microsecond=0)
    def nxt(weekday, hh, mm):      # next given weekday (0=Mon) at hh:mm, strictly in the future, inside Dr. Asha Rao's chart (Mon-Fri 10-13)
        d = (datetime.now() + timedelta(days=1)).replace(hour=hh, minute=mm, second=0, microsecond=0)
        while d.weekday() != weekday: d += timedelta(days=1)
        return d
    def appt(doc, pat, start, status, reason, note=None):
        db.execute("INSERT INTO appointments(doctor_id,patient_id,start_ts,end_ts,status,reason,note,note_ts) VALUES(?,?,?,?,?,?,?,?)",
                   (doc, pat, start.strftime("%Y-%m-%d %H:%M"), (start + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M"), status, reason, note, now if note else None))
    appt(docs[0], pats[0], t0 - timedelta(days=9), "completed", "Forehead breakouts", "Advised a lighter evening meal, warm water, and triphala at night. Review in 2 weeks.")
    appt(docs[0], pats[0], nxt(1, 10, 20), "booked", "Follow-up")
    appt(docs[2], pats[1], t0 - timedelta(days=100), "completed", "General check-up", "Mild seasonal skin dryness; advised hydration and sesame-oil massage. Review if forehead spots appear.")
    appt(docs[0], pats[1], nxt(2, 11, 0), "booked", "New spots on forehead")
    db.commit(); db.close(); print("demo database ready:", A.DB_PATH)


def main():
    if "--quick" in sys.argv: return quick()
    import demo_data, reminders
    demo_data.build_big_demo(A, insights, reminders)


if __name__ == "__main__":
    main()
