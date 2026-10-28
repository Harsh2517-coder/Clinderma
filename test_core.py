"""Core rules of the platform. Run: python3 -m pytest -q test_core.py   (uses a throw-away database)"""
import os, tempfile
os.environ["WELLNESS_DB"] = os.path.join(tempfile.mkdtemp(), "t.db"); os.environ["WELLNESS_DEV"] = "1"; os.environ.pop("SMTP_HOST", None)
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import app as A
import insights

PW = "test-password-1"


def client(): return TestClient(A.app)


def login(email, pw=PW):
    c = client(); r = c.post("/api/login", json={"email": email, "password": pw}); assert r.status_code == 200, r.text; return c


@pytest.fixture(scope="module")
def world():
    db = A.connect()
    def user(role, name, email):
        return db.execute("INSERT INTO users(role,name,email,pw_hash,created) VALUES(?,?,?,?,'now')", (role, name, email, A.hash_pw(PW))).lastrowid
    admin = user("admin", "Admin", "admin@t.io")
    d1, d2 = user("doctor", "Dr One", "d1@t.io"), user("doctor", "Dr Two", "d2@t.io")
    for d in (d1, d2):
        db.execute("INSERT INTO doctors(user_id, specialization, focus) VALUES(?,?,?)", (d, "Ayurveda", '["digestive"]' if d == d1 else '["endocrine"]'))
        for wd in range(7): db.execute("INSERT INTO availability(doctor_id,weekday,start,end,slot_minutes) VALUES(?,?,?,?,?)", (d, wd, "00:00", "23:00", 60))
    db.commit(); db.close()
    return {"d1": d1, "d2": d2}


def new_patient(name, email):
    c = client(); assert c.post("/api/register", json={"name": name, "email": email, "password": PW}).status_code == 200; return c


def pick(c, did, zones=None):
    """first scan, then choose a doctor (the order the platform enforces)"""
    assert c.post("/api/patient/scan-result", json={"zone_counts": zones or {"forehead": 6, "chin": 1}}).status_code == 200
    r = c.post("/api/patient/doctor", json={"doctor_id": did}); assert r.status_code == 200, r.text


def first_slot(c):
    s = c.get("/api/slots").json(); day = sorted(s)[0]; return f"{day} {s[day][0]}"


def test_auth_and_roles(world):
    assert client().get("/api/me").status_code == 401
    assert client().post("/api/login", json={"email": "admin@t.io", "password": "wrong"}).status_code == 401
    p = new_patient("Pat", "pat@t.io")
    assert p.get("/api/me").json()["onboarding"] == {"needs_intake": True, "needs_scan": True, "needs_doctor": True, "doctor_id": None}
    assert p.get("/api/admin/doctors").status_code == 403 and p.get("/api/doctor/summary").status_code == 403
    assert login("d1@t.io").get("/api/admin/stats").status_code == 403
    assert login("admin@t.io").get("/api/admin/stats").status_code == 200
    assert client().post("/api/register", json={"name": "x", "email": "pat@t.io", "password": PW}).status_code == 409
    assert client().post("/api/register", json={"name": "x", "email": "short@t.io", "password": "short"}).status_code == 422


def test_onboarding_order_scan_before_doctor_and_intake(world):
    p = new_patient("Ona", "ona@t.io")
    assert p.get("/api/me").json()["onboarding"] == {"needs_intake": True, "needs_scan": True, "needs_doctor": True, "doctor_id": None}
    assert p.post("/api/patient/doctor", json={"doctor_id": world["d1"]}).status_code == 409          # scan first
    assert p.get("/api/patient/recommended-doctors").status_code == 409 and p.get("/api/slots").status_code == 409
    r = p.put("/api/patient/intake", json={"height_cm": 170, "weight_kg": 72.25, "conditions": ["digestive"]}); assert r.status_code == 200 and r.json()["bmi"] == 25.0
    assert p.put("/api/patient/intake", json={"conditions": ["not-a-system"]}).status_code == 422
    p.post("/api/patient/scan-result", json={"zone_counts": {"forehead": 9, "chin": 1}})
    rec = p.get("/api/patient/recommended-doctors").json()
    assert rec[0]["id"] == world["d1"] and rec[0]["recommended"] and rec[0]["why"] and rec[1]["score"] == 0       # digestive doctor first for a forehead scan + digestive history
    assert p.post("/api/patient/doctor", json={"doctor_id": 9999}).status_code == 404
    assert p.post("/api/patient/doctor", json={"doctor_id": world["d1"]}).status_code == 200
    assert p.post("/api/patient/doctor", json={"doctor_id": world["d2"]}).status_code == 409          # later changes go through switch-doctor
    assert p.get("/api/me").json()["onboarding"] == {"needs_intake": False, "needs_scan": False, "needs_doctor": False, "doctor_id": world["d1"]}


def test_switch_doctor_keeps_history_visible_to_both(world):
    p = new_patient("Swi", "swi@t.io"); pick(p, world["d1"]); pid = p.get("/api/me").json()["id"]
    slot = first_slot(p); old = p.post("/api/appointments", json={"start": slot, "reason": "forehead"}).json()["id"]
    d1, d2 = login("d1@t.io"), login("d2@t.io")
    assert d1.post(f"/api/doctor/appointments/{old}/note", json={"summary": "Digestive advice given."}).status_code == 200
    upcoming = p.post("/api/appointments", json={"start": [f"{d} {t[1]}" for d, t in sorted(p.get("/api/slots").json().items())][0]}).json()["id"]
    assert p.post("/api/patient/switch-doctor", json={"doctor_id": world["d1"]}).status_code == 409        # same doctor
    assert p.post("/api/patient/switch-doctor", json={"doctor_id": world["d2"], "reason": "new hormonal concern"}).status_code == 200
    assert upcoming not in [a["id"] for a in p.get("/api/patient/appointments").json() if a["status"] == "booked"]   # old doctor's upcoming booking cancelled
    new = d2.get(f"/api/doctor/patients/{pid}").json()                                                     # new doctor sees the old doctor's note
    assert new["is_current_doctor"] and any(a["note"] == "Digestive advice given." for a in new["appointments"]) and len(new["assignments"]) == 2
    prev = d1.get(f"/api/doctor/patients/{pid}").json()                                                    # previous doctor can still read, but is no longer current
    assert not prev["is_current_doctor"] and prev["scans"] and prev["appointments"]
    assert pid in [x["id"] for x in d1.get("/api/doctor/summary").json()["former_patients"]] and pid not in [x["id"] for x in d1.get("/api/doctor/summary").json()["patients"]]
    assert d1.post(f"/api/doctor/appointments/{old}/note", json={"summary": "late edit"}).status_code == 200  # own past appointment note stays editable by its author
    s = p.get("/api/slots").json(); day = sorted(s)[0]                                                          # the patient already holds the first hour; take another
    assert p.post("/api/appointments", json={"start": f"{day} {s[day][-1]}"}).status_code == 200                    # bookings now go to the new doctor
    assert [h["doctor"] for h in p.get("/api/patient/doctors-history").json()] == ["Dr Two", "Dr One"]
    stranger = login("d2@t.io"); other = new_patient("Zed", "zed@t.io"); pick(other, world["d1"])
    assert stranger.get(f"/api/doctor/patients/{other.get('/api/me').json()['id']}").status_code == 404          # no link, no access
    a = login("admin@t.io").get(f"/api/admin/patients/{pid}").json(); assert len(a["assignments"]) == 2 and a["current_doctor_id"] == world["d2"]


def test_booking_blocks_double_booking_and_frees_on_cancel(world):
    a, b = new_patient("A", "a@t.io"), new_patient("B", "b@t.io")
    for c in (a, b): pick(c, world["d1"])
    slot = first_slot(a); day, hhmm = slot.split()
    r = a.post("/api/appointments", json={"start": slot, "reason": "acne"}); assert r.status_code == 200
    assert hhmm not in b.get("/api/slots").json().get(day, [])                      # hidden from the next patient
    assert b.post("/api/appointments", json={"start": slot}).status_code == 409     # and refused if forced
    assert a.post("/api/appointments", json={"start": slot}).status_code == 409
    assert a.delete(f"/api/appointments/{r.json()['id']}").status_code == 200
    assert b.post("/api/appointments", json={"start": slot}).status_code == 200      # cancelled slot is free again


def test_cannot_book_outside_template_or_in_past(world):
    p = new_patient("C", "c@t.io"); pick(p, world["d2"])
    assert p.post("/api/appointments", json={"start": "2020-01-06 10:00"}).status_code == 409                     # past
    tomorrow = datetime.now() + timedelta(days=1)
    assert p.post("/api/appointments", json={"start": tomorrow.strftime("%Y-%m-%d") + " 23:30"}).status_code == 409   # outside 00:00-23:00 chart
    assert p.post("/api/appointments", json={"start": "garbage"}).status_code == 422


def test_weekday_only_chart(world):
    d = login("d2@t.io"); assert d.put("/api/doctor/availability", json=[{"weekday": wd, "start": "09:00", "end": "12:00", "slot_minutes": 30} for wd in range(5)]).status_code == 200
    p = new_patient("W", "w@t.io"); pick(p, world["d2"]); s = p.get("/api/slots?days=21").json()
    assert s and all(datetime.strptime(day, "%Y-%m-%d").weekday() < 5 for day in s)                                # no weekend slots
    assert all(t in ("09:00", "09:30", "10:00", "10:30", "11:00", "11:30") for ts in s.values() for t in ts)       # 30-min slots inside the hours
    assert d.put("/api/doctor/availability", json=[{"weekday": 9, "start": "09:00", "end": "12:00"}]).status_code == 422
    assert d.put("/api/doctor/availability", json=[{"weekday": 1, "start": "12:00", "end": "09:00"}]).status_code == 422


def test_doctor_sees_only_own_patients_and_notes_reach_patient(world):
    p = new_patient("N", "n@t.io"); pick(p, world["d1"]); slot = first_slot(p)
    aid = p.post("/api/appointments", json={"start": slot, "reason": "forehead spots"}).json()["id"]
    d1, d2 = login("d1@t.io"), login("d2@t.io"); pid = p.get("/api/me").json()["id"]
    s = d1.get("/api/doctor/summary").json(); assert any(x["id"] == pid for x in s["patients"]) and any(a["id"] == aid for a in s["upcoming"])
    assert not any(x["id"] == pid for x in d2.get("/api/doctor/summary").json()["patients"])
    assert d2.get(f"/api/doctor/patients/{pid}").status_code == 404                                                  # another doctor's patient
    assert d2.post(f"/api/doctor/appointments/{aid}/note", json={"summary": "x"}).status_code == 404
    assert d1.post(f"/api/doctor/appointments/{aid}/note", json={"summary": "Advised diet change; review in 2 weeks."}).status_code == 200
    mine = [a for a in p.get("/api/patient/appointments").json() if a["id"] == aid][0]
    assert mine["note"].startswith("Advised") and mine["status"] == "completed"                                       # patient sees the doctor's note
    assert any(a["id"] == aid for a in d1.get("/api/doctor/summary").json()["previous"])


def test_admin_sees_everything(world):
    ad = login("admin@t.io"); docs = ad.get("/api/admin/doctors").json(); assert {d["id"] for d in docs} >= {world["d1"], world["d2"]}
    assert len(ad.get(f"/api/admin/doctors/{world['d1']}/patients").json()) >= 1
    cal = ad.get("/api/admin/calendar", params={"start": "2000-01-01 00:00", "end": "2100-01-01 00:00"}).json(); assert cal and {"doctor", "patient"} <= set(cal[0])
    assert ad.post("/api/admin/doctors", json={"name": "Dr New", "email": "new@t.io", "password": PW, "specialization": "TCM"}).status_code == 200
    assert login("new@t.io").get("/api/doctor/availability").json()                                                  # default weekday chart created


def test_scan_insights_combine_zones_with_history(world):
    p = new_patient("S", "s@t.io"); pick(p, world["d1"])
    fit = p.post("/api/patient/scan-result", json={"zone_counts": {"forehead": 9, "chin": 1}}).json()
    assert fit["findings"][0]["zone"] == "forehead" and not fit["findings"][0]["matches_reported_conditions"] and "did not report" in fit["findings"][0]["personal_note"]
    p.put("/api/patient/intake", json={"conditions": ["digestive"]})
    sick = p.post("/api/patient/scan-result", json={"zone_counts": {"forehead": 9, "chin": 1}}).json()
    f = sick["findings"][0]; assert f["matches_reported_conditions"] and "already told us" in f["personal_note"] and sick["suggest_appointment"]
    assert f["evidence"] == "traditional" and "not a medical diagnosis" in sick["disclaimer"]
    assert insights.analyse_zones({"nose": 1})["findings"] == []                                                       # too few spots: no pattern
    assert p.post("/api/patient/scan-result", json={"zone_counts": {"elbow": 3}}).status_code == 422
    assert len(p.get("/api/patient/scans").json()) == 3                                                  # pick() scan + 2 above


def _img_b64():
    import base64, cv2, numpy as np
    return base64.b64encode(cv2.imencode(".jpg", np.full((60, 60, 3), 128, np.uint8))[1].tobytes()).decode()


def _fake_scan(img, side=None):
    import numpy as np
    return {"status": "ok", "zone_counts": {"forehead": 8, "left_cheek": 2, "right_cheek": 2, "nose": 1, "chin": 1}, "zone_area": {z: 100000 for z in insights.ZONES}, "total_spots": 14, "preview": np.zeros((50, 50, 3), np.uint8)}


def test_scan_endpoint_validation_and_no_face(world):
    import base64
    p = new_patient("Scn", "scn@t.io"); img = _img_b64()
    assert p.post("/api/patient/scan", json={}).status_code == 422                                   # front photo required
    assert p.post("/api/patient/scan", json={"front": base64.b64encode(b"not an image").decode()}).status_code == 422
    r = p.post("/api/patient/scan", json={"front": img, "left": img, "right": img}); assert r.status_code == 422 and "front photo" in r.json()["detail"]   # blank image: no face found
    assert p.get("/api/patient/scans").json() == []                                                      # failed scans store nothing


def test_first_scan_needs_three_angles_later_scans_do_not(world, monkeypatch):
    import scan as scan_mod
    monkeypatch.setattr(scan_mod, "scan_image", _fake_scan); p = new_patient("Ang", "ang@t.io"); img = _img_b64()
    for views in ({"front": img}, {"front": img, "left": img}, {"front": img, "right": img}):
        r = p.post("/api/patient/scan", json=views); assert r.status_code == 422 and "all three angles" in r.json()["detail"]
    r = p.post("/api/patient/scan", json={"front": img, "left": img, "right": img}); assert r.status_code == 200 and r.json()["report"]["views"] == ["front", "left", "right"]
    assert "accuracy" in r.json()["report"] and r.json()["report"]["accuracy"]["photos"] > 100                 # measured accuracy travels with every result
    assert all(f["reliability"] is not None for f in r.json()["report"]["findings"])
    assert p.post("/api/patient/scan", json={"front": img}).status_code == 200                            # a later quick scan may be front only
    assert len(p.get("/api/patient/scans").json()) == 2


def test_raw_counts_endpoint_is_off_outside_dev_mode(world, monkeypatch):
    monkeypatch.setattr(A, "DEV", False); p = new_patient("Dev", "dev@t.io")
    assert p.post("/api/patient/scan-result", json={"zone_counts": {"forehead": 9}}).status_code == 404       # patients cannot inject fake counts


def _add_scan(pid, days_ago, total):
    db = A.connect(); rep = insights.analyse_zones({"forehead": total, "chin": 1}); rep["total_spots"] = total + 1
    db.execute("INSERT INTO scans(patient_id, created, report_json) VALUES(?,?,?)", (pid, (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds"), __import__("json").dumps(rep))); db.commit(); db.close()


def test_progress_trend_and_weekly_due_status(world):
    p = new_patient("Prg", "prg@t.io"); pick(p, world["d1"]); pid = p.get("/api/me").json()["id"]
    db = A.connect(); db.execute("DELETE FROM scans WHERE patient_id=?", (pid,)); db.commit(); db.close()
    assert p.get("/api/patient/progress").json()["status"] == "no_scan"
    _add_scan(pid, 22, 30); _add_scan(pid, 15, 24); _add_scan(pid, 9, 20)
    pr = p.get("/api/patient/progress").json()
    assert pr["n"] == 3 and pr["due"] and pr["status"] == "due" and pr["days_since"] == 9 and pr["change_vs_first_pct"] < 0 and [s["total"] for s in pr["series"]] == [31, 25, 21]
    assert p.get("/api/me").json()["scan_status"]["due"] is True                                                   # shown as a banner after login
    p.post("/api/patient/scan-result", json={"zone_counts": {"forehead": 5}}); assert p.get("/api/patient/progress").json()["due"] is False
    assert login("d1@t.io").get(f"/api/doctor/patients/{pid}").json()["progress"]["n"] == 4                         # the doctor sees the same trend


def test_reminders_once_per_week_respect_opt_out_and_retry_failures(world, monkeypatch):
    import mailer, reminders
    p = new_patient("Rem", "rem@t.io"); pick(p, world["d1"]); pid = p.get("/api/me").json()["id"]
    db = A.connect(); db.execute("DELETE FROM scans WHERE patient_id=?", (pid,)); db.commit(); db.close()
    _add_scan(pid, 9, 10); slot = first_slot(p); p.post("/api/appointments", json={"start": slot})
    db = A.connect(); now = datetime.strptime(slot, "%Y-%m-%d %H:%M") - timedelta(hours=5)                            # 5 hours before the appointment
    out = reminders.run_reminders(db, now); kinds = sorted(k for k, to, st in out if to == "rem@t.io"); assert kinds == ["appt_reminder", "scan_reminder"] and all(st == "dry_run" for _, _, st in out)
    assert all(st == "duplicate" for k, to, st in reminders.run_reminders(db, now) if to == "rem@t.io")               # never repeated
    log = db.execute("SELECT subject, body FROM email_log WHERE to_addr='rem@t.io'").fetchall(); assert len(log) >= 2 and not any("forehead" in (r["body"] or "") for r in log)   # no health findings in emails
    p.put("/api/patient/reminders", json={"enabled": False}); nxt = now + timedelta(days=8)
    assert [x for x in reminders.run_reminders(db, nxt) if x[1] == "rem@t.io"] == []                                  # opted out: nothing sent
    p.put("/api/patient/reminders", json={"enabled": True})
    class Broken:                                                                                                    # SMTP server that fails: logged as failed, then retried next run
        def __init__(self, *a, **k): raise OSError("smtp down")
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid"); monkeypatch.setenv("SMTP_USER", "x"); monkeypatch.setenv("SMTP_PASSWORD", "y"); monkeypatch.setattr(mailer.smtplib, "SMTP", Broken)
    st = [x[2] for x in reminders.run_reminders(db, nxt) if x[1] == "rem@t.io"]; assert st and set(st) == {"failed"}
    sent = []
    class Good:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, u, p): pass
        def send_message(self, m): sent.append((m["To"], m["Subject"]))
    monkeypatch.setattr(mailer.smtplib, "SMTP", Good); st2 = [x[2] for x in reminders.run_reminders(db, nxt) if x[1] == "rem@t.io"]
    assert set(st2) == {"sent"} and any(to == "rem@t.io" for to, _ in sent); db.close()


def test_booking_logs_a_confirmation_email_and_admin_can_see_it(world):
    p = new_patient("Cnf", "cnf@t.io"); pick(p, world["d1"]); p.post("/api/appointments", json={"start": first_slot(p)})
    rows = login("admin@t.io").get("/api/admin/emails").json()["rows"]; assert any(r["to_addr"] == "cnf@t.io" and r["kind"] == "appt_confirm" for r in rows)
    assert p.get("/api/admin/emails").status_code == 403


def test_even_spread_gives_no_regional_finding_instead_of_crashing():
    r = insights.analyse_zones({"forehead": 6, "left_cheek": 6, "right_cheek": 6, "nose": 6, "chin": 6}, None, {z: 100000 for z in insights.ZONES})
    assert r["findings"] == [] and "evenly" in r["summary"] and r["suggest_appointment"] is False
    r2 = insights.analyse_zones({"forehead": 6, "left_cheek": 6, "right_cheek": 6, "nose": 6, "chin": 6})            # same without area weighting
    assert r2["findings"] == []


def test_generalist_ranks_between_matching_and_non_matching_specialists():
    docs = [{"id": 1, "name": "Digestive", "specialization": "", "focus": ["digestive"]}, {"id": 2, "name": "Hormonal", "specialization": "", "focus": ["endocrine"]}, {"id": 3, "name": "General", "specialization": "", "focus": []}]
    rep = insights.analyse_zones({"forehead": 9, "chin": 1})
    r = insights.recommend_doctors(docs, rep, None)
    assert [d["name"] for d in r] == ["Digestive", "General", "Hormonal"] and r[0]["recommended"] and r[1]["recommended"] and not r[2]["recommended"]


def test_outlook_lists_conditions_with_evidence_tiers_and_personal_risk_factors():
    it = {"conditions": ["digestive"], "bmi": 27.1, "family_history": "Father has diabetes", "lifestyle": {"stress": "High", "sleep_hours": 5}}
    r = insights.analyse_zones({"forehead": 9, "chin": 8, "nose": 1}, it); f = {x["zone"]: x for x in r["findings"]}
    fo = f["forehead"]["outlook"][0]; assert fo["already_reported"] and fo["evidence"] == "traditional" and "high stress level" in fo["risk_factors"] and fo["watch_for"]
    ch = f["chin"]["outlook"][0]; assert "partial modern" in ch["evidence"] and any("BMI" in x for x in ch["risk_factors"]) and any("diabetes" in x for x in ch["risk_factors"])
    assert r["skin_outlook"]["evidence"] == "established" and len(r["skin_outlook"]["watch_for"]) == 3
    assert insights.analyse_zones({"forehead": 9})["findings"][0]["outlook"][0]["risk_factors"] == []                       # no intake form: no personal factors, no crash
    assert all(o["system"] and o["watch_for"] for x in r["findings"] for o in x["outlook"])


def test_detector_parsers_nms_and_config():
    import numpy as np, detectors as D
    dets = np.array([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]); lab = np.full((1, 2, 3), -8.0); lab[0, 0, 1] = 4.0
    r = D.parse_rfdetr(dets, lab, 1000, 800, 0.3, ["comedone", "papule", "pustule"]); assert len(r) == 1 and r[0]["cls"] == "papule" and r[0]["box"][:2] == (400.0, 320.0)
    y = np.zeros((1, 6, 2100)); y[0, :4, 0] = [256, 256, 100, 100]; y[0, 4, 0] = 0.9; y[0, :4, 1:] = 10; y[0, 4:, 1:] = 0.05
    r2 = D.parse_yolo_onnx(y, 1024, 512, 512, 0.25, ["a", "b"]); assert len(r2) == 1 and r2[0]["box"][0] == 412.0
    assert len(D.nms([{"box": (0, 0, 10, 10), "conf": .9, "cls": "a"}, {"box": (1, 1, 11, 11), "conf": .8, "cls": "a"}, {"box": (50, 50, 60, 60), "conf": .7, "cls": "a"}])) == 2


def test_simulated_outlook_ranks_obvious_risk_higher_and_is_labelled_simulated():
    import risk_sim
    sick = {"conditions": ["digestive"], "age": 35, "sex": "Female", "bmi": 28, "family_history": "mother had a stomach ulcer", "lifestyle": {"stress": "High", "sleep_hours": 4.5}}
    well = {"conditions": [], "age": 24, "sex": "Female", "bmi": 22, "family_history": "", "lifestyle": {"stress": "Low", "sleep_hours": 8}}
    a = risk_sim.simulated_outlook(sick, {"forehead": 9, "left_cheek": 2, "right_cheek": 2, "nose": 1, "chin": 1}); b = risk_sim.simulated_outlook(well, {"forehead": 1, "left_cheek": 2, "right_cheek": 2, "nose": 0, "chin": 1})
    ta = {x["system"]: x["tier"] for x in a["tiers"]}; tb = {x["system"]: x["tier"] for x in b["tiers"]}
    assert a["simulated"] is True and "SYNTHETIC" in a["note"] and ta["digestive"] == "higher" and tb["digestive"] != "higher" and ta["stress_sleep"] == "higher"
    assert risk_sim.simulated_outlook(None, {})["tiers"]                                       # no intake form and no spots: still works with defaults
    d = risk_sim.generate(2000, 0.0, 5); assert d["future"].shape == (2000, 8) and 0.03 < d["future"].mean() < 0.3                  # cohort generator sanity


def test_scan_report_carries_simulated_outlook(world, monkeypatch):
    import scan as scan_mod
    monkeypatch.setattr(scan_mod, "scan_image", _fake_scan); p = new_patient("Sim", "sim@t.io"); img = _img_b64()
    r = p.post("/api/patient/scan", json={"front": img, "left": img, "right": img}).json()["report"]; assert r["simulated_outlook"]["simulated"] is True and len(r["simulated_outlook"]["tiers"]) == 8
