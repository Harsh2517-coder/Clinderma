"""Three-level wellness platform API: admin -> doctors -> patients.  Run: python3 app.py  (http://127.0.0.1:7890)
SQLite + signed-cookie sessions + scrypt password hashes.  Prototype: local data only, not for real patient records without a security review."""
import base64, hashlib, hmac, json, os, secrets, sqlite3, time
from datetime import datetime, timedelta

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import insights, mailer, reminders, risk_sim

HERE = os.path.dirname(os.path.abspath(__file__))
mailer.load_env(os.path.join(HERE, ".env"))       # SMTP settings for real email; without them emails are only logged (dry run)
DEV = os.environ.get("WELLNESS_DEV") == "1"       # enables the test-only endpoint that accepts raw zone counts
DB_PATH = os.environ.get("WELLNESS_DB", os.path.join(HERE, "data", "platform.db"))
SESSION_SECONDS = 8 * 3600
BOOKING_WINDOW_DAYS = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('admin','doctor','patient')), name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE, pw_hash TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS doctors(user_id INTEGER PRIMARY KEY REFERENCES users(id), specialization TEXT DEFAULT '', bio TEXT DEFAULT '',
  focus TEXT NOT NULL DEFAULT '[]');   -- JSON list of insights.SYSTEMS keys this doctor treats; empty = generalist
CREATE TABLE IF NOT EXISTS patients(user_id INTEGER PRIMARY KEY REFERENCES users(id), doctor_id INTEGER REFERENCES doctors(user_id), intake_json TEXT, email_reminders INTEGER NOT NULL DEFAULT 1);  -- doctor_id = CURRENT doctor
-- every doctor a patient has had; a doctor in this table can still READ the whole record after the patient switches away
CREATE TABLE IF NOT EXISTS patient_doctors(id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(user_id), doctor_id INTEGER NOT NULL REFERENCES doctors(user_id),
  start_ts TEXT NOT NULL, end_ts TEXT, reason TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS availability(id INTEGER PRIMARY KEY, doctor_id INTEGER NOT NULL REFERENCES doctors(user_id), weekday INTEGER NOT NULL CHECK(weekday BETWEEN 0 AND 6),
  start TEXT NOT NULL, end TEXT NOT NULL, slot_minutes INTEGER NOT NULL DEFAULT 20);
CREATE TABLE IF NOT EXISTS appointments(id INTEGER PRIMARY KEY, doctor_id INTEGER NOT NULL REFERENCES doctors(user_id), patient_id INTEGER NOT NULL REFERENCES patients(user_id),
  start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'booked' CHECK(status IN ('booked','cancelled','completed')),
  reason TEXT DEFAULT '', note TEXT, note_ts TEXT);
-- a doctor's slot can be held by only one live booking, and a patient cannot hold two at the same time
CREATE UNIQUE INDEX IF NOT EXISTS ux_doctor_slot ON appointments(doctor_id, start_ts) WHERE status <> 'cancelled';
CREATE UNIQUE INDEX IF NOT EXISTS ux_patient_slot ON appointments(patient_id, start_ts) WHERE status <> 'cancelled';
CREATE TABLE IF NOT EXISTS email_log(id INTEGER PRIMARY KEY, to_addr TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT DEFAULT '', created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(user_id), created TEXT NOT NULL, report_json TEXT NOT NULL);
"""


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False); c.row_factory = sqlite3.Row; c.execute("PRAGMA foreign_keys=ON"); return c


def init_db():
    c = connect(); c.executescript(SCHEMA)
    try: c.execute("ALTER TABLE patients ADD COLUMN email_reminders INTEGER NOT NULL DEFAULT 1")      # upgrade an older database
    except sqlite3.OperationalError: pass
    c.commit(); c.close()


def get_db():
    c = connect()
    try: yield c
    finally: c.close()


# ---------- auth ----------
def _secret():
    p = os.path.join(os.path.dirname(DB_PATH), "secret.key")
    if not os.path.exists(p): open(p, "wb").write(secrets.token_bytes(32)); os.chmod(p, 0o600)
    return open(p, "rb").read()


def hash_pw(pw, salt=None):
    salt = salt or secrets.token_bytes(16)
    return "scrypt$" + salt.hex() + "$" + hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1).hex()


def check_pw(pw, stored):
    try: _, salt, h = stored.split("$"); return hmac.compare_digest(hash_pw(pw, bytes.fromhex(salt)).split("$")[2], h)
    except Exception: return False


def make_token(uid):
    body = f"{uid}.{int(time.time()) + SESSION_SECONDS}"; return base64.urlsafe_b64encode(body.encode()).decode() + "." + hmac.new(_secret(), body.encode(), "sha256").hexdigest()


def read_token(tok):
    try:
        b64, sig = tok.split("."); body = base64.urlsafe_b64decode(b64).decode()
        if not hmac.compare_digest(hmac.new(_secret(), body.encode(), "sha256").hexdigest(), sig): return None
        uid, exp = body.split("."); return int(uid) if int(exp) > time.time() else None
    except Exception: return None


def current_user(request: Request, db=Depends(get_db)):
    uid = read_token(request.cookies.get("session", ""))
    u = db.execute("SELECT id, role, name, email FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    if not u: raise HTTPException(401, "Please log in")
    return dict(u)


def need(role):
    def dep(u=Depends(current_user)):
        if u["role"] != role: raise HTTPException(403, f"{role} access only")
        return u
    return dep


# ---------- slots ----------
def _slots_for(db, doctor_id, day):
    """Free slot start strings ('YYYY-MM-DD HH:MM') for one doctor on one date, from the weekly template minus live bookings and the past."""
    out = []
    for a in db.execute("SELECT start, end, slot_minutes FROM availability WHERE doctor_id=? AND weekday=? ORDER BY start", (doctor_id, day.weekday())):
        t = datetime.strptime(f"{day:%Y-%m-%d} {a['start']}", "%Y-%m-%d %H:%M"); end = datetime.strptime(f"{day:%Y-%m-%d} {a['end']}", "%Y-%m-%d %H:%M")
        while t + timedelta(minutes=a["slot_minutes"]) <= end:
            out.append((t, t + timedelta(minutes=a["slot_minutes"]))); t += timedelta(minutes=a["slot_minutes"])
    taken = {r[0] for r in db.execute("SELECT start_ts FROM appointments WHERE doctor_id=? AND start_ts LIKE ? AND status<>'cancelled'", (doctor_id, f"{day:%Y-%m-%d}%"))}
    now = datetime.now()
    return [(s, e) for s, e in out if s > now and s.strftime("%Y-%m-%d %H:%M") not in taken]


app = FastAPI(title="Face wellness platform")


class Login(BaseModel): email: str; password: str
class Register(BaseModel): name: str; email: str; password: str
class Choose(BaseModel): doctor_id: int
class Intake(BaseModel):
    height_cm: float | None = None; weight_kg: float | None = None; age: int | None = None; sex: str = ""
    conditions: list[str] = []; condition_notes: str = ""; medications: str = ""; allergies: str = ""; family_history: str = ""
    lifestyle: dict = {}; skin_notes: str = ""
class Scan(BaseModel): zone_counts: dict[str, int]
class Book(BaseModel): start: str; reason: str = ""
class Note(BaseModel): summary: str
class Hours(BaseModel): weekday: int; start: str; end: str; slot_minutes: int = 20
class NewDoctor(BaseModel): name: str; email: str; password: str; specialization: str = ""; focus: list[str] = []


@app.post("/api/login")
def login(b: Login, response: Response, db=Depends(get_db)):
    u = db.execute("SELECT * FROM users WHERE email=?", (b.email.strip().lower(),)).fetchone()
    if not u or not check_pw(b.password, u["pw_hash"]): raise HTTPException(401, "Wrong email or password")
    response.set_cookie("session", make_token(u["id"]), httponly=True, samesite="lax", max_age=SESSION_SECONDS)
    return {"role": u["role"], "name": u["name"]}


@app.post("/api/logout")
def logout(response: Response): response.delete_cookie("session"); return {"ok": True}


@app.post("/api/register")
def register(b: Register, response: Response, db=Depends(get_db)):
    if len(b.password) < 8: raise HTTPException(422, "Password must be at least 8 characters")
    try:
        cur = db.execute("INSERT INTO users(role,name,email,pw_hash,created) VALUES('patient',?,?,?,?)", (b.name.strip(), b.email.strip().lower(), hash_pw(b.password), datetime.now().isoformat(timespec="seconds")))
        db.execute("INSERT INTO patients(user_id) VALUES(?)", (cur.lastrowid,)); db.commit()
    except sqlite3.IntegrityError: raise HTTPException(409, "That email is already registered")
    response.set_cookie("session", make_token(cur.lastrowid), httponly=True, samesite="lax", max_age=SESSION_SECONDS)
    return {"role": "patient", "name": b.name}


@app.get("/api/me")
def me(u=Depends(current_user), db=Depends(get_db)):
    if u["role"] == "patient":
        p = db.execute("SELECT doctor_id, intake_json FROM patients WHERE user_id=?", (u["id"],)).fetchone()
        scans = db.execute("SELECT COUNT(*) FROM scans WHERE patient_id=?", (u["id"],)).fetchone()[0]
        # order on first login: intake form -> first face scan -> choose a doctor (recommended from the scan)
        u["onboarding"] = {"needs_intake": p["intake_json"] is None, "needs_scan": scans == 0, "needs_doctor": p["doctor_id"] is None, "doctor_id": p["doctor_id"]}
        pr = insights.progress(_scan_rows(db, u["id"])); u["scan_status"] = {"status": pr["status"], "due": pr["due"], "days_since": pr["days_since"], "next_due": pr["next_due"]}
    return u


# ---------- patient ----------
def _doctor_rows(db):
    return [{**dict(r), "focus": json.loads(r["focus"])} for r in db.execute("SELECT u.id, u.name, d.specialization, d.bio, d.focus FROM doctors d JOIN users u ON u.id=d.user_id ORDER BY u.name")]


@app.get("/api/doctors")
def list_doctors(u=Depends(current_user), db=Depends(get_db)): return _doctor_rows(db)


def _scan_rows(db, pid):
    return [{"created": r["created"], "report": json.loads(r["report_json"])} for r in db.execute("SELECT created, report_json FROM scans WHERE patient_id=? ORDER BY id", (pid,))]


def _latest_report(db, pid):
    r = db.execute("SELECT report_json FROM scans WHERE patient_id=? ORDER BY id DESC LIMIT 1", (pid,)).fetchone(); return json.loads(r["report_json"]) if r else None


@app.get("/api/patient/recommended-doctors")
def recommended(u=Depends(need("patient")), db=Depends(get_db)):
    """Doctors ranked by how well their focus matches the first scan and the intake form. Needs a scan first."""
    rep = _latest_report(db, u["id"])
    if rep is None: raise HTTPException(409, "Do your face scan first")
    return insights.recommend_doctors(_doctor_rows(db), rep, _intake(db, u["id"]))


def _assign(db, pid, did, reason=""):
    """Closes the current assignment, opens a new one, updates the patient's current doctor. Upcoming bookings with the previous doctor are cancelled."""
    now = datetime.now().isoformat(timespec="seconds"); old = db.execute("SELECT doctor_id FROM patients WHERE user_id=?", (pid,)).fetchone()["doctor_id"]
    db.execute("UPDATE patient_doctors SET end_ts=? WHERE patient_id=? AND end_ts IS NULL", (now, pid))
    db.execute("INSERT INTO patient_doctors(patient_id,doctor_id,start_ts,reason) VALUES(?,?,?,?)", (pid, did, now, reason))
    db.execute("UPDATE patients SET doctor_id=? WHERE user_id=?", (did, pid))
    if old is not None: db.execute("UPDATE appointments SET status='cancelled' WHERE patient_id=? AND doctor_id=? AND status='booked' AND start_ts>=?", (pid, old, datetime.now().strftime("%Y-%m-%d %H:%M")))
    db.commit()


@app.post("/api/patient/doctor")
def choose_doctor(b: Choose, u=Depends(need("patient")), db=Depends(get_db)):
    """First choice only, and only after the first scan (the scan decides which doctors are suggested). Later changes use /switch-doctor."""
    p = db.execute("SELECT doctor_id FROM patients WHERE user_id=?", (u["id"],)).fetchone()
    if _latest_report(db, u["id"]) is None: raise HTTPException(409, "Do your face scan first")
    if p["doctor_id"] is not None: raise HTTPException(409, "You already have a doctor; use switch-doctor to change")
    if not db.execute("SELECT 1 FROM doctors WHERE user_id=?", (b.doctor_id,)).fetchone(): raise HTTPException(404, "No such doctor")
    _assign(db, u["id"], b.doctor_id, "first choice"); return {"doctor_id": b.doctor_id}


class Switch(BaseModel): doctor_id: int; reason: str = ""


@app.post("/api/patient/switch-doctor")
def switch_doctor(b: Switch, u=Depends(need("patient")), db=Depends(get_db)):
    """Changes the current doctor. The previous doctor keeps read access to the record; the new doctor sees the full history including old notes."""
    cur = db.execute("SELECT doctor_id FROM patients WHERE user_id=?", (u["id"],)).fetchone()["doctor_id"]
    if cur is None: raise HTTPException(409, "Choose a doctor first")
    if b.doctor_id == cur: raise HTTPException(409, "That is already your doctor")
    if not db.execute("SELECT 1 FROM doctors WHERE user_id=?", (b.doctor_id,)).fetchone(): raise HTTPException(404, "No such doctor")
    _assign(db, u["id"], b.doctor_id, b.reason or "switched"); return {"doctor_id": b.doctor_id}


@app.get("/api/patient/doctors-history")
def my_doctor_history(u=Depends(need("patient")), db=Depends(get_db)):
    return _appt_rows(db.execute("SELECT du.name AS doctor, pd.doctor_id, pd.start_ts, pd.end_ts, pd.reason FROM patient_doctors pd JOIN users du ON du.id=pd.doctor_id WHERE pd.patient_id=? ORDER BY pd.id DESC", (u["id"],)))


@app.put("/api/patient/intake")
def save_intake(b: Intake, u=Depends(need("patient")), db=Depends(get_db)):
    bad = [c for c in b.conditions if c not in insights.SYSTEMS]
    if bad: raise HTTPException(422, f"Unknown condition keys: {bad}")
    d = b.model_dump(); d["bmi"] = insights.bmi(b.height_cm, b.weight_kg); d["saved"] = datetime.now().isoformat(timespec="seconds")
    db.execute("UPDATE patients SET intake_json=? WHERE user_id=?", (json.dumps(d), u["id"])); db.commit(); return d


@app.get("/api/patient/intake")
def get_intake(u=Depends(need("patient")), db=Depends(get_db)):
    r = db.execute("SELECT intake_json FROM patients WHERE user_id=?", (u["id"],)).fetchone()["intake_json"]; return json.loads(r) if r else None


@app.get("/api/systems")
def systems(u=Depends(current_user)): return insights.SYSTEMS


def _intake(db, pid):
    r = db.execute("SELECT intake_json FROM patients WHERE user_id=?", (pid,)).fetchone(); return json.loads(r["intake_json"]) if r and r["intake_json"] else None


@app.post("/api/patient/scan-result")
def scan_result(b: Scan, u=Depends(need("patient")), db=Depends(get_db)):
    """TEST ONLY (needs WELLNESS_DEV=1): stores a scan from raw zone counts. Real scans go through /api/patient/scan, which runs the detector."""
    if not DEV: raise HTTPException(404, "Not found")
    bad = [z for z in b.zone_counts if z not in insights.ZONES]
    if bad: raise HTTPException(422, f"Unknown zones: {bad}")
    rep = insights.analyse_zones(b.zone_counts, _intake(db, u["id"]))
    db.execute("INSERT INTO scans(patient_id, created, report_json) VALUES(?,?,?)", (u["id"], datetime.now().isoformat(timespec="seconds"), json.dumps(rep))); db.commit(); return rep


class ScanReq(BaseModel): front: str | None = None; left: str | None = None; right: str | None = None     # data-URL or base64 JPEG/PNG per view


def _decode_image(s):
    import cv2, numpy as np
    raw = base64.b64decode(s.split(",", 1)[-1])
    if len(raw) > 15_000_000: raise HTTPException(413, "Photo is too large (max 15 MB)")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(422, "That file could not be read as an image")
    return img


def finalize_report(rep, intake, counts, area):
    """Adds what every real scan report carries: the simulated outlook (research prototype) and the measured accuracy of the scan, with low-reliability flags per zone.
    Shared by the scan endpoint and the demo seed so demo reports look exactly like real ones."""
    import scan as scanner
    rep["simulated_outlook"] = risk_sim.simulated_outlook(intake, counts, area)      # trained on a SYNTHETIC cohort; qualitative tiers only
    acc = scanner.reliability()       # measured accuracy of the active detector per zone; low-recall zones are flagged
    if acc:
        rep["accuracy"] = {"validation": acc["validation"], "photos": acc["photos"], "recall": acc["filtered"]["recall"], "precision": acc["filtered"]["precision"], "zone_match": acc["filtered"]["agree"]}
        for f in rep["findings"]:
            z = acc["zones"].get(f["zone"]); f["reliability"] = {**z, "low": z["recall"] < 0.3} if z else None
    return rep


@app.post("/api/patient/scan")
def scan(b: ScanReq, u=Depends(need("patient")), db=Depends(get_db)):
    """Front photo (required) plus optional left/right profile photos. Photos are analysed in memory and NOT stored; only zone counts and the report are kept."""
    import cv2, scan as scanner
    views = {k: v for k, v in (("front", b.front), ("left", b.left), ("right", b.right)) if v}
    if "front" not in views: raise HTTPException(422, "A front photo is required")
    first = db.execute("SELECT COUNT(*) FROM scans WHERE patient_id=?", (u["id"],)).fetchone()[0] == 0
    if first and len(views) < 3: raise HTTPException(422, "Your first scan needs all three angles: front, left side and right side")
    counts, area, previews = {z: 0 for z in insights.ZONES}, {z: 0 for z in insights.ZONES}, []
    for name, data in views.items():
        s = scanner.scan_image(_decode_image(data), None if name == "front" else name)
        if s["status"] != "ok": raise HTTPException(422, f"{name} photo: {s['message']}")
        for z in insights.ZONES: counts[z] += s["zone_counts"][z]; area[z] += s["zone_area"][z]
        ok, jpg = cv2.imencode(".jpg", s["preview"], [cv2.IMWRITE_JPEG_QUALITY, 80]); previews.append({"view": name, "spots": s["total_spots"], "image": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})
    rep = insights.analyse_zones(counts, _intake(db, u["id"]), area); rep["views"] = list(views); rep["zone_area"] = area
    finalize_report(rep, _intake(db, u["id"]), counts, area)
    db.execute("INSERT INTO scans(patient_id, created, report_json) VALUES(?,?,?)", (u["id"], datetime.now().isoformat(timespec="seconds"), json.dumps(rep))); db.commit()
    return {"report": rep, "previews": previews}


@app.get("/api/patient/scans")
def my_scans(u=Depends(need("patient")), db=Depends(get_db)):
    return [{"id": r["id"], "created": r["created"], "report": json.loads(r["report_json"])} for r in db.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY id DESC", (u["id"],))]


@app.get("/api/patient/progress")
def my_progress(u=Depends(need("patient")), db=Depends(get_db)):
    pr = insights.progress(_scan_rows(db, u["id"])); pr["email_reminders"] = bool(db.execute("SELECT email_reminders FROM patients WHERE user_id=?", (u["id"],)).fetchone()[0])
    pr["email_live"] = mailer.configured(); return pr


class Remind(BaseModel): enabled: bool


@app.put("/api/patient/reminders")
def set_reminders(b: Remind, u=Depends(need("patient")), db=Depends(get_db)):
    db.execute("UPDATE patients SET email_reminders=? WHERE user_id=?", (1 if b.enabled else 0, u["id"])); db.commit(); return {"email_reminders": b.enabled}


@app.get("/api/slots")
def slots(days: int = 14, u=Depends(need("patient")), db=Depends(get_db)):
    p = db.execute("SELECT doctor_id FROM patients WHERE user_id=?", (u["id"],)).fetchone()
    if p["doctor_id"] is None: raise HTTPException(409, "Choose your doctor first")
    out, today = {}, datetime.now().date()
    for i in range(min(days, BOOKING_WINDOW_DAYS) + 1):
        day = today + timedelta(days=i); s = [t.strftime("%H:%M") for t, _ in _slots_for(db, p["doctor_id"], day)]
        if s: out[f"{day:%Y-%m-%d}"] = s
    return out


@app.post("/api/appointments")
def book(b: Book, bg: BackgroundTasks, u=Depends(need("patient")), db=Depends(get_db)):
    p = db.execute("SELECT doctor_id FROM patients WHERE user_id=?", (u["id"],)).fetchone()
    if p["doctor_id"] is None: raise HTTPException(409, "Choose your doctor first")
    try: t = datetime.strptime(b.start, "%Y-%m-%d %H:%M")
    except ValueError: raise HTTPException(422, "start must look like 2026-11-02 10:20")
    ok = [(s, e) for s, e in _slots_for(db, p["doctor_id"], t.date()) if s == t]
    if not ok or t > datetime.now() + timedelta(days=BOOKING_WINDOW_DAYS): raise HTTPException(409, "That time is not available")
    try:
        cur = db.execute("INSERT INTO appointments(doctor_id,patient_id,start_ts,end_ts,reason) VALUES(?,?,?,?,?)",
                         (p["doctor_id"], u["id"], b.start, ok[0][1].strftime("%Y-%m-%d %H:%M"), b.reason)); db.commit()
    except sqlite3.IntegrityError: raise HTTPException(409, "That slot was just taken")     # the unique index is the real guard against double booking
    doc = db.execute("SELECT name FROM users WHERE id=?", (p["doctor_id"],)).fetchone()["name"]; em = db.execute("SELECT email FROM users WHERE id=?", (u["id"],)).fetchone()["email"]
    bg.add_task(_email_in_background, em, u["name"], doc, t.strftime("%a %d %b, %I:%M %p"), f"appt-confirm:{cur.lastrowid}")
    return {"id": cur.lastrowid, "start": b.start}


def _email_in_background(to, patient, doctor, when, key):
    c = connect()
    try: subj, body = mailer.appointment_confirmation(patient, doctor, when); mailer.send_email(c, to, subj, body, "appt_confirm", key)
    finally: c.close()


def _appt_rows(rows): return [dict(r) for r in rows]


@app.get("/api/patient/appointments")
def my_appointments(u=Depends(need("patient")), db=Depends(get_db)):
    return _appt_rows(db.execute("SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, a.note, a.note_ts, du.name AS doctor FROM appointments a JOIN users du ON du.id=a.doctor_id "
                                 "WHERE a.patient_id=? ORDER BY a.start_ts DESC", (u["id"],)))


@app.delete("/api/appointments/{aid}")
def cancel(aid: int, u=Depends(current_user), db=Depends(get_db)):
    a = db.execute("SELECT * FROM appointments WHERE id=?", (aid,)).fetchone()
    if not a or (u["role"] == "patient" and a["patient_id"] != u["id"]) or (u["role"] == "doctor" and a["doctor_id"] != u["id"]): raise HTTPException(404, "Not found")
    if a["status"] != "booked": raise HTTPException(409, "Only booked appointments can be cancelled")
    db.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (aid,)); db.commit(); return {"ok": True}


# ---------- doctor ----------
@app.get("/api/doctor/summary")
def doctor_summary(u=Depends(need("doctor")), db=Depends(get_db)):
    now = datetime.now().strftime("%Y-%m-%d %H:%M"); pats = []
    for p in db.execute("SELECT u.id, u.name, u.email, p.intake_json FROM patients p JOIN users u ON u.id=p.user_id WHERE p.doctor_id=? ORDER BY u.name", (u["id"],)):
        scan = db.execute("SELECT created, report_json FROM scans WHERE patient_id=? ORDER BY id DESC LIMIT 1", (p["id"],)).fetchone()
        nxt = db.execute("SELECT start_ts FROM appointments WHERE patient_id=? AND doctor_id=? AND status='booked' AND start_ts>=? ORDER BY start_ts LIMIT 1", (p["id"], u["id"], now)).fetchone()
        rep = json.loads(scan["report_json"]) if scan else None
        pats.append({"id": p["id"], "name": p["name"], "email": p["email"], "intake": json.loads(p["intake_json"]) if p["intake_json"] else None,
                     "last_scan": {"created": scan["created"], "summary": rep["summary"], "suggest_appointment": rep["suggest_appointment"]} if scan else None,
                     "next_appointment": nxt["start_ts"] if nxt else None})
    q = "SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, a.note, pu.name AS patient, a.patient_id FROM appointments a JOIN users pu ON pu.id=a.patient_id WHERE a.doctor_id=? AND "
    former = _appt_rows(db.execute("SELECT u.id, u.name, MAX(pd.end_ts) AS left_on FROM patient_doctors pd JOIN users u ON u.id=pd.patient_id JOIN patients p ON p.user_id=pd.patient_id "
                                   "WHERE pd.doctor_id=? AND (p.doctor_id IS NULL OR p.doctor_id<>?) GROUP BY u.id ORDER BY u.name", (u["id"], u["id"])))      # read-only: moved to another doctor
    return {"patients": pats, "former_patients": former,
            "upcoming": _appt_rows(db.execute(q + "a.status='booked' AND a.start_ts>=? ORDER BY a.start_ts", (u["id"], now))),
            "previous": _appt_rows(db.execute(q + "(a.status='completed' OR (a.status='booked' AND a.start_ts<?)) ORDER BY a.start_ts DESC", (u["id"], now)))}


@app.get("/api/doctor/patients/{pid}")
def doctor_patient(pid: int, u=Depends(need("doctor")), db=Depends(get_db)):
    # any doctor the patient has ever had may read the full record (all doctors' notes, scans, intake); only the current doctor can write
    p = db.execute("SELECT u.id, u.name, u.email, p.doctor_id AS current_doctor_id FROM patients p JOIN users u ON u.id=p.user_id "
                   "WHERE p.user_id=? AND EXISTS(SELECT 1 FROM patient_doctors pd WHERE pd.patient_id=p.user_id AND pd.doctor_id=?)", (pid, u["id"])).fetchone()
    if not p: raise HTTPException(404, "Not your patient")
    return {**dict(p), "is_current_doctor": p["current_doctor_id"] == u["id"], "intake": _intake(db, pid), "progress": insights.progress(_scan_rows(db, pid)),
            "scans": [{"created": r["created"], "report": json.loads(r["report_json"])} for r in db.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY id DESC", (pid,))],
            "assignments": _appt_rows(db.execute("SELECT du.name AS doctor, pd.start_ts, pd.end_ts, pd.reason FROM patient_doctors pd JOIN users du ON du.id=pd.doctor_id WHERE pd.patient_id=? ORDER BY pd.id", (pid,))),
            "appointments": _appt_rows(db.execute("SELECT a.id, a.start_ts, a.status, a.reason, a.note, a.note_ts, du.name AS doctor FROM appointments a JOIN users du ON du.id=a.doctor_id "
                                                  "WHERE a.patient_id=? AND a.status<>'cancelled' ORDER BY a.start_ts DESC", (pid,)))}


@app.post("/api/doctor/appointments/{aid}/note")
def add_note(aid: int, b: Note, u=Depends(need("doctor")), db=Depends(get_db)):
    a = db.execute("SELECT * FROM appointments WHERE id=? AND doctor_id=?", (aid, u["id"])).fetchone()
    if not a or a["status"] == "cancelled": raise HTTPException(404, "Not found")
    db.execute("UPDATE appointments SET note=?, note_ts=?, status='completed' WHERE id=?", (b.summary, datetime.now().isoformat(timespec="seconds"), aid)); db.commit(); return {"ok": True}


def _valid_hhmm(s):
    try: datetime.strptime(s, "%H:%M"); return True
    except ValueError: return False


@app.get("/api/doctor/availability")
def get_hours(u=Depends(need("doctor")), db=Depends(get_db)):
    return _appt_rows(db.execute("SELECT id, weekday, start, end, slot_minutes FROM availability WHERE doctor_id=? ORDER BY weekday, start", (u["id"],)))


@app.put("/api/doctor/availability")
def set_hours(rows: list[Hours], u=Depends(need("doctor")), db=Depends(get_db)):
    """Replaces the whole weekly chart. Existing bookings are kept even if their slot is no longer offered."""
    for r in rows:
        if not (0 <= r.weekday <= 6 and _valid_hhmm(r.start) and _valid_hhmm(r.end) and r.start < r.end and 5 <= r.slot_minutes <= 120): raise HTTPException(422, f"Bad row: {r}")
    db.execute("DELETE FROM availability WHERE doctor_id=?", (u["id"],))
    db.executemany("INSERT INTO availability(doctor_id,weekday,start,end,slot_minutes) VALUES(?,?,?,?,?)", [(u["id"], r.weekday, r.start, r.end, r.slot_minutes) for r in rows]); db.commit(); return {"rows": len(rows)}


# ---------- admin ----------
@app.get("/api/admin/doctors")
def admin_doctors(u=Depends(need("admin")), db=Depends(get_db)):
    return _appt_rows(db.execute("SELECT u.id, u.name, u.email, d.specialization, (SELECT COUNT(*) FROM patients p WHERE p.doctor_id=u.id) AS patients, "
                                 "(SELECT COUNT(*) FROM appointments a WHERE a.doctor_id=u.id AND a.status='booked' AND a.start_ts>=datetime('now','localtime')) AS upcoming "
                                 "FROM doctors d JOIN users u ON u.id=d.user_id ORDER BY u.name"))


@app.get("/api/admin/doctors/{did}/patients")
def admin_doctor_patients(did: int, u=Depends(need("admin")), db=Depends(get_db)):
    return _appt_rows(db.execute("SELECT u.id, u.name, u.email, (p.intake_json IS NOT NULL) AS intake_done, (SELECT COUNT(*) FROM scans s WHERE s.patient_id=u.id) AS scans "
                                 "FROM patients p JOIN users u ON u.id=p.user_id WHERE p.doctor_id=? ORDER BY u.name", (did,)))


@app.get("/api/admin/patients/{pid}")
def admin_patient(pid: int, u=Depends(need("admin")), db=Depends(get_db)):
    p = db.execute("SELECT u.id, u.name, u.email, p.doctor_id AS current_doctor_id FROM patients p JOIN users u ON u.id=p.user_id WHERE p.user_id=?", (pid,)).fetchone()
    if not p: raise HTTPException(404, "No such patient")
    return {**dict(p), "intake": _intake(db, pid), "progress": insights.progress(_scan_rows(db, pid)), "scans": [{"created": r["created"], "report": json.loads(r["report_json"])} for r in db.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY id DESC", (pid,))],
            "assignments": _appt_rows(db.execute("SELECT du.name AS doctor, pd.start_ts, pd.end_ts, pd.reason FROM patient_doctors pd JOIN users du ON du.id=pd.doctor_id WHERE pd.patient_id=? ORDER BY pd.id", (pid,))),
            "appointments": _appt_rows(db.execute("SELECT a.id, a.start_ts, a.status, a.reason, a.note, du.name AS doctor FROM appointments a JOIN users du ON du.id=a.doctor_id WHERE a.patient_id=? ORDER BY a.start_ts DESC", (pid,)))}


@app.get("/api/admin/emails")
def admin_emails(limit: int = 100, u=Depends(need("admin")), db=Depends(get_db)):
    return {"live": mailer.configured(), "rows": _appt_rows(db.execute("SELECT id, to_addr, subject, kind, status, error, created FROM email_log ORDER BY id DESC LIMIT ?", (min(limit, 500),)))}


@app.post("/api/admin/run-reminders")
def admin_run_reminders(u=Depends(need("admin")), db=Depends(get_db)):
    out = reminders.run_reminders(db); return {"processed": [{"kind": k, "to": to, "status": st} for k, to, st in out]}


@app.get("/api/admin/calendar")
def admin_calendar(start: str, end: str, u=Depends(need("admin")), db=Depends(get_db)):
    return _appt_rows(db.execute("SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, du.name AS doctor, a.doctor_id, pu.name AS patient FROM appointments a "
                                 "JOIN users du ON du.id=a.doctor_id JOIN users pu ON pu.id=a.patient_id WHERE a.start_ts>=? AND a.start_ts<? AND a.status<>'cancelled' ORDER BY a.start_ts", (start, end)))


@app.get("/api/admin/stats")
def admin_stats(u=Depends(need("admin")), db=Depends(get_db)):
    q = lambda s: db.execute(s).fetchone()[0]
    return {"doctors": q("SELECT COUNT(*) FROM doctors"), "patients": q("SELECT COUNT(*) FROM patients"), "unassigned_patients": q("SELECT COUNT(*) FROM patients WHERE doctor_id IS NULL"),
            "scans": q("SELECT COUNT(*) FROM scans"), "booked": q("SELECT COUNT(*) FROM appointments WHERE status='booked'")}


@app.post("/api/admin/doctors")
def admin_new_doctor(b: NewDoctor, u=Depends(need("admin")), db=Depends(get_db)):
    if len(b.password) < 8: raise HTTPException(422, "Password must be at least 8 characters")
    if [f for f in b.focus if f not in insights.SYSTEMS]: raise HTTPException(422, "Unknown focus area")
    try:
        cur = db.execute("INSERT INTO users(role,name,email,pw_hash,created) VALUES('doctor',?,?,?,?)", (b.name.strip(), b.email.strip().lower(), hash_pw(b.password), datetime.now().isoformat(timespec="seconds")))
        db.execute("INSERT INTO doctors(user_id, specialization, focus) VALUES(?,?,?)", (cur.lastrowid, b.specialization, json.dumps(b.focus)))
        db.executemany("INSERT INTO availability(doctor_id,weekday,start,end,slot_minutes) VALUES(?,?,?,?,20)", [(cur.lastrowid, wd, "10:00", "13:00") for wd in range(5)]); db.commit()   # default Mon-Fri 10-13
    except sqlite3.IntegrityError: raise HTTPException(409, "That email is already registered")
    return {"id": cur.lastrowid}


@app.get("/")
def index():
    p = os.path.join(HERE, "static", "index.html")
    return FileResponse(p) if os.path.exists(p) else Response("API is running. UI not built yet.", media_type="text/plain")


if os.path.isdir(os.path.join(HERE, "static")): app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
init_db()
if __name__ == "__main__":
    import uvicorn
    reminders.start_background(connect, int(os.environ.get("REMINDER_INTERVAL_SECONDS", "600")))      # weekly-scan and appointment emails
    uvicorn.run(app, host="127.0.0.1", port=7890, log_level="warning")
