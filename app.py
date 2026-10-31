"""Three-level wellness platform API: admin -> doctors -> patients.  Run: python3 app.py  (http://127.0.0.1:7890)
SQLite + signed-cookie sessions + scrypt password hashes.  Prototype: local data only, not for real patient records without a security review."""
import base64, hashlib, hmac, json, os, secrets, sqlite3, time
from datetime import datetime, timedelta

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import acne_lab, gags, insights, mailer, reminders, risk_sim

HERE = os.path.dirname(os.path.abspath(__file__))
mailer.load_env(os.path.join(HERE, ".env"))       # SMTP settings for real email; without them emails are only logged (dry run)
DEV = os.environ.get("WELLNESS_DEV") == "1"       # enables the test-only endpoint that accepts raw zone counts
DB_PATH = os.environ.get("WELLNESS_DB", os.path.join(HERE, "data", "platform.db"))
SESSION_SECONDS = 8 * 3600
VIEW_AS_SECONDS = 2 * 3600          # an admin "view as" session is shorter and read-only
PHOTO_DIR = os.path.join(os.path.dirname(DB_PATH), "scan_photos")
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
  reason TEXT DEFAULT '', note TEXT, note_ts TEXT,
  reschedule_count INTEGER NOT NULL DEFAULT 0, rescheduled_from TEXT, rescheduled_by TEXT, reschedule_reason TEXT);
-- every time an appointment is moved (by the patient or by the doctor): audit trail and analytics
CREATE TABLE IF NOT EXISTS appointment_changes(id INTEGER PRIMARY KEY, appt_id INTEGER NOT NULL REFERENCES appointments(id), old_start TEXT NOT NULL, new_start TEXT NOT NULL,
  by_role TEXT NOT NULL, by_id INTEGER NOT NULL, reason TEXT DEFAULT '', created TEXT NOT NULL);
-- a doctor's slot can be held by only one live booking, and a patient cannot hold two at the same time
CREATE UNIQUE INDEX IF NOT EXISTS ux_doctor_slot ON appointments(doctor_id, start_ts) WHERE status <> 'cancelled';
CREATE UNIQUE INDEX IF NOT EXISTS ux_patient_slot ON appointments(patient_id, start_ts) WHERE status <> 'cancelled';
CREATE TABLE IF NOT EXISTS email_log(id INTEGER PRIMARY KEY, to_addr TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT DEFAULT '', created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(user_id), created TEXT NOT NULL, report_json TEXT NOT NULL);
-- face photos of a scan (original + annotated), stored once by content hash. Only the patient and doctors the patient has had may read them; admins never can.
CREATE TABLE IF NOT EXISTS scan_photos(scan_id INTEGER NOT NULL REFERENCES scans(id), view TEXT NOT NULL, kind TEXT NOT NULL, sha TEXT NOT NULL, PRIMARY KEY(scan_id, view, kind));
"""


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False); c.row_factory = sqlite3.Row; c.execute("PRAGMA foreign_keys=ON"); return c


def init_db():
    c = connect(); c.executescript(SCHEMA)
    for ddl in ("ALTER TABLE patients ADD COLUMN email_reminders INTEGER NOT NULL DEFAULT 1", "ALTER TABLE appointments ADD COLUMN reschedule_count INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE appointments ADD COLUMN rescheduled_from TEXT", "ALTER TABLE appointments ADD COLUMN rescheduled_by TEXT", "ALTER TABLE appointments ADD COLUMN reschedule_reason TEXT"):
        try: c.execute(ddl)                                                                       # upgrade an older database
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


def make_token(uid, via=None):
    life = VIEW_AS_SECONDS if via else SESSION_SECONDS; body = f"{uid}.{int(time.time()) + life}" + (f".{via}" if via else "")
    return base64.urlsafe_b64encode(body.encode()).decode() + "." + hmac.new(_secret(), body.encode(), "sha256").hexdigest()


def read_token(tok):
    """Returns (user_id, via_admin_id or None), or (None, None) if the token is invalid or expired."""
    try:
        b64, sig = tok.split("."); body = base64.urlsafe_b64decode(b64).decode()
        if not hmac.compare_digest(hmac.new(_secret(), body.encode(), "sha256").hexdigest(), sig): return None, None
        parts = body.split("."); return (int(parts[0]), int(parts[2]) if len(parts) > 2 else None) if int(parts[1]) > time.time() else (None, None)
    except Exception: return None, None


def current_user(request: Request, db=Depends(get_db)):
    uid, via = read_token(request.cookies.get("session", ""))
    u = db.execute("SELECT id, role, name, email FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    if not u: raise HTTPException(401, "Please log in")
    return {**dict(u), "via": via}


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


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """Front-end files are re-checked on every load, so an update is never hidden by a stale browser cache."""
    resp = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/": resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.middleware("http")
async def read_only_when_viewing_as(request: Request, call_next):
    """While an admin is viewing as another user nothing can be changed: every non-GET call is refused except leaving that view or signing out."""
    tok = request.cookies.get("session")
    if tok and request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path not in ("/api/view-as/exit", "/api/logout"):
        if read_token(tok)[1]: return JSONResponse({"detail": "Read-only: you are viewing as another user. Return to your admin view to make changes."}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def acne_lab_login_gate(request: Request, call_next):
    """The Acne Lab (page and API) is only for signed-in FaceVeda users."""
    if request.url.path.startswith("/acne-lab") and not read_token(request.cookies.get("session", ""))[0]:
        return JSONResponse({"detail": "Please log in"}, status_code=401) if request.url.path.startswith("/acne-lab/api") else RedirectResponse("/")
    return await call_next(request)


@app.api_route("/acne-lab/api/{path:path}", methods=["GET", "POST"])
async def acne_lab_api(path: str, request: Request):
    """Forwards the Acne Lab's API calls to the prototype backend (photo upload, preprocessing, detection, GAGS)."""
    import httpx
    headers = {"content-type": request.headers["content-type"]} if "content-type" in request.headers else {}
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.request(request.method, f"{acne_lab.LAB_URL}/api/{path}", params=dict(request.query_params), content=await request.body(), headers=headers)
    except httpx.HTTPError: raise HTTPException(503, "The Acne Lab engine is not running. Start it with the run script or set ACNE_LAB_DIR in .env.")
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


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


class PwChange(BaseModel): current: str; new: str


@app.put("/api/me/password")
def change_password(b: PwChange, u=Depends(current_user), db=Depends(get_db)):
    row = db.execute("SELECT pw_hash FROM users WHERE id=?", (u["id"],)).fetchone()
    if not check_pw(b.current, row["pw_hash"]): raise HTTPException(403, "Your current password is wrong")
    if len(b.new) < 8: raise HTTPException(422, "The new password must be at least 8 characters")
    if b.new == b.current: raise HTTPException(422, "The new password must be different from the current one")
    db.execute("UPDATE users SET pw_hash=? WHERE id=?", (hash_pw(b.new), u["id"])); db.commit(); return {"ok": True}


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
    u["created"] = db.execute("SELECT created FROM users WHERE id=?", (u["id"],)).fetchone()["created"]
    via = u.pop("via", None); adm = db.execute("SELECT id, name FROM users WHERE id=?", (via,)).fetchone() if via else None
    u["viewing_as"] = {"admin_id": adm["id"], "admin_name": adm["name"]} if adm else None
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


def _jpeg(img, max_side, quality):
    import cv2
    h, w = img.shape[:2]; k = max_side / max(h, w)
    if k < 1: img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def save_photo(db, scan_id, view, kind, data):
    """Stores one JPEG by content hash (identical photos are kept once) with owner-only file permissions; access is only through scan_photo() below."""
    sha = hashlib.sha256(data).hexdigest(); d = os.path.join(PHOTO_DIR, sha[:2]); os.makedirs(d, mode=0o700, exist_ok=True); p = os.path.join(d, sha + ".jpg")
    if not os.path.exists(p):
        with open(p, "wb") as fh: fh.write(data)
        os.chmod(p, 0o600)
    db.execute("INSERT OR REPLACE INTO scan_photos(scan_id, view, kind, sha) VALUES(?,?,?,?)", (scan_id, view, kind, sha))


def photo_views(db, scan_ids):
    """{scan_id: [views with stored photos]} for a list of scans."""
    out = {i: [] for i in scan_ids}
    if scan_ids:
        for r in db.execute(f"SELECT scan_id, view FROM scan_photos WHERE kind='original' AND scan_id IN ({','.join('?' * len(scan_ids))}) ORDER BY view", list(scan_ids)): out[r["scan_id"]].append(r["view"])
    return out


def run_scan(images, intake):
    """images: {view: BGR image}. Returns (report, parts) where parts = [{view, spots, preview, original}]. Raises 422 if a photo has no usable face."""
    import scan as scanner
    counts, area, parts, typed = {z: 0 for z in insights.ZONES}, {z: 0 for z in insights.ZONES}, [], []
    for name, img in images.items():
        s_ = scanner.scan_image(img, None if name == "front" else name)
        if s_["status"] != "ok": raise HTTPException(422, f"{name} photo: {s_['message']}")
        for z in insights.ZONES: counts[z] += s_["zone_counts"][z]; area[z] += s_["zone_area"][z]
        parts.append({"view": name, "spots": s_["total_spots"], "preview": s_["preview"], "original": img}); typed.append({"view": name, "zone_types": s_.get("zone_types")})
    rep = insights.analyse_zones(counts, intake, area); rep["views"] = list(images); rep["zone_area"] = area; rep["gags"] = gags.compute(typed); finalize_report(rep, intake, counts, area)
    return rep, parts


@app.post("/api/patient/scan")
def scan(b: ScanReq, u=Depends(need("patient")), db=Depends(get_db)):
    """Front photo (required) plus left/right profile photos (all three on the first scan). The photos are kept, visible only to the patient and doctors the patient has had."""
    views = {k: v for k, v in (("front", b.front), ("left", b.left), ("right", b.right)) if v}
    if "front" not in views: raise HTTPException(422, "A front photo is required")
    first = db.execute("SELECT COUNT(*) FROM scans WHERE patient_id=?", (u["id"],)).fetchone()[0] == 0
    if first and len(views) < 3: raise HTTPException(422, "Your first scan needs all three angles: front, left side and right side")
    rep, parts = run_scan({k: _decode_image(v) for k, v in views.items()}, _intake(db, u["id"]))
    cur = db.execute("INSERT INTO scans(patient_id, created, report_json) VALUES(?,?,?)", (u["id"], datetime.now().isoformat(timespec="seconds"), json.dumps(rep))); sid = cur.lastrowid
    previews = []
    for p in parts:
        ann = _jpeg(p["preview"], 1000, 80); save_photo(db, sid, p["view"], "original", _jpeg(p["original"], 1400, 85)); save_photo(db, sid, p["view"], "annotated", ann)
        previews.append({"view": p["view"], "spots": p["spots"], "image": "data:image/jpeg;base64," + base64.b64encode(ann).decode()})
    db.commit(); return {"scan_id": sid, "report": rep, "previews": previews}


@app.get("/api/scans/{sid}/photo/{view}/{kind}")
def scan_photo(sid: int, view: str, kind: str, u=Depends(current_user), db=Depends(get_db)):
    """A stored face photo. Allowed for the patient who took it and for every doctor that patient has had. Never for admins, including an admin viewing as someone else."""
    if u.get("via") or u["role"] == "admin": raise HTTPException(403, "Face photos are private to the patient and their doctors")
    row = db.execute("SELECT s.patient_id, ph.sha FROM scans s JOIN scan_photos ph ON ph.scan_id=s.id WHERE s.id=? AND ph.view=? AND ph.kind=?", (sid, view, kind)).fetchone()
    if not row: raise HTTPException(404, "Not found")
    ok = (u["role"] == "patient" and u["id"] == row["patient_id"]) or (u["role"] == "doctor" and db.execute("SELECT 1 FROM patient_doctors WHERE patient_id=? AND doctor_id=?", (row["patient_id"], u["id"])).fetchone())
    if not ok: raise HTTPException(404, "Not found")
    return FileResponse(os.path.join(PHOTO_DIR, row["sha"][:2], row["sha"] + ".jpg"), media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@app.get("/api/patient/scans")
def my_scans(u=Depends(need("patient")), db=Depends(get_db)):
    rows = db.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY id DESC", (u["id"],)).fetchall(); pv = photo_views(db, [r["id"] for r in rows])
    return [{"id": r["id"], "created": r["created"], "photos": pv[r["id"]], "report": json.loads(r["report_json"])} for r in rows]


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


def _appt_rows(rows):
    """Rows as dicts. Adds display_status: a booked appointment whose time has passed without being completed is shown as 'missed'."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M"); out = []
    for r in rows:
        d = dict(r)
        if "status" in d and d.get("end_ts"): d["display_status"] = "missed" if d["status"] == "booked" and d["end_ts"] < now else d["status"]
        out.append(d)
    return out


RS = "a.reschedule_count, a.rescheduled_from, a.rescheduled_by, a.reschedule_reason"


@app.get("/api/patient/appointments")
def my_appointments(u=Depends(need("patient")), db=Depends(get_db)):
    return _appt_rows(db.execute(f"SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, a.note, a.note_ts, {RS}, du.name AS doctor FROM appointments a JOIN users du ON du.id=a.doctor_id "
                                 "WHERE a.patient_id=? ORDER BY a.start_ts DESC", (u["id"],)))


class Resched(BaseModel): start: str; reason: str = ""


def _own_appointment(db, aid, u):
    a = db.execute("SELECT a.*, pu.name AS patient_name, pu.email AS patient_email, du.name AS doctor_name, du.email AS doctor_email FROM appointments a "
                   "JOIN users pu ON pu.id=a.patient_id JOIN users du ON du.id=a.doctor_id WHERE a.id=?", (aid,)).fetchone()
    mine = a and not u.get("via") and ((u["role"] == "patient" and a["patient_id"] == u["id"]) or (u["role"] == "doctor" and a["doctor_id"] == u["id"]))
    if not mine: raise HTTPException(404, "Not found")
    return a


@app.get("/api/appointments/{aid}/slots")
def appointment_slots(aid: int, days: int = 28, u=Depends(current_user), db=Depends(get_db)):
    """Free slots of this appointment's doctor, for choosing a new time (patient or doctor of the appointment)."""
    a = _own_appointment(db, aid, u); out, today = {}, datetime.now().date()
    for i in range(min(days, BOOKING_WINDOW_DAYS) + 1):
        day = today + timedelta(days=i); sl = [t.strftime("%H:%M") for t, _ in _slots_for(db, a["doctor_id"], day)]
        if sl: out[f"{day:%Y-%m-%d}"] = sl
    return out


@app.post("/api/appointments/{aid}/reschedule")
def reschedule(aid: int, b: Resched, bg: BackgroundTasks, u=Depends(current_user), db=Depends(get_db)):
    """Moves an upcoming booked appointment to another free slot of the same doctor. The patient can do it, and so can the doctor (for example when suddenly unavailable); the other side is emailed."""
    a = _own_appointment(db, aid, u); now = datetime.now()
    if a["status"] != "booked" or a["end_ts"] < now.strftime("%Y-%m-%d %H:%M"): raise HTTPException(409, "Only an upcoming booked appointment can be rescheduled")
    try: t = datetime.strptime(b.start, "%Y-%m-%d %H:%M")
    except ValueError: raise HTTPException(422, "start must look like 2026-11-02 10:20")
    if b.start == a["start_ts"]: raise HTTPException(409, "That is the current time")
    ok = [(s_, e) for s_, e in _slots_for(db, a["doctor_id"], t.date()) if s_ == t]
    if not ok or t > now + timedelta(days=BOOKING_WINDOW_DAYS): raise HTTPException(409, "That time is not available")
    reason = b.reason.strip()[:200]
    try:
        db.execute("UPDATE appointments SET start_ts=?, end_ts=?, reschedule_count=reschedule_count+1, rescheduled_from=?, rescheduled_by=?, reschedule_reason=? WHERE id=?",
                   (b.start, ok[0][1].strftime("%Y-%m-%d %H:%M"), a["start_ts"], u["role"], reason, aid))
        db.execute("INSERT INTO appointment_changes(appt_id, old_start, new_start, by_role, by_id, reason, created) VALUES(?,?,?,?,?,?,?)", (aid, a["start_ts"], b.start, u["role"], u["id"], reason, now.isoformat(timespec="seconds")))
        db.commit()
    except sqlite3.IntegrityError: db.rollback(); raise HTTPException(409, "That slot was just taken, or the patient already has an appointment at that time")
    fmt = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M").strftime("%a %d %b, %I:%M %p")
    bg.add_task(_resched_emails, a["patient_email"], a["patient_name"], a["doctor_email"], a["doctor_name"], fmt(a["start_ts"]), fmt(b.start), u["role"], reason, aid)
    return {"id": aid, "start": b.start, "end": ok[0][1].strftime("%Y-%m-%d %H:%M"), "rescheduled_by": u["role"]}


def _resched_emails(p_email, p_name, d_email, d_name, old, new, by, reason, aid):
    c = connect()
    try:
        subj, body = mailer.appointment_rescheduled(p_name, d_name, old, new, by, reason if by == "doctor" else ""); mailer.send_email(c, p_email, subj, body, "appt_resched", f"appt-resched:{aid}:{new}:patient")
        if by == "patient": subj, body = mailer.appointment_rescheduled_for_doctor(d_name, p_name, old, new); mailer.send_email(c, d_email, subj, body, "appt_resched_doctor", f"appt-resched:{aid}:{new}:doctor")
    finally: c.close()


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
    q = f"SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, a.note, {RS}, pu.name AS patient, a.patient_id FROM appointments a JOIN users pu ON pu.id=a.patient_id WHERE a.doctor_id=? AND "
    former = _appt_rows(db.execute("SELECT u.id, u.name, MAX(pd.end_ts) AS left_on FROM patient_doctors pd JOIN users u ON u.id=pd.patient_id JOIN patients p ON p.user_id=pd.patient_id "
                                   "WHERE pd.doctor_id=? AND (p.doctor_id IS NULL OR p.doctor_id<>?) GROUP BY u.id ORDER BY u.name", (u["id"], u["id"])))      # read-only: moved to another doctor
    return {"patients": pats, "former_patients": former,
            "upcoming": _appt_rows(db.execute(q + "a.status='booked' AND a.start_ts>=? ORDER BY a.start_ts", (u["id"], now))),
            "previous": _appt_rows(db.execute(q + "(a.status='completed' OR (a.status='booked' AND a.start_ts<?)) ORDER BY a.start_ts DESC", (u["id"], now)))}


def _scans_with_photos(db, pid):
    rows = db.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY id DESC", (pid,)).fetchall(); pv = photo_views(db, [r["id"] for r in rows])
    return [{"id": r["id"], "created": r["created"], "photos": pv[r["id"]], "report": json.loads(r["report_json"])} for r in rows]


@app.get("/api/doctor/patients/{pid}")
def doctor_patient(pid: int, u=Depends(need("doctor")), db=Depends(get_db)):
    # any doctor the patient has ever had may read the full record (all doctors' notes, scans, intake); only the current doctor can write
    p = db.execute("SELECT u.id, u.name, u.email, p.doctor_id AS current_doctor_id FROM patients p JOIN users u ON u.id=p.user_id "
                   "WHERE p.user_id=? AND EXISTS(SELECT 1 FROM patient_doctors pd WHERE pd.patient_id=p.user_id AND pd.doctor_id=?)", (pid, u["id"])).fetchone()
    if not p: raise HTTPException(404, "Not your patient")
    return {**dict(p), "is_current_doctor": p["current_doctor_id"] == u["id"], "intake": _intake(db, pid), "progress": insights.progress(_scan_rows(db, pid)),
            "scans": _scans_with_photos(db, pid),
            "assignments": _appt_rows(db.execute("SELECT du.name AS doctor, pd.start_ts, pd.end_ts, pd.reason FROM patient_doctors pd JOIN users du ON du.id=pd.doctor_id WHERE pd.patient_id=? ORDER BY pd.id", (pid,))),
            "appointments": _appt_rows(db.execute(f"SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, a.note, a.note_ts, {RS}, du.name AS doctor FROM appointments a JOIN users du ON du.id=a.doctor_id "
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
    last = db.execute("SELECT created FROM scans WHERE patient_id=? ORDER BY id DESC LIMIT 1", (pid,)).fetchone(); n = db.execute("SELECT COUNT(*) FROM scans WHERE patient_id=?", (pid,)).fetchone()[0]
    # face scans (photos AND results) are private to the patient and their doctors: the admin only sees how many exist
    return {**dict(p), "intake": _intake(db, pid), "scan_summary": {"count": n, "last": last["created"] if last else None, "private": True},
            "assignments": _appt_rows(db.execute("SELECT du.name AS doctor, pd.start_ts, pd.end_ts, pd.reason FROM patient_doctors pd JOIN users du ON du.id=pd.doctor_id WHERE pd.patient_id=? ORDER BY pd.id", (pid,))),
            "appointments": _appt_rows(db.execute(f"SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, a.note, {RS}, du.name AS doctor FROM appointments a JOIN users du ON du.id=a.doctor_id WHERE a.patient_id=? ORDER BY a.start_ts DESC", (pid,)))}


def admin_or_previewing_admin(u=Depends(current_user), db=Depends(get_db)):
    """The admin, or an admin who is currently previewing someone else's screens (so they can jump straight to another preview)."""
    if u["role"] == "admin" or (u.get("via") and db.execute("SELECT 1 FROM users WHERE id=? AND role='admin'", (u["via"],)).fetchone()): return u
    raise HTTPException(403, "admin access only")


@app.get("/api/admin/users")
def admin_users(role: str = "doctor", u=Depends(admin_or_previewing_admin), db=Depends(get_db)):
    """Doctors or patients the admin can view the interface as (read-only)."""
    if role not in ("doctor", "patient"): raise HTTPException(422, "role must be doctor or patient")
    if role == "doctor": q = "SELECT u.id, u.name, u.email, d.specialization AS detail FROM users u JOIN doctors d ON d.user_id=u.id ORDER BY u.name"
    else: q = "SELECT u.id, u.name, u.email, COALESCE(du.name, 'no doctor yet') AS detail FROM users u JOIN patients p ON p.user_id=u.id LEFT JOIN users du ON du.id=p.doctor_id ORDER BY u.name"
    return _appt_rows(db.execute(q))


class ViewAs(BaseModel): user_id: int


@app.post("/api/admin/view-as")
def view_as(b: ViewAs, response: Response, u=Depends(need("admin")), db=Depends(get_db)):
    t = db.execute("SELECT id, role, name FROM users WHERE id=? AND role IN ('doctor','patient')", (b.user_id,)).fetchone()
    if not t: raise HTTPException(404, "No such doctor or patient")
    response.set_cookie("session", make_token(t["id"], via=u["id"]), httponly=True, samesite="lax", max_age=VIEW_AS_SECONDS); return {"role": t["role"], "name": t["name"]}


@app.post("/api/view-as/exit")
def exit_view_as(request: Request, response: Response, db=Depends(get_db)):
    uid, via = read_token(request.cookies.get("session", ""))
    adm = db.execute("SELECT id, role FROM users WHERE id=? AND role='admin'", (via,)).fetchone() if via else None
    if not adm: raise HTTPException(409, "You are not viewing as another user")
    response.set_cookie("session", make_token(adm["id"]), httponly=True, samesite="lax", max_age=SESSION_SECONDS); return {"role": "admin"}


@app.get("/api/admin/analytics")
def admin_analytics(u=Depends(need("admin")), db=Depends(get_db)):
    """Platform-wide analytics. Everything is an aggregate: no patient names and no scan results or photos (those stay private to the patient and their doctors)."""
    now = datetime.now(); nows = now.strftime("%Y-%m-%d %H:%M"); today = now.date(); monday = today - timedelta(days=today.weekday()); WB, WF = 8, 4
    kind = lambda r: "cancelled" if r["status"] == "cancelled" else "completed" if r["status"] == "completed" else ("missed" if r["end_ts"] < nows else "upcoming")
    appts = db.execute("SELECT doctor_id, patient_id, start_ts, end_ts, status, reschedule_count FROM appointments").fetchall(); K = ("completed", "upcoming", "missed", "cancelled")
    tot = {k: 0 for k in K}; weekly = [{"week": (monday + timedelta(days=7 * w)).isoformat(), **{k: 0 for k in K}} for w in range(-WB, WF)]; wd = [0] * 7
    per = {r["id"]: {"id": r["id"], "name": r["name"], "specialization": r["specialization"], "patients": 0, **{k: 0 for k in K}, "rescheduled": 0, "capacity": 0, "booked": 0}
           for r in db.execute("SELECT u.id, u.name, d.specialization FROM doctors d JOIN users u ON u.id=d.user_id")}
    lo, hi = (monday - timedelta(days=7 * WB)).isoformat(), (monday + timedelta(days=7 * WF)).isoformat()
    for r in appts:
        k = kind(r); tot[k] += 1; d = per.get(r["doctor_id"])
        if d is not None: d[k] += 1; d["rescheduled"] += 1 if r["reschedule_count"] else 0
        day = r["start_ts"][:10]
        if lo <= day < hi:
            weekly[(datetime.strptime(day, "%Y-%m-%d").date() - (monday - timedelta(days=7 * WB))).days // 7][k] += 1
            if k != "cancelled" and d is not None: d["booked"] += 1
        if k != "cancelled": wd[datetime.strptime(day, "%Y-%m-%d").weekday()] += 1
    for did, wins in _chart_slots(db).items():
        if did in per: per[did]["capacity"] = wins * (WB + WF)
    for r in db.execute("SELECT doctor_id, COUNT(*) n FROM patients WHERE doctor_id IS NOT NULL GROUP BY doctor_id"):
        if r["doctor_id"] in per: per[r["doctor_id"]]["patients"] = r["n"]
    docs = []
    for d in per.values():
        att = d["completed"] + d["missed"]; d["attendance_rate"] = round(d["completed"] / att, 3) if att else None; d["utilisation"] = round(min(1.0, d["booked"] / d["capacity"]), 3) if d["capacity"] else None
        docs.append({k: v for k, v in d.items() if k not in ("capacity", "booked")})
    # recovery from scan trends: mean of the first vs the last few scans (aggregate only, never individual results)
    rec = {"improved": 0, "stable": 0, "worsened": 0, "insufficient": 0}; by_doc = {did: {"improved": 0, "stable": 0, "worsened": 0} for did in per}; edges = [-40, -15, 15, 40]; hist = [0] * 5
    cur = {r["user_id"]: r["doctor_id"] for r in db.execute("SELECT user_id, doctor_id FROM patients")}; series = {}
    for r in db.execute("SELECT patient_id, report_json FROM scans ORDER BY created"): series.setdefault(r["patient_id"], []).append(json.loads(r["report_json"]).get("total_spots", 0))
    scan_state = {"scanned_last_7_days": 0, "due": 0, "overdue": 0, "never": 0}; last = {r["patient_id"]: r["m"] for r in db.execute("SELECT patient_id, MAX(created) m FROM scans GROUP BY patient_id")}
    for pid in cur:
        T = series.get(pid, [])
        if last.get(pid) is None: scan_state["never"] += 1
        else:
            days = (now - datetime.fromisoformat(last[pid])).days; scan_state["scanned_last_7_days" if days < 7 else "due" if days < 10 else "overdue"] += 1
        if len(T) < 2: rec["insufficient"] += 1; continue
        m = max(1, min(3, len(T) // 2)); a0, a1 = sum(T[:m]) / m, sum(T[-m:]) / m; ch = 100 * (a1 - a0) / a0 if a0 else 0
        cls = "improved" if ch <= -15 else "worsened" if ch >= 15 else "stable"; rec[cls] += 1
        if cur[pid] in by_doc: by_doc[cur[pid]][cls] += 1
        hist[sum(1 for e in edges if ch > e)] += 1
    scored = rec["improved"] + rec["stable"] + rec["worsened"]; reasons = {}
    for r in db.execute("SELECT by_role, reason FROM appointment_changes"):
        if r["reason"]: reasons[(r["by_role"], r["reason"].strip().lower())] = reasons.get((r["by_role"], r["reason"].strip().lower()), 0) + 1
    chg = {r["by_role"]: r["n"] for r in db.execute("SELECT by_role, COUNT(*) n FROM appointment_changes GROUP BY by_role")}
    att = tot["completed"] + tot["missed"]
    return {"totals": {"doctors": len(per), "patients": len(cur), "scans": sum(len(v) for v in series.values()), "appointments": sum(tot.values())},
            "appointments": {**tot, "rescheduled": sum(chg.values()), "attendance_rate": round(tot["completed"] / att, 3) if att else None}, "weekly": weekly, "by_weekday": wd,
            "doctors": sorted(docs, key=lambda d: d["name"]), "recovery": {**rec, "rate": round(rec["improved"] / scored, 3) if scored else None, "histogram": {"edges": edges, "counts": hist},
            "by_doctor": [{"doctor_id": did, "name": per[did]["name"], **v} for did, v in by_doc.items()]}, "scanning": scan_state,
            "reschedules": {"by_patient": chg.get("patient", 0), "by_doctor": chg.get("doctor", 0), "top_reasons": [{"by": k[0], "reason": k[1], "count": n} for k, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]]}}


def _chart_slots(db):
    """Slots per week in each doctor's availability chart: {doctor_id: slots a week}."""
    out = {}
    for r in db.execute("SELECT doctor_id, start, end, slot_minutes FROM availability"):
        a = datetime.strptime(r["start"], "%H:%M"); b = datetime.strptime(r["end"], "%H:%M"); out[r["doctor_id"]] = out.get(r["doctor_id"], 0) + int((b - a).total_seconds() // 60 // r["slot_minutes"])
    return out


@app.get("/api/admin/emails")
def admin_emails(limit: int = 100, u=Depends(need("admin")), db=Depends(get_db)):
    return {"live": mailer.configured(), "rows": _appt_rows(db.execute("SELECT id, to_addr, subject, kind, status, error, created FROM email_log ORDER BY id DESC LIMIT ?", (min(limit, 500),)))}


@app.post("/api/admin/run-reminders")
def admin_run_reminders(u=Depends(need("admin")), db=Depends(get_db)):
    out = reminders.run_reminders(db); return {"processed": [{"kind": k, "to": to, "status": st} for k, to, st in out]}


@app.get("/api/admin/calendar")
def admin_calendar(start: str, end: str, u=Depends(need("admin")), db=Depends(get_db)):
    return _appt_rows(db.execute(f"SELECT a.id, a.start_ts, a.end_ts, a.status, a.reason, a.reschedule_count, du.name AS doctor, a.doctor_id, pu.name AS patient FROM appointments a "
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


if os.path.isdir(os.path.join(HERE, "static", "acne-lab")): app.mount("/acne-lab", StaticFiles(directory=os.path.join(HERE, "static", "acne-lab"), html=True), name="acne-lab")
if os.path.isdir(os.path.join(HERE, "static")): app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
init_db()
if __name__ == "__main__":
    import uvicorn
    print(acne_lab.start())
    reminders.start_background(connect, int(os.environ.get("REMINDER_INTERVAL_SECONDS", "600")))      # weekly-scan and appointment emails
    uvicorn.run(app, host="127.0.0.1", port=7890, log_level="warning")
