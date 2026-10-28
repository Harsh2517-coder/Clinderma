"""Scheduled reminders: weekly scan nudges and appointment reminders. `run_reminders` is a plain function (testable with a fake clock and a fake sender);
`start_background` runs it every few minutes on a daemon thread when the app starts."""
import threading, time
from datetime import datetime, timedelta

import mailer

SCAN_EVERY_DAYS = 7
APPT_REMIND_HOURS = 24


def run_reminders(db, now=None, send=mailer.send_email):
    """Returns a list of (kind, to, status). A reminder is sent only if the patient has reminders on, and at most once per (kind, key)."""
    now = now or datetime.now(); out = []
    # weekly scan: last scan at least a week ago -> at most one email per calendar week
    for p in db.execute("SELECT u.id, u.name, u.email FROM patients p JOIN users u ON u.id=p.user_id WHERE p.email_reminders=1 AND p.doctor_id IS NOT NULL").fetchall():
        r = db.execute("SELECT created FROM scans WHERE patient_id=? ORDER BY id DESC LIMIT 1", (p["id"],)).fetchone()
        if not r: continue
        days = (now - datetime.fromisoformat(r["created"])).days
        if days >= SCAN_EVERY_DAYS:
            subj, body = mailer.weekly_scan_reminder(p["name"], days)
            y, w, _ = now.isocalendar(); out.append(("scan_reminder", p["email"], send(db, p["email"], subj, body, "scan_reminder", f"{p['id']}:{y}-W{w}")))
    # appointments starting within the next 24 hours
    lo, hi = now.strftime("%Y-%m-%d %H:%M"), (now + timedelta(hours=APPT_REMIND_HOURS)).strftime("%Y-%m-%d %H:%M")
    for a in db.execute("SELECT a.id, a.start_ts, pu.name AS patient, pu.email, du.name AS doctor FROM appointments a JOIN patients p ON p.user_id=a.patient_id JOIN users pu ON pu.id=a.patient_id "
                        "JOIN users du ON du.id=a.doctor_id WHERE a.status='booked' AND p.email_reminders=1 AND a.start_ts>? AND a.start_ts<=?", (lo, hi)).fetchall():
        when = datetime.strptime(a["start_ts"], "%Y-%m-%d %H:%M").strftime("%a %d %b, %I:%M %p")
        subj, body = mailer.appointment_reminder(a["patient"], a["doctor"], when); out.append(("appt_reminder", a["email"], send(db, a["email"], subj, body, "appt_reminder", f"appt:{a['id']}")))
    return out


def start_background(connect, interval=600):
    def loop():
        while True:
            try:
                db = connect()
                try: run_reminders(db)
                finally: db.close()
            except Exception as e: print("reminder run failed:", e)
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True, name="reminders").start()
