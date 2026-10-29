"""Email sending (free: any SMTP server, e.g. Gmail with an app password, or Brevo's free plan) with a built-in dry-run mode.

Configure in a local `.env` file next to app.py (never commit it):
    SMTP_HOST=smtp.gmail.com     SMTP_PORT=587     SMTP_USER=you@gmail.com     SMTP_PASSWORD=<16-character app password>
    MAIL_FROM=FaceVeda <you@gmail.com>     APP_URL=http://127.0.0.1:7890
With SMTP_HOST unset every email is only recorded in the email_log table (status `dry_run`), so the whole flow can be tested without sending anything.
Every attempt is logged; a (kind, key) pair is sent at most once, so a reminder never repeats. Reminder emails contain no health findings, only times and names.
"""
import os, smtplib
from datetime import datetime
from email.message import EmailMessage


def load_env(path):
    """Minimal .env reader (KEY=VALUE per line); real environment variables win."""
    if not os.path.exists(path): return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def configured(): return bool(os.environ.get("SMTP_HOST"))


def already_sent(db, kind, key):
    return db.execute("SELECT 1 FROM email_log WHERE kind=? AND key=? AND status IN ('sent','dry_run')", (kind, key)).fetchone() is not None


def send_email(db, to, subject, body, kind, key):
    """Sends (or dry-runs) one email and records it. Returns 'sent' | 'dry_run' | 'failed' | 'duplicate'."""
    if already_sent(db, kind, key): return "duplicate"
    status, err = "dry_run", ""
    if configured():
        try:
            msg = EmailMessage(); msg["From"] = os.environ.get("MAIL_FROM") or os.environ["SMTP_USER"]; msg["To"] = to; msg["Subject"] = subject; msg.set_content(body)
            host, port = os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587"))
            if port == 465:
                with smtplib.SMTP_SSL(host, port, timeout=20) as s: s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"]); s.send_message(msg)
            else:
                with smtplib.SMTP(host, port, timeout=20) as s:
                    s.starttls(); s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"]); s.send_message(msg)
            status = "sent"
        except Exception as e:      # never let a mail problem break the app; it is logged and retried on the next run
            status, err = "failed", f"{type(e).__name__}: {e}"[:300]
    db.execute("INSERT INTO email_log(to_addr, subject, body, kind, key, status, error, created) VALUES(?,?,?,?,?,?,?,?)",
               (to, subject, body, kind, key, status, err, datetime.now().isoformat(timespec="seconds")))
    db.commit(); return status


def app_url(): return os.environ.get("APP_URL", "http://127.0.0.1:7890")


# ---- templates: plain, no health findings ----
def appointment_confirmation(patient, doctor, when):
    return (f"Appointment confirmed: {when}", f"Hello {patient},\n\nYour appointment with {doctor} is confirmed for {when}.\nYou can view or cancel it here: {app_url()}\n\nFaceVeda")


def appointment_reminder(patient, doctor, when):
    return (f"Reminder: appointment with {doctor} on {when}", f"Hello {patient},\n\nThis is a reminder of your appointment with {doctor} on {when}.\nManage it here: {app_url()}\n\nTo stop reminder emails, turn them off on your Progress page.\n\nFaceVeda")


def weekly_scan_reminder(patient, days):
    return ("Time for your weekly face scan", f"Hello {patient},\n\nIt has been {days} days since your last face scan. A scan once a week lets you and your doctor see how your skin is changing.\nTake it here: {app_url()}/#/scan\n\n"
                                              "To stop reminder emails, turn them off on your Progress page.\n\nFaceVeda")


def appointment_rescheduled(patient, doctor, old, new, by, reason=""):
    who = "your doctor" if by == "doctor" else "you"
    return (f"Appointment moved to {new}", f"Hello {patient},\n\nYour appointment with {doctor} was moved by {who}.\nWas: {old}\nNow: {new}\n" + (f"Reason: {reason}\n" if reason else "") +
            f"\nYou can view or change it here: {app_url()}\n\nFaceVeda")


def appointment_rescheduled_for_doctor(doctor, patient, old, new):
    return (f"{patient} moved an appointment to {new}", f"Hello {doctor},\n\n{patient} moved their appointment.\nWas: {old}\nNow: {new}\n\nSee your schedule: {app_url()}\n\nFaceVeda")
