"""Builds docs/faceveda_overview.mp4 (captioned explainer, no audio) from docs/screenshots/*.png and the project's real result files.  python3 docs/make_video.py
Uses static/logo.png on the title slide if it exists.  Slide text is in SLIDES below; docs/video_script.md has the same text as a narration script."""
import json, os, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); SHOTS = os.path.join(HERE, "screenshots"); OUT = os.path.join(HERE, "faceveda_overview.mp4")
W, H, FPS = 1920, 1080, 24; NAVY, ORANGE, BG, INK, MUTED, LINE = (50, 83, 133), (184, 74, 22), (250, 250, 250), (28, 28, 28), (95, 109, 122), (217, 217, 217)
FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]; BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]


def font(size, bold=False):
    for p in (BOLD if bold else FONTS):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def jload(rel, default=None):
    p = os.path.join(ROOT, rel); return json.load(open(p)) if os.path.exists(p) else default


ACC = jload("models/spot_filter_metrics.json", {}); SIM = jload("results/simulation_results.json", {})
pct = lambda x: f"{round(100 * x)}%"
fa, ua = ACC.get("filtered", {}), ACC.get("unfiltered", {})
mean_gain = lambda sc: (sum(v["gain"] for v in SIM["auc"][sc].values()) / 8) if SIM else 0

SLIDES = [
    ("title",),
    ("text", "What is FaceVeda?", ["A three-level platform: **admin** manages doctors, **doctors** manage patients, **patients** scan their face and book visits.",
                                    "A face scan splits the face into five zones and counts spots in each one.",
                                    "Spots are linked to body areas using **Ayurveda and Traditional Chinese Medicine face maps**, always shown as traditional and unproven, never as a diagnosis.",
                                    "Patients repeat the scan weekly; a progress dashboard shows the trend. A college research prototype."]),
    ("img", "01_login", "Start", "Three roles, one login", "Patients can register themselves. Doctors and admins are created by the admin."),
    ("img", "02_setup_health_form", "Patient setup · step 1", "Health form", "History by body system, medicines, family history and lifestyle. It personalises every insight."),
    ("img", "03_setup_scan_three_angles", "Patient setup · step 2", "First scan: three angles", "Front, left and right photos are all required the first time. Photos are analysed in memory and not stored."),
    ("img", "04_setup_choose_doctor", "Patient setup · step 3", "Doctors ranked from the scan", "Each doctor has focus areas. The scan and health form decide who is recommended; the patient chooses."),
    ("img", "05_patient_home", "Patient", "Home", "A weekly-scan reminder, the next appointment and the doctor's latest note."),
    ("img", "06_scan_zones_and_findings", "Patient · scan result", "Spots per face zone", "Zones are compared by spot density, then matched with what the patient reported, for example a digestive issue."),
    ("img", "07_if_not_addressed", "Patient · scan result", "If this is not addressed", "Conditions doctors watch for, tagged traditional, partly supported or well established. Personal risk factors are listed."),
    ("img", "08_simulated_outlook", "Patient · research prototype", "Simulated outlook", "Trained on a synthetic population, clearly labelled as a simulation. Qualitative tiers only, never probabilities."),
    ("img", "10_progress_dashboard", "Patient · progress", "Progress over time", "Total spots per scan, weekly streak and next scan due. Reminder emails can be switched off."),
    ("img", "11_progress_zones", "Patient · progress", "Zone trends and scanning habit", "Which zones are changing, and how regularly the patient scans."),
    ("img", "12_appointments_booking", "Patient", "Booking", "Only free slots from the doctor's weekly chart are shown. A slot cannot be double booked."),
    ("img", "13_my_doctor_switch", "Patient", "Switching doctors", "A patient can switch later. The previous doctor keeps read access; the new doctor sees the full history."),
    ("img", "14_doctor_dashboard", "Doctor", "Dashboard", "My patients, upcoming visits, and a flag when the latest scan suggests a review."),
    ("img", "15_doctor_patient_record", "Doctor", "Full patient record", "Health form, scan progress, all doctors' notes. Notes written here are visible to the patient."),
    ("img", "16_doctor_availability", "Doctor", "Weekly availability", "Hours per weekday and slot length. Existing bookings are never cancelled by a change."),
    ("img", "17_admin_overview", "Admin", "Overview", "All doctors, their patients and upcoming visits."),
    ("img", "18_admin_master_calendar", "Admin", "Master calendar", "Every doctor's appointments in one week view."),
    ("img", "19_admin_email_log", "Admin", "Email reminders", "Weekly scan and appointment reminders. Test mode until real email (SMTP) is configured; emails never contain health findings."),
    ("text", "How accurate is the scan?", [f"Measured on {ACC.get('photos', 555)} expert-labelled photos of different people (cross-validated).",
                                           f"The raw spot finder found {pct(ua.get('recall', .86))} of real lesions but only {pct(ua.get('precision', .06))} of its spots were real.",
                                           f"A trained filter raised precision to **{pct(fa.get('precision', .34))}** (recall {pct(fa.get('recall', .53))}) and the right most-affected zone from {pct(ua.get('agree', .28))} to **{pct(fa.get('agree', .48))}** (guessing: 20%).",
                                           "Weak spot: the nose (recall about 11%). The app flags low-reliability zones.",
                                           "A friend's Roboflow detector can be plugged in later; it runs locally so photos never leave the machine."]),
    ("text", "Simulation study: future-disease prediction", ["No real dataset links face patterns to later disease, so we **simulated** a population with an assumed link and measured scanner noise.",
                                                            f"If the face map had no effect, zone data added {mean_gain('null'):+.3f} AUC. Even with a moderate real effect it added only {mean_gain('moderate'):+.3f}.",
                                                            "A real study would need about **3,000 people** to detect a moderate link for digestive or stress outcomes.",
                                                            "These numbers describe the method, not reality. The app labels the outlook as simulated."]),
    ("text", "What is proven and what is not", ["**Tested:** roles and access rules, booking without double bookings, doctor switching, reminders, progress maths (21 automated tests).",
                                                "**Measured:** scan accuracy per face zone on labelled photos.",
                                                "**Not proven:** that face zones relate to body organs or later disease. Everything linking zones to organs is traditional and shown as such.",
                                                "**Not clinical:** needs HTTPS, audit logging, consent rules and clinician review before real patients."]),
    ("text", "Next steps and what we need", ["From the Roboflow friend: model file (ONNX, .pth or YOLO .pt), class names, what the model was trained on, and held-out test photos with labels.",
                                              "Add the FaceVeda logo and real email settings.",
                                              "Clinician review of the traditional-medicine wording.",
                                              "Optional: a PCOS risk model trained on the real Kaggle dataset."]),
]
DUR = {"title": 5.0, "text": 11.0, "img": 5.5}


def wrap(draw, text, fnt, width):
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width: line = t
        else: out.append(line); line = w
    return out + ([line] if line else [])


def rich(draw, x, y, text, size, width, color=INK, gap=10):
    """Draws text where **bold** segments are bold; returns the new y."""
    parts = text.split("**"); words = []
    for i, seg in enumerate(parts):
        for w in seg.split(): words.append((w, i % 2 == 1))
    cur, cw = [], 0
    fr, fb = font(size), font(size, True)
    for w, b in words:
        f = fb if b else fr; ww = draw.textlength(w + " ", font=f)
        if cw + ww > width and cur:
            xx = x
            for cw_, cb in cur: draw.text((xx, y), cw_, font=fb if cb else fr, fill=color); xx += draw.textlength(cw_ + " ", font=fb if cb else fr)
            y += int(size * 1.35); cur, cw = [], 0
        cur.append((w, b)); cw += ww
    xx = x
    for cw_, cb in cur: draw.text((xx, y), cw_, font=fb if cb else fr, fill=color); xx += draw.textlength(cw_ + " ", font=fb if cb else fr)
    return y + int(size * 1.35) + gap


def mark(draw, cx, cy, r, ring=(255, 255, 255), accent=ORANGE):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ring, width=max(3, r // 7)); e = r // 3
    for dx in (-e, e): draw.ellipse([cx + dx - r // 9, cy - e // 2 - r // 9, cx + dx + r // 9, cy - e // 2 + r // 9], fill=ring)
    draw.arc([cx - r // 2, cy - r // 4, cx + r // 2, cy + r // 2], 20, 160, fill=accent, width=max(3, r // 7))


def chrome(img, idx, total, tag):
    d = ImageDraw.Draw(img); d.rectangle([0, 0, W, 90], fill=NAVY); lm = os.path.join(ROOT, "static", "logo-mark.png")
    if os.path.exists(lm):
        d.rounded_rectangle([34, 12, 102, 78], 12, fill="white"); img.paste(Image.open(lm).convert("RGB").resize((60, 60), Image.LANCZOS), (38, 15)); d.text((118, 24), "FaceVeda", font=font(36, True), fill="white")
    else: mark(d, 56, 45, 22); d.text((96, 24), "FaceVeda", font=font(36, True), fill="white")
    tw = d.textlength(tag, font=font(26)); d.text((W - 48 - tw, 31), tag, font=font(26), fill=(215, 224, 240))
    d.rectangle([0, H - 8, W, H], fill=LINE); d.rectangle([0, H - 8, int(W * idx / total), H], fill=ORANGE); d.text((W - 150, H - 48), f"{idx} / {total}", font=font(22), fill=MUTED)


def slide_frame(s, idx, total):
    img = Image.new("RGB", (W, H), BG)
    if s[0] == "title":
        img = Image.new("RGB", (W, H), NAVY); d = ImageDraw.Draw(img); logo = os.path.join(ROOT, "static", "logo.png")
        if os.path.exists(logo):
            lg = Image.open(logo).convert("RGB"); k = 640 / lg.height; lg = lg.resize((int(lg.width * k), 640)); card = Image.new("RGB", (lg.width + 60, 700), "white"); card.paste(lg, (30, 30)); img.paste(card, (W // 2 - card.width // 2, 70))
            y = 810
        else:
            mark(d, W // 2, 330, 150); d.text((W // 2 - d.textlength("FaceVeda", font=font(120, True)) / 2, 520), "FaceVeda", font=font(120, True), fill="white"); y = 700
        for t, f, c in (("Face wellness platform: what we built and how it works", font(44), (235, 240, 250)), ("Three roles · weekly face scans · traditional-medicine insights · research prototype", font(30), (190, 205, 230))):
            d.text((W // 2 - d.textlength(t, font=f) / 2, y), t, font=f, fill=c); y += 70
        return np.array(img)[:, :, ::-1]
    chrome(img, idx, total, "Overview for the team"); d = ImageDraw.Draw(img)
    if s[0] == "text":
        d.text((96, 150), s[1], font=font(64, True), fill=NAVY); d.rectangle([96, 240, 260, 246], fill=ORANGE); y = 300
        for b in s[2]:
            d.ellipse([100, y + 16, 116, y + 32], fill=ORANGE); y = rich(d, 150, y, b, 38, 1640, gap=26)
        return np.array(img)[:, :, ::-1]
    _, fn, tag, title, cap = s; shot = Image.open(os.path.join(SHOTS, fn + ".png")).convert("RGB").resize((1296, 729), Image.LANCZOS)
    d.rectangle([44, 146, 44 + 1296 + 8, 146 + 729 + 8], fill=(200, 205, 212)); img.paste(shot, (48, 150))
    x0 = 1390; chip = tag.upper(); cw = d.textlength(chip, font=font(22, True)) + 36; d.rounded_rectangle([x0, 150, x0 + cw, 194], 22, fill=ORANGE); d.text((x0 + 18, 158), chip, font=font(22, True), fill="white")
    y = 228
    for line in wrap(d, title, font(46, True), 480): d.text((x0, y), line, font=font(46, True), fill=INK); y += 58
    y += 14
    for line in wrap(d, cap, font(29), 480): d.text((x0, y), line, font=font(29), fill=MUTED); y += 42
    return np.array(img)[:, :, ::-1]


def main():
    total = len(SLIDES); vw = None
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*fourcc), FPS, (W, H))
        if vw.isOpened(): print("encoder:", fourcc); break
    prev = None; frames = 0
    for i, s in enumerate(SLIDES, 1):
        f = slide_frame(s, i, total)
        if prev is not None:
            for k in range(1, 13): vw.write(cv2.addWeighted(prev, 1 - k / 12, f, k / 12, 0)); frames += 1      # half-second crossfade
        n = int(DUR[s[0]] * FPS)
        for _ in range(n): vw.write(f)
        frames += n; prev = f
    vw.release(); print(f"wrote {OUT}: {frames} frames, {frames / FPS:.0f} s")
    cap = cv2.VideoCapture(OUT); print("readable frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "size", int(cap.get(3)), "x", int(cap.get(4))); cap.release()
    with open(os.path.join(HERE, "video_script.md"), "w") as fh:
        fh.write("# FaceVeda overview: narration script (one block per slide, about 5 to 11 seconds each)\n\n")
        for i, s in enumerate(SLIDES, 1):
            if s[0] == "title": fh.write(f"## {i}. Title\nThis is FaceVeda, a face wellness platform we built as a research prototype. Here is what it does and how it works.\n\n")
            elif s[0] == "text": fh.write(f"## {i}. {s[1]}\n" + "\n".join("- " + b.replace("**", "") for b in s[2]) + "\n\n")
            else: fh.write(f"## {i}. {s[3]} ({s[2]})\n{s[4]}\n\n")


if __name__ == "__main__":
    main()
