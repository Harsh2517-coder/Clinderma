# FaceVeda: face wellness platform (research prototype)

Three-level SaaS (admin -> doctor -> patient). Patients scan their face (front, left, right) weekly; the scan splits the face into five zones and counts spots per zone; zones are linked to body areas using
**Ayurveda and Traditional Chinese Medicine face maps, shown as traditional and unproven, never as a diagnosis**. Doctors get weekly availability charts, patient records and notes; admins get everything plus a master calendar.

**Watch the overview:** [docs/faceveda_overview.mp4](docs/faceveda_overview.mp4) (about 3 minutes, captioned) | narration script: [docs/video_script.md](docs/video_script.md) | screenshots: [docs/screenshots/](docs/screenshots/)

**Quick start**
```
pip install fastapi uvicorn opencv-python numpy scikit-learn scipy joblib onnxruntime ultralytics pillow
python3 seed.py          # demo data (demo logins are listed at the top of seed.py)
python3 app.py           # http://127.0.0.1:7890
python3 -m pytest -q     # 21 tests
```
The scan needs the Clinderma V_3 pipeline and its two model files (set `CLINDERMA_V3` to its folder, default `../Clinderma/V_3`).

**What is measured, what is simulated, what is not proven**
- Measured: scan accuracy per face zone on expert-labelled photos (`models/spot_filter_metrics.json`, `results/spot_filter_cv.json`).
- Simulated: the "future outlook" tiers come from a synthetic population with an assumed face-map link (`risk_sim.py`, `SIMULATION.md`). They describe a method, not reality.
- Not proven: any link between face zones and body organs or later disease.

**Notices**
- `static/css/` (design tokens, layout, components) is adapted from the Satkaam TAP three-level SaaS web UI, an internship project. Confirm you may reuse it before sharing this beyond the team.
- `models/spot_filter.joblib` was trained on the Clinderma labelled acne photos. No photos or patient identifiers are in this repository. Never commit `.env`, `data/` or `results/` logs (they can contain patient filenames).
- Not ready for real patients: needs HTTPS, audit logging, consent and retention rules, and clinician review of the wording.

---

Three roles: **admin** (all doctors, their patients, master calendar) -> **doctor** (own patients, weekly availability chart, upcoming/previous appointments, notes) -> **patient**
(picks a doctor once, fills the intake form, scans their face, books free slots, reads the doctor's notes).

- `app.py` FastAPI + SQLite API (`python3 app.py` -> http://127.0.0.1:7890), `insights.py` knowledge base + zone/intake logic, `seed.py` demo data, `test_core.py` rules (`python3 -m pytest -q`).
- Double booking is prevented by unique partial indexes in the database, not only by checks in code.
- `scan.py` turns photos into per-zone spot counts using the Clinderma V_3 pipeline (set `CLINDERMA_V3` if the folder moves); `static/` is the UI (vanilla JS, Satkaam-style design tokens).
- Patient flow: register -> health form -> first face scan (front + optional left/right) -> doctors ranked from the scan -> pick one. Later the patient can switch doctor; the
  previous doctor keeps read access, the new doctor sees the whole history, and upcoming bookings with the previous doctor are cancelled.
- Run: `python3 seed.py` (demo data; logins are listed at the top of seed.py) then `python3 app.py` -> http://127.0.0.1:7890. Tests: `python3 -m pytest -q`.

## How the indications are produced
Per-zone spot counts are combined with the patient's intake form. Face-zone -> body-system links come from **Traditional Chinese Medicine face mapping and Ayurveda (mukha pariksha)**;
they are shown as *traditional, not clinically validated*, with the modern view next to each. Only the chin/jaw hormonal-metabolic link has partial modern support
(acne severity vs hyperandrogenism / PCOS / insulin resistance). If the patient already reported an issue in the linked system the wording says so and suggests a doctor review.
Output is never a diagnosis. Each screen must show the disclaimer.

## Before real patients
Prototype security only (scrypt hashes, signed cookie). Needs HTTPS, audit logging, backups, consent and retention rules, and a clinician-approved wording review before real use.


## Email reminders (free)
- Weekly scan reminder: sent when a patient's last scan is 7+ days old (at most once per calendar week). Appointment reminder: 24 hours before. Booking confirmation: immediately.
  Patients can switch reminders off on the My progress page. Emails contain no health findings.
- With no `.env`, the platform is in **test mode**: every email is recorded in the Email log (admin menu) with status `dry_run` and nothing is sent.
- To send real emails for free: copy `.env.example` to `.env`, create a Gmail **app password** (Google account > Security > 2-Step Verification > App passwords), and fill in the values. Restart the app.
  A personal Gmail is fine for a demo (about 500 emails a day); a real product should send from its own verified domain.

## Scan accuracy
`eval_accuracy.py` + `eval_report.py` measure the scan against expert-labelled lesions by face zone (results in `results/accuracy_by_zone.json`). `build_spot_dataset.py` and `train_spot_filter.py` train
the filter that keeps only the proposed spots that look like real acne-type lesions. The training data are the Clinderma labelled images and are read, never copied or modified.

## Logo
The FaceVeda logo is `static/logo.png`; `python3 make_logo_assets.py` rebuilds the small monogram and favicon from it. `python3 docs/make_video.py` rebuilds the overview video from `docs/screenshots/`.
