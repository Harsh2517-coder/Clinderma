# FaceVeda overview: narration script (one block per slide, about 5 to 11 seconds each)

## 1. Title
This is FaceVeda, a face wellness platform we built as a research prototype. Here is what it does and how it works.

## 2. What is FaceVeda?
- A three-level platform: admin manages doctors, doctors manage patients, patients scan their face and book visits.
- A face scan splits the face into five zones and counts spots in each one.
- Spots are linked to body areas using Ayurveda and Traditional Chinese Medicine face maps, always shown as traditional and unproven, never as a diagnosis.
- Patients repeat the scan weekly; a progress dashboard shows the trend. A college research prototype.

## 3. Three roles, one login (Start)
Patients can register themselves. Doctors and admins are created by the admin.

## 4. Health form (Patient setup · step 1)
History by body system, medicines, family history and lifestyle. It personalises every insight.

## 5. First scan: three angles (Patient setup · step 2)
Front, left and right photos are all required the first time. Photos are analysed in memory and not stored.

## 6. Doctors ranked from the scan (Patient setup · step 3)
Each doctor has focus areas. The scan and health form decide who is recommended; the patient chooses.

## 7. Home (Patient)
A weekly-scan reminder, the next appointment and the doctor's latest note.

## 8. Spots per face zone (Patient · scan result)
Zones are compared by spot density, then matched with what the patient reported, for example a digestive issue.

## 9. If this is not addressed (Patient · scan result)
Conditions doctors watch for, tagged traditional, partly supported or well established. Personal risk factors are listed.

## 10. Simulated outlook (Patient · research prototype)
Trained on a synthetic population, clearly labelled as a simulation. Qualitative tiers only, never probabilities.

## 11. Progress over time (Patient · progress)
Total spots per scan, weekly streak and next scan due. Reminder emails can be switched off.

## 12. Zone trends and scanning habit (Patient · progress)
Which zones are changing, and how regularly the patient scans.

## 13. Booking (Patient)
Only free slots from the doctor's weekly chart are shown. A slot cannot be double booked.

## 14. Switching doctors (Patient)
A patient can switch later. The previous doctor keeps read access; the new doctor sees the full history.

## 15. Dashboard (Doctor)
My patients, upcoming visits, and a flag when the latest scan suggests a review.

## 16. Full patient record (Doctor)
Health form, scan progress, all doctors' notes. Notes written here are visible to the patient.

## 17. Weekly availability (Doctor)
Hours per weekday and slot length. Existing bookings are never cancelled by a change.

## 18. Overview (Admin)
All doctors, their patients and upcoming visits.

## 19. Master calendar (Admin)
Every doctor's appointments in one week view.

## 20. Email reminders (Admin)
Weekly scan and appointment reminders. Test mode until real email (SMTP) is configured; emails never contain health findings.

## 21. How accurate is the scan?
- Measured on 555 expert-labelled photos of different people (cross-validated).
- The raw spot finder found 86% of real lesions but only 6% of its spots were real.
- A trained filter raised precision to 34% (recall 53%) and the right most-affected zone from 28% to 48% (guessing: 20%).
- Weak spot: the nose (recall about 11%). The app flags low-reliability zones.
- A friend's Roboflow detector can be plugged in later; it runs locally so photos never leave the machine.

## 22. Simulation study: future-disease prediction
- No real dataset links face patterns to later disease, so we simulated a population with an assumed link and measured scanner noise.
- If the face map had no effect, zone data added -0.000 AUC. Even with a moderate real effect it added only +0.006.
- A real study would need about 3,000 people to detect a moderate link for digestive or stress outcomes.
- These numbers describe the method, not reality. The app labels the outlook as simulated.

## 23. What is proven and what is not
- Tested: roles and access rules, booking without double bookings, doctor switching, reminders, progress maths (21 automated tests).
- Measured: scan accuracy per face zone on labelled photos.
- Not proven: that face zones relate to body organs or later disease. Everything linking zones to organs is traditional and shown as such.
- Not clinical: needs HTTPS, audit logging, consent rules and clinician review before real patients.

## 24. Next steps and what we need
- From the Roboflow friend: model file (ONNX, .pth or YOLO .pt), class names, what the model was trained on, and held-out test photos with labels.
- Add the FaceVeda logo and real email settings.
- Clinician review of the traditional-medicine wording.
- Optional: a PCOS risk model trained on the real Kaggle dataset.

