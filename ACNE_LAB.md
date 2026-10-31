# Acne Lab (Clinderma prototype) inside FaceVeda

- **Model:** `models/acne_rfdetr_small.onnx` (RF-DETR small, classes comedone, nodule, papule, pustule), plugged in through `.env` (`DETECTOR=onnx`, `DETECTOR_INPUT=full`, `DETECTOR_CONF=0.15`). The weights are git-ignored (over GitHub's 100 MB limit): share them privately.
- **Patient scans** now also return a GAGS block (`report.gags`, see `gags.py`): worst lesion per zone x zone factor, acne types, and a lesion-count safeguard so widespread acne is not reported as mild. Shown on every scan report.
- **Acne Lab** (doctors and admins, sidebar "Acne Lab (GAGS)"): the prototype UI at `/acne-lab/`, restyled to FaceVeda. Its API is the prototype backend, started by `app.py` (`acne_lab.py`) and reached through `/acne-lab/api/*`, so only signed-in users can use it. Photos are not stored.
- **Run:** `START_FACEVEDA.bat` (or `python app.py` with the prototype's venv). Demo logins: see the top of `seed.py`.
- **Rebuild the Acne Lab UI** (from the prototype folder): `set ACNE_LAB_OUT=<this folder>\static\acne-lab` then `npm run build`.
- Needs `scikit-learn` and `joblib` in the environment (installed in the prototype venv).
