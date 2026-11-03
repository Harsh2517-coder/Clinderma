"""Face landmarks for photos, from any interpreter: runs landmark_worker.py in the interpreter that has mediapipe 0.10 (env FACEFUSION_PYTHON, default ../facefusion_venv/bin/python)
and caches the result per photo content, so a photo is only analysed once."""
import hashlib, json, os, subprocess, sys, tempfile
import cv2, numpy as np
from . import model as M

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = os.environ.get("FACEFUSION_PYTHON") or os.path.join(ROOT, "..", "facefusion_venv", "bin", "python")
CACHE = os.path.join(ROOT, "data", "landmark_cache")


def _key(img): s = 640 / max(img.shape[:2]); return hashlib.sha1(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).tobytes()).hexdigest()


def unmirror(entry, perm):
    """Landmarks found on a mirrored copy -> landmarks of the real photo (x mirrored, and every landmark swapped with its left/right partner)."""
    lm = np.asarray(entry["lm"], float).copy(); lm[:, 0] = 1 - lm[:, 0]; out = np.empty_like(lm); out[perm] = lm; return out


def available(): return os.path.exists(PY) and os.path.exists(os.path.join(ROOT, "models", "face_landmarker.task"))


def detect(images, m=None):
    """images: {name: BGR image}. Returns {name: (478,3) normalised landmarks in the photo's own orientation, or None if no face mesh could be fitted}."""
    m = m or M.load(); os.makedirs(CACHE, exist_ok=True); out, todo = {}, {}
    for name, img in images.items():
        k = _key(img); f = os.path.join(CACHE, k + ".json")
        if os.path.exists(f):
            e = json.load(open(f)); out[name] = (unmirror(e, m["perm"]) if e.get("flipped") else np.asarray(e["lm"])) if e.get("ok") else None
        else: todo[name] = (k, img)
    if todo:
        if not available(): raise RuntimeError("face_fusion needs mediapipe 0.10 (set FACEFUSION_PYTHON) and models/face_landmarker.task")
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, (k, img) in todo.items():
                s = min(1.0, 2400 / max(img.shape[:2])); p = os.path.join(d, k + ".jpg"); cv2.imwrite(p, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img); paths.append(p)
            res_f = os.path.join(d, "res.json"); r = subprocess.run([PY, os.path.join(HERE, "landmark_worker.py"), res_f] + paths, capture_output=True, text=True, timeout=600)
            if not os.path.exists(res_f): raise RuntimeError("landmark worker failed: " + r.stderr[-400:])
            res = {os.path.basename(e["path"])[:-4]: e for e in json.load(open(res_f))}
        for name, (k, img) in todo.items():
            e = res[k]; json.dump({kk: e[kk] for kk in ("ok", "flipped", "lm") if kk in e}, open(os.path.join(CACHE, k + ".json"), "w"))
            out[name] = (unmirror(e, m["perm"]) if e.get("flipped") else np.asarray(e["lm"])) if e.get("ok") else None
    return out
