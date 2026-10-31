"""Acne Lab: the Clinderma acne prototype (React UI + FastAPI pipeline) inside FaceVeda.

The UI is a static build in static/acne-lab, served at /acne-lab/. Its API runs as a separate process (the prototype backend, default http://127.0.0.1:8000)
and FaceVeda forwards /acne-lab/api/* to it, so everything is one origin and only signed-in FaceVeda users can reach it. Photos are processed in memory and never stored.

Environment (.env):
    ACNE_LAB_DIR=D:\CLINDERMA\prototype\backend   folder of the prototype backend (has the `app` package). Unset = do not start it automatically.
    ACNE_LAB_URL=http://127.0.0.1:8000               where the prototype backend listens
"""
import atexit, os, socket, subprocess, sys, time
from urllib.parse import urlparse

LAB_URL = os.environ.get("ACNE_LAB_URL", "http://127.0.0.1:8000").rstrip("/")
_proc = None


def _up():
    u = urlparse(LAB_URL)
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=0.5): return True
    except OSError: return False


def start():
    """Starts the prototype backend if ACNE_LAB_DIR is set and nothing is listening yet. Returns a short status string."""
    global _proc
    d = os.environ.get("ACNE_LAB_DIR", "")
    if _up(): return f"Acne Lab backend already running at {LAB_URL}"
    if not d or not os.path.isdir(d): return "Acne Lab backend not started (set ACNE_LAB_DIR in .env to the prototype's backend folder)"
    port = urlparse(LAB_URL).port or 8000
    _proc = subprocess.Popen([os.environ.get("ACNE_LAB_PYTHON", sys.executable), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], cwd=d)
    atexit.register(stop)
    for _ in range(60):
        if _up(): return f"Acne Lab backend started at {LAB_URL}"
        time.sleep(0.5)
    return "Acne Lab backend is still starting (first start loads the models)"


def stop():
    global _proc
    if _proc and _proc.poll() is None: _proc.terminate()
    _proc = None
