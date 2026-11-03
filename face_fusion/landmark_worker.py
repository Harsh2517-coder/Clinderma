"""Face landmarks + head pose for a list of photos. Runs in an interpreter that has mediapipe 0.10.x (the system mediapipe 1.0 crashes at start-up on macOS).
    <python-with-mediapipe> landmark_worker.py out.json photo1.jpg photo2.jpg ...
Writes [{path, ok, flipped, lm: 478 x [x, y, z] (x,y normalised 0..1 of the photo, z scaled like x), pose: 4x4}].
The detector is much better on faces turned one way than the other, so on failure it retries on a mirrored copy; `flipped` says that happened and the
landmarks are then in the MIRRORED photo (landmarks.py maps them back)."""
import json, os, sys
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_landmarker.task")
TRIES = [(1600, 0.25, False), (1600, 0.25, True), (1600, 0.05, False), (1600, 0.05, True), (900, 0.25, False), (900, 0.25, True), (2400, 0.05, False), (2400, 0.05, True)]


def main(out, paths):
    dets = {c: vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL, delegate=mpp.BaseOptions.Delegate.CPU), num_faces=1, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=c, min_face_presence_confidence=c, running_mode=vision.RunningMode.IMAGE)) for c in (0.25, 0.05)}
    res = []
    for p in paths:
        img = cv2.imread(p); got = None
        if img is not None:
            for side, conf, flip in TRIES:
                s = side / max(img.shape[:2]); im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                if flip: im = cv2.flip(im, 1)
                r = dets[conf].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                if r.face_landmarks:
                    got = {"path": p, "ok": True, "flipped": flip, "lm": [[l.x, l.y, l.z] for l in r.face_landmarks[0]], "pose": np.array(r.facial_transformation_matrixes[0]).tolist()}; break
        res.append(got or {"path": p, "ok": False})
    json.dump(res, open(out, "w"))


if __name__ == "__main__": main(sys.argv[1], sys.argv[2:])
