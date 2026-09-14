"""Face outline segmentation: return just the face skin (+ nose + ears),
0% background -- hair, neck, clothing, and the scene are all masked out.
Eyes, eyebrows, and mouth/lips are *also* physically cut out (left as
transparent holes), not just excluded from spot detection: those regions
are never skin, so removing the pixels entirely -- rather than relying on
detection-side exclusion zones -- guarantees no false-positive "spot" can
ever land there. Two stages:
  1. YOLOv8-face (face_crop.py) finds the face and gives a generous crop --
     this is what makes it robust to front/left/right profile shots and to
     a face that's small within a larger photo.
  2. A BiSeNet face-parsing model (yakhyo/face-parsing, CelebAMask-HQ
     19-class) segments that crop pixel-by-pixel, so the mask follows the
     real jaw/hairline contour instead of an approximated oval.
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

from face_crop import crop_face

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_parsing_resnet18.onnx")

# CelebAMask-HQ class indices (0 = background)
# skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, neck_l, cloth, hair, hat
FACE_CLASSES = {1, 7, 8, 9, 10}  # skin, ears, ear ring, nose --
# eyebrows(2,3), eyes(4,5), glasses(6), mouth(11), lips(12,13) deliberately excluded

_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name
_INPUT_SIZE = (512, 512)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _parse_mask(img_bgr):
    """Per-pixel CelebAMask-HQ class index, same size as img_bgr."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    normed = (resized - _MEAN) / _STD
    batch = np.expand_dims(np.transpose(normed, (2, 0, 1)), 0).astype(np.float32)
    output = _session.run(None, {_input_name: batch})[0]
    small_mask = output.squeeze(0).argmax(0).astype(np.uint8)
    return cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def analyze_face(img_bgr, conf=0.3, margin=0.3):
    """Detect + parse the face once. Returns a dict with:
      crop        -- BGR face crop (from face_crop.crop_face)
      class_mask  -- per-pixel CelebAMask-HQ class index, same size as crop
      face_mask   -- cleaned binary mask (uint8 0/255) of FACE_CLASSES,
                     largest connected component only
    or None if no face is found. Shared by segment_face() and spot detection
    so the parsing model only runs once per image."""
    crop = crop_face(img_bgr, conf=conf, margin=margin)
    if crop is None:
        return None

    class_mask = _parse_mask(crop)
    face_mask = np.isin(class_mask, list(FACE_CLASSES)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(face_mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        face_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    if not face_mask.any():
        return None

    return {"crop": crop, "class_mask": class_mask, "face_mask": face_mask}


def segment_face(img_bgr, conf=0.3, margin=0.3, feather=3):
    """Detect the face, then segment it precisely. Returns a BGRA crop with
    alpha=0 outside the true face outline (hair/neck/background removed), or
    None if no face is found."""
    analysis = analyze_face(img_bgr, conf=conf, margin=margin)
    if analysis is None:
        return None
    crop, face_mask = analysis["crop"], analysis["face_mask"]

    if feather > 0:
        face_mask = cv2.GaussianBlur(face_mask, (feather * 2 + 1, feather * 2 + 1), 0)

    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = face_mask

    ys, xs = np.where(face_mask > 0)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return bgra[y0:y1, x0:x1]


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    out_img = segment_face(img)
    if out_img is None:
        print("no face detected")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else "segmented.png"
        cv2.imwrite(out, out_img)
        print("saved", out, out_img.shape)
