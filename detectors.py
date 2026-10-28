"""Pluggable acne/spot detector. The platform runs any detector locally (patient photos never leave the machine).

Configure with environment variables (or in .env):
    DETECTOR=onnx | yolo | rfdetr     (unset = built-in spot finder + trained filter)
    DETECTOR_PATH=models/acne_detector.onnx            path to the model file
    DETECTOR_CLASSES=comedone,papule,pustule,nodule,cyst   class names in the model's class order (optional)
    DETECTOR_CONF=0.15                                  minimum confidence
    DETECTOR_INPUT=cutout | crop | full                 what the model sees: cutout = CLAHE face-only (background, hair, eyes blacked out: how V_3 trains it),
                                                        crop = the untouched face crop, full = the whole photo
    DETECTOR_SIZE=512                                   ONNX only: square input size if the model has a dynamic input shape
Every detector returns [{"box": (x0, y0, x1, y1), "conf": float, "cls": str}] in the coordinates of the image it was given.
"""
import os

import numpy as np

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)    # RF-DETR / ImageNet normalisation


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def parse_rfdetr(dets, labels, W, H, conf, classes=None, class_offset=0):
    """RF-DETR ONNX outputs: dets (1, Q, 4) as normalised cx,cy,w,h and labels (1, Q, C) logits. Returns detections in pixel coordinates of a W x H image."""
    d, l = np.asarray(dets)[0], np.asarray(labels)[0]; p = _sigmoid(l); score = p.max(axis=1); cls = p.argmax(axis=1) - class_offset; out = []
    for (cx, cy, w, h), s, c in zip(d, score, cls):
        if s < conf or c < 0: continue
        name = classes[c] if classes and c < len(classes) else str(int(c))
        out.append({"box": (float((cx - w / 2) * W), float((cy - h / 2) * H), float((cx + w / 2) * W), float((cy + h / 2) * H)), "conf": float(s), "cls": name})
    return out


def parse_yolo_onnx(out, W, H, size, conf, classes=None):
    """Ultralytics-exported ONNX: (1, 4+C, N) with cx,cy,w,h in input-pixel units. Boxes are scaled from the square input back to the W x H image."""
    a = np.asarray(out)[0]
    if a.shape[0] > a.shape[1]: a = a.T                                   # tolerate (N, 4+C): make it (4+C, N)
    res = []
    for col in a.T:
        cx, cy, w, h = col[:4]; sc = col[4:]; s, c = float(sc.max()), int(sc.argmax())
        if s < conf: continue
        name = classes[c] if classes and c < len(classes) else str(c)
        res.append({"box": (float((cx - w / 2) * W / size), float((cy - h / 2) * H / size), float((cx + w / 2) * W / size), float((cy + h / 2) * H / size)), "conf": s, "cls": name})
    return res


def nms(dets, iou_thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        ok = True
        for k in keep:
            a, b = d["box"], k["box"]; ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
            if i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9) > iou_thr: ok = False; break
        if ok: keep.append(d)
    return keep


class OnnxDetector:
    def __init__(self, path, classes=None, conf=0.15, size=None, class_offset=0):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); self.classes, self.conf, self.class_offset = classes, conf, class_offset
        inp = self.s.get_inputs()[0]; self.in_name = inp.name; shp = inp.shape; self.H = shp[2] if isinstance(shp[2], int) else (size or 512); self.W = shp[3] if isinstance(shp[3], int) else (size or 512)
        self.out_names = [o.name for o in self.s.get_outputs()]; self.rfdetr = "dets" in self.out_names and "labels" in self.out_names; self.name = f"onnx:{os.path.basename(path)}"

    def detect(self, bgr, conf=None):
        import cv2
        conf = self.conf if conf is None else conf; H0, W0 = bgr.shape[:2]
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.W, self.H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        if self.rfdetr: x = (x - MEAN) / STD                                   # RF-DETR: ImageNet normalisation; ultralytics-exported ONNX: plain 0..1
        outs = dict(zip(self.out_names, self.s.run(None, {self.in_name: x.transpose(2, 0, 1)[None].astype(np.float32)})))
        if self.rfdetr: return parse_rfdetr(outs["dets"], outs["labels"], W0, H0, conf, self.classes, self.class_offset)
        return nms(parse_yolo_onnx(next(iter(outs.values())), W0, H0, self.W, conf, self.classes))


class YoloDetector:
    def __init__(self, path, classes=None, conf=0.15, device=None):
        from ultralytics import YOLO
        self.m = YOLO(path); self.classes, self.conf, self.device = classes, conf, device; self.name = f"yolo:{os.path.basename(path)}"

    def detect(self, bgr, conf=None):
        r = self.m.predict(bgr, conf=self.conf if conf is None else conf, verbose=False, device=self.device)[0]; names = self.classes or r.names
        return [{"box": tuple(float(v) for v in b), "conf": float(c), "cls": (names[int(k)] if isinstance(names, (list, tuple)) else names.get(int(k), str(int(k))))} for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


class RFDETRDetector:
    """Runs a Roboflow/RF-DETR .pth checkpoint with the open-source `rfdetr` package (pip install rfdetr)."""
    def __init__(self, path, classes=None, conf=0.15, variant="small"):
        try: import rfdetr
        except ImportError as e: raise RuntimeError("DETECTOR=rfdetr needs `pip install rfdetr`; or export the checkpoint to ONNX and use DETECTOR=onnx") from e
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "base": rfdetr.RFDETRBase, "medium": rfdetr.RFDETRMedium}[variant]
        self.m = cls(pretrain_weights=path); self.classes, self.conf = classes, conf; self.name = f"rfdetr:{os.path.basename(path)}"

    def detect(self, bgr, conf=None):
        import cv2
        d = self.m.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=self.conf if conf is None else conf)
        return [{"box": tuple(float(v) for v in b), "conf": float(c), "cls": (self.classes[int(k)] if self.classes and int(k) < len(self.classes) else str(int(k)))} for b, c, k in zip(d.xyxy, d.confidence, d.class_id)]


_cache = {"d": None, "loaded": False}


def set_detector(d): _cache.update(d=d, loaded=True)                    # tests / custom code


def get():
    """The configured detector, or None to use the built-in spot finder + trained filter. Config errors raise, so a wrong path is noticed immediately."""
    if _cache["loaded"]: return _cache["d"]
    kind = os.environ.get("DETECTOR", "").strip().lower(); _cache["loaded"] = True
    if not kind: return None
    path = os.environ.get("DETECTOR_PATH", ""); here = os.path.dirname(os.path.abspath(__file__)); path = path if os.path.isabs(path) else os.path.join(here, path)
    if not os.path.exists(path): raise FileNotFoundError(f"DETECTOR_PATH not found: {path}")
    classes = [c.strip() for c in os.environ.get("DETECTOR_CLASSES", "").split(",") if c.strip()] or None; conf = float(os.environ.get("DETECTOR_CONF", "0.15"))
    if kind == "onnx": d = OnnxDetector(path, classes, conf, int(os.environ["DETECTOR_SIZE"]) if os.environ.get("DETECTOR_SIZE") else None, int(os.environ.get("DETECTOR_CLASS_OFFSET", "0")))
    elif kind == "yolo": d = YoloDetector(path, classes, conf, os.environ.get("DETECTOR_DEVICE") or None)
    elif kind == "rfdetr": d = RFDETRDetector(path, classes, conf, os.environ.get("DETECTOR_VARIANT", "small"))
    else: raise ValueError("DETECTOR must be onnx, yolo or rfdetr")
    _cache["d"] = d; return d


def input_mode(): return os.environ.get("DETECTOR_INPUT", "cutout").strip().lower()
