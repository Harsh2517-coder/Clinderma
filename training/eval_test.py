"""Score final/rfdetr-small.onnx on the held-out test split (71 photos, never used in training or model selection). IoU>=0.5 matching."""
import glob, os, sys, json
import numpy as np, cv2, onnxruntime as ort

NAMES = ["comedone", "nodule", "papule", "pustule"]
sess = ort.InferenceSession("final/rfdetr-small.onnx", providers=["CPUExecutionProvider"])
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def run(img):
    x = cv2.resize(img, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    dets, logits = sess.run(None, {sess.get_inputs()[0].name: x})
    p = 1 / (1 + np.exp(-logits[0]))          # (300, 5) sigmoid scores
    return dets[0], p                          # dets = cx,cy,w,h normalised


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1)); i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def load(split):
    out = []
    for f in sorted(glob.glob(f"dataset_1x/{split}/images/*.jpg")):
        gt = [list(map(float, l.split())) for l in open(f"dataset_1x/{split}/labels/{os.path.basename(f)[:-4]}.txt") if l.strip()]
        out.append((f, [(int(g[0]), g[1:]) for g in gt]))
    return out


def score(data, conf, offset, class_aware):
    tp = fp = fn = 0; per = {c: [0, 0, 0] for c in range(4)}
    for f, gt in data:
        d, p = run(cv2.imread(f)); cls = p.argmax(1); sc = p.max(1)
        keep = [(sc[i], cls[i] - offset, d[i]) for i in range(300) if sc[i] >= conf and 0 <= cls[i] - offset < 4]
        keep.sort(key=lambda t: -t[0]); used = set()
        for s, c, b in keep:
            best, bj = 0, -1
            for j, (gc, gb) in enumerate(gt):
                if j in used or (class_aware and gc != c): continue
                v = iou(b, gb)
                if v > best: best, bj = v, j
            if best >= 0.5:
                used.add(bj); tp += 1; per[gt[bj][0]][0] += 1
            else: fp += 1; per[c][1] += 1
        fn += len(gt) - len(used)
        for j, (gc, gb) in enumerate(gt):
            if j not in used: per[gc][2] += 1
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(2 * P * R / max(P + R, 1e-9), 3), tp=tp, fp=fp, fn=fn,
                per_class={NAMES[c]: dict(precision=round(v[0] / max(v[0] + v[1], 1), 3), recall=round(v[0] / max(v[0] + v[2], 1), 3), gt=v[0] + v[2]) for c, v in per.items()})


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"; data = load(split)
    print(split, len(data), "images,", sum(len(g) for _, g in data), "lesions")
    for off in (0, 1):
        r = score(data, 0.3, off, True); print(f"class offset {off}: class-aware @conf0.3 ->", {k: r[k] for k in ("precision", "recall", "f1")})
