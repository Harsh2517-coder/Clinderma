"""Turn a YOLO acne dataset into a *preprocessed* one, using the exact same step the detector will get at inference:
CLAHE-enhanced face cutout (background/hair/neck/eyes/brows/lips blacked out, tight crop) -- see ../preprocess.py `face_cutout`.
Labels are re-mapped into the cutout; boxes whose centre lands outside the face skin are dropped; images with no detectable face are
skipped. Also makes a leakage-aware 80/10/10 split (same person never in two splits).

usage: python3 prepare_dataset.py --src <dataset dir with images/ labels/ data.yaml> --out <output dir> [--seed 0]
"""
import argparse, os, random, re, shutil, sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from preprocess import face_cutout, to_cutout_coords  # noqa: E402

JPG = [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def read_yaml_names(path):
    """Minimal reader for `nc:` and `names: [...]` -- avoids a yaml dependency."""
    txt = open(path).read()
    names = re.search(r"names:\s*\[(.*?)\]", txt, re.S).group(1)
    return [n.strip().strip("'\"") for n in names.split(",") if n.strip()]


def person_groups(stems):
    """Group photos of the same person so train/val/test never share a face. Clinderma naming: v1_* -> phone-number prefix,
    v2_*/v3_* -> HHMMSS timestamps (<=120 s apart = same session). Anything else: each image is its own group."""
    groups, by_src = defaultdict(list), defaultdict(list)
    for s in stems:
        m = re.match(r"(v\d_[a-z0-9]+?)_(.*)", s)
        if not m:
            groups[("other", s)].append(s)
        elif m.group(1).startswith("v1"):
            groups[(m.group(1), m.group(2).rsplit("_", 1)[0])].append(s)
        elif re.match(r"\d{6}", m.group(2)):
            r = m.group(2)
            by_src[m.group(1)].append((int(r[:2]) * 3600 + int(r[2:4]) * 60 + int(r[4:6]), s))
        else:
            groups[(m.group(1), s)].append(s)
    for src, items in by_src.items():
        items.sort()
        g, last = 0, None
        for t, s in items:
            if last is not None and t - last > 120:
                g += 1
            groups[(src, g)].append(s)
            last = t
    return groups


def remap_labels(lines, w, h, origin, mask):
    """YOLO lines in the original image -> YOLO lines in the cutout. Returns (lines, n_dropped)."""
    boxes = []
    for l in lines:
        c, x, y, bw, bh = l.split()[:5]
        boxes.append(((float(x) - float(bw) / 2) * w, (float(y) - float(bh) / 2) * h, (float(x) + float(bw) / 2) * w, (float(y) + float(bh) / 2) * h, int(c)))
    ch, cw = mask.shape
    keep_zone = cv2.dilate(mask.astype(np.uint8), np.ones((2 * max(3, ch // 100) + 1,) * 2, np.uint8)) > 0  # tolerate centres just past the outline
    out, dropped = [], 0
    for x0, y0, x1, y1, c in to_cutout_coords(boxes, origin):
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)
        if not (0 <= cx < cw and 0 <= cy < ch and keep_zone[cy, cx]) or x1 - x0 < 2 or y1 - y0 < 2:
            dropped += 1
            continue
        out.append(f"{c} {(x0 + x1) / 2 / cw:.6f} {(y0 + y1) / 2 / ch:.6f} {(x1 - x0) / cw:.6f} {(y1 - y0) / ch:.6f}")
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = read_yaml_names(f"{args.src}/data.yaml")
    ext = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in os.listdir(f"{args.src}/images")}
    stems = sorted(ext)
    groups = person_groups(stems)
    keys = sorted(groups, key=str)
    random.Random(args.seed).shuffle(keys)
    split, n = {}, 0
    for k in keys:
        sp = "train" if n < 0.8 * len(stems) else "val" if n < 0.9 * len(stems) else "test"
        for s in groups[k]:
            split[s] = sp
        n += len(groups[k])

    for sp in ("train", "val", "test"):
        os.makedirs(f"{args.out}/images/{sp}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{sp}", exist_ok=True)

    no_face, dropped_total, in_total = [], 0, 0
    counts = {sp: Counter() for sp in ("train", "val", "test")}
    n_img = Counter()
    for i, s in enumerate(stems):
        img = cv2.imread(f"{args.src}/images/{s}{ext[s]}")
        lines = [l for l in open(f"{args.src}/labels/{s}.txt").read().splitlines() if l.strip()]
        cut = face_cutout(img)
        if cut is None:
            no_face.append(s)
            continue
        new, dropped = remap_labels(lines, img.shape[1], img.shape[0], cut["origin"], cut["mask"])
        dropped_total, in_total = dropped_total + dropped, in_total + len(lines)
        sp = split[s]
        cv2.imwrite(f"{args.out}/images/{sp}/{s}.jpg", cut["image"], JPG)
        open(f"{args.out}/labels/{sp}/{s}.txt", "w").write("\n".join(new) + ("\n" if new else ""))
        n_img[sp] += 1
        counts[sp].update(int(l.split()[0]) for l in new)
        if i % 50 == 0:
            print(f"{i}/{len(stems)}", flush=True)

    with open(f"{args.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: {len(names)}\nnames: {names}\n")
    report = [f"images written: {dict(n_img)}   (skipped, no face found: {len(no_face)})",
              f"boxes dropped (centre outside face skin): {dropped_total}/{in_total}"]
    for sp in ("train", "val", "test"):
        report.append(f"{sp:5} boxes per class: " + ", ".join(f"{names[c]}={counts[sp][c]}" for c in range(len(names))))
    report.append("no face: " + ", ".join(no_face))
    open(f"{args.out}/prep_report.txt", "w").write("\n".join(report) + "\n")
    print("\n".join(report[:-1]))


if __name__ == "__main__":
    main()
