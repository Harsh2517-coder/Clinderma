"""Builds dataset_1x/ from a Roboflow YOLOv8 export (train/valid/test with images/ and labels/): keeps ONE copy per source photo in train
(Roboflow's 3x augmentation makes near-identical flip/brightness copies, 3x slower per epoch for no new information); valid and test are copied as they are.
    python3 make_1x_dataset.py <roboflow_export_dir> dataset_1x"""
import os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
seen = set(); os.makedirs(f"{dst}/train/images", exist_ok=True); os.makedirs(f"{dst}/train/labels", exist_ok=True)
for f in sorted(os.listdir(f"{src}/train/images")):
    base = f.split("_jpg.rf.")[0]
    if base in seen: continue
    seen.add(base); stem = os.path.splitext(f)[0]
    shutil.copy(f"{src}/train/images/{f}", f"{dst}/train/images/{f}"); shutil.copy(f"{src}/train/labels/{stem}.txt", f"{dst}/train/labels/{stem}.txt")
for s in ("valid", "test"): shutil.copytree(f"{src}/{s}", f"{dst}/{s}", dirs_exist_ok=True)
open(f"{dst}/data.yaml", "w").write("train: train/images\nval: valid/images\ntest: test/images\nnc: 4\nnames: ['comedone', 'nodule', 'papule', 'pustule']\n")
print("train photos kept:", len(seen))
