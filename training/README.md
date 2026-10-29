# Acne detector training (RF-DETR small, trained locally)

A local retrain of the acne detector, because the Roboflow-hosted model's weights could not be downloaded. **Not Tanaya's original model**; accuracy was measured fresh.

## Get the trained model (not on GitHub: the files are over the 100 MB limit)
`rfdetr-small.onnx` (123 MB) and `acne_rfdetr_small.pth` (128 MB) are sent directly by Harsh (folder `FaceVeda_acne_detector`). Check the sha256 in `notes.txt`, and put the ONNX in `models/`.

## Results (71 never-seen test photos, IoU 0.5), see `notes.txt` and `test_results.json`
Best balance at confidence 0.30: about 42% of detections correct, about 39% of lesions found. Classes: comedone, nodule, papule, pustule (no cyst).
Measured on ACNE04 close-ups only, not on the 3-view patient photos.

## Reproduce
1. Get the data: Roboflow project `tanayaaa/acne-detection-v1-tjjbs`, version 2, export **YOLOv8** (the dataset is CC BY 4.0, and is not in this repo).
2. `python3.12 -m venv venv && ./venv/bin/pip install "rfdetr[train,loggers,onnx]"`
3. `python3 make_1x_dataset.py <export_dir> dataset_1x`
4. `./run_training.sh` (20 epochs, resumes after a crash; about 8 min per epoch on an Apple M5 with 16 GB, MPS). Watch it with `./watch.sh`.
5. Export: `RFDETRSmall(pretrain_weights="output/checkpoint_best_total.pth", num_classes=4).export(output_dir="final")` gives `final/rfdetr-small.onnx`.
6. Score: `./venv/bin/python eval_test.py test` (expects `final/rfdetr-small.onnx` and `dataset_1x/test`).

## Notes learned the hard way
- On macOS the training script needs the `if __name__ == "__main__"` guard (data-loader workers are spawned).
- On a 16 GB Mac use batch 2 with gradient accumulation 8 and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85`; the default memory use swapped the machine to a halt, and 0.6 ran out of GPU memory.
- ONNX output: `dets` (1,300,4) = cx,cy,w,h normalised; `labels` (1,300,5) logits, apply sigmoid; classes are channels 0-3, no offset. Input: RGB, stretched to 512x512, ImageNet mean/std.

## What would actually improve it (all training happens on Harsh's Mac; nobody needs a GPU)
- **The curve has plateaued**: validation mAP@50 reached about 0.29 at epoch 6 and stayed there to epoch 20 (`training_metrics.csv`). More epochs on this data will not help.
- **More and different data is the lever.** Tanaya's 5-class `ACNE_COMBINED_DATASET` (with cyst; 6976 train / 1220 val / 989 test) is about 2x the data. She sends it to Harsh privately (zip, not GitHub); Harsh trains on it here (a few hours).
- **Photos like FaceVeda's.** All 71 test photos are ACNE04 close-ups, so accuracy on the 3-view patient photos is unmeasured. Labelled patient photos would be the most useful data of all.
- The images in the Roboflow export are already shrunk to 512x512, so training at a higher resolution on them would add no detail.
